# main.py
"""
Main Application Script
----------------------------
Example code for the MP1 RRMC implementation

Edited to add FiveDOFRobot class that implements calc_velocity_kinematics
"""

import time
from math import cos, sin, pi
import numpy as np
import traceback

from funrobo_hiwonder.core.hiwonder import HiwonderRobot
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate
import funrobo_kinematics.core.utils as ut

class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
        #hw_robot = HiwonderRobot()
        #self.joint_limits = hw_robot.joint_limits
        self.joint_limits = [[-120, 120],[-90,90],[-120,120],[-100,100],[-90,90],[-120,30]]
        self.joint_limits = [[val[0] * pi / 180, val[1] * pi / 180] for val in self.joint_limits]
        print(f"\n\njoint limits {self.joint_limits}\n\n")
        #del(hw_robot)
    
    def dh_to_H(self,dh_table):
        H_list = []
        for i in range(dh_table.shape[0]):
            theta = dh_table[i,0]
            d = dh_table[i,1]
            a = dh_table[i,2]
            alpha = dh_table[i,3]
            H_list.append(np.array([
                [cos(theta),-sin(theta)*cos(alpha),sin(theta)*sin(alpha),a*cos(theta)],
                [sin(theta),cos(theta)*cos(alpha),-cos(theta)*sin(alpha),a*sin(theta)],
                [0,sin(alpha),cos(alpha),d],
                [0,0,0,1]
            ]))

            if (i == 0):
                H_ee = H_list[0]
            else:
                # build on previous H
                H_ee = H_ee @ H_list[i]
        
        return H_ee, H_list

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        dh_table = np.array([
            [joint_values[0],          self.l1,            0,        0.5 * pi],
            [joint_values[1] - 0.5*pi, 0,                 -self.l2,  pi      ],
            [joint_values[2],          0,                 -self.l3,  pi      ],
            [joint_values[3] + 0.5*pi, 0,                  0,       -0.5*pi  ],
            [joint_values[4],          self.l4 + self.l5,  0,        0       ],
        ])

        H_ee, H_list = self.dh_to_H(dh_table=dh_table)

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]

        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_list
    
    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy, vz].
        """
        new_joint_values = joint_values.copy()

        J = self.calc_jacobians(new_joint_values)
        Jv = J[0:3,:] # Only includes linear velocity, shape (3,5)

        # Shift away from singularities using damped inverse
        lam = 0.05
        JJt = Jv @ Jv.T # shape (3,3)
        JJt = JJt + (lam**2 * np.eye(3))
        J_inv_damped = Jv.T @ self.inv_jacobian(JJt)
        joint_vel = J_inv_damped @ vel

        joint_vel = np.clip(joint_vel, 
                            [limit[0] for limit in self.joint_vel_limits], 
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        print(f"Commanded linear vel: {vel}, resulting joint vel (rad): {joint_vel}")

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        print(f"Almost final joint values sent: {new_joint_values}")
        new_joint_values = np.clip(new_joint_values, 
                               [limit[0] for limit in self.joint_limits], 
                               [limit[1] for limit in self.joint_limits]
                            )
        
        print(f"Final joint values sent: {new_joint_values}")
        return new_joint_values
    
    def calc_numerical_ik(self, ee, joint_values, tol=0.02, ilimit=1000):
        # Numerical IK
        n = ilimit
        eps = tol
        attempts = 1000

        p_des = np.array([ee.x,ee.y,ee.z])
        for j in range(attempts):
            if j == 0:
                # Start with our current joint values
                curr_joint_vals = np.array(joint_values[:5])
            else:
                # After that, do a random guess
                curr_joint_vals = np.array(ut.sample_valid_joints(self))[:5]

            for i in range(n):
                p_ee, _ = self.calc_forward_kinematics(curr_joint_vals)
                err = p_des - np.array([p_ee.x, p_ee.y, p_ee.z]) # error vector (position)

                if np.linalg.norm(err) < eps:
                    print(f"Found solution at iter {i} with error {np.linalg.norm(err)}")
                    return curr_joint_vals
                
                J = self.calc_jacobians(curr_joint_vals)
                J = J[0:3,0:5]
                
                # Damped inverse
                lam = 0.001
                JJt = J @ J.T
                JJt = JJt + (lam**2 * np.eye(3))
                J_inv = J.T @ np.linalg.pinv(JJt)
                step = J_inv @ err

                curr_joint_vals = curr_joint_vals + step
                if not ut.check_joint_limits(curr_joint_vals, self.joint_limits):
                    break
                    
        print("Failed to converge")
        return None

    def calc_jacobians(self,joint_values):
        """
        Calculates Jacobian matrix given joint angles
        """
        # Jacobian will be 6 rows by 5 colums
        k_hat = np.array([0,0,1])
        J = np.zeros(shape=(6,5))

        # Get ee position and H transform matrices
        ee, H_list = self.calc_forward_kinematics(joint_values=joint_values,radians=True)

        pos_ee = np.array([ee.x, ee.y, ee.z])

        H_0_current = np.eye(4)
        for i,H in enumerate(H_list):
            R_i = H_0_current[0:3,0:3]
            d_i = H_0_current[0:3,3]
            
            pos_joint_to_ee = pos_ee - d_i # lever arm of joint to ee
            z_joint = R_i @ k_hat # axis of rotation of joint

            Jv = np.cross(z_joint,pos_joint_to_ee) # cross product is jacobian
            Jw = z_joint

            J[0:3,i] = Jv
            J[3:6,i] = Jw

            H_0_current = H_0_current @ H
        
        return J
    
    def inv_jacobian(self,J):
        """
        Inverts a provided Jacobian matrix using numpy
        """
        return np.linalg.pinv(J)

def follow_waypts(model, robot, waypt_list):
    """
    Use numerical IK to command the Hiwonder to follow a given path of waypoints.
    First pre-computes all IK solutions, then executes them sequentially.
    """
    ee, _ = model.calc_forward_kinematics(robot.get_joint_values())
    home = np.array([0.2, 0, 0.3])
    print(home)

    curr_joint_values_full = robot.get_joint_values()  # deg
    curr_joint_values_rad_full = [v * pi / 180 for v in curr_joint_values_full]  # rad
    prev_joints_5dof = curr_joint_values_rad_full[:5]

    solutions = []
    for i in range(len(waypt_list)):
        waypt = waypt_list[i, :] / 1000  # mm to m
        waypt = home + waypt
        print(waypt)

        ee_target = ut.EndEffector()
        ee_target.x, ee_target.y, ee_target.z = waypt[0], waypt[1], waypt[2]

        t_compute = time.perf_counter()
        commanded_joints = model.calc_numerical_ik(ee_target, prev_joints_5dof, tol=0.002, ilimit=1000)

        if commanded_joints is None:
            solutions.append(None)
        else:
            commanded_joints_full = list(commanded_joints) + [curr_joint_values_rad_full[5]]
            solutions.append(commanded_joints_full)
            prev_joints_5dof = commanded_joints  # warm-start next solve

    for i, joints in enumerate(solutions):
        robot.set_joint_values(joints, duration=2, radians=True)
        time.sleep(2)


def main():
    """ Main loop that reads gamepad commands and updates the robot accordingly. """
    try:

        # Initialize components
        robot = HiwonderRobot()
        model = FiveDOFRobot()

        curr_joint_values = None # Initialize to none
        
        control_hz = 20 
        dt = 1 / control_hz
        t0 = time.time()
        waypts_square = np.array([[0,-60,-60],[0,-60,60],[0,60,60],[0,60,-60],[0,-60,-60]])
        waypts_star = np.array([[0,-38,-60],[0,0,60],[0,38,-60],[0,-60,13],[0,60,13],[0,-38,-60]])
        waypts_IN = np.array([[0,60,60],[0,60,-60],[0,0,-60],[0,0,60],[0,-60,-60],[0,-60,60]])

        while True:
            t_start = time.time()

            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break

            # joints = robot.get_joint_values()[:5]

            # ee_target = ut.EndEffector()
            # ee_target.x, ee_target.y, ee_target.z = 0.2, 0, 0.2

            # commanded_joints = model.calc_numerical_ik(ee_target, joints, tol=0.002, ilimit=1000)
            # robot.set_joint_values(list(commanded_joints) + [0], duration=4, radians=True)

            # time.sleep(4)

            # Square
            follow_waypts(model,robot,waypts_square)
            time.sleep(2)

            # Star
            follow_waypts(model,robot,waypts_star)
            time.sleep(2)

            # IN
            follow_waypts(model,robot,waypts_IN)
            time.sleep(2)

            # if robot.gamepad.cmdlist:
            #     cmd = robot.gamepad.cmdlist[-1]

            #     if cmd.arm_home:
            #         robot.move_to_home_position()
                
            #     if curr_joint_values is None:
            #         curr_joint_values = robot.get_joint_values() # deg
            #         curr_joint_values_rad = [v * pi / 180 for v in curr_joint_values] # rad
                
                

            #     #curr_joint_values = robot.get_joint_values()

            #     ### Convert to radians 
            #     #curr_joint_values = [v * pi / 180 for v in curr_joint_values]

            #     vel = [cmd.arm_vx, cmd.arm_vy, cmd.arm_vz]
            #     curr_joint_values_rad = model.calc_velocity_kinematics(curr_joint_values_rad, vel)
            #     curr_joint_values = [v * 180 / pi for v in curr_joint_values_rad]
            #     ### Convert to degrees
            #     #new_joint_values = [v * 180 / pi for v in new_joint_values]

            #     # set new joint angles
            #     print(f"Final values sent (deg): {curr_joint_values}")
            #     robot.set_joint_values(curr_joint_values, duration=dt, radians=False)

            # elapsed = time.time() - t_start
            # remaining_time = dt - elapsed
            # if remaining_time > 0:
            #     time.sleep(remaining_time)

            
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        robot.shutdown_robot()




if __name__ == "__main__":
    main()