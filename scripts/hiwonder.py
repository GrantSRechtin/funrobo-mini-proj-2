from math import *
import numpy as np
import time
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate


class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    
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

    def calc_forward_kinematics(self,joint_values: list, radians=True):
        
        dh_table = np.array([
            [joint_values[0],self.l1, 0, -0.5 * pi],
            [joint_values[1] - 0.5*pi,0,self.l2,pi],
            [joint_values[2],0,self.l3,pi],
            [joint_values[3] + 0.5*pi,0,0,0.5*pi],
            [joint_values[4],self.l4 + self.l5,0,0],
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

        # move robot slightly out of zeros singularity
        #if all(theta == 0.0 for theta in new_joint_values):
        #    new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate joint velocities using the inverse Jacobian
        # For this, we don't care about rotation, so we want only a 3 element velocity vector
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

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values, 
                               [limit[0] for limit in self.joint_limits], 
                               [limit[1] for limit in self.joint_limits]
                            )
        
        return new_joint_values
    
    def calc_inverse_kinematics(self, ee, joint_values, soln):
        """
        Calculates analytical inverse kinematics for the joint
        """
        # We have 4 solutions
        t0 = time.time()
        sols = []
        errors = []
        for solution in range(4):
            opt1 = (solution >> 1) & 1
            opt2 = solution & 1

            R_0_ee = ut.euler_to_rotm((ee.rotx,ee.roty,ee.rotz))
            p_ee = np.array([ee.x,ee.y,ee.z])
            z_ee = R_0_ee @ np.array([0,0,1]) # Z axis of end effector
            p_wrist = p_ee - (self.l4 + self.l5)*z_ee # Bend in wrist position

            # The joints before the wrist consist only of the two DOF arm on a pivot
            if opt2:
                theta1 = np.atan2(p_wrist[1],p_wrist[0])
            else:
                theta1 = np.atan2(p_wrist[1],p_wrist[0]) + pi
            theta1 = ut.wraptopi(theta1)

            # Shift p_wrist to correspond to translation from joint 1
            p_wrist_transformed = p_wrist - np.array([0,0,self.l1])
            L = np.linalg.norm(p_wrist_transformed)

            X = np.sqrt(p_wrist_transformed[0]**2 + p_wrist_transformed[1]**2)
            cosB = (-L**2 + self.l2**2 + self.l3**2)/(2*self.l2*self.l3)
            beta = np.acos(cosB)

            # Two solutions: arm bends up or down
            if opt1:
                theta3 = pi - beta
            else:
                theta3 = beta - pi
            theta3 = ut.wraptopi(theta3)
            
            gam = np.atan2(p_wrist_transformed[2],X)
            perpendicular = self.l3 * sin(theta3)
            alpha = np.arcsin(perpendicular / L)

            # Different value for theta2 if theta1 is the alternate orientation
            if opt2:
                theta2 = (-(gam - alpha - np.pi / 2))
            if not opt2:
                theta2 = gam + alpha - np.pi / 2

            theta2 = ut.wraptopi(theta2)
            
            # Get the orientation of the 3rd frame (wrist) w.r.t. base frame
            dh_table_partial = np.array([
                [theta1,self.l1, 0, -0.5 * pi],
                [theta2 - 0.5*pi,0,self.l2,pi],
                [theta3,0,self.l3,pi]
            ])

            H_0_3,_ = self.dh_to_H(dh_table_partial)
            R_0_3 = H_0_3[0:3,0:3]

            # We want rotation of EE w.r.t. frame 3 (before wrist)
            R_3_ee = R_0_3.T @ R_0_ee
            
            theta4 = ut.wraptopi(np.atan2(R_3_ee[1,2],R_3_ee[0,2]))
            theta5 = ut.wraptopi(np.atan2(R_3_ee[2,0],R_3_ee[2,1]))

            ee_pose,_ = self.calc_forward_kinematics([theta1,theta2,theta3,theta4,theta5])
            ee_pose_diff = np.array([ee.x - ee_pose.x, ee.y - ee_pose.y, ee.z - ee_pose.z])
            #print(f"Error for sol {soln}: {np.linalg.norm(ee_pose_diff)}")
            #print(f"Returned angles: {[theta1,theta2,theta3,theta4,theta5]}\n")
            if ut.check_valid_ik_soln([theta1,theta2,theta3,theta4,theta5],ee,self,tol=0.002):
                sols.append([theta1,theta2,theta3,theta4,theta5])
                errors.append(np.linalg.norm(ee_pose_diff))
            else:
                print(f"\n\nREJECTED INVALID SOL")
                print([theta1,theta2,theta3,theta4,theta5])
                print(f"error: {np.linalg.norm(ee_pose_diff)}\n\n")
        
        print(f"Error list: {errors}")
        print(f"Solutions: {sols}")
        print(f"Elapsed time: {time.time() - t0}")
        sols_ordered = [s for s, _ in sorted(zip(sols, errors), key=lambda x: x[1])]

        return sols_ordered[soln]

    def calc_numerical_ik(self, ee, joint_values, tol=0.002, ilimit=1000):
        # Numerical IK
        n = ilimit
        eps = tol
        attempts = 1000
        t0 = time.time()

        p_des = np.array([ee.x,ee.y,ee.z])
        for j in range(attempts):
            if j == 0:
                # Start with our current joint values
                curr_joint_vals = joint_values
            else:
                # After that, do a random guess
                curr_joint_vals = ut.sample_valid_joints(self)
            for i in range(n):
                print(f"Curr joint vals: {curr_joint_vals}")
                p_ee, _ = self.calc_forward_kinematics(curr_joint_vals)
                err = p_des - np.array([p_ee.x, p_ee.y, p_ee.z]) # error vector (position)
                print(f"Error {np.linalg.norm(err)}")

                if np.linalg.norm(err) < eps:
                    print(f"Found solution at iter {i} with error {np.linalg.norm(err)}")
                    print(f"Elapsed time: {time.time() - t0}")
                    return curr_joint_vals
                
                J = self.calc_jacobians(curr_joint_vals)
                J = J[0:3,0:5]
                
                # Damped inverse
                lam = 0.001
                JJt = J @ J.T
                JJt = JJt + (lam**2 * np.eye(3))
                J_inv = J.T @ np.linalg.pinv(JJt)
                #J_inv = np.linalg.pinv(J)
                #print(f"jinv shape {J_inv.shape}, err shape {err.shape}")
                step = J_inv @ err
                print(f"err {err}, step {step}")

                curr_joint_vals = curr_joint_vals + step
                #curr_joint_vals = [ut.wraptopi(val) for val in curr_joint_vals]
                if not ut.check_joint_limits(curr_joint_vals, self.joint_limits):
                    break
                    
        print("Failed to converge")
        print(f"Elapsed time: {time.time() - t0}")
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

            #print(f"Jv: {Jv}, Jw: {Jw}")

            J[0:3,i] = Jv
            J[3:6,i] = Jw

            H_0_current = H_0_current @ H
        
        #print(f"\nMy Jacobian was: {J}\n")
        return J
    
    def inv_jacobian(self,J):
        """
        Inverts a provided Jacobian matrix using numpy
        """
        return np.linalg.pinv(J)


if __name__ == "__main__":
    model = FiveDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()