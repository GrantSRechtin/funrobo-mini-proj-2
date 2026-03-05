import sys
sys.path.insert(0, '/home/grant/Documents/funrobo-mini-proj-2')

from math import *
import numpy as np
from typing import List, Tuple
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate

class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    
    def calc_numerical_ik(
        self, ee: ut.EndEffector, joint_values: List[float], tol: float = 0.01, ilimit: int = 100
    ) -> List[float]:
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (list[float]): Initial guess for joint angles.
            tol (float, optional): Convergence tolerance on pose/position error. Defaults to 0.01.
            ilimit (int, optional): Maximum number of iterations. Defaults to 100.

        Returns:
            list[float]: Estimated joint angles in radians.
        """
        new_joint_values = np.array(joint_values, dtype=float)

        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = np.array([
                theta + np.random.rand() * 0.02 for theta in new_joint_values])

        for i in range(ilimit*5):
            curr_ee, _ = self.calc_forward_kinematics(new_joint_values)
            x_error = ee.x - curr_ee.x
            y_error = ee.y - curr_ee.y
            z_error = ee.z - curr_ee.z
            rx_error = ee.rotx - curr_ee.rotx
            ry_error = ee.roty - curr_ee.roty
            rz_error = ee.rotz - curr_ee.rotz

            error = np.array([x_error, y_error, z_error, rx_error, ry_error, rz_error])  # ← also make this an array

            if np.linalg.norm(error) < tol/5:
                break

            new_joint_values = new_joint_values + 0.1 * self.inv_jacobian(self.calc_jacobians(new_joint_values)) @ error

        return new_joint_values
    
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
            [joint_values[0],self.l1, 0, 0.5 * pi],
            [joint_values[1] - 0.5*pi,0,-self.l2,pi],
            [joint_values[2],0,-self.l3,pi],
            [joint_values[3] + 0.5*pi,0,0,-0.5*pi],
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

            print(f"Jv: {Jv}, Jw: {Jw}")

            J[0:3,i] = Jv
            J[3:6,i] = Jw

            H_0_current = H_0_current @ H
        
        print(f"\nMy Jacobian was: {J}\n")
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