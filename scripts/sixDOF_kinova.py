import sys
sys.path.insert(0, '/home/grant/Documents/funrobo-mini-proj-2')

from math import *
from typing import List, Tuple
import numpy as np
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import KinovaRobotTemplate

class KinovaRobot(KinovaRobotTemplate):
    def __init__(self):
        super().__init__()

    def calc_inverse_kinematics(
        self, ee: ut.EndEffector, joint_values: List[float], soln: int = 0
    ) -> List[float]:
        """
        Compute an analytical inverse kinematics solution.

        Given a desired end-effector pose, this method computes a set of joint
        angles that achieve the pose, if a valid solution exists.

        Args:
            ee (EndEffector): Desired end-effector pose.
            joint_values (List[float]): Initial or previous joint angles, used
                for solution selection or continuity.
            soln (int, optional): Solution branch index (e.g., elbow-up vs
                elbow-down). Defaults to 0.

        Returns:
            list[float]: Joint angles [theta1, theta2] in radians that achieve
            the desired end-effector pose.
        """
        l1,l2,l3,l4,l5,l6,l7 = self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7

        curr_ee, _ = self.calc_forward_kinematics(new_joint_values)
        d6 = l6+l7

        w_ee = ee - d6*ee*np.transpose([0,0,1])
        xw = w_ee.x
        yw = w_ee.x
        zw = w_ee.z

        th1 = tan(yw/xw)

        x1 = sqrt(xw^2 + yw^2)
        y1 = zw-l1
        L = sqrt(x1^2 + y1^2)

        beta = acos(l2^2+(l3+l4)^2-L^2,2*l2*(l3+l4))
        th3 = np.pi - beta

        alpha = atan((l3+l4)*sin(th3),l2+(l3+l4)*cos(th3))
        gamma = atan(y1/x1)

        th2 = gamma-alpha
        
        dh_table = np.array([
            [0,0,0,pi],
            [th1,-self.l2, 0, pi/2],
            [th2 + pi/2,0,-self.l3,pi],
            [th3 + pi/2,0,0,pi/2],
        ])

        H03 = self.dh_to_H(dh_table)
        R03 = H03[:3, :3]

        R36 = ut.euler_to_rotm((curr_ee.rotx,curr_ee.roty,curr_ee.rotz)).T @ R03
        

        new_joint_values = joint_values.copy()
        return new_joint_values

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

            # ← also make this an array
            error = np.array([x_error, y_error, z_error,
                             rx_error, ry_error, rz_error])

            if np.linalg.norm(error) < tol/5:
                break

            new_joint_values = new_joint_values + \
                0.1 * self.inverse_jacobian(new_joint_values) @ error

        return new_joint_values

    def dh_to_H(self, dh_table):
        H_list = []
        for i in range(dh_table.shape[0]):
            theta = dh_table[i, 0]
            d = dh_table[i, 1]
            a = dh_table[i, 2]
            alpha = dh_table[i, 3]
            H_list.append(np.array([
                [cos(theta), -sin(theta)*cos(alpha),
                 sin(theta)*sin(alpha), a*cos(theta)],
                [sin(theta), cos(theta)*cos(alpha), -
                 cos(theta)*sin(alpha), a*sin(theta)],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1]
            ]))

            if (i == 0):
                H_ee = H_list[0]
            else:
                # build on previous H
                H_ee = H_ee @ H_list[i]

        return H_ee, H_list

    def calc_forward_kinematics(self, joint_values: list, radians=True):

        dh_table = np.array([
            [0,                          self.l1,           0,        pi],
            [joint_values[0],           -self.l2,            0,  0.5*pi],
            [joint_values[1] + 0.5*pi,  0,           -self.l3,        pi],
            [joint_values[2] + 0.5*pi,  0,                0,  0.5*pi],
            [joint_values[3],           -self.l4 - self.l5,  0, -0.5*pi],
            [joint_values[4],            0,                0,  0.5*pi],
            [joint_values[5],           -self.l6 - self.l7,  0,        pi]
        ])

        H_ee, H_list = self.dh_to_H(dh_table=dh_table)

        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]

        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, H_list

    def calc_jacobians(self, joint_values):
        """
        Calculates Jacobian matrix given joint angles
        """
        # Jacobian will be 6 rows by 5 colums
        k_hat = np.array([0, 0, 1])
        J = np.zeros(shape=(6, 6))

        # Get ee position and H transform matrices
        ee, H_list = self.calc_forward_kinematics(
            joint_values=joint_values, radians=True)

        pos_ee = np.array([ee.x, ee.y, ee.z])

        H_0_current = np.eye(4)
        H_0_current = H_0_current @ H_list[0]  # apply base transform first

        for i, H in enumerate(H_list[1:]):  # ← skip the base row
            R_i = H_0_current[0:3, 0:3]
            d_i = H_0_current[0:3, 3]

            pos_joint_to_ee = pos_ee - d_i
            z_joint = R_i @ k_hat

            J[0:3, i] = np.cross(z_joint, pos_joint_to_ee)
            J[3:6, i] = z_joint

            H_0_current = H_0_current @ H

        # print(f"\nMy Jacobian was: {J}\n")
        return J

    def inverse_jacobian(self, joint_values):
        """
        Inverts a provided Jacobian matrix using numpy
        """
        return np.linalg.pinv(self.calc_jacobians(joint_values))


if __name__ == "__main__":
    model = KinovaRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()
