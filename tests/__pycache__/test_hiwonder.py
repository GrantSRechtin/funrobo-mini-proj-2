import random, math
import pytest
import yaml

import funrobo_kinematics.core.utils as ut


#TODO Import your robot model script
# ---------------------------------------
from scripts.fiveDOF_hiwonder import FiveDOFRobot
# ---------------------------------------


robot_model = FiveDOFRobot()
N = 100 # number of sample tries


# -----------------------------------------------------------------------------
# Test description
# -----------------------------------------------------------------------------
# - This script uses pytest to run unit tests on the inverse and forward kinematics 
#   functions we implement.
#
# - STEPS FOR INVERSE KINEMATICS:
#   1. Make sure you import the right robot model class into the script
#   2. The script randomly generates N number of valid joint position and end-effector pose pairs
#   3. For each pair, it computes the joint values using the IK given the end-effector pose
#   4. The test checks if the computed solution is valid


joint_values_list = [ut.sample_valid_joints(robot_model) for _ in range(N)]
ee_list = []

for joint_values in joint_values_list:
    ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)
    ee_list.append([float(ee.x), float(ee.y)])

ids = [f"joint_values_{i}={[round(x,2) for x in q]}" for i, q in enumerate(joint_values_list)]


# -----------------------------------------------------------------------------
# Python test for analytical inverse kinematics
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("joint_values", joint_values_list, ids=ids)
def test_analytical_ik(joint_values):
    ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)

    init_joint_values = ut.sample_valid_joints(robot_model)
    new_joint_values = robot_model.calc_inverse_kinematics(ee, init_joint_values, soln=0)

    assert ut.check_valid_ik_soln(new_joint_values, ee, robot_model)


# -----------------------------------------------------------------------------
# Python test for numerical inverse kinematics
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("joint_values", joint_values_list, ids=ids)
def test_numerical_ik(joint_values):
    ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)

    init_joint_values = ut.sample_valid_joints(robot_model)
    new_joint_values = robot_model.calc_numerical_ik(ee, init_joint_values)

    assert ut.check_valid_ik_soln(new_joint_values, ee, robot_model)


# @pytest.mark.parametrize("joint_values", joint_values_list, ids=ids)
# def test_numerical_ik_restarts(joint_values):
#     ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)

#     init_joint_values = ut.sample_valid_joints(robot_model)
#     new_joint_values = robot_model.calc_numerical_ik_restarts(ee, init_joint_values, 0.001)

#     assert ut.check_valid_ik_soln(new_joint_values, ee, robot_model)


# -----------------------------------------------------------------------------
# Python test for forward position kinematics
# -----------------------------------------------------------------------------

# with open('tests/data/two_dof_fk_test_data.yaml', 'r') as file:
#     data = yaml.safe_load(file)
#     joint_values_list = data['joint_values']
#     ee_list = data['ee']

# @pytest.mark.parametrize(
#     "joint_values, ee",
#     list(zip(joint_values_list, ee_list)),
#     ids=[f"position_{i}={[round(x,2) for x in ee_list[i]]}" for i, q in enumerate(joint_values_list)]
# )
# def test_forward_kinematics(joint_values, ee):
#     new_ee, _ = robot_model.calc_forward_kinematics(joint_values, radians=True)

#     assert [new_ee.x, new_ee.y, new_ee.z] == pytest.approx(ee, abs=1e-3)