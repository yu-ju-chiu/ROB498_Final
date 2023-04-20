import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym


class CartpoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cartpole = None
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (6,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (6,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        p.resetSimulation()
        # p.setAdditionalSearchPath(pd.getDataPath())
        self.cartpole = p.loadURDF('cartpole2.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 2, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (6,) representing cartpole state [x, theta, theta2, x_dot, theta_dot, theta2_dot]

        """

        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot = p.getJointState(self.cartpole, 1)[0:2]
        theta2, theta2_dot = p.getJointState(self.cartpole, 2)[0:2]
        return np.array([x, theta, theta2, x_dot, theta_dot, theta2_dot])

    def set_state(self, state):
        x, theta, theta2, x_dot, theta_dot, theta2_dot = state
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta, targetVelocity=theta_dot)
        p.resetJointState(self.cartpole, 2, targetValue=theta2, targetVelocity=theta2_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(low=np.array([x_lims[0], theta_lims[0], theta_lims[0], x_dot_lims[0], theta_dot_lims[0], theta_dot_lims[0]]),
                                     high=np.array([x_lims[1], theta_lims[1], theta_lims[1], x_dot_lims[1], theta_dot_lims[1], theta_dot_lims[1]])) # linear force 
        # TODO: Verifty that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 480
        self.render_w = 640
        base_pos = [0, 0, 0]
        cam_dist = 4
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (6,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (6, 6) representing Jacobian df/dx for dynamics f
            B: np.array of shape (6, 1) representing Jacobian df/du for dynamics f
        """
        A, B = None, None
        # --- Your code here
        idx1 = np.array([1, 0, 0, 0, 0, 0])
        idx2 = np.array([0, 1, 0, 0, 0, 0])
        idx3 = np.array([0, 0, 1, 0, 0, 0])
        idx4 = np.array([0, 0, 0, 1, 0, 0])
        idx5 = np.array([0, 0, 0, 0, 1, 0])
        idx6 = np.array([0, 0, 0, 0, 0, 1])
        A1 = (self.dynamics(state+eps*idx1, control)-self.dynamics(state-eps*idx1, control)) /2 /eps
        A2 = (self.dynamics(state+eps*idx2, control)-self.dynamics(state-eps*idx2, control)) /2 /eps
        A3 = (self.dynamics(state+eps*idx3, control)-self.dynamics(state-eps*idx3, control)) /2 /eps
        A4 = (self.dynamics(state+eps*idx4, control)-self.dynamics(state-eps*idx4, control)) /2 /eps
        A5 = (self.dynamics(state+eps*idx5, control)-self.dynamics(state-eps*idx5, control)) /2 /eps
        A6 = (self.dynamics(state+eps*idx6, control)-self.dynamics(state-eps*idx6, control)) /2 /eps
        B = (self.dynamics(state, control+eps)-self.dynamics(state, control-eps)) /2 /eps
        A = np.vstack((A1, A2, A3, A4, A5, A6))
        A = A.T
        B = B[:, None]
        # ---
        return A, B