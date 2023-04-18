import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym
from torch.autograd.functional import jacobian

def dynamics_analytic(state, action):
    """
    Function to define the system of ordinary differential equations (ODEs) for the double pendulum on a cart.
    
    Args:
        t (float): Current time.
        y (array): Array containing the current values of the state variables [x, dx/dt, theta1, d(theta1)/dt, theta2, d(theta2)/dt].
        m1 (float): Mass of pendulum 1.
        m2 (float): Mass of pendulum 2.
        l1 (float): Length of pendulum 1.
        l2 (float): Length of pendulum 2.
        g (float): Acceleration due to gravity.
    
    Returns:
        dydt (array): Array containing the derivatives of the state variables [dx/dt, d^2x/dt^2, d(theta1)/dt, d^2(theta1)/dt^2, d(theta2)/dt, d^2(theta2)/dt^2].
    """
    next_state = None
    M = 1
    m1 = 0.1
    m2 = 0.1
    L1 = 0.5
    L2 = 0.5
    g = 9.8
    dt = 0.05
    # x, dxdt, theta1, dtheta1dt, theta2, dtheta2dt = y
    x, theta1, theta2, dx, dtheta1, dtheta2 = torch.chunk(state, 6, 1)

    Mt = torch.tensor([[ M+m1+m2,                  L1*(m1+m2)*torch.cos(theta1),     m2*L2*torch.cos(theta2)],\
          [L1*(m1+m2)*torch.cos(theta1),     L1**2*(m1+m2),             L1*L2*m2*torch.cos(theta1-theta2)],\
          [L2*m2*torch.cos(theta2),          L1*L2*m2*torch.cos(theta1-theta2),  L2**2*m2        ]], dtype=torch.float64)

    f1 = L1*(m1+m2)*dtheta1**2*torch.sin(theta1) + m2*L2*theta2**2*torch.sin(theta2) + action 
    f2 = -L1*L2*m2*dtheta2**2*torch.sin(theta1-theta2) + g*(m1+m2)*L1*torch.sin(theta1)
    f3 =  L1*L2*m2*dtheta1**2*torch.sin(theta1-theta2) + g*L2*m2*torch.sin(theta2)

    xacc, t1acc, t2acc = torch.linalg.inv(Mt) @ torch.tensor([[f1],[f2],[f3]], dtype=torch.float64) 
    # Update velocities.
    dx = dx + xacc * dt
    dtheta1 = dtheta1 + t1acc * dt
    dtheta2 = dtheta2 + t2acc * dt

    # Update position/angle.
    x = x + dt * dx
    theta1= theta1 + dt * dtheta1
    theta2 = theta2 + dt * dtheta2                 
    next_state = torch.cat([x, theta1, theta2, dx, dtheta1, dtheta2], dim = 1)
    # dydt = [dxdt, ddxdtdt, dtheta1dt, ddtheta1dtdt, dtheta2dt, ddtheta2dtdt]
    
    return next_state

def linearize_pytorch(state, control):
    """f shape (4,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

    """
    A, B = None, None
    # --- Your code here
    state = torch.reshape(state, (1, 6))
    control = torch.reshape(control, (1, 1))
    A, B = jacobian(dynamics_analytic,(state,control))
    A, B = A[0, :, 0, :,0, :], B[0, :, 0, :,0, :]
    # ---
    return A, B
