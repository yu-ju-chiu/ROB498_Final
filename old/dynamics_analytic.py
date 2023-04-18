import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym
from torch.autograd.functional import jacobian
def dynamics_analytic_ex(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 6) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 6) representing the next cartpole state

    """
    next_state = None
    M = 1
    p = 0.1
    q = 0.1
    lp = 0.5
    lq = 0.5
    g = 9.8
    dt = 0.05

    # state
    x, theta_1, theta_2, dx, dtheta_1, dtheta_2 = torch.chunk(state, 6, 1)
    # print(x, theta, dx, dtheta)
    st1 = torch.sin(theta_1)
    ct1 = torch.cos(theta_1)
    st2 = torch.sin(theta_2)
    ct2 = torch.cos(theta_2)

    # Compute accelerations
    # tacc = (g * st - ct * (action + mp * l * dtheta ** 2 * st) / mt) / (l * ((4.0 / 3.0) - ((mp * ct ** 2) / mt)))
    # xacc = (action + mp * l * (dtheta ** 2 * st - tacc * ct)) / mt
    # F = action
    # xacc =  (F + m1 * l1 * dtheta_1**2 * st1 + m2 * l2 * dtheta_2**2 * st2 *- m2 * l1 * dtheta_1 ** 2 * torch.cos(theta_1-theta_2)\
    #         * torch.sin(theta_1 - theta_2) - (m1 + m2) * g * st1 - m2 * l2 * dtheta_2**2 * torch.cos(theta_1 - theta_2)\
    #              * torch.sin(theta_1 - theta_2)) / M
    # t1acc = (g * st1 + torch.cos(theta_1 - theta_2) * ((l1 * dtheta_1 * torch.sin(theta_1 - theta_2))\
    #     - ( g * st2 + l2 * (dtheta_2)**2 * torch.sin(theta_1-theta_2)))) / l1
    # t2acc = (g * st2 + l1 * dtheta_1**2 * torch.sin(theta_1 - theta_2) + torch.cos(theta_1 - theta_2) * ((m1 - m2) * g\
    #     *st1 +l2 * (m2 * dtheta_2**2 * torch.sin(theta_1 - theta_2)))) / l2
    D = (p**2) + 2*M*p+M*q+p*q-M*q*torch.cos(2*theta_1-2*theta_2)-p*q*torch.cos(2*theta_1)-(p**2)*torch.cos(2*theta_1)
    xacc =  -((2*lp*(p**2)*st1)+(2*lp*p*q*st1))*(dtheta_1**2)/D\
        -(p*q*lp*t*lp*torch.sin(theta_1-theta_2)-2*M*q*lq*torch.sin(theta_1-theta_2))*dtheta_1**2/(lq*D)\
        +(M*q*lq*torch.sin(2*theta_1-2*theta_2))*dtheta_2**2/(lq*D)\
        -(M*p*torch.sin(2*theta_1-theta_2)-M*p*st2+M*q*torch.sin(2*theta_1-theta_2)-M*q*st2)*g/(lq*D)\
        +(M*p*ct2-M*p*torch.cos(2*theta_1-theta_2)+M*q*ct2-M*q*torch.cos(2*theta_1-theta_2))*F/(lq*D)
    
    t1acc = ((lp*p**2*torch.sin(2*theta_1)) + lq*M*q*torch.sin(2*theta_1-2*theta_2))*dtheta_1**2/(lp*D)\
        -(p*q*lp*torch.sin(theta_1+theta_2)+p*q*lq*torch.sin(theta_1-theta_2)+2*M*q*lq*torch.sin(theta_1-theta_2))*dtheta_2**2/(lp*D)\
        -(M*p*(-2*st1-M*q*st1-2*p**2*st1-2*p*q*st1))-M*q*torch.sin(theta_1-2*theta_2)*g/(lp*D)\
        +(2*M*p*ct1+M*q*ct1-M*q*torch.cos(theta_1-theta_2))*F/(lp*D)
    t2acc = -(-2*M*p*lp*torch.sin(theta_1-theta_2)-2*M*q*lq*torch.sin(theta_1-theta_2))*dtheta_1**2/(lq*D)\
        +(M*q*lq*torch.sin(2*theta_1-2*theta_2))*dtheta_2**2/(lq*D)\
        -(M*p*torch.sin(2*theta_1-theta_2)-M*p*st2+M*q*torch.sin(2*theta_1-theta_2)-M*q*st2)*g/(lq*D)\
        +(M*p*ct2-M*p*torch.cos(2*theta_1-theta_2)+M*q*ct2-M*q*torch.cos(2*theta_1-theta_2))*F/(lq*D)
    
    
    #### be careful the order

    # Update velocities.
    dx = dx + xacc * dt
    dtheta_1 = dtheta_1 + t1acc * dt
    dtheta_2 = dtheta_2 + t2acc * dt

    # Update position/angle.
    x = x + dt * dx
    theta_1= theta_1 + dt * dtheta_1
    theta_2 = theta_2 + dt * dtheta_2



    next_state = torch.cat([x, theta_1, theta_2, dx, dtheta_1, dtheta_2], dim = 1)
    # ---

    return next_state

def dynamics_analytic_ex(state, action):
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
    l1 = 0.5
    l2 = 0.5
    g = 9.8
    dt = 0.05
    # x, dxdt, theta1, dtheta1dt, theta2, dtheta2dt = y
    x, theta1, theta2, dx, dtheta1dt, dtheta2dt = torch.chunk(state, 6, 1)
    
    # Equations of motion for the cart
    ddxdtdt = ((m1 * l1 * (dtheta1dt ** 2) * torch.sin(theta1)) +
               (m2 * l2 * (dtheta2dt ** 2) * torch.sin(theta2)) +
               (m2 * l1 * (dtheta1dt ** 2) * torch.cos(theta1 - theta2) * torch.sin(theta1 - theta2))) / \
              (m1 + m2 - (m1 + m2) * torch.cos(theta1 - theta2) ** 2)
    
    # Equations of motion for pendulum 1
    ddtheta1dtdt = ((-g * (m1 + m2) * torch.sin(theta1)) -
                   (m2 * l1 * (dtheta1dt ** 2) * torch.sin(theta1 - theta2) * torch.cos(theta1 - theta2))) / \
                  (l1 * (m1 + m2 - (m1 + m2) * torch.cos(theta1 - theta2) ** 2))
    
    # Equations of motion for pendulum 2
    ddtheta2dtdt = ((-g * m2 * torch.sin(theta2)) +
                   (m2 * l2 * (dtheta2dt ** 2) * torch.sin(theta1 - theta2) * torch.cos(theta1 - theta2))) / \
                  (l2 * (m1 + m2 - (m1 + m2) * torch.cos(theta1 - theta2) ** 2))
    # Update velocities.
    dx = dx + ddxdtdt * dt
    dtheta1dt = dtheta1dt + ddtheta1dtdt * dt
    dtheta2dt = dtheta2dt + ddtheta2dtdt * dt

    # Update position/angle.
    x = x + dt * dx
    theta1= theta1 + dt * dtheta1dt
    theta2 = theta2 + dt * dtheta2dt                 
    next_state = torch.cat([x, theta1, theta2, dx, dtheta1dt, dtheta2dt], dim = 1)
    # dydt = [dxdt, ddxdtdt, dtheta1dt, ddtheta1dtdt, dtheta2dt, ddtheta2dtdt]
    
    return next_state

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
