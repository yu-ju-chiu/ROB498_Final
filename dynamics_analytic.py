def dynamics_analytic(state, action):
    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 4) representing the cartpole state
        control: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 4) representing the next cartpole state

    """
    next_state = None
    M = 1000
    m1 = 100
    m2 = 100
    l1 = 1
    l2 = 1
    g = 9.8

    # state
    x, theta_1, theta_2, dx, dtheta_1, dtheta_2 = torch.chunk(state, 6, 1)
    # print(x, theta, dx, dtheta)
    st1 = torch.sin(theta_1)
    ct2 = torch.cos(theta_1)
    st2 = torch.sin(theta_2)
    ct2 = torch.cos(theta_2)

    # Compute accelerations
    # tacc = (g * st - ct * (action + mp * l * dtheta ** 2 * st) / mt) / (l * ((4.0 / 3.0) - ((mp * ct ** 2) / mt)))
    # xacc = (action + mp * l * (dtheta ** 2 * st - tacc * ct)) / mt
    F = action
    xacc =  F + m1 * l1 * dtheta_1**2 * st1 + m2 * l2 * dtheta_2**2 * st2 *- m2 * l1 * dtheta_1 ** 2 * torch.cos(theta_1-theta_2)\
            * torch.sin(theta_1 - theta_2) - (m1 + m2) * g * st1 - m2 * l2 * dtheta_2**2 * torch.cos(theta_1 - theta_2) * torch.sin(theta_1 - theta_2)
    t1acc = g * st1 + torch.cos(theta_1 - theta_2) * (((l1 * dtheta_1 * torch.sin(theta_1 - theta_2))\
        - ( g * st2 + l2 * (dtheta_2)**2 * torch.sin(theta_1-theta_2)))
    t2acc = g * st2 + l1 * dtheta_1**2 * torch.sin(theta_1 - theta_2) + torch.cos(theta_1 - theta_2) * ((m1 - m2) * g\
        *st1 +l2 * (m2 * dtheta_2**2 * torch.sin(theta_1 - theta_2)))
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
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (4,) representing cartpole state
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
