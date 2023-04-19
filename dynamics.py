import torch

def dynamics_analytic(state, action):
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
    dt = 0.05
    g = 9.81
    mc = 1
    mp = 0.1
    l = 0.5
    
    m0 = 1
    m1 = 0.1
    m2 = 0.1
    L1 = 0.5
    L2 = 0.5
    l1 = L1/2
    l2 = L2/2
    I1 = m1*L1**2/12
    I2 = m2*L2**2/12

    # --- Your code here
    # find accel
    state
    
    
    d1 = m0 + m1 + m2
    d2 = m1 * l1 + m2 * L1
    d3 = m2 * l2
    d4 = m1 * l1**2 + m2 * L2**2 + I1
    d5 = m2 * L1 * l2
    d6 = m2 * l2**2 + I2
    d7 = (m1 * l1 + m2 * L1) * g
    d8 = m2 * l2 * g
    d1, d2, d3 = 0.1, 0.1, 0.1

    batched_state = []

    for i in range(state.shape[0]):
        x, t1, t2, x_d, t1_d, t2_d = state[i]

        D = torch.tensor([[d1, d2 * torch.cos(t1), d3 * torch.cos(t2)],
                            [d2 * torch.cos(t1), d4, d5 * torch.cos(t1 - t2)],
                            [d3 * torch.cos(t2), d5 * torch.cos(t1 - t2), d6]])
        
        C = torch.tensor([[0, -d2 * torch.sin(t1) * t1_d, -d3 * torch.sin(t2) * t2_d],
                            [0, 0, d5 * torch.sin(t1 - t2) * t2_d],
                            [0, -d5 * torch.sin(t1 - t2) * t1_d, 0]])
        
        G = torch.tensor([[0], [-d7 * torch.sin(t1)], [-d8 * torch.sin(t2)]])
        
        # G = torch.tensor([[d2 * torch.sin(t1) * t1_d**2 + d3 * torch.sin(t2) * t2_d**2 + action[i] ],\
        #                   [-d5 * torch.sin(t1 - t2) * t2_d**2 + d7 * torch.sin(t1)],\
        #                   [d5 * torch.sin(t1 - t2) * t1_d**2 + d8 * torch.sin(t2)]], dtype=torch.float32)

        H = torch.tensor([[1.0], [0], [0]])
        u = action[i].type(torch.float32)

        state_dd = -torch.linalg.inv(D) @ C @state[i][3:6][:,None] \
                    -torch.linalg.inv(D) @ G + torch.linalg.inv(D) @ (H*u).type(torch.float32)
        # state_dd = torch.linalg.inv(D) @ G

        # A = torch.cat([torch.cat([torch.zeros(3, 3), torch.eye(3)], dim=1),
        #               torch.cat([torch.zeros(3, 3), -torch.linalg.inv(D) @ C], dim=1)], dim=0)
        
        # B = torch.cat([torch.zeros(3, 1), torch.linalg.inv(D) @ H], dim=0)

        # L = torch.cat([torch.zeros(3, 1), -torch.linalg.inv(D) @ G], dim=0)
        # print(A)
        # state_d = (A @ state[i][:, None] + B * u + L).squeeze()
        # state_d = torch.hstack((state_d[3:6], state_d[0:3]))
        # state_update = state[i] + state_d * dt
        # batched_state.append(state_update)

        dx = state[i][3] + dt * state_dd[0]
        dt1 = state[i][4] + dt * state_dd[1]
        dt2 = state[i][5] + dt * state_dd[2]
        x = state[i][0] + dt * dx
        t1 = state[i][1] + dt * dt1
        t2 = state[i][2] + dt * dt2
        
        next_state_single = torch.tensor([x, t1, t2, dx, dt1, dt2])
        batched_state.append(next_state_single)


    next_state = torch.vstack(batched_state)
    # ---

    return next_state