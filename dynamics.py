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
    
    m0 = 1
    m1 = 0.1
    m2 = 0.1
    L1 = 1.0
    L2 = 1.0
    l1 = L1/2
    l2 = L2/2
    I1 = m1*L1**2/12
    I2 = m2*L2**2/12

    x, t1, t2, x_d, t1_d, t2_d = torch.chunk(state, 6, 1)

    F = action.type(torch.float64)

    # x_dd = (4*F + 12*F*torch.cos(t2)**2 + 12*F*torch.sin(t2)**2 + 6*L2*m2*t2_d**2*torch.sin(t2)**3 + \
    #         2*L1*m1*t1_d**2*torch.sin(t1) + 2*L1*m2*t1_d**2*torch.sin(t1) + 2*L2*m2*t2_d**2*torch.sin(t2) - \
    #         3*g*m1*torch.cos(t1)*torch.sin(t1) - 3*g*m2*torch.cos(t1)*torch.sin(t1) - 12*g*m2*torch.cos(t2)*torch.sin(t2) + \
    #         6*L1*m1*t1_d**2*torch.cos(t2)**2*torch.sin(t1) + 6*L2*m2*t2_d**2*torch.cos(t2)**2*torch.sin(t2) + \
    #         6*L1*m1*t1_d**2*torch.sin(t1)*torch.sin(t2)**2 + 6*L1*m2*t1_d**2*torch.sin(t1)*torch.sin(t2)**2 - \
    #         9*g*m1*torch.cos(t1)*torch.cos(t2)**2*torch.sin(t1) - 9*g*m1*torch.cos(t1)*torch.sin(t1)*torch.sin(t2)**2 - \
    #         9*g*m2*torch.cos(t1)*torch.sin(t1)*torch.sin(t2)**2)/(4*m0 + 4*m1 + 4*m2 + 12*m0*torch.cos(t2)**2 - \
    #         3*m1*torch.cos(t1)**2 + 12*m1*torch.cos(t2)**2 - 3*m2*torch.cos(t1)**2 + 12*m0*torch.sin(t2)**2 + \
    #         12*m1*torch.sin(t2)**2 + 12*m2*torch.sin(t2)**2 - 9*m1*torch.cos(t1)**2*torch.cos(t2)**2 - \
    #         9*m1*torch.cos(t1)**2*torch.sin(t2)**2 - 9*m2*torch.cos(t1)**2*torch.sin(t2)**2)
    
    # t1_dd = -(3*(2*F*torch.cos(t1) - 2*g*m0*torch.sin(t1) - 2*g*m1*torch.sin(t1) - 2*g*m2*torch.sin(t1) + \
    #         6*F*torch.cos(t1)*torch.cos(t2)**2 + 6*F*torch.cos(t1)*torch.sin(t2)**2 - 6*g*m0*torch.cos(t2)**2*torch.sin(t1) - \
    #         6*g*m1*torch.cos(t2)**2*torch.sin(t1) - 6*g*m0*torch.sin(t1)*torch.sin(t2)**2 - 6*g*m1*torch.sin(t1)*torch.sin(t2)**2 - \
    #         6*g*m2*torch.sin(t1)*torch.sin(t2)**2 + 3*L2*m2*t2_d**2*torch.cos(t1)*torch.sin(t2)**3 + \
    #         L1*m1*t1_d**2*torch.cos(t1)*torch.sin(t1) + L1*m2*t1_d**2*torch.cos(t1)*torch.sin(t1) + \
    #         L2*m2*t2_d**2*torch.cos(t1)*torch.sin(t2) - 6*g*m2*torch.cos(t1)*torch.cos(t2)*torch.sin(t2) + \
    #         3*L1*m1*t1_d**2*torch.cos(t1)*torch.cos(t2)**2*torch.sin(t1) + 3*L2*m2*t2_d**2*torch.cos(t1)*torch.cos(t2)**2*torch.sin(t2) + \
    #         3*L1*m1*t1_d**2*torch.cos(t1)*torch.sin(t1)*torch.sin(t2)**2 + \
    #         3*L1*m2*t1_d**2*torch.cos(t1)*torch.sin(t1)*torch.sin(t2)**2))/(L1*(4*m0 + 4*m1 + 4*m2 + 12*m0*torch.cos(t2)**2 - \
    #         3*m1*torch.cos(t1)**2 + 12*m1*torch.cos(t2)**2 - 3*m2*torch.cos(t1)**2 + 12*m0*torch.sin(t2)**2 + 12*m1*torch.sin(t2)**2 + \
    #         12*m2*torch.sin(t2)**2 - 9*m1*torch.cos(t1)**2*torch.cos(t2)**2 - 9*m1*torch.cos(t1)**2*torch.sin(t2)**2 - \
    #         9*m2*torch.cos(t1)**2*torch.sin(t2)**2))

    # t2_dd = (3*(8*g*m0*torch.sin(t2) - 8*F*torch.cos(t2) + 8*g*m1*torch.sin(t2) + 8*g*m2*torch.sin(t2) + \
    #         6*F*torch.cos(t1)**2*torch.cos(t2) - 6*g*m1*torch.cos(t1)**2*torch.sin(t2) - 6*g*m2*torch.cos(t1)**2*torch.sin(t2) + \
    #         4*L1*m0*t1_d**2*torch.cos(t2)*torch.sin(t1) - 4*L2*m2*t2_d**2*torch.cos(t2)*torch.sin(t2) - \
    #         6*g*m0*torch.cos(t1)*torch.cos(t2)*torch.sin(t1) + 3*L2*m2*t2_d**2*torch.cos(t1)**2*torch.cos(t2)*torch.sin(t2)))/(4*L2*m0 + \
    #         4*L2*m1 + 4*L2*m2 + 12*L2*m0*torch.cos(t2)**2 - 3*L2*m1*torch.cos(t1)**2 + 12*L2*m1*torch.cos(t2)**2 - 3*L2*m2*torch.cos(t1)**2 + \
    #         12*L2*m0*torch.sin(t2)**2 + 12*L2*m1*torch.sin(t2)**2 + 12*L2*m2*torch.sin(t2)**2 - 9*L2*m1*torch.cos(t1)**2*torch.cos(t2)**2 - \
    #         9*L2*m1*torch.cos(t1)**2*torch.sin(t2)**2 - 9*L2*m2*torch.cos(t1)**2*torch.sin(t2)**2)
    
    x_dd = (F*I1*I2 + F*I2*L1**2*m2 + F*I2*l1**2*m1 + F*I1*l2**2*m2 + F*L1**2*l2**2*m2**2 + F*l1**2*l2**2*m1*m2 - F*L1**2*l2**2*m2**2*torch.cos(t1 - t2)**2 + L1**3*l2**2*m2**3*t1_d**2*torch.sin(t1) + L1**2*l2**3*m2**3*t2_d**2*torch.sin(t2) + I2*L1**3*m2**2*t1_d**2*torch.sin(t1) + I2*l1**3*m1**2*t1_d**2*torch.sin(t1) + I1*l2**3*m2**2*t2_d**2*torch.sin(t2) - I2*L1**2*g*m2**2*torch.cos(t1)*torch.sin(t1) + I1*L1*l2**2*m2**2*t1_d**2*torch.sin(t1) + I2*L1**2*l2*m2**2*t2_d**2*torch.sin(t2) + I1*I2*L1*m2*t1_d**2*torch.sin(t1) - I2*g*l1**2*m1**2*torch.cos(t1)*torch.sin(t1) + I1*g*l2**2*m2**2*torch.cos(t2)*torch.sin(t2) + L1**2*l2**3*m2**3*t2_d**2*torch.sin(t1 - t2)*torch.cos(t1) - L1**3*l2**2*m2**3*t1_d**2*torch.sin(t1 - t2)*torch.cos(t2) + I1*I2*l1*m1*t1_d**2*torch.sin(t1) + I1*I2*l2*m2*t2_d**2*torch.sin(t2) - L1**3*l2**2*m2**3*t1_d**2*torch.cos(t1 - t2)**2*torch.sin(t1) - L1**2*l2**3*m2**3*t2_d**2*torch.cos(t1 - t2)**2*torch.sin(t2) - L1**2*g*l2**2*m2**3*torch.cos(t1)*torch.sin(t1) + L1**2*g*l2**2*m2**3*torch.cos(t2)*torch.sin(t2) + l1**3*l2**2*m1**2*m2*t1_d**2*torch.sin(t1) + l1**2*l2**3*m1*m2**2*t2_d**2*torch.sin(t2) + L1*l1**2*l2**2*m1*m2**2*t1_d**2*torch.sin(t1) + L1**2*l1*l2**2*m1*m2**2*t1_d**2*torch.sin(t1) - I1*L1*l2**2*m2**2*t1_d**2*torch.sin(t1 - t2)*torch.cos(t2) + I2*L1**2*l2*m2**2*t2_d**2*torch.sin(t1 - t2)*torch.cos(t1) + I2*L1*l1**2*m1*m2*t1_d**2*torch.sin(t1) + I2*L1**2*l1*m1*m2*t1_d**2*torch.sin(t1) - g*l1**2*l2**2*m1**2*m2*torch.cos(t1)*torch.sin(t1) + g*l1**2*l2**2*m1*m2**2*torch.cos(t2)*torch.sin(t2) + L1**3*l2**2*m2**3*t1_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2)*torch.cos(t1) - L1**2*l2**3*m2**3*t2_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2)*torch.cos(t2) + I1*l1*l2**2*m1*m2*t1_d**2*torch.sin(t1) + I2*l1**2*l2*m1*m2*t2_d**2*torch.sin(t2) - L1**2*g*l2**2*m2**3*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t2) + L1**2*g*l2**2*m2**3*torch.cos(t1 - t2)*torch.cos(t2)*torch.sin(t1) - L1*l1**2*l2**2*m1*m2**2*t1_d**2*torch.sin(t1 - t2)*torch.cos(t2) - 2*L1*g*l1*l2**2*m1*m2**2*torch.cos(t1)*torch.sin(t1) - 2*I2*L1*g*l1*m1*m2*torch.cos(t1)*torch.sin(t1) - L1**2*l1*l2**2*m1*m2**2*t1_d**2*torch.cos(t1 - t2)**2*torch.sin(t1) + L1*l1*l2**3*m1*m2**2*t2_d**2*torch.sin(t1 - t2)*torch.cos(t1) + L1**2*l1*l2**2*m1*m2**2*t1_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2)*torch.cos(t1) - L1*g*l1*l2**2*m1*m2**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t2) + L1*g*l1*l2**2*m1*m2**2*torch.cos(t1 - t2)*torch.cos(t2)*torch.sin(t1) + I2*L1*l1*l2*m1*m2*t2_d**2*torch.sin(t1 - t2)*torch.cos(t1))/(I2*L1**2*m2**2 + I2*l1**2*m1**2 + I1*l2**2*m2**2 + I1*I2*m0 + I1*I2*m1 + I1*I2*m2 + L1**2*l2**2*m2**3 - L1**2*l2**2*m2**3*torch.cos(t1 - t2)**2 + L1**2*l2**2*m0*m2**2 + L1**2*l2**2*m1*m2**2 + I2*L1**2*m0*m2 + I2*L1**2*m1*m2 + l1**2*l2**2*m1*m2**2 + l1**2*l2**2*m1**2*m2 + I2*l1**2*m0*m1 + I1*l2**2*m0*m2 + I1*l2**2*m1*m2 + I2*l1**2*m1*m2 - I2*L1**2*m2**2*torch.cos(t1)**2 - I2*l1**2*m1**2*torch.cos(t1)**2 - I1*l2**2*m2**2*torch.cos(t2)**2 - L1**2*l2**2*m2**3*torch.cos(t1)**2 - L1**2*l2**2*m2**3*torch.cos(t2)**2 + l1**2*l2**2*m0*m1*m2 - l1**2*l2**2*m1**2*m2*torch.cos(t1)**2 - l1**2*l2**2*m1*m2**2*torch.cos(t2)**2 - L1**2*l2**2*m0*m2**2*torch.cos(t1 - t2)**2 - L1**2*l2**2*m1*m2**2*torch.cos(t1 - t2)**2 - 2*L1*l1*l2**2*m1*m2**2*torch.cos(t1)**2 + 2*L1**2*l2**2*m2**3*torch.cos(t1 - t2)*torch.cos(t1)*torch.cos(t2) - 2*I2*L1*l1*m1*m2*torch.cos(t1)**2 + 2*L1*l1*l2**2*m1*m2**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.cos(t2))

    t1_dd = -(L1*l2**3*m2**3*t2_d**2*torch.sin(t1 - t2) - I2*g*l1*m1**2*torch.sin(t1) - I2*L1*g*m2**2*torch.sin(t1) + F*L1*l2**2*m2**2*torch.cos(t1) + F*I2*L1*m2*torch.cos(t1) - L1*g*l2**2*m2**3*torch.sin(t1) + F*I2*l1*m1*torch.cos(t1) + I2*L1*l2*m2**2*t2_d**2*torch.sin(t1 - t2) - I2*L1*g*m0*m2*torch.sin(t1) - I2*L1*g*m1*m2*torch.sin(t1) - g*l1*l2**2*m1*m2**2*torch.sin(t1) - g*l1*l2**2*m1**2*m2*torch.sin(t1) + L1**2*l2**2*m2**3*t1_d**2*torch.cos(t1)*torch.sin(t1) - I2*g*l1*m0*m1*torch.sin(t1) - I2*g*l1*m1*m2*torch.sin(t1) - L1*l2**3*m2**3*t2_d**2*torch.sin(t1 - t2)*torch.cos(t2)**2 - F*L1*l2**2*m2**2*torch.cos(t1 - t2)*torch.cos(t2) + F*l1*l2**2*m1*m2*torch.cos(t1) + L1**2*l2**2*m2**3*t1_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2) - L1*g*l2**2*m2**3*torch.cos(t1 - t2)*torch.sin(t2) + L1*l2**3*m0*m2**2*t2_d**2*torch.sin(t1 - t2) + L1*l2**3*m1*m2**2*t2_d**2*torch.sin(t1 - t2) + I2*L1**2*m2**2*t1_d**2*torch.cos(t1)*torch.sin(t1) + L1*g*l2**2*m2**3*torch.cos(t2)**2*torch.sin(t1) + I2*l1**2*m1**2*t1_d**2*torch.cos(t1)*torch.sin(t1) + L1*l2**3*m2**3*t2_d**2*torch.cos(t1)*torch.sin(t2) - L1*g*l2**2*m0*m2**2*torch.sin(t1) - L1*g*l2**2*m1*m2**2*torch.sin(t1) + L1**2*l2**2*m0*m2**2*t1_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2) + L1**2*l2**2*m1*m2**2*t1_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2) - L1*g*l2**2*m0*m2**2*torch.cos(t1 - t2)*torch.sin(t2) - L1*g*l2**2*m1*m2**2*torch.cos(t1 - t2)*torch.sin(t2) + I2*L1*l2*m0*m2*t2_d**2*torch.sin(t1 - t2) + I2*L1*l2*m1*m2*t2_d**2*torch.sin(t1 - t2) - g*l1*l2**2*m0*m1*m2*torch.sin(t1) - L1**2*l2**2*m2**3*t1_d**2*torch.cos(t1 - t2)*torch.cos(t2)*torch.sin(t1) - L1**2*l2**2*m2**3*t1_d**2*torch.sin(t1 - t2)*torch.cos(t1)*torch.cos(t2) + L1*g*l2**2*m2**3*torch.cos(t1)*torch.cos(t2)*torch.sin(t2) + g*l1*l2**2*m1*m2**2*torch.cos(t2)**2*torch.sin(t1) + l1*l2**3*m1*m2**2*t2_d**2*torch.cos(t1)*torch.sin(t2) + I2*L1*l2*m2**2*t2_d**2*torch.cos(t1)*torch.sin(t2) + l1**2*l2**2*m1**2*m2*t1_d**2*torch.cos(t1)*torch.sin(t1) - L1*l2**3*m2**3*t2_d**2*torch.cos(t1 - t2)*torch.cos(t2)*torch.sin(t2) + 2*L1*l1*l2**2*m1*m2**2*t1_d**2*torch.cos(t1)*torch.sin(t1) + 2*I2*L1*l1*m1*m2*t1_d**2*torch.cos(t1)*torch.sin(t1) + g*l1*l2**2*m1*m2**2*torch.cos(t1)*torch.cos(t2)*torch.sin(t2) + I2*l1*l2*m1*m2*t2_d**2*torch.cos(t1)*torch.sin(t2) - L1*l1*l2**2*m1*m2**2*t1_d**2*torch.cos(t1 - t2)*torch.cos(t2)*torch.sin(t1) - L1*l1*l2**2*m1*m2**2*t1_d**2*torch.sin(t1 - t2)*torch.cos(t1)*torch.cos(t2))/(I2*L1**2*m2**2 + I2*l1**2*m1**2 + I1*l2**2*m2**2 + I1*I2*m0 + I1*I2*m1 + I1*I2*m2 + L1**2*l2**2*m2**3 - L1**2*l2**2*m2**3*torch.cos(t1 - t2)**2 + L1**2*l2**2*m0*m2**2 + L1**2*l2**2*m1*m2**2 + I2*L1**2*m0*m2 + I2*L1**2*m1*m2 + l1**2*l2**2*m1*m2**2 + l1**2*l2**2*m1**2*m2 + I2*l1**2*m0*m1 + I1*l2**2*m0*m2 + I1*l2**2*m1*m2 + I2*l1**2*m1*m2 - I2*L1**2*m2**2*torch.cos(t1)**2 - I2*l1**2*m1**2*torch.cos(t1)**2 - I1*l2**2*m2**2*torch.cos(t2)**2 - L1**2*l2**2*m2**3*torch.cos(t1)**2 - L1**2*l2**2*m2**3*torch.cos(t2)**2 + l1**2*l2**2*m0*m1*m2 - l1**2*l2**2*m1**2*m2*torch.cos(t1)**2 - l1**2*l2**2*m1*m2**2*torch.cos(t2)**2 - L1**2*l2**2*m0*m2**2*torch.cos(t1 - t2)**2 - L1**2*l2**2*m1*m2**2*torch.cos(t1 - t2)**2 - 2*L1*l1*l2**2*m1*m2**2*torch.cos(t1)**2 + 2*L1**2*l2**2*m2**3*torch.cos(t1 - t2)*torch.cos(t1)*torch.cos(t2) - 2*I2*L1*l1*m1*m2*torch.cos(t1)**2 + 2*L1*l1*l2**2*m1*m2**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.cos(t2))

    t2_dd = -(l2*m2*(F*I1*torch.cos(t2) + L1**2*g*m2**2*torch.sin(t2) + g*l1**2*m1**2*torch.sin(t2) + I1*g*m0*torch.sin(t2) + I1*g*m1*torch.sin(t2) + I1*g*m2*torch.sin(t2) + F*L1**2*m2*torch.cos(t2) + F*l1**2*m1*torch.cos(t2) - L1**3*m2**2*t1_d**2*torch.sin(t1 - t2) + L1**2*g*m2**2*torch.cos(t1 - t2)*torch.sin(t1) - L1*l1**2*m1**2*t1_d**2*torch.sin(t1 - t2) + L1**2*g*m0*m2*torch.sin(t2) + L1**2*g*m1*m2*torch.sin(t2) - I1*L1*m0*t1_d**2*torch.sin(t1 - t2) - I1*L1*m1*t1_d**2*torch.sin(t1 - t2) - I1*L1*m2*t1_d**2*torch.sin(t1 - t2) + g*l1**2*m0*m1*torch.sin(t2) + g*l1**2*m1*m2*torch.sin(t2) - L1**2*g*m2**2*torch.cos(t1)**2*torch.sin(t2) + L1**3*m2**2*t1_d**2*torch.cos(t2)*torch.sin(t1) - g*l1**2*m1**2*torch.cos(t1)**2*torch.sin(t2) + l1**3*m1**2*t1_d**2*torch.cos(t2)*torch.sin(t1) - F*L1**2*m2*torch.cos(t1 - t2)*torch.cos(t1) - L1**3*m0*m2*t1_d**2*torch.sin(t1 - t2) - L1**3*m1*m2*t1_d**2*torch.sin(t1 - t2) + L1**3*m2**2*t1_d**2*torch.sin(t1 - t2)*torch.cos(t1)**2 + I1*l1*m1*t1_d**2*torch.cos(t2)*torch.sin(t1) + I1*l2*m2*t2_d**2*torch.cos(t2)*torch.sin(t2) - L1**2*l2*m2**2*t2_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2) + L1*g*l1*m1**2*torch.cos(t1 - t2)*torch.sin(t1) + L1**2*g*m0*m2*torch.cos(t1 - t2)*torch.sin(t1) + L1**2*g*m1*m2*torch.cos(t1 - t2)*torch.sin(t1) - L1*l1**2*m0*m1*t1_d**2*torch.sin(t1 - t2) - L1*l1**2*m1*m2*t1_d**2*torch.sin(t1 - t2) + L1*l1**2*m1**2*t1_d**2*torch.sin(t1 - t2)*torch.cos(t1)**2 - L1**3*m2**2*t1_d**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t1) - F*L1*l1*m1*torch.cos(t1 - t2)*torch.cos(t1) - L1**2*g*m2**2*torch.cos(t1)*torch.cos(t2)*torch.sin(t1) + L1**2*l2*m2**2*t2_d**2*torch.cos(t2)*torch.sin(t2) + I1*L1*m2*t1_d**2*torch.cos(t2)*torch.sin(t1) - g*l1**2*m1**2*torch.cos(t1)*torch.cos(t2)*torch.sin(t1) + l1**2*l2*m1*m2*t2_d**2*torch.cos(t2)*torch.sin(t2) - L1**2*l2*m0*m2*t2_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2) - L1**2*l2*m1*m2*t2_d**2*torch.cos(t1 - t2)*torch.sin(t1 - t2) + L1*g*l1*m0*m1*torch.cos(t1 - t2)*torch.sin(t1) + L1*g*l1*m1*m2*torch.cos(t1 - t2)*torch.sin(t1) + 2*L1**2*l1*m1*m2*t1_d**2*torch.sin(t1 - t2)*torch.cos(t1)**2 - 2*L1*g*l1*m1*m2*torch.cos(t1)**2*torch.sin(t2) - L1*l1**2*m1**2*t1_d**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t1) - L1**2*l2*m2**2*t2_d**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t2) + L1**2*l2*m2**2*t2_d**2*torch.sin(t1 - t2)*torch.cos(t1)*torch.cos(t2) + L1*l1**2*m1*m2*t1_d**2*torch.cos(t2)*torch.sin(t1) + L1**2*l1*m1*m2*t1_d**2*torch.cos(t2)*torch.sin(t1) - 2*L1**2*l1*m1*m2*t1_d**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t1) - 2*L1*g*l1*m1*m2*torch.cos(t1)*torch.cos(t2)*torch.sin(t1) - L1*l1*l2*m1*m2*t2_d**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.sin(t2) + L1*l1*l2*m1*m2*t2_d**2*torch.sin(t1 - t2)*torch.cos(t1)*torch.cos(t2)))/(I2*L1**2*m2**2 + I2*l1**2*m1**2 + I1*l2**2*m2**2 + I1*I2*m0 + I1*I2*m1 + I1*I2*m2 + L1**2*l2**2*m2**3 - L1**2*l2**2*m2**3*torch.cos(t1 - t2)**2 + L1**2*l2**2*m0*m2**2 + L1**2*l2**2*m1*m2**2 + I2*L1**2*m0*m2 + I2*L1**2*m1*m2 + l1**2*l2**2*m1*m2**2 + l1**2*l2**2*m1**2*m2 + I2*l1**2*m0*m1 + I1*l2**2*m0*m2 + I1*l2**2*m1*m2 + I2*l1**2*m1*m2 - I2*L1**2*m2**2*torch.cos(t1)**2 - I2*l1**2*m1**2*torch.cos(t1)**2 - I1*l2**2*m2**2*torch.cos(t2)**2 - L1**2*l2**2*m2**3*torch.cos(t1)**2 - L1**2*l2**2*m2**3*torch.cos(t2)**2 + l1**2*l2**2*m0*m1*m2 - l1**2*l2**2*m1**2*m2*torch.cos(t1)**2 - l1**2*l2**2*m1*m2**2*torch.cos(t2)**2 - L1**2*l2**2*m0*m2**2*torch.cos(t1 - t2)**2 - L1**2*l2**2*m1*m2**2*torch.cos(t1 - t2)**2 - 2*L1*l1*l2**2*m1*m2**2*torch.cos(t1)**2 + 2*L1**2*l2**2*m2**3*torch.cos(t1 - t2)*torch.cos(t1)*torch.cos(t2) - 2*I2*L1*l1*m1*m2*torch.cos(t1)**2 + 2*L1*l1*l2**2*m1*m2**2*torch.cos(t1 - t2)*torch.cos(t1)*torch.cos(t2))


    dx = state[:, 3:4] + dt * x_dd
    dt1 = state[:, 4:5] + dt * t1_dd
    dt2 = state[:, 5:6] + dt * t2_dd
    x = state[:, 0:1] + dt * dx
    t1 = state[:, 1:2] + dt * dt1
    t2 = state[:, 2:3] + dt * dt2

    next_state = torch.hstack((x, t1, t2, dx, dt1, dt2))

    return next_state

def linearize_pytorch(state, control):
    """ shape (6,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (6, 6) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (6, 1) representing Jacobian df/du for dynamics f

    """
    A, B = None, None
    # --- Your code here
    state_B = state[None, :]
    control_B = control[None, :]
    J = torch.autograd.functional.jacobian(dynamics_analytic, (state_B, control_B))
    A = torch.squeeze(J[0])
    B = torch.squeeze(J[1])
    B = B[:, None]
    # ---
    return A, B



# # find accel
# d1 = m0 + m1 + m2
# d2 = m1 * l1 + m2 * L1
# d3 = m2 * l2
# d4 = m1 * l1**2 + m2 * L2**2 + I1
# d5 = m2 * L1 * l2
# d6 = m2 * l2**2 + I2
# d7 = (m1 * l1 + m2 * L1) * g
# d8 = m2 * l2 * g

# batched_state = []
# for i in range(state.shape[0]):
#     x, t1, t2, x_d, t1_d, t2_d = state[i].type(torch.double)

#     # D = torch.tensor([[d1, d2 * torch.cos(t1), d3 * torch.cos(t2)],
#     #                     [d2 * torch.cos(t1), d4, d5 * torch.cos(t1 - t2)],
#     #                     [d3 * torch.cos(t2), d5 * torch.cos(t1 - t2), d6]], dtype=torch.double)
    
#     # C = torch.tensor([[0, -d2 * torch.sin(t1) * t1_d, -d3 * torch.sin(t2) * t2_d],
#     #                     [0, 0, d5 * torch.sin(t1 - t2) * t2_d],
#     #                     [0, -d5 * torch.sin(t1 - t2) * t1_d, 0]], dtype=torch.double)
    
#     # G = torch.tensor([[0], [-d7 * torch.sin(t1)], [-d8 * torch.sin(t2)]], dtype=torch.double)
    
#     # # G = torch.tensor([[d2 * torch.sin(t1) * t1_d**2 + d3 * torch.sin(t2) * t2_d**2 + action[i] ],\
#     # #                   [-d5 * torch.sin(t1 - t2) * t2_d**2 + d7 * torch.sin(t1)],\
#     # #                   [d5 * torch.sin(t1 - t2) * t1_d**2 + d8 * torch.sin(t2)]], dtype=torch.double)

#     # H = torch.tensor([[1.0], [0], [0]], dtype=torch.double)
#     # u = action[i].type(torch.float32)

#     # state_dd = -torch.linalg.inv(D) @ C @state[i][3:6][:,None].type(torch.double) \
#     #             -torch.linalg.inv(D) @ G + torch.linalg.inv(D) @ H*u
    
#     # state_dd = torch.linalg.inv(D) @ G

#     # A = torch.cat([torch.cat([torch.zeros(3, 3), torch.eye(3)], dim=1),
#     #               torch.cat([torch.zeros(3, 3), -torch.linalg.inv(D) @ C], dim=1)], dim=0)
    
#     # B = torch.cat([torch.zeros(3, 1), torch.linalg.inv(D) @ H], dim=0)

#     # L = torch.cat([torch.zeros(3, 1), -torch.linalg.inv(D) @ G], dim=0)
#     # print(A)
#     # state_d = (A @ state[i][:, None] + B * u + L).squeeze()
#     # state_d = torch.hstack((state_d[3:6], state_d[0:3]))
#     # state_update = state[i] + state_d * dt
#     # batched_state.append(state_update)
#     dx = state[i][3] + dt * x_dd
#     dt1 = state[i][4] + dt * t1_dd
#     dt2 = state[i][5] + dt * t2_dd
#     x = state[i][0] + dt * dx
#     t1 = state[i][1] + dt * dt1
#     t2 = state[i][2] + dt * dt2
    
#     next_state_single = torch.tensor([x, t1, t2, dx, dt1, dt2])
#     batched_state.append(next_state_single)


# next_state = torch.vstack(batched_state)