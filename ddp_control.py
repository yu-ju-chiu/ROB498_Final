import torch
from dynamics import *

class DDPController(object):

    def __init__(self, env, horizon):
        """
        :param env: Simulation environment. Must have an action_space and a state_space.
        :param horizon: <int> Number of control steps into the future
        """
        self.env = env
        self.T = horizon
        self.action_size = env.action_space.shape[-1]
        self.state_size = env.state_space.shape[-1]
        self.goal_state = torch.zeros(self.state_size)  # This is just a container for later use
        # self.Q = torch.diag(torch.tensor([20.0, 0.5, 5.0, 1.0, 1.0, 1.0]))
        self.Q = torch.diag(10*torch.tensor([5.0, 5.0, 5.0, 0.1, 0.1, 0.1]))
        self.R = torch.diag(torch.tensor([0.01]))
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.X = torch.zeros((self.T, self.state_size))
        self.x_init = torch.zeros(self.state_size)
        self.k = torch.zeros((self.T, self.action_size))
        self.K = torch.zeros((self.T, self.action_size, self.state_size))

    def reset(self, state):
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.X = self._rollout_dynamics(state, actions=self.U) # nominal state sequence (T, state_size)

    def command(self, state):
        """
        Run a DDP step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return: action: torch tensor of shape (1,)
        """
        self._backward_pass()
        # line search
        initial_cost =  self._compute_trajectory_cost(self.X, self.U)
        # print("initial", initial_cost)
        eps=1
        if initial_cost.item() is 0:
            x, u = self._foward_pass(state, eps)
        else:
            x, u = self._foward_pass(state, eps)
            # after_cost = self._compute_trajectory_cost(x, u)
            # # print(after_cost)
            # min_cost, min_idx = after_cost, 10
            # for i in reversed(range(10)):
            #     x,u = self._foward_pass(state, i*0.1*eps)
            #     after_cost = self._compute_trajectory_cost(x, u)
            #     # print(after_cost)
            #     if after_cost.item() < min_cost.item():
            #         # print("less than")
            #         min_cost = after_cost
            #         min_idx = i
            # # print(min_cost, min_idx)
            # x, u = self._foward_pass(state, min_idx*0.1*eps)
        self.X = x
        self.U = u 
        # print("result", self._compute_trajectory_cost(self.X, self.U))

        # select optimal action
        action = self.U[0]
        # final update nominal trajectory
        self.U = torch.roll(self.U, -1, dims=0) # move u_t to u_{t-1}
        self.U[-1] = self.u_init # Initialize new end action
        self.X = torch.roll(self.X, -1, dims=0) # move x_t to x_{t-1}
        self.X[-1] = self.x_init # Initialize new end state
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        Roll out the environment dynamics from state_0 and taking the control actions given by actions
        :param state_0: torch tensor of shape (state_size,)
        :param actions: torch tensor of shape (T, action_size)
        :return:
         * trajectory: torch tensor of shape (T, state_size) containing the states along the trajectories given by
                       starting at state_0 and taking actions.
                       This tensor contains 1 trajectories of T length.
        """
        state_1 = self._dynamics(state_0, actions[0])
        trajectory = state_1.unsqueeze(1)
        for t in range(1, self.T):
          state = self._dynamics(state_1, actions[t])
          trajectory = torch.hstack((trajectory, state.unsqueeze(1)))
          state_1 = state
        return trajectory.T

    def _backward_pass(self):
        V_0 = torch.zeros(self.T+1)
        V_x = torch.zeros((self.state_size, self.T+1))
        V_xx = torch.zeros((self.state_size, self.state_size, self.T+1))

        for i in reversed(range(self.T-1)):
            mu_1 = 0.1
            mu_2 = 0.1
            f_x, f_u, f_xx, f_ux, f_uu = self.find_f_gredient(self.X[i], self.U[i])
            L_x, L_u, L_xx, L_xu, L_uu = self.find_cost_gredient(self.X[i], self.U[i])
            # print(L_x.shape, f_x.T.shape, V_x[:, i+1:i+2].shape)
            Q_x = L_x + f_x.T @ V_x[:, i+1:i+2]
            # print(L_u.shape, f_u.T.shape, V_x[:, i+1:i+2].shape)
            Q_u = L_u + f_u.T @ V_x[:, i+1:i+2]
            # print(L_xx.shape, f_x.T.shape, V_xx[:, :, i+1].shape, V_x[:, i+1:i+2].shape, f_xx.shape)
            Q_xx = L_xx + f_x.T @ V_xx[:, :, i+1] @ f_x + (V_x[:, i+1:i+2].T @ f_xx).squeeze()
            # print(L_xu.shape, f_u.T.shape, V_xx[:, :, i+1].shape, V_x[:, i+1:i+2].shape, f_ux.shape)
            Q_ux = L_xu.T + f_u.T @ V_xx[:, :, i+1] @ f_x + V_x[:, i+1:i+2].T @ f_ux
            # print(L_uu.shape, f_u.T.shape, V_xx[:, :, i+1].shape, V_x[:, i+1:i+2].shape, f_uu.shape)
            Q_uu = L_uu + f_u.T @ V_xx[:, :, i+1] @ f_u + V_x[:, i+1:i+2].T @ f_uu
            # # v2
            # Q_xx = L_xx + f_x.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_x + (V_x[:, i+1:i+2].T @ f_xx).squeeze()
            # Q_ux = L_xu.T + f_u.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_x + V_x[:, i+1:i+2].T @ f_ux
            # Q_uu = L_uu + f_u.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_u + V_x[:, i+1:i+2].T @ f_uu + mu_2 * torch.eye(1)

            inv_Q_uu = None
            while inv_Q_uu is None:
                try:
                    inv_Q_uu = torch.linalg.inv(Q_uu)
                except:
                    mu_1 += 0.1
                    mu_2 += 0.1
                    Q_xx = L_xx + f_x.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_x
                    Q_ux = L_xu.T + f_u.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_x
                    Q_uu = L_uu + f_u.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_u + mu_2 * torch.eye(1)
            # Q_xx = L_xx + f_x.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_x
            # Q_ux = L_xu.T + f_u.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_x
            # Q_uu = L_uu + f_u.T @ (V_xx[:, :, i+1] + mu_1 * torch.eye(6)) @ f_u + mu_2 * torch.eye(1)
            # inv_Q_uu = torch.linalg.inv(Q_uu)
            # print(inv_Q_uu, Q_u, Q_ux, L_uu)
            # lower_b, upper_b= -10-self.U, 5-self.U
            # num = 50
            # min_tar = 100000
            # argmin_du = 0
            # for j in range(num):
            #     du = lower_b[i] + (upper_b[i]-lower_b[i])*j/num
            #     tar = du @ Q_uu @ du /2 + torch.mean(Q_x @ du)
            #     if tar<min_tar: argmin_du = du
            # self.k[i] = argmin_du
            self.k[i] = -inv_Q_uu @ Q_u
            self.K[i] = -inv_Q_uu @ Q_ux

            V_0[i] = V_0[i+1] - Q_u.T @ inv_Q_uu @ Q_u /2 # +L_0
            # print(V_x[:, i].shape, Q_x.shape, Q_ux.T.shape, inv_Q_uu.shape, Q_u.shape)
            V_x[:, i] = (Q_x - Q_ux.T @ inv_Q_uu @ Q_u).squeeze()
            V_xx[:, :, i] = Q_xx - Q_ux.T @ inv_Q_uu @ Q_ux

    def _foward_pass(self, state, eps):
        x = torch.zeros_like(self.X)
        u = torch.zeros_like(self.U)
        x[0] = state
        lr = 1.0
        for i in range(self.T-1):
            dx = x[i] - self.X[i]
            # print(self.k[i], self.K[i], dx)
            u[i] = self.U[i] + eps*self.k[i] + lr*self.K[i] @ dx
            # print(self.U[i])
            x[i+1] = self._dynamics(x[i], u[i])
        return x, u

    # def _compute_cost(self, states, actions):
    #     """
    #     Compute the costs for the K different trajectories
    #     :param state: torch tensor of shape (state_size)
    #     :param action: torch tensor of shape (action_size)
    #     :return:
    #      - total_trajectory_cost: torch tensor of shape (1,) containing the total trajectory costs for the K trajectories
    #     Observations:
    #     * The trajectory cost be the sum of the state costs and action costs along the trajectories
    #     * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
    #     * Action costs "TODO"
    #     """
    #     total_cost = None
    #     # Define the state cost weight matrix Q
    #     Q = self.Q

    #     # Define the control cost weight matrix R
    #     R = self.R
    #     # Compute the state costs
    #     actions = actions[:, None]
    #     states = states[:, None]

    #     state_errors = states - self.goal_state[:, None]
    #     state_costs = state_errors.T @ Q @ state_errors
    #     control_costs = actions.T @ R @ actions
    #     total_cost = state_costs + control_costs
    #     # ---
    #     return total_cost

    def _compute_cost(self, states, actions):
        Q = self.Q
        R = self.R
        state_errors = states - self.goal_state
        state_costs = 0
        for i in range(Q.shape[0]):
            state_costs += state_errors[i]**2 * Q[i,i]
        control_costs = actions**2 * R
        total_cost = state_costs + control_costs
        return total_cost

    def _compute_trajectory_cost(self, trajectory_states, trajectory_acrions):
        """
        Compute the costs for the K different trajectories
        :param state: torch tensor of shape (T, state_size)
        :param action: torch tensor of shape (T, action_size)
        :return:
         - total_trajectory_cost: torch tensor of shape (1,) containing the total trajectory costs for the K trajectories
        Observations:
        * The trajectory cost be the sum of the state costs and action costs along the trajectories
        * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
        * Action costs "TODO"
        """
        # Define the state cost weight matrix Q
        Q = self.Q

        # Define the control cost weight matrix R
        R = self.R

        # Compute the state costs
        state_errors = trajectory_states - self.goal_state[None, :].repeat(self.T, 1)
        state_costs = torch.sum(state_errors @ Q @ state_errors.T, axis=1)
        control_costs = torch.sum(trajectory_acrions @ R @ trajectory_acrions.T, axis=1)

        # Compute the total cost
        total_cost = state_costs + control_costs
        total_cost = torch.sum(total_cost)

        return total_cost

    def _dynamics(self, state, action):
        """
        Query the environment dynamics to obtain the next_state in a batched format.
        :param state: torch tensor of size (...., state_size)
        :param action: torch tensor of size (..., action_size)
        :return: next_state: torch tensor of size (..., state_size)
        """
        next_state = self.env.dynamics(state.cpu().detach().numpy(), action.cpu().detach().numpy())
        next_state = torch.tensor(next_state, dtype=state.dtype)
        return next_state
    
    def find_f_gredient(self, state, action):
        f_x, f_u = self.env.linearize_numerical(state, action)
        f_x = torch.from_numpy(f_x).type(torch.float32)
        f_u = torch.from_numpy(f_u).type(torch.float32)
        # f_x, f_u = linearize_pytorch(state, action)
        f = self.env.dynamics

        eps = 1e-3
        f_xx = torch.zeros(6,6,6)
        # idx = torch.eye(6)
        # for i in range(6):
        #     for j in range(6):
        #         f1 = f(state+eps*idx[i]+eps*idx[j], action)
        #         f2 = f(state+eps*idx[i]-eps*idx[j], action)
        #         f3 = f(state-eps*idx[i]+eps*idx[j], action)
        #         f4 = f(state-eps*idx[i]-eps*idx[j], action)
        #         f_xx[i, j] = torch.from_numpy(f1-f2-f3+f4) /4 /eps /eps
        
        f_xu = torch.zeros(6,6)
        # for i in range(6):
        #     f1 = f(state+eps*idx[i], action+eps)
        #     f2 = f(state+eps*idx[i], action-eps)
        #     f3 = f(state-eps*idx[i], action+eps)
        #     f4 = f(state-eps*idx[i], action-eps)
        #     f_xu[i] = torch.from_numpy(f1-f2-f3+f4) /4 /eps /eps

        f_uu = torch.zeros(6,1)
        # idx = torch.eye(6)
        # f1 = -f(state, action+2*eps)
        # f2 = 16*f(state, action+eps)
        # f3 = -30*f(state, action)
        # f4 = 16*f(state, action-eps)
        # f5 = -f(state, action-2*eps)
        # f_uu[:, 0] = torch.from_numpy(f1+f2+f3+f4+f5)/12/eps/eps
        
        return f_x, f_u, f_xx, f_xu, f_uu


    # def find_cost_gredient(self, state, action):
    #     # Set the finite difference step size
    #     eps = 1e-3
    #     f = self._compute_cost
    #     idx1 = torch.tensor([1, 0, 0, 0, 0, 0])
    #     idx2 = torch.tensor([0, 1, 0, 0, 0, 0])
    #     idx3 = torch.tensor([0, 0, 1, 0, 0, 0])
    #     idx4 = torch.tensor([0, 0, 0, 1, 0, 0])
    #     idx5 = torch.tensor([0, 0, 0, 0, 1, 0])
    #     idx6 = torch.tensor([0, 0, 0, 0, 0, 1])
    #     A1 = (f(state+eps*idx1, action)-f(state-eps*idx1, action)) /2 /eps
    #     A2 = (f(state+eps*idx2, action)-f(state-eps*idx2, action)) /2 /eps
    #     A3 = (f(state+eps*idx3, action)-f(state-eps*idx3, action)) /2 /eps
    #     A4 = (f(state+eps*idx4, action)-f(state-eps*idx4, action)) /2 /eps
    #     A5 = (f(state+eps*idx5, action)-f(state-eps*idx5, action)) /2 /eps
    #     A6 = (f(state+eps*idx6, action)-f(state-eps*idx6, action)) /2 /eps
    #     L_u = (f(state, action+eps)-f(state, action-eps)) /2 /eps
    #     L_x = torch.vstack((A1, A2, A3, A4, A5, A6))
    #     # print(L_x.shape, L_u.shape)

    #     L_xx = torch.zeros(6,6)
    #     L_xu = torch.zeros(6,1)
    #     L_uu = torch.zeros(1,1)
    #     return L_x, L_u, L_xx, L_xu, L_uu

    def find_cost_gredient(self, state, action):
        # Set the finite difference step size
        # eps = 1e-3
        # idx = torch.eye(6)
        f = self._compute_cost
        # L_xx = torch.zeros(6, 6)
        # for i in range(state.shape[0]):
        #     for j in range(state.shape[0]):
        #         L_xx[i,j] = (f(state + eps*idx[i] + eps*idx[j], action)\
        #             - f(state + eps*idx[i] - eps*idx[j], action)\
        #             - f(state - eps*idx[i] + eps*idx[j], action)\
        #             + f(state - eps*idx[i] - eps*idx[j], action)) /4 /eps /eps
        # L_xu = torch.zeros(6,1)
        # for i in range(state.shape[0]):
        #     for j in range(action.shape[0]):
        #         L_xu[i,j] = (f(state + eps*idx[i], action + eps)\
        #             - f(state + eps*idx[i], action - eps)\
        #             - f(state - eps*idx[i], action + eps)\
        #             + f(state - eps*idx[i], action - eps)) /4 /eps /eps
        # L_uu = torch.zeros(1,1)
        # for i in range(action.shape[0]):
        #     for j in range(action.shape[0]):
        #         L_xu[i,j] = (f(state + eps*idx[i], action + eps)\
        #             - f(state + eps*idx[i], action - eps)\
        #             - f(state - eps*idx[i], action + eps)\
        #             + f(state - eps*idx[i], action - eps)) /4 /eps /eps

        # idx1 = torch.tensor([1, 0, 0, 0, 0, 0])
        # idx2 = torch.tensor([0, 1, 0, 0, 0, 0])
        # idx3 = torch.tensor([0, 0, 1, 0, 0, 0])
        # idx4 = torch.tensor([0, 0, 0, 1, 0, 0])
        # idx5 = torch.tensor([0, 0, 0, 0, 1, 0])
        # idx6 = torch.tensor([0, 0, 0, 0, 0, 1])
        # A1 = (f(state+eps*idx1, action)-f(state-eps*idx1, action)) /2 /eps
        # A2 = (f(state+eps*idx2, action)-f(state-eps*idx2, action)) /2 /eps
        # A3 = (f(state+eps*idx3, action)-f(state-eps*idx3, action)) /2 /eps
        # A4 = (f(state+eps*idx4, action)-f(state-eps*idx4, action)) /2 /eps
        # A5 = (f(state+eps*idx5, action)-f(state-eps*idx5, action)) /2 /eps
        # A6 = (f(state+eps*idx6, action)-f(state-eps*idx6, action)) /2 /eps
        # L_u = (f(state, action+eps)-f(state, action-eps)) /2 /eps
        # L_x = torch.vstack((A1, A2, A3, A4, A5, A6))
        # print(L_x.shape, L_u.shape)

        # L_xx = torch.zeros(6,6)
        # L_xu = torch.zeros(6,1)
        # L_uu = torch.zeros(1,1)
        J = torch.autograd.functional.jacobian(f, (state, action))
        L_x = J[0].reshape(6,1)
        L_u = J[1].reshape(1,1)
        H = torch.autograd.functional.hessian(f, (state, action))
        L_xx = H[0][0]
        L_xu = H[0][1]
        L_uu = H[1][1]
        return L_x, L_u, L_xx, L_xu, L_uu