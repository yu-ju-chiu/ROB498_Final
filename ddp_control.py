import torch

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
        self.Q = torch.diag(torch.tensor([5.0, 5.0, 5.0, 0.1, 0.1, 0.1]))
        self.R = torch.diag(torch.tensor([0.0]))
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.X = torch.zeros((self.T, self.state_size))
        self.x_init = torch.zeros(self.state_size)
        self.k = torch.zeros((self.T, self.action_size))
        self.K = torch.zeros((self.T, self.action_size, self.state_size))

    def reset(self):
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.X = self._rollout_dynamics(self.x_init, actions=self.U) # nominal state sequence (T, state_size)

    def command(self, state):
        """
        Run a DDP step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return: action: torch tensor of shape (1,)
        """
        self._backward_pass()
        self._foward_pass(state)
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
        return trajectory

    def _backward_pass(self):
        V_0 = torch.zeros(self.T+1)
        V_x = torch.zeros((self.state_size, self.T+1))
        V_xx = torch.zeros((self.state_size, self.state_size, self.T+1))

        for i in reversed(range(self.T-1)):
            mu_1 = 0
            mu_2 = 0
            f_x, f_u, f_xx, f_ux, f_uu = self.find_f_gredient(self.X[i], self.U[i])
            L_x, L_u, L_xx, L_xu, L_uu = self.find_cost_gredient(self.X[i], self.U[i])
            # print(L_x.shape, f_x.T.shape, V_x[:, i+1:i+2].shape)
            Q_x = L_x + f_x.T @ V_x[:, i+1:i+2]
            # print(L_u.shape, f_u.T.shape, V_x[:, i+1:i+2].shape)
            Q_u = L_u + f_u.T @ V_x[:, i+1:i+2]
            # print(L_xx.shape, f_x.T.shape, V_xx[:, :, i+1].shape, V_x[:, i+1:i+2].shape, f_xx.shape)
            Q_xx = L_xx + f_x.T @ V_xx[:, :, i+1] @ f_x + V_x[:, i+1:i+2].T @ f_xx
            # print(L_xu.shape, f_u.T.shape, V_xx[:, :, i+1].shape, V_x[:, i+1:i+2].shape, f_ux.shape)
            Q_ux = L_xu.T + f_u.T @ V_xx[:, :, i+1] @ f_x + V_x[:, i+1:i+2].T @ f_ux
            # print(L_uu.shape, f_u.T.shape, V_xx[:, :, i+1].shape, V_x[:, i+1:i+2].shape, f_uu.shape)
            Q_uu = L_uu + f_u.T @ V_xx[:, :, i+1] @ f_u + V_x[:, i+1:i+2].T @ f_uu
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
                
            self.k[i] = -torch.linalg.inv(Q_uu) @ Q_u
            self.K[i] = -torch.linalg.inv(Q_uu) @ Q_ux

            V_0[i] = V_0[i+1] - Q_u.T @ inv_Q_uu @ Q_u /2 # +L_0
            # print(V_x[:, i].shape, Q_x.shape, Q_ux.T.shape, inv_Q_uu.shape, Q_u.shape)
            V_x[:, i] = (Q_x - Q_ux.T @ inv_Q_uu @ Q_u).squeeze()
            V_xx[:, :, i] = Q_xx - Q_ux.T @ inv_Q_uu @ Q_ux

    def _foward_pass(self, state):
        x = torch.zeros_like(self.X)
        x[0] = state
        eps = 1
        for i in range(self.T-1):
            dx = x[i] - self.X[i]
            print(self.k[i], self.K[i])
            self.U[i] += eps*self.k[i] + self.K[i] @ dx
            x[i+1] = self._dynamics(x[i], self.U[i])
        self.X = x

    def _compute_cost(self, states, actions):
        """
        Compute the costs for the K different trajectories
        :param state: torch tensor of shape (state_size)
        :param action: torch tensor of shape (action_size)
        :return:
         - total_trajectory_cost: torch tensor of shape (1,) containing the total trajectory costs for the K trajectories
        Observations:
        * The trajectory cost be the sum of the state costs and action costs along the trajectories
        * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
        * Action costs "TODO"
        """
        total_cost = None
        # Define the state cost weight matrix Q
        Q = self.Q

        # Define the control cost weight matrix R
        R = self.R
        # Compute the state costs
        actions = actions[:, None]
        states = states[:, None]

        state_errors = states - self.goal_state[:, None]
        state_costs = state_errors.T @ Q @ state_errors
        control_costs = actions.T @ R @ actions
        total_cost = state_costs + control_costs
        # ---
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
        state_costs = torch.sum(state_errors.T @ Q @ state_errors, axis=1)
        control_costs = torch.sum(trajectory_acrions.T @ R @ trajectory_acrions, axis=1)

        # Compute the total cost
        total_cost = state_costs + control_costs

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
        f = self.env.dynamics
        eps = 1e-3
        f_xx = torch.zeros(6,6)
        f_xu = torch.zeros(6,1)
        f_uu = torch.zeros(6,1)
        idx1 = torch.tensor([1, 0, 0, 0, 0, 0])
        idx2 = torch.tensor([0, 1, 0, 0, 0, 0])
        idx3 = torch.tensor([0, 0, 1, 0, 0, 0])
        idx4 = torch.tensor([0, 0, 0, 1, 0, 0])
        idx5 = torch.tensor([0, 0, 0, 0, 1, 0])
        idx6 = torch.tensor([0, 0, 0, 0, 0, 1])
        A1 = (f(state+eps*idx1, action)-f(state-eps*idx1, action)) /2 /eps
        A2 = (f(state+eps*idx2, action)-f(state-eps*idx2, action)) /2 /eps
        A3 = (f(state+eps*idx3, action)-f(state-eps*idx3, action)) /2 /eps
        A4 = (f(state+eps*idx4, action)-f(state-eps*idx4, action)) /2 /eps
        A5 = (f(state+eps*idx5, action)-f(state-eps*idx5, action)) /2 /eps
        A6 = (f(state+eps*idx6, action)-f(state-eps*idx6, action)) /2 /eps
        L_u = (f(state, action+eps)-f(state, action-eps)) /2 /eps
        L_x = torch.vstack((A1, A2, A3, A4, A5, A6))
        return f_x, f_u, f_xx, f_xu, f_uu


    def find_cost_gredient(self, state, action):
        # Set the finite difference step size
        eps = 1e-3
        f = self._compute_cost
        idx1 = torch.tensor([1, 0, 0, 0, 0, 0])
        idx2 = torch.tensor([0, 1, 0, 0, 0, 0])
        idx3 = torch.tensor([0, 0, 1, 0, 0, 0])
        idx4 = torch.tensor([0, 0, 0, 1, 0, 0])
        idx5 = torch.tensor([0, 0, 0, 0, 1, 0])
        idx6 = torch.tensor([0, 0, 0, 0, 0, 1])
        A1 = (f(state+eps*idx1, action)-f(state-eps*idx1, action)) /2 /eps
        A2 = (f(state+eps*idx2, action)-f(state-eps*idx2, action)) /2 /eps
        A3 = (f(state+eps*idx3, action)-f(state-eps*idx3, action)) /2 /eps
        A4 = (f(state+eps*idx4, action)-f(state-eps*idx4, action)) /2 /eps
        A5 = (f(state+eps*idx5, action)-f(state-eps*idx5, action)) /2 /eps
        A6 = (f(state+eps*idx6, action)-f(state-eps*idx6, action)) /2 /eps
        L_u = (f(state, action+eps)-f(state, action-eps)) /2 /eps
        L_x = torch.vstack((A1, A2, A3, A4, A5, A6))
        # print(L_x.shape, L_u.shape)

        L_xx = torch.zeros(6,6)
        L_xu = torch.zeros(6,1)
        L_uu = torch.zeros(1,1)
        return L_x, L_u, L_xx, L_xu, L_uu
