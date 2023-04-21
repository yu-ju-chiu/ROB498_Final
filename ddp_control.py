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
        self.R = torch.diag(torch.tensor([0.1]))
        self.U = torch.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = torch.zeros(self.action_size)
        self.X = torch.zeros((self.T, self.state_size))
        self.x_init = torch.zeros(self.state_size)
        self.k = torch.zeros((self.T, self.action_size))
        self.K = torch.zeros((self.T, self.action_size, self.state_size))

    def reset(self):
        self.U = torch.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

    def command(self, state):
        """
        Run a DDP step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return: action: torch tensor of shape (1,)
        """
        self.X = self._rollout_dynamics(state, actions=self.U)
        trajectory_cost = self._compute_trajectory_cost(self.X)
        self._backward_pass()
        self._foward_pass()
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
            f_x, f_u, f_ux, f_xx, f_uu = self.find_f_gredient(self.X[i], self.U[i])
            L_x, L_u, L_xx, L_xu, L_uu = self.find_cost_gredient(self.X[i], self.U[i])
            Q_x = L_x + f_x.T @ V_x[i+1]
            Q_u = L_u + f_u.T @ V_x[i+1]
            Q_xx = L_xx + f_x.T @ V_xx[i+1] @ f_x + V_x[i+1] @ f_xx
            Q_ux = L_xu + f_u.T @ V_xx[i+1] @ f_x + V_x[i+1] @ f_ux
            Q_uu = L_uu + f_u.T @ V_xx[i+1] @ f_u + V_x[i+1] @ f_uu
            inv_Q_uu = None
            while inv_Q_uu is None:
                try:
                    inv_Q_uu = torch.linalg.inv(Q_uu)
                except:
                    mu_1 += 0.1
                    mu_2 += 0.1
            self.k[i] = -torch.linalg.inv(Q_uu) @ Q_u
            self.K[i] = -torch.linalg.inv(Q_uu) @ Q_ux

            V_0[i] = V_0[i+1] - Q_u.T @ inv_Q_uu @ Q_u /2 # +L_0
            V_x[i] = Q_x - Q_ux.T @ inv_Q_uu @ Q_u
            V_xx[i] = Q_xx - Q_ux.T @ inv_Q_uu @ Q_ux

    def _foward_pass(self):
        x = self.X
        dx = torch.zeros_like(self.X)
        eps = 1
        for i in range(self.T-1):
            dx[i] = x[i] - self.X[i]
            self.U[i] += eps*self.k[i] + self.K[i] @ dx[i]
            x[i+1] = self._dynamics(x[i], self.U[i])

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


        state_errors = states - self.goal_state
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
        state_errors = trajectory_states - self.goal_state.repeat(self.T, 1)
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