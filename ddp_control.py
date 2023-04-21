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

    def _foward_pass(self, state):
        x = torch.zeros_like(self.X)
        x[0] = state
        eps = 1
        for i in range(self.T-1):
            dx = x[i] - self.X[i]
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
    
    def find_f_gredient(self, state, action):
        f_x, f_u = self.env.linearize_numerical(state, action)
        f = self.env.dynamics
        # Set the finite difference step size
        eps = 1e-3
        # Compute the Hessian matrix of the output state with respect to the input state
        f_xx = torch.zeros(state.shape[0], state.shape[0])
        for i in range(state.shape[0]):
            for j in range(i, state.shape[0]):
                # Compute the second-order partial derivative with respect to state_i and state_j
                x_plus_i = state.clone()
                x_plus_i[i] += eps
                x_minus_i = state.clone()
                x_minus_i[i] -= eps
                x_plus_j = state.clone()
                x_plus_j[j] += eps
                x_minus_j = state.clone()
                x_minus_j[j] -= eps

                output_plus_ii = f(x_plus_i, action)
                output_minus_ii = f(x_minus_i, action)
                output_plus_jj = f(x_plus_j, action)
                output_minus_jj = f(x_minus_j, action)
                output_plus_ij = f(x_plus_i, action)
                output_plus_ij[i] = output_plus_jj[i]
                output_minus_ij = f(x_minus_i, action)
                output_minus_ij[i] = output_minus_jj[i]

                f_xx[i, j] = (output_plus_ii - output_minus_ii - output_plus_ij + output_minus_ij) / (4*eps*eps)
                f_xx[j, i] = f_xx[i, j]

        # Compute the Hessian matrix of the output state with respect to the input action and state
        f_xu = torch.zeros(state.shape[0], action.shape[0])
        for i in range(state.shape[0]):
            for j in range(action.shape[0]):
                # Compute the second-order partial derivative with respect to state_i and action_j
                x_plus_i = state.clone()
                x_plus_i[i] += eps
                x_minus_i = state.clone()
                x_minus_i[i] -= eps
                u_plus_j = action.clone()
                u_plus_j[j] += eps
                u_minus_j = action.clone()
                u_minus_j[j] -= eps

                output_plus_ui = f(x_plus_i, u_plus_j)
                output_minus_ui = f(x_minus_i, u_plus_j)
                output_plus_uj = f(state, u_plus_j)
                output_plus_uj[j] = f(state, u_minus_j)[j]
                output_minus_uj = f(state, u_minus_j)
                output_minus_uj[j] = f(state, u_plus_j)[j]

                f_xu[i, j] = (output_plus_ui - output_minus_ui - output_plus_uj + output_minus_uj) / (4*eps*eps)

        f_uu = torch.zeros(action.shape[0], action.shape[0])
        for i in range(action.shape[0]):
            for j in range(i, action.shape[0]):
                # Compute the second-order partial derivative with respect to action_i and action_j
                u_plus_i = action.clone()
                u_plus_i[i] += eps
                u_minus_i = action.clone()
                u_minus_i[i] -= eps
                u_plus_j = action.clone()
                u_plus_j[j] += eps
                u_minus_j = action.clone()
                u_minus_j[j] -= eps

                output_plus_ii = f(state, u_plus_i)
                output_minus_ii = f(state, u_minus_i)
                output_plus_jj = f(state, u_plus_j)
                output_minus_jj = f(state, u_minus_j)
                output_plus_ij = f(state, u_plus_i)
                output_plus_ij[i] = f(state, u_plus_j)[i]
                output_minus_ij = f(state, u_minus_i)
                output_minus_ij[i] = f(state, u_minus_j)[i]

                f_uu[i, j] = (output_plus_ii - output_minus_ii - output_plus_ij + output_minus_ij) / (4*eps*eps)
                f_uu[j, i] = f_uu[i, j]

        return f_x, f_u, f_xx, f_xu, f_uu


    def find_cost_gredient(self, state, action):
        # Set the finite difference step size
        eps = 1e-3
        f = self.cost_function
        # Compute the partial derivatives of the output state with respect to the input tensors
        L_x = torch.zeros_like(state)
        L_u = torch.zeros_like(action)

        for i in range(state.shape[0]):
            # Compute the partial derivative with respect to state_i
            x_plus = state.clone()
            x_plus[i] += eps
            x_minus = state.clone()
            x_minus[i] -= eps
            output_plus = f(x_plus, action)
            output_minus = f(x_minus, action)
            L_x[i] = (output_plus - output_minus) / (2*eps)

        for i in range(action.shape[0]):
            # Compute the partial derivative with respect to action_i
            u_plus = action.clone()
            u_plus[i] += eps
            u_minus = action.clone()
            u_minus[i] -= eps
            output_plus = f(state, u_plus)
            output_minus = f(state, u_minus)
            L_u[i] = (output_plus - output_minus) / (2*eps)
        
        # Compute the Hessian matrix of the output state with respect to the input state
        L_xx = torch.zeros(state.shape[0], state.shape[0])
        for i in range(state.shape[0]):
            for j in range(i, state.shape[0]):
                # Compute the second-order partial derivative with respect to state_i and state_j
                x_plus_i = state.clone()
                x_plus_i[i] += eps
                x_minus_i = state.clone()
                x_minus_i[i] -= eps
                x_plus_j = state.clone()
                x_plus_j[j] += eps
                x_minus_j = state.clone()
                x_minus_j[j] -= eps

                output_plus_ii = f(x_plus_i, action)
                output_minus_ii = f(x_minus_i, action)
                output_plus_jj = f(x_plus_j, action)
                output_minus_jj = f(x_minus_j, action)
                output_plus_ij = f(x_plus_i, action)
                output_plus_ij[i] = output_plus_jj[i]
                output_minus_ij = f(x_minus_i, action)
                output_minus_ij[i] = output_minus_jj[i]

                L_xx[i, j] = (output_plus_ii - output_minus_ii - output_plus_ij + output_minus_ij) / (4*eps*eps)
                L_xx[j, i] = L_xx[i, j]

        # Compute the Hessian matrix of the output state with respect to the input action and state
        L_xu = torch.zeros(state.shape[0], action.shape[0])
        for i in range(state.shape[0]):
            for j in range(action.shape[0]):
                # Compute the second-order partial derivative with respect to state_i and action_j
                x_plus_i = state.clone()
                x_plus_i[i] += eps
                x_minus_i = state.clone()
                x_minus_i[i] -= eps
                u_plus_j = action.clone()
                u_plus_j[j] += eps
                u_minus_j = action.clone()
                u_minus_j[j] -= eps

                output_plus_ui = f(x_plus_i, u_plus_j)
                output_minus_ui = f(x_minus_i, u_plus_j)
                output_plus_uj = f(state, u_plus_j)
                output_plus_uj[j] = f(state, u_minus_j)[j]
                output_minus_uj = f(state, u_minus_j)
                output_minus_uj[j] = f(state, u_plus_j)[j]

                L_xu[i, j] = (output_plus_ui - output_minus_ui - output_plus_uj + output_minus_uj) / (4*eps*eps)

        L_uu = torch.zeros(action.shape[0], action.shape[0])
        for i in range(action.shape[0]):
            for j in range(i, action.shape[0]):
                # Compute the second-order partial derivative with respect to action_i and action_j
                u_plus_i = action.clone()
                u_plus_i[i] += eps
                u_minus_i = action.clone()
                u_minus_i[i] -= eps
                u_plus_j = action.clone()
                u_plus_j[j] += eps
                u_minus_j = action.clone()
                u_minus_j[j] -= eps

                output_plus_ii = f(state, u_plus_i)
                output_minus_ii = f(state, u_minus_i)
                output_plus_jj = f(state, u_plus_j)
                output_minus_jj = f(state, u_minus_j)
                output_plus_ij = f(state, u_plus_i)
                output_plus_ij[i] = f(state, u_plus_j)[i]
                output_minus_ij = f(state, u_minus_i)
                output_minus_ij[i] = f(state, u_minus_j)[i]

                L_uu[i, j] = (output_plus_ii - output_minus_ii - output_plus_ij + output_minus_ij) / (4*eps*eps)
                L_uu[j, i] = L_uu[i, j]
        return L_x, L_u, L_xx, L_xu, L_uu
