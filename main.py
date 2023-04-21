import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm import tqdm

from cartpole_env import *
from dynamics import *
from mppi_control import *
from ddp_control import *

def main():
    env = CartpoleEnv()
    env.reset(state = np.array([0.0, 0.25, 0.25, 0.0, 0.0, 0.0]))

    # frames=[] #frames to create animated png
    # frames.append(env.render())
    # for i in tqdm(range(100)):
    #     action = env.action_space.sample()
    #     s = env.step(action)
    #     img = env.render()
    #     frames.append(img)

    # write_apng("cartpole_example.png", frames, delay=10)
    # Image(filename="cartpole_example.png")

    # Linearization around 
    # env = CartpoleEnv()
    # B = 4
    # states = 0.1 * torch.randn(B, 6)
    # actions = torch.randn(B, 1)

    # # first lets see what the pybullet dynamics are
    # print('Next states from simulator are: ')
    # for state, action in zip(states, actions):
    #     print(env.dynamics(state.numpy(), action.numpy()))

    # print('')
    # print('Batched next states from analytical dynamics are')
    # print(dynamics_analytic(states, actions))


    # A_numerical, B_numerical = env.linearize_numerical(np.zeros(6), np.zeros(1))

    # print('Numerical Linearizations are ')
    # print(A_numerical)
    # print(B_numerical)
    # print('')
    # A_autograd, B_autograd = linearize_pytorch(torch.zeros(6), torch.zeros(1))

    # print('Autograd linearizations are ')
    # print(A_autograd)
    # print(B_autograd)


    # # Let's test to see if your analytic dynamics matches the simulator

    # # first let's generate a random control sequence
    # T = 50
    # control_sequence = np.random.randn(T, 1)
    # start_state = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0])

    # # We use the simulator to simulate a trajectory
    # env = CartpoleEnv()
    # env.reset(start_state)
    # states_pybullet = np.zeros((T+1, 6))
    # states_pybullet[0] = start_state
    # for t in range(T):
    #     states_pybullet[t+1] = env.step(control_sequence[t])

    # # Now we will use your analytic dynamics to simulate a trajectory
    # states_analytic = torch.zeros(T+1, 1, 6) # Need an extra 1 which is the batch dimension (T x B x 4)
    # states_analytic[0] = torch.from_numpy(start_state).reshape(1, 6)
    # for t in range(T):
    #     current_state = states_analytic[t]
    #     current_control = torch.from_numpy(control_sequence[t]).reshape(1, 1) # add batch dimension to control    
    #     states_analytic[t+1] = dynamics_analytic(current_state, current_control)
        
    # # convert back to numpy for plotting
    # states_analytic = states_analytic.reshape(T+1, 6).numpy()

    # # Plot and compare - They should be indistinguishable 
    # fig, axes = plt.subplots(2, 3, figsize=(8, 8))
    # axes[0][0].plot(states_analytic[:, 0], label='analytic')
    # axes[0][0].plot(states_pybullet[:, 0], '--', label='pybullet')
    # axes[0][0].title.set_text('x')

    # axes[0][1].plot(states_analytic[:, 1])
    # axes[0][1].plot(states_pybullet[:, 1], '--')
    # axes[0][1].title.set_text('theta_1')

    # axes[1][0].plot(states_analytic[:, 3])
    # axes[1][0].plot(states_pybullet[:, 3], '--')
    # axes[1][0].title.set_text('x_dot')

    # axes[1][1].plot(states_analytic[:, 4])
    # axes[1][1].plot(states_pybullet[:, 4], '--')
    # axes[1][1].title.set_text('theta_1_dot')

    # axes[0][2].plot(states_analytic[:, 2])
    # axes[0][2].plot(states_pybullet[:, 2], '--')
    # axes[0][2].title.set_text('theta_2')

    # axes[1][2].plot(states_analytic[:, 5])
    # axes[1][2].plot(states_pybullet[:, 5], '--')
    # axes[1][2].title.set_text('theta_2_dot')

    # axes[0][0].legend()
    # plt.show()

    # MPPI
    # env = CartpoleEnv()
    # env.reset(np.array([0, np.pi, 0, 0, 0, 0]) + np.random.rand(6,)) 
    # goal_state = np.zeros(6)
    # controller = MPPIController(env, num_samples=500, horizon=30, hyperparams=get_cartpole_mppi_hyperparams())
    # controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
    # frames = []
    # num_steps = 70
    # pbar = tqdm(range(num_steps))
    # for i in pbar:
    #     state = env.get_state()
    #     state = torch.tensor(state, dtype=torch.float32)
    #     control = controller.command(state)
    #     s = env.step(control) 
    #     error_i = np.linalg.norm(s-goal_state[:7])
    #     pbar.set_description(f'Goal Error: {error_i:.4f}')
    #     img = env.render()
    #     frames.append(img)
    #     if error_i < .1:
    #         break
    # print("creating animated gif, please wait about 10 seconds")
    # write_apng("cartpole_mppi.gif", frames, delay=10)
    # Image(filename="cartpole_mppi.gif")

    # DDP
    # Define problem parameters
    env = CartpoleEnv()
    env.reset(np.array([0, np.pi, 0, 0, 0, 0]) + np.random.rand(6,)) 
    goal_state = np.zeros(6)
    controller = DDPController(env, horizon=30)
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
    frames = []
    num_steps = 2
    pbar = tqdm(range(num_steps))
    for i in pbar:
        state = env.get_state()
        state = torch.tensor(state, dtype=torch.float32)
        control = controller.command(state)
        s = env.step(control) 
        error_i = np.linalg.norm(s-goal_state[:7])
        pbar.set_description(f'Goal Error: {error_i:.4f}')
        pbar.update()
        img = env.render()
        frames.append(img)
        if error_i < .1:
            break
    
    print("creating animated gif, please wait about 10 seconds")
    write_apng("cartpole_ddp.gif", frames, delay=10)
    Image(filename="cartpole_ddp.gif")


if __name__ == '__main__':
    main()
