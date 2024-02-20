import torch
import numpy as np
from numpngw import write_apng
from IPython.display import Image
from tqdm import tqdm

from cartpole_env import *
from mppi_control import *
from ddp_control import *

def main():
    # DDP
    # Define problem parameters
    env = CartpoleEnv()
    start_state = np.array([0, np.pi, 0, 0, 0, 0]) + 0.5*np.random.rand(6,)
    env.reset(start_state)
    goal_state = np.zeros(6)
    controller = DDPController(env, horizon=30)
    controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
    controller.reset(torch.tensor(start_state, dtype=torch.float32))
    frames = []
    num_steps = 100

    # motion start
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
        if error_i < .05:
            break
    
    print("creating animated gif, please wait about 10 seconds")
    write_apng("cartpole_ddp.gif", frames, delay=10)
    Image(filename="cartpole_ddp.gif")

    if 1: # you can turn on mppi here
        # MPPI
        env = CartpoleEnv()
        env.reset(start_state) 
        goal_state = np.zeros(6)
        controller = MPPIController(env, num_samples=500, horizon=30, hyperparams=get_cartpole_mppi_hyperparams())
        controller.goal_state = torch.tensor(goal_state, dtype=torch.float32)
        frames = []
        num_steps = 100
        pbar = tqdm(range(num_steps))
        for i in pbar:
            state = env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            control = controller.command(state)
            s = env.step(control) 
            error_i = np.linalg.norm(s-goal_state[:7])
            pbar.set_description(f'Goal Error: {error_i:.4f}')
            img = env.render()
            frames.append(img)
            if error_i < .5:
                break
        print("creating animated gif, please wait about 10 seconds")
        write_apng("cartpole_mppi.gif", frames, delay=10)
        Image(filename="cartpole_mppi.gif")

if __name__ == '__main__':
    main()
