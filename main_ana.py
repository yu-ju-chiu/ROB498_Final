import torch
import numpy as np
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm import tqdm

from dynamics import *
from mppi_control import *
from ddp_control import *

def main():
    # DDP
    # Define problem parameters
    start_state = 
    env = CartpoleEnv()
    env.reset(np.array([0, np.pi, 0, 0, 0, 0]) + np.random.rand(6,)) 
    goal_state = np.zeros(6)
    controller = DDPController(env, horizon=10)
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
        pbar.update()
        img = env.render()
        frames.append(img)
        if error_i < .1:
            break
    
    print("creating animated gif, please wait about 10 seconds")
    write_apng("cartpole_ddp.gif", frames, delay=10)
    Image(filename="cartpole_ddp_ana.gif")

    x1, y1 = [-1, 12], [1, 10]
    x2, y2 = [-1, 10], [3, -1]
    plt.xlim(0, 8), plt.ylim(-2, 8)
    plt.plot(x1, y1, x2, y2, marker = 'o')
    plt.show()


if __name__ == '__main__':
    main()
