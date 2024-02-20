import matplotlib.pyplot as plt

def state_plot(state1, state2, state3, state4, state5, state6):
    plt.plot(state1, linewidth=1.0, label=r'$x$')
    plt.plot(state2, linewidth=1.0, label=r'$\theta_1$')
    plt.plot(state3, linewidth=1.0, label=r'$\theta_2$')
    plt.plot(state4, linewidth=1.0, label=r'$dx$')
    plt.plot(state5, linewidth=1.0, label=r'$d\theta_1$')
    plt.plot(state6, linewidth=1.0, label=r'$d\theta_2$')
    plt.xlabel('time')
    plt.ylabel('states')
    plt.title('State Convergence (MPPI)')
    plt.legend()
    plt.show()

def cost_plot(cost):
    plt.plot(cost, linewidth=1.0, label=r'$cost$')
    plt.xlabel('time')
    plt.ylabel('cost')
    plt.title('Cost Convergence (MPPI)')
    plt.legend()
    plt.show()

def gain_plot(k, K1, K2, K3, K4, K5, K6):
    plt.plot(K1, linewidth=1.0, label=r'$K[1]$')
    plt.plot(K2, linewidth=1.0, label=r'$K[2]$')
    plt.plot(K3, linewidth=1.0, label=r'$K[3]$')
    plt.plot(K4, linewidth=1.0, label=r'$K[4]$')
    plt.plot(K5, linewidth=1.0, label=r'$K[5]$')
    plt.plot(K6, linewidth=1.0, label=r'$K[6]$')
    plt.plot(k, linewidth=1.0, label=r'$k$')
    plt.xlabel('time')
    plt.ylabel('gain')
    plt.title('Gain Over Time (DDP)')
    plt.legend()
    plt.show()