import numpy as np
import matplotlib.pyplot as plt

# seed = 4
# np.random.seed(seed)

p1 = np.random.uniform(low=0.1, high=0.5, size=1)
p2 = np.random.uniform(low=0.1, high=0.5, size=1)
p3 = np.random.uniform(low=0.1, high=0.25, size=1)
p4 = np.random.uniform(low=0.1, high=0.25, size=1)
p5 = np.random.uniform(low=0.5, high=1.5, size=1)

def unknown_reward_function(x, noise = 0):
    r = (p5*np.exp(-np.power(x+p1,2.)/(2*np.power(p3, 2.)))
         +np.exp(-np.power(x-p2,2.)/(2*np.power(p4,2.))))

    return r+(noise*(np.random.random()-0.5)/10.)

def solution():
    x_max = 0
    dev = 0.40
    for i in range(5):
        x = np.linspace(-1, 1, 200)
        y = unknown_reward_function(x, 0)
        plt.xlabel('X', size=15)
        plt.ylabel('R', size=15)
        plt.plot(x, y, color='red')
        x = np.random.normal(loc=x_max, scale=dev, size=15)
        y = unknown_reward_function(x, 0)
        plt.plot(x, y, 'bo', color='blue')
        x_max = x[y.argmax()]
        if dev < 0.03:
            dev = 0
        else:
            dev -= 0.13
        # plt.savefig('seeds/%d/%d.png' % (seed, i))
        plt.show()

solution()