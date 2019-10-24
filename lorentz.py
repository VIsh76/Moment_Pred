import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

N=100
state0 = [1.0, 1.0, 1.0]
states0 = np.random.randn((N,3))
t = np.arange(0.0, 5000.0, 0.01)
for i,s in enumerate(states0):
    states[i] = odeint(f, s, t)
print(states.shape)
states = states[:,1000:]

fig = plt.figure()
ax = fig.gca(projection='3d')
for i,s in enumerate(states):
    ax.plot(states[i,0], states[:,1], states[:,2])
plt.show()
