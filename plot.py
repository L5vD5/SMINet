import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

inputs = np.loadtxt('Interpolate/Interpolate_1')
f = open('KURILAB/n=1.csv', 'r', encoding='utf-8')

points = pd.read_csv('KURILAB/n=1.csv', header=5, index_col=0)
points = points.to_numpy()
# points.pop('Time (Seconds)')
times = points[0,:]
points = points[:,1:]
points = points.reshape(-1, 6, 3)
# Set figure
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('plot')

max = 0.4
ax.set_xlim3d([-max, max])
ax.set_ylim3d([-max, max])
ax.set_zlim3d([-max, max])

def update(frame):
  scatter._offsets3d = points[frame,:,0], points[frame,:,1], points[frame,:,2]
  scatter.set_offsets(points[frame,:,0:2])
  # ax.set_title("1: "+str(math.floor(inputs[frame,0]))+" 2: "+str(math.floor(inputs[frame,1]))+" 3: "+str(math.floor(inputs[frame,2])))
  ax.set_title(frame)
  # scatter.set_3d_properties(data[frame, 2])
  return scatter

# First plot
scatter = ax.scatter(points[0,:,0], points[0,:,1], points[0,:,2], c='k')

ani = FuncAnimation(fig, update, frames=len(points), interval=1)

plt.show()
