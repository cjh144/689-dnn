'''Example of Python code reading the skeletons'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
loaded = np.load('DIRO_skeletons.npz')

#get skeleton data of size (n_subject, n_gait, n_frame, 25*3)
data = loaded['data']

#get joint coordinates of a specific skeleton
skel = data[0,0,0,:]
x = [skel[i] for i in range(0, len(skel), 3)]
y = [skel[i] for i in range(1, len(skel), 3)]
z = [skel[i] for i in range(2, len(skel), 3)]

#get default separation
separation = loaded['split']

#print information
print(data.shape)
print(separation)
#expected results:
#(9, 9, 1200, 75)
# meaning:
# 9subj, 9gaits, 1200frames
#['train' 'test' 'train' 'test' 'train' 'train' 'test' 'test' 'train']

#plt.scatter(x,z)
#plt.show()

'''
Code for bones and plot bones
'''
bone_list = [(0,1), (1,20), (20,2), (2,3), \
    (20,4), (4,5), (5,6), (6,7), (7,21), (6,22), \
    (20,8), (8,9), (9,10), (10,11), (11,23), (10,24), \
    (0,12), (12,13), (13,14), (14,15),\
    (0,16), (16,17), (17,18), (18,19)]

'''
for bone in bone_list:
    joint1 = bone[0]
    joint2 = bone[1]
    bone_x = [x[joint1], x[joint2]]
    bone_y = [y[joint1], y[joint2]]
    plt.plot(bone_x,bone_y)
    plt.text(x[joint2], y[joint2], str(joint2))

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()
'''

'''
Select part of the joints to be compatible with body-25 skeleton in OpenPose
'''
selected_joints = [0, 20, 3, 4, 5, 6, \
    8,9,10, 12, 13, 14, 15, 16, 17, 18, 19]

selected_bone_list = []
for bone in bone_list:
    if bone[0] in selected_joints and bone[1] in selected_joints:
        selected_bone_list.append(bone)

selected_bone_list.append((0,20))
selected_bone_list.append((3,20))
print(selected_bone_list)

'''
#Code for Plot the selected bones
for bone in selected_bone_list:
    joint1 = bone[0]
    joint2 = bone[1]
    bone_x = [x[joint1], x[joint2]]
    bone_y = [y[joint1], y[joint2]]
    bone_z = [z[joint1], z[joint2]]
    print(bone_z)
    plt.plot(bone_z,bone_y)
    plt.text(z[joint2], y[joint2], str(joint2))

#plt.xlim([0,4])
plt.ylim([-1,1])
plt.show()

'''

'''
Code for plot animation of selected bones
'''
fig = plt.figure()
ax = plt.axes(xlim=(1,3), ylim=(-1,1))

lines = []
for bone in selected_bone_list:
    #joint1 = bone[0]
    #joint2 = bone[1]
    #bone_x = [x[joint1], x[joint2]]
    #bone_y = [y[joint1], y[joint2]]
    line, = ax.plot([],[])
    lines.append(line)

def init():
    for line in lines:
        line.set_data([],[])
    return lines

def animate(j):
    skel = data[0,2,j,:]
    x = [skel[i] for i in range(0, len(skel), 3)]
    y = [skel[i] for i in range(1, len(skel), 3)]
    z = [skel[i] for i in range(2, len(skel), 3)]
    for k, line in enumerate(lines):
        bone = selected_bone_list[k]
        joint1 = bone[0]
        joint2 = bone[1]
        xdata = (x[joint1], x[joint2])
        ydata = (y[joint1], y[joint2])
        zdata = (z[joint1], z[joint2])
        line.set_data(zdata, ydata)
    return lines

anim = FuncAnimation(fig, animate, init_func=init, frames=4, interval=50, blit=True)

plt.show()

