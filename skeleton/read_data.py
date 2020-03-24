import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from random import random
import math

def read_data_3d():
    loaded = np.load('DIRO_skeletons.npz')
    data = loaded['data']
    
    normal_data = data[:, 0, :, :]
    normal_data = normal_data.reshape([-1,1200,75])
    asym_data = data[:, 1:9, :, :]
    asym_data = asym_data.reshape([-1,1200,75])

    # cut videos to 50 frame clips
    normal_data =  normal_data.reshape([-1,50,75])
    asym_data = asym_data.reshape([-1,50,75])

    # scaling the speed of each clip
    # and randomly cut the clip
    normal_data = speed_scaling_batch(normal_data)
    asym_data = speed_scaling_batch(asym_data)

    # mark the clips and shuffle the data
    # ratio of normal samples and abnormal samples:
    ratio = 1
    sample_asym = math.ceil(ratio * len(normal_data))
    asym_id = list(range(len(asym_data)))
    np.random.shuffle(asym_id)
    asym_id = asym_id[:sample_asym]
    asym_data = [asym_data[x] for x in asym_id]

    data = normal_data + asym_data
    label = [0 for x in range(len(normal_data))] + \
        [1 for x in range(len(asym_data))]

    # shuffle the data
    index = list(range(len(data)))
    np.random.shuffle(index)
    data_new = []
    label_new = []
    for i in index:
        data_new.append(data[i])
        label_new.append(label[i])

    # training test split
    ratio = 0.8
    split = math.ceil(ratio * len(label_new))
    x_train = data_new[:split]
    y_train = label_new[:split]
    x_test = data_new[split:]
    y_test = label_new[split:]

    return x_train, y_train, x_test, y_test 

    '''
    # split data and shuffle data
    print(normal_data.shape)
    print(asym_data.shape)
    #plot_data( asym_data[56,:,:].squeeze())
    plot_data( normal_data[21,:,:].squeeze())
    '''

def speed_scaling_batch(data):
    sample = data.shape[0]
    data_new = []
    for i in range(sample):
        # do scaling to the frame
        new_clip = speed_scaling(data[i,:,:].squeeze())
        length = new_clip.shape[0]
        # cut the clip randomly
        lim = math.ceil(length * (random() * 0.5 + 0.5))
        new_clip = new_clip[:lim, :]
        data_new.append(new_clip)
    return data_new



def speed_scaling(data):
    num_frame, num_point = data.shape
    data_new = []
    # randomly choose a scaling factor
    n = random()
    if n > 0.5:
        scaling = random() * 0.5 + 0.5
    else:
        scaling = random() + 1
    # assume the original interval is 1,
    # the new interval is the scaling factor
    interval = scaling
    for i in range(num_point):
        # get trace of ith node
        trace = data[:,i]
        # interpolate the points
        f_interp = interpolate.interp1d(np.arange(num_frame), trace)
        x_new = np.arange(0, num_frame-1, interval)
        y_new = f_interp(x_new)
        data_new.append(y_new)
    data_new = np.array(data_new)
    return data_new.T

def plot_data(data):
    bone_list = [(20, 4), (4, 5), (5, 6), (20, 8), (8, 9), (9, 10), (0, 12), (12, 13), \
        (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (0, 20), (3, 20)]

    fig = plt.figure()
    ax = plt.axes(xlim=(1,3), ylim=(-1,1))

    lines = []
    for bone in bone_list:
        line, = ax.plot([],[])
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(j):
        skel = data[j,:]
        x = [skel[i] for i in range(0, len(skel), 3)]
        y = [skel[i] for i in range(1, len(skel), 3)]
        z = [skel[i] for i in range(2, len(skel), 3)]
        for k, line in enumerate(lines):
            bone = bone_list[k]
            joint1 = bone[0]
            joint2 = bone[1]
            xdata = (x[joint1], x[joint2])
            ydata = (y[joint1], y[joint2])
            zdata = (z[joint1], z[joint2])
            line.set_data(zdata, ydata)
        return lines

    length = data.shape[0]
    anim = FuncAnimation(fig, animate, init_func=init, frames=length, interval=50, blit=True)

    plt.show()



'''
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
selected_joints = [0, 20, 3, 4, 5, 6, \
    8,9,10, 12, 13, 14, 15, 16, 17, 18, 19]
bone_list = [(0,1), (1,20), (20,2), (2,3), \
    (20,4), (4,5), (5,6), (6,7), (7,21), (6,22), \
    (20,8), (8,9), (9,10), (10,11), (11,23), (10,24), \
    (0,12), (12,13), (13,14), (14,15),\
    (0,16), (16,17), (17,18), (18,19)]
selected_bone_list = []
for bone in bone_list:
    if bone[0] in selected_joints and bone[1] in selected_joints:
        selected_bone_list.append(bone)

selected_bone_list.append((0,20))
selected_bone_list.append((3,20))
'''

if __name__ == '__main__':
    read_data_3d()
