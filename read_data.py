import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from random import random
import math

def read_data_2d():
    loaded = np.load('./skeleton/DIRO_skeletons.npz')
    data = loaded['data']
    
    # for i in range(2, 75, 3):
    #     data[:,:,:,i] = data[:,:,:,i]-2.25

    skel_dim = data.shape[3]
    normal_data = data[:, 0, :, :]
    normal_data = normal_data.reshape([-1,1200,skel_dim])
    asym_data = data[:, [1,2,4,5,6,8], :, :]
    asym_data = asym_data.reshape([-1,1200,skel_dim])

    # cut videos to 50 frame clips
    frame = 100
    normal_data =  normal_data.reshape([-1,frame,75])
    asym_data = asym_data.reshape([-1,frame,75])

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

    data_2d= []
    label_2d = []
    num_deg = 2
    selected_joints = [0, 20, 4, 5, 6, \
        8,9,10, 12, 13, 14, 16, 17, 18] # 15,19
    for iter_sample, sample in enumerate(data):
        for iter_k in range(num_deg):
            #theta = random() * 2 * math.pi
            #theta =  0
            if random() > 0.5:
                theta = math.pi * (random()/3 +1/3)
            else:
                theta = math.pi * (random()/3 +1/3)
                #theta = math.pi * (random()/2 + 0.75 )
            out = []
            for i in range(sample.shape[0]):
                skel = sample[i,:]
                x = [skel[i] for i in range(0, len(skel), 3)]
                y = [skel[i] for i in range(1, len(skel), 3)]
                z = [skel[i]-2.25 for i in range(2, len(skel), 3)]
                u = [x[i]*math.cos(theta)+z[i]*math.sin(theta) for i in range(len(x))]

                dx = u[1] - u[0]
                dy = y[1] - y[0]
                degree = np.arctan(dx/dy)
                for i in range(len(u)):
                    ur = u[i] * np.cos(degree) - y[i] * np.sin(degree)
                    yr = u[i] * np.sin(degree) + y[i] * np.cos(degree)
                    u[i] = ur
                    y[i] = yr

                u_mean = np.mean(u)
                u_var = np.amax(u) - np.amin(u)
                y_mean = np.mean(y)
                y_var = np.amax(y) - np.amin(y)
                u = (u-u_mean)/u_var
                y = (y-y_mean)/y_var

                out.append(np.array([u[i] for i in selected_joints]+[y[i] for i in selected_joints]))
            data_2d.append(np.array(out))
            label_2d.append(label[iter_sample])

    # shuffle the data
    index = list(range(len(data_2d)))
    np.random.shuffle(index)
    data_new = []
    label_new = np.zeros([len(data_2d),2])
    for i,j in enumerate(index):
        data_new.append(data_2d[j])
        label_new[i, label_2d[j]] = 1 # when normal, first bit is 1

    # for k in range(100):
    #     print(label_new[k])
    #     plot_data_2d(data_new[k])

    # training test split
    ratio = 0.8
    split = math.ceil(ratio * len(label_new))
    x_train = data_new[:split]
    y_train = label_new[:split,:]
    x_test = data_new[split:]
    y_test = label_new[split:,:]
    return x_train, y_train, x_test, y_test 

def read_data_3d_selected():
    loaded = np.load('./skeleton/DIRO_skeletons.npz')
    data = loaded['data']
    
    for i in range(2, 75, 3):
        data[:,:,:,i] = data[:,:,:,i]-2.25
    '''
    for j in ran
        skel = data[j,:]
        x = [skel[i] for i in range(0, len(skel), 3)]
        y = [skel[i] for i in range(1, len(skel), 3)]
        z = [skel[i] for i in range(2, len(skel), 3)]
    '''

    skel_dim = data.shape[3]
    normal_data = data[:, 0, :, :]
    normal_data = normal_data.reshape([-1,1200,skel_dim])
    asym_data = data[:, 1:9, :, :]
    asym_data = asym_data.reshape([-1,1200,skel_dim])

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
    label_new = np.zeros([len(data),2])
    for i,j in enumerate(index):
        data_new.append(data[j])
        #print(str(i)+str(j) + str(label[j]))
        label_new[i, label[j]] = 1 # when normal, first bit is 1

    #selected_joints = [0, 20, 4, 5, 6, \
    #    8,9,10, 12, 13, 14, 15, 16, 17, 18, 19]
    selected_joints = [0, 20, 4, 5, 6, \
        8,9,10, 12, 13, 14, 16, 17, 18]
    joint_index = list()
    f = lambda x:[3*x, 3*x+1, 3*x+2]
    for x in selected_joints:
        joint_index = joint_index + f(x)
    data_new = [x[:,joint_index] for x in data_new]
    #for k in range(10):
    #    plot_data_selected(data_new[k])

    # training test split
    ratio = 0.8
    split = math.ceil(ratio * len(label_new))
    x_train = data_new[:split]
    y_train = label_new[:split,:]
    x_test = data_new[split:]
    y_test = label_new[split:,:]
    return x_train, y_train, x_test, y_test 

def read_data_3d():
    loaded = np.load('./skeleton/DIRO_skeletons.npz')
    data = loaded['data']
    
    '''
        skel = data[j,:]
        x = [skel[i] for i in range(0, len(skel), 3)]
        y = [skel[i] for i in range(1, len(skel), 3)]
        z = [skel[i] for i in range(2, len(skel), 3)]
    '''
    for i in range(2, 75, 3):
        data[:,:,:,i] = data[:,:,:,i]-2.25

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
    label_new = np.zeros([len(data),2])
    for i,j in enumerate(index):
        data_new.append(data[j])
        #print(str(i)+str(j) + str(label[j]))
        label_new[i, label[j]] = 1 # when normal, first bit is 1

    # training test split
    ratio = 0.8
    split = math.ceil(ratio * len(label_new))
    x_train = data_new[:split]
    y_train = label_new[:split,:]
    x_test = data_new[split:]
    y_test = label_new[split:,:]
    return x_train, y_train, x_test, y_test 

    '''
    # split data and shuffle data
    print(normal_data.shape)
    print(asym_data.shape)
    #plot_data( asym_data[56,:,:].squeeze())
    plot_data( normal_data[21,:,:].squeeze())
    '''

def read_data_3d_50():
    loaded = np.load('./skeleton/DIRO_skeletons.npz')
    data = loaded['data']
    
    '''
        skel = data[j,:]
        x = [skel[i] for i in range(0, len(skel), 3)]
        y = [skel[i] for i in range(1, len(skel), 3)]
        z = [skel[i] for i in range(2, len(skel), 3)]
    '''
    for i in range(2, 75, 3):
        data[:,:,:,i] = data[:,:,:,i]-2.25

    normal_data = data[:, 0, :, :]
    normal_data = normal_data.reshape([-1,1200,75])
    asym_data = data[:, 1:9, :, :]
    asym_data = asym_data.reshape([-1,1200,75])

    # cut videos to 50 frame clips
    normal_data =  normal_data.reshape([-1,50,75])
    asym_data = asym_data.reshape([-1,50,75])

    # scaling the speed of each clip
    # and randomly cut the clip
    #normal_data = speed_scaling_batch(normal_data)
    #asym_data = speed_scaling_batch(asym_data)

    # mark the clips and shuffle the data
    # ratio of normal samples and abnormal samples:
    ratio = 1
    sample_asym = math.ceil(ratio * len(normal_data))
    asym_id = list(range(len(asym_data)))
    np.random.shuffle(asym_id)
    asym_id = asym_id[:sample_asym]
    asym_data = [asym_data[x] for x in asym_id]

    normal_data = [np.array(x) for x in normal_data.tolist()]
    #asym_data = [np.array(x) for x in asym_data.tolist()]

    data = normal_data + asym_data
    label = [0 for x in range(len(normal_data))] + \
        [1 for x in range(len(asym_data))]

    # shuffle the data
    index = list(range(len(data)))
    np.random.shuffle(index)
    data_new = []
    label_new = np.zeros([len(data),2])
    for i,j in enumerate(index):
        data_new.append(data[j])
        #print(str(i)+str(j) + str(label[j]))
        label_new[i, label[j]] = 1 # when normal, first bit is 1

    # training test split
    ratio = 0.8
    split = math.ceil(ratio * len(label_new))
    x_train = data_new[:split]
    y_train = label_new[:split,:]
    x_test = data_new[split:]
    y_test = label_new[split:,:]
    return np.array(x_train), y_train, np.array(x_test), y_test 

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
    interval = 0.3 + n*0.7 # 0.5 to 1 uniform distribution
    # this range garentees the quality of the data won't be damaged
    # assume the original interval is 1,
    # the new interval is the scaling factor
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

def plot_data_2d(data):
    selected_joints = [0, 20, 3, 4, 5, 6, \
       8,9,10, 12, 13, 14, 15, 16, 17, 18, 19]
    bone_list = [(20, 4), (4, 5), (5, 6), (20, 8), (8, 9), (9, 10), (0, 12), (12, 13), \
       (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (0, 20), (3, 20)]
    # bone_list = [(0,1), (1,20), (20,2), (2,3), \
    # (20,4), (4,5), (5,6), (6,7), (7,21), (6,22), \
    # (20,8), (8,9), (9,10), (10,11), (11,23), (10,24), \
    # (0,12), (12,13), (13,14), (14,15),\
    # (0,16), (16,17), (17,18), (18,19)]

    fig = plt.figure()
    ax = plt.axes(xlim=(-1,1), ylim=(-1,1))

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
        x = [skel[i] for i in range(17)]
        y = [skel[i] for i in range(17,34)]
        for k, line in enumerate(lines):
            bone = bone_list[k]
            joint1 = bone[0]
            joint1 = selected_joints.index(joint1)
            joint2 = bone[1]
            joint2 = selected_joints.index(joint2)
            xdata = (x[joint1], x[joint2])
            ydata = (y[joint1], y[joint2])
            line.set_data(xdata, ydata)
        return lines
    length = data.shape[0]
    anim = FuncAnimation(fig, animate, init_func=init, frames=length, interval=50, blit=True)

    plt.show()

def plot_data_selected(data):
    selected_joints = [0, 20, 3, 4, 5, 6, \
        8,9,10, 12, 13, 14, 15, 16, 17, 18, 19]
    bone_list = [(20, 4), (4, 5), (5, 6), (20, 8), (8, 9), (9, 10), (0, 12), (12, 13), \
        (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (0, 20), (3, 20)]

    fig = plt.figure()
    ax = plt.axes(xlim=(-1,1), ylim=(-1,1))

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
            joint1 = selected_joints.index(joint1)
            joint2 = bone[1]
            joint2 = selected_joints.index(joint2)
            xdata = (x[joint1], x[joint2])
            ydata = (y[joint1], y[joint2])
            zdata = (z[joint1], z[joint2])
            line.set_data(zdata, ydata)
        return lines

    length = data.shape[0]
    anim = FuncAnimation(fig, animate, init_func=init, frames=length, interval=50, blit=True)

    plt.show()

def plot_data(data):
    bone_list = [(20, 4), (4, 5), (5, 6), (20, 8), (8, 9), (9, 10), (0, 12), (12, 13), \
        (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), (0, 20), (3, 20)]

    fig = plt.figure()
    ax = plt.axes(xlim=(-1,1), ylim=(-1,1))

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

def test():
    x_train, y_train, x_test, y_test = read_data_2d()
    for i in range(200):
        #print(y_train[i])
        plot_data_2d(x_train[i])


if __name__ == '__main__':
    test()
