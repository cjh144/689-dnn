import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation 
#from IPython.display import HTML
import tensorflow.keras as keras
from read_data import read_data_2d

bone_list = [(0,1), (1,20), (20,2), (2,3), \
    (20,4), (4,5), (5,6), (6,7), (7,21), (6,22), \
    (20,8), (8,9), (9,10), (10,11), (11,23), (10,24), \
    (0,12), (12,13), (13,14), (14,15),\
    (0,16), (16,17), (17,18), (18,19)]
joint_list = [0, 20, 4, 5, 6, \
        8,9,10, 12, 13, 14, 16, 17, 18] # 15,19

selected_bone_list = []
for bone in bone_list:
    if bone[0] in joint_list and bone[1] in joint_list:
        selected_bone_list.append(bone)


def test(model, x_data, y_data):
    x_feed = np.array([x_data])
    result = model.predict(x_feed)
    label = y_data[1]
    show_animate(x_data, result, label)
    # print(y_data)
    # ax[0].plot(np.arange(result.shape[1]), result[0,:,1].squeeze())
    # ax[0].xlabel('frames')
    # fig.show()


def show_animate(x_data, result, label):
    frames = x_data.shape[0]
    name = 'normal' if label == 0 else 'limping'

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    # ax1 is the animation of skeleton
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title(name)
    # ax2 is the animation of possibility
    ax2.set_xlim(0, frames)
    ax2.set_ylim(0,1)
    ax2.set_title('Possibility of limping')

    lines = []
    for i in range(frames):
        line_i = []
        line_i.append(ax2.plot( np.arange(i+1), result[0,:(i+1),1].squeeze(), 'C0')[0])
        skel = x_data[i,:]
        x = [skel[j] for j in range(14)]
        y = [skel[j] for j in range(14,28)]
        for k, bone in enumerate(selected_bone_list):
            joint1 = bone[0]
            joint2 = bone[1]
            joint1 = joint_list.index(joint1)
            joint2 = joint_list.index(joint2)
            xdata = (x[joint1], x[joint2])
            ydata = (y[joint1], y[joint2])
            c = 'C' + str(k)
            line_i.append(ax1.plot(xdata, ydata, c=c)[0])
        lines.append(line_i)
    ani = animation.ArtistAnimation(fig, lines, interval=50, blit=True)
    plt.show()


    '''
    for bone in selected_bone_list:
        line, = ax.plot([],[])
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    def animate(j):
        skel = data[j,:]
        x = [skel[i] for i in range(16)]
        y = [skel[i] for i in range(16,32)]
        for k, line in enumerate(lines):
            bone = selected_bone_list[k]
            joint1 = bone[0]
            joint2 = bone[1]
            joint1 = joint_list.index(joint1)
            joint2 = joint_list.index(joint2)
            xdata = (x[joint1], x[joint2])
            ydata = (-y[joint1], -y[joint2])
            line.set_data(xdata, ydata)
        return lines

    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)
    #anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)
    '''


def main():
    #x_train, y_train, x_test, y_test = read_data_2d()
    data = np.load('models/front-slow/training_data.npz', allow_pickle=True)
    x_train = data['x_train']
    x_test= data['x_test']
    y_train = data['y_train']
    y_test= data['y_test']
    print(len(x_test))
    print(len(x_train))
    print(len(y_test))
    print(len(y_train))
    model = keras.models.load_model('./models/front-90/model.h5')
    for i in range(len(x_test)):
    # for i in range(1):
        x_data = x_test[i]
        y_data = y_test[i]
        test(model, x_data, y_data)

if __name__ == '__main__':
    main()