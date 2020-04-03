import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow.keras as keras

#path = '/content/drive/My Drive/DL-dataset/limp/27/'
path = './test/walk1/output/'
file_list = os.listdir(path)
file_list.sort()

#file_path = path + file_list[0]
#with open(file_path) as fp:
def read_json(fp, person_id):
    data = json.load(fp)
    person = data['people'][person_id]
    keypoints = person['pose_keypoints_2d']
    x = []
    y = []
    c = []
    data_iter = iter(keypoints)
    for i in range(25):
        x.append(next(data_iter))
        y.append(next(data_iter))
        c.append(next(data_iter))
    return x, y, c

bone_list = ( (0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
    (1,8), (8,9), (9,10), (10,11), (11,24), (11,22), (22,23),
    (8,12), (12,13), (13,14), (14,21), (14,19), (19,20),
    (0,15), (15,17), (0,16), (16,18))
joint_list = ( 8,1,2,3,4,5,6,7,9,10,11,12,13,14 )

selected_bone_list = []
for bone in bone_list:
    if bone[0] in joint_list and bone[1] in joint_list:
        selected_bone_list.append(bone)

# f=file_list[200]
data = []
for f in file_list:
    if f.endswith('.json'):
        file_path = path+f 
        with open(file_path) as fp:
            x,y,_ = read_json(fp,0)
            # for bone in bone_list:
            #     joint1 = bone[0]
            #     joint2 = bone[1]
            #     if (x[joint1]!=0 or y[joint1]!=0) and (x[joint2]!=0 or y[joint2]!=0):
            #         x_data = ( x[joint1], x[joint2])
            #         y_data = ( -y[joint1], -y[joint2])
            #         plt.plot(x_data, y_data)
            # plt.show()

            x = [ x[i] for i in joint_list ]
            y = [ y[i] for i in joint_list ]

            x_mean = np.mean(x)
            x_var = np.amax(x) - np.amin(x)
            y_mean = np.mean(y)
            y_var = np.amax(y) - np.amin(y)

            x = (x-x_mean)/x_var
            y = -(y-y_mean)/y_var
            data.append(np.concatenate([x,y]))

data = np.array(data)
print(data.shape)
frames = data.shape[0]

# code for prediction
data = np.array([data])
model = keras.models.load_model('./models/360.h5')
result = model.predict(data)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
plt.plot(np.arange(result.shape[1]), result[0,:,1].squeeze())
plt.xlabel('frames')
plt.show()

time_data = []
for i in range(result.shape[1]):
    time_data.append([str(i), str(result[0,i,1])])
json_content = {"limping":time_data}
json_dump = json.dumps(json_content)
with open('./timeLabel.json','w') as outfile:
    outfile.write(json_dump)

'''
### Code for plot one image
skel = data[0,:].squeeze()
x = [skel[i] for i in range(16)]
y = [skel[i] for i in range(16,32)]
for k, line in enumerate(selected_bone_list):
    bone = selected_bone_list[k]
    joint1 = bone[0]
    joint2 = bone[1]
    joint1 = joint_list.index(joint1)
    joint2 = joint_list.index(joint2)
    xdata = (x[joint1], x[joint2])
    ydata = (y[joint1], y[joint2])
    plt.plot(xdata, ydata)
'''
### Code for plot animation of selected bones
data = data.squeeze()
fig = plt.figure()
ax = plt.axes(xlim=(-1.5,1.5), ylim=(-1.5,1.5))

lines = []
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
        ydata = (y[joint1], y[joint2])
        line.set_data(xdata, ydata)
    return lines

anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)


HTML(anim.to_html5_video())

