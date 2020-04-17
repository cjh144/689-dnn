import matplotlib.pyplot as plt
import numpy as np

fp = open('./log')

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []

for line in fp:
    #if line.startswith('173'):
    if '=' in line:
        loss = line.split(' ')[-4]
        acc = line.split(' ')[-1]
        train_loss.append( float(loss))
        train_acc.append( float(acc))

    if line.startswith('['):
        loss = line.split(',')[0].strip()[1:]
        acc = line.split(',')[1].strip()[:-2]
        valid_loss.append( float(loss))
        valid_acc.append( float(acc))

print(len(train_loss))
print(len(train_acc))
print(len(valid_loss))
print(len(valid_acc))

train_label = np.arange(360) + 1
valid_label = 10 * (np.arange(36) + 1)

plt.figure()
plt.plot(train_label, train_loss, label='Train loss')
plt.plot(valid_label, valid_loss, label='Valid loss')
plt.legend()
plt.xlabel('Epochs')
#plt.show()
plt.savefig('loss.pdf')

plt.figure()
plt.plot(train_label, train_acc, label='Train accuracy')
plt.plot(valid_label, valid_acc, label='Valid accuracy')
plt.legend()
plt.xlabel('Epochs')
#plt.show()
plt.savefig('acc.pdf')
