from read_data import read_data_3d_50,read_data_3d,read_data_3d_selected,read_data_2d
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, Input, TimeDistributed
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

def main_v1():
    # this version has fixed time steps: 50
    # and only one output
    x_train, y_train, x_test, y_test = read_data_3d_50()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    input_length = x_train[0].shape[1]
    output_length = y_train.shape[1]
    print(input_length)
    print(output_length)

    train_samples = len(x_train)
    test_samples = len(x_test)

    # model defination
    model = Sequential()
    #model.add(TimeDistributed(Dense(200, input_shape = (None, 75), activation='relu')))
    #model.add(Dense(200, activation='relu'))
    model.add(SimpleRNN(400, #return_sequences=True,
        input_shape = (50, 75)))
    #model.add(Dropout(0.5))
    #model.add(TimeDistributed(Dense(200, activation='relu')))
    #model.add(TimeDistributed(Dense(2, activation='softmax')))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    opt = SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy'])
    epochs = 150
    model.summary()

    def train_generator():
        while True:
            i = np.random.randint(0, train_samples)
            x_out = np.array([x_train[i]])
            steps = x_out.shape[1]
            y_out = []
            '''
            for k in range(steps):
                y_out.append(y_train[i])
            y_out = np.array([y_out])
            '''
            y_out = np.array([y_train[i]])
            yield x_out, y_out

    def test_generator():
        x_iter = iter(x_test)
        y_iter = iter(y_test)
        while True:
            x_out = np.array([next(x_iter)])
            y_out = np.array([next(y_iter)])
            yield x_out, y_out


    for i in range(10):
        model.fit(x_train, y_train, epochs = 10, 
            batch_size=10, verbose=True)
        data, accuracy = model.evaluate(x_test, y_test)
        print(data)
        print(accuracy)
    #print(model.predict(x_test))
    '''
    #model.fit_generator(train_generator(),steps_per_epoch=train_samples, 
    #    epochs=epochs, verbose=1)


    for xt, yt in test_generator():
        result = model.predict(xt)
        print(result)
        print(yt)

    #result = model.evaluate_generator(test_generator(), steps = test_samples)
    '''

def main_3d():
    # this version has varaible number of timesteps
    # and sequence output
    x_train, y_train, x_test, y_test = read_data_3d_selected()
    #x_train, y_train, x_test, y_test = read_data_2d()
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(y_train.shape)
    print(y_test.shape)

    input_length = x_train[0].shape[1]
    output_length = y_train.shape[1]
    print(input_length)
    print(output_length)

    train_samples = len(x_train)
    test_samples = len(x_test)
    rnn_size = 200
    dense_size = 200

    # model defination
    model = Sequential()
    #model.add(TimeDistributed(Dense(200, input_shape = (None, 75), activation='relu')))
    #model.add(Dense(200, activation='relu'))
    model.add(SimpleRNN(rnn_size, return_sequences=True,
        input_shape = (None, input_length)))
    #model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(dense_size, activation='relu')))
    model.add(TimeDistributed(Dense(output_length, activation='softmax')))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    opt = SGD(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy'])
    epochs = 200
    model.summary()

    def train_generator():
        while True:
            i = np.random.randint(0, train_samples)
            x_out = np.array([x_train[i]])
            steps = x_out.shape[1]
            y_out = []
            for k in range(steps):
                y_out.append(y_train[i])
            y_out = np.array([y_out])
            #y_out = np.array([y_train[i]])
            yield x_out, y_out

    def test_generator():
        x_iter = iter(x_test)
        y_iter = iter(y_test)
        while True:
            x_out = np.array([next(x_iter)])
            steps = x_out.shape[1]
            y_sample = next(y_iter)
            y_out = []
            for k in range(steps):
                y_out.append(y_sample)
            y_out = np.array([y_out])
            #y_out = np.array([next(y_iter)])
            yield x_out, y_out

    def scheduler(epoch):
        if epoch == 70:
            K.set_value(model.optimizer.lr, 0.0005)
        return K.get_value(model.optimizer.lr)

    #print(model.predict(x_test))
    change_lr = keras.callbacks.LearningRateScheduler(scheduler)

    for i in range(15):
        model.fit_generator(train_generator(),steps_per_epoch=train_samples, 
            epochs=10, verbose=1)
        result = model.evaluate_generator(test_generator(), steps = test_samples)
        if result[0] < 0.3:
            K.set_value(model.optimizer.lr, 0.00005)
        print(result)

    '''
    model.save('test.h5')
    new_model = keras.models.load_model('./test.h5')

    print(new_model.evaluate_generator(test_generator(), steps = test_samples))


    for xt, yt in test_generator():
        result = model.predict(xt)
        print(result)
        print(yt)
    '''

def main_2d():
    # this version has varaible number of timesteps
    # and sequence output
    #x_train, y_train, x_test, y_test = read_data_3d_selected()
    x_train, y_train, x_test, y_test = read_data_2d()
    np.savez('training_data.npz', x_train = x_train, y_train = y_train,
        x_test = x_test, y_test = y_test)
    print(x_train[0].shape)
    print(x_train[1].shape)
    print(y_train.shape)
    print(y_test.shape)

    input_length = x_train[0].shape[1]
    output_length = y_train.shape[1]
    print(input_length)
    print(output_length)

    train_samples = len(x_train)
    test_samples = len(x_test)
    rnn_size = 200
    dense_size = 100

    # model defination
    model = Sequential()
    #model.add(TimeDistributed(Dense(200, input_shape = (None, 75), activation='relu')))
    #model.add(Dense(200, activation='relu'))
    model.add( keras.layers.Dropout(0.02, noise_shape = (None, None, input_length)))
    model.add(SimpleRNN(rnn_size, return_sequences=True,
    #model.add(LSTM(rnn_size, return_sequences=True,
        input_shape = (None, input_length)))
    #model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(dense_size, activation='relu')))
    model.add(TimeDistributed(Dense(output_length, activation='softmax')))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dense(100, activation='relu'))
    opt = SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
        metrics=['accuracy'])
    epochs = 200
    #model.summary()

    def train_generator():
        while True:
            i = np.random.randint(0, train_samples)
            x_out = np.array([x_train[i]])
            steps = x_out.shape[1]
            y_out = []
            for k in range(steps):
                y_out.append(y_train[i])
            y_out = np.array([y_out])
            #y_out = np.array([y_train[i]])
            yield x_out, y_out

    def test_generator():
        x_iter = iter(x_test)
        y_iter = iter(y_test)
        while True:
            x_out = np.array([next(x_iter)])
            steps = x_out.shape[1]
            y_sample = next(y_iter)
            y_out = []
            for k in range(steps):
                y_out.append(y_sample)
            y_out = np.array([y_out])
            #y_out = np.array([next(y_iter)])
            yield x_out, y_out

    def scheduler(epoch):
        if epoch == 60:
            K.set_value(model.optimizer.lr, 0.001)
        if epoch == 120:
            K.set_value(model.optimizer.lr, 0.0001)
        return K.get_value(model.optimizer.lr)

    #print(model.predict(x_test))
    change_lr = keras.callbacks.LearningRateScheduler(scheduler)
    for i in range(40):
        train_result = model.fit_generator(train_generator(),steps_per_epoch=train_samples, 
            epochs=10, verbose=1)
        result = model.evaluate_generator(test_generator(), steps = test_samples)
        print('Epochs:'+str(i*10))
        print(str(result))
        if result[0] < 0.6:
            K.set_value(model.optimizer.lr, 0.0001)
        if result[0] < 0.5:
            K.set_value(model.optimizer.lr, 0.00001)
        if result[0] < 0.4:
            K.set_value(model.optimizer.lr, 0.000001)
        model.save('./models/'+str(10+i*10)+'.h5')


    '''
    model.save('test.h5')
    new_model = keras.models.load_model('./test.h5')

    print(new_model.evaluate_generator(test_generator(), steps = test_samples))


    for xt, yt in test_generator():
        result = model.predict(xt)
        print(result)
        print(yt)
    '''


if __name__ == '__main__':
    main_2d()