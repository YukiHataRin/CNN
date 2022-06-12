from keras.engine import sequential
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
(train_feature, train_label), \
(test_feature, test_label) = mnist.load_data()

np.random.seed(10)



def show_images_labels_predictions(images, labels, predictions, start_id, num = 10):
    win = plt.gcf()
    win.set_size_inches(12, 20)
    if num > 25: num = 25
    
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[start_id], cmap = 'binary')

        if (len(predictions) > 0):
            title = 'ai = ' + str(predictions[start_id])
            if predictions[start_id] == labels[start_id]:
                title += '(o)'
            else:
                title += '(x)'
            title += '\nLabel = ' + str(labels[start_id])

        else:
            title = 'label = ' + str(labels[start_id])

        ax.set_title(title, fontsize = 12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1
    plt.show()

train_feature_vector = train_feature.reshape(len(train_feature), 784).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 784).astype('float32')

train_feature_normalize = train_feature_vector / 255
test_feature_normalize = test_feature_vector / 255

train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)

model = Sequential()
model.add(Dense(units =  256, input_dim = 784, kernel_initializer = 'normal', activation = 'relu'))
model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'softmax' ))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

train_history = model.fit(x = train_feature_normalize, y = train_label_onehot, validation_split = 0.2,
                          epochs = 10, batch_size = 200, verbose = 2)

scores = model.evaluate(test_feature_normalize, test_label_onehot)
print(">> \n準確度 = ", scores[1])

prediction = np.argmax(model.predict(test_feature_normalize), axis = -1)

show_images_labels_predictions(test_feature, test_label, prediction, 0)
try:
    print(">> Saving Model...")
    model.save('CNN\\Mnist_cnn_model.h5')
    print(">> Success!\n>> Saving weight...")
    model.save_weights('CNN\\Mnist_cnn_model.weight')
    print(">> Success!")
    del model
except:
    print(">> Error!")