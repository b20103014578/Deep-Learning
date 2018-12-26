import numpy as np
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# building a deep learning model
model = Sequential()
# Add a input layer and hidden layers
model.add(Dense(units=256, input_dim = 28*28, kernel_initializer='normal', activation='relu'))
# Add a output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
# Select compile params
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert y to one-hot
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

# Convert x to 2D
x_train_2D = x_train.reshape(60000, 28*28).astype('float32')
x_test_2D = x_test.reshape(10000, 28*28).astype('float32')

x_train_norm = x_train_2D/256
x_test_norm = x_test_2D/256

# Training and save in train_history
train_history = model.fit(x=x_train_norm, y=y_trainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

# Training result
scores = model.evaluate(x_test_norm, y_testOneHot)
print('\t[Info] Accuracy of testing datas = {:2.1f}%'.format(scores[1]*100.))

# Prediction
x = x_test_norm[0:10, :]
predictions = model.predict_classes(x)
print(predictions)

# display training result
# plt.imshow(x_test[0])
# plt.show()
