# nn_npy
2 layer neural net with numpy

equivalent to the following in keras
model = Sequential()
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
