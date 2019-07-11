# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:57:43 2019

@author: Microtc
"""

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras import backend as K
#from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Dropout
import numpy as np
import cv2
import sys
import os
import time

#print("ZERO")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Numero de classes")
print(np.unique(y_train))

x_train = x_train.reshape(60000,28, 28, 1)
x_test = x_test.reshape(10000,28,28, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

global num_pixels
num_pixels = x_train.shape[1] * x_train.shape[2]

model_name = str(sys.argv[0]).split(".")[0]+".h5"

batch_size  = 32 #menos exemplos, mais updates
num_classes = 10
epochz      = 6 #Nao precisa mais que 1 epoch para o MNIST. Evite overfitting

global density
density     = 128

def img_to_array(img, image_data_format='default'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    if image_data_format == "default":
        image_data_format = K.image_data_format()

    if image_data_format not in ['channels_first', 'channels_last']:
        raise Exception('Unknown image_data_format: ', image_data_format)

    x = np.asarray(img, dtype=K.floatx())

    # colored image
    # (channel, height, width)
    if len(x.shape) == 3:
        if image_data_format == 'channels_first':
            x = x.transpose(2, 0, 1)

    # grayscale
    elif len(x.shape) == 2:
        # already for th format
        x = np.expand_dims(x, axis=0)
        if image_data_format == 'channels_last':
            # (height, width, channel)
            #o canal esta no 2. estranho, mas...
            #(height, channel, width)
            x = x.reshape((x.shape[1], x.shape[2], x.shape[0]))
            #print("CHANNEL")
            #print(x.shape[2])
    # unknown
    else:
        raise Exception('Unsupported image shape: ', x.shape)

    return x

def base_model():
    model = Sequential()
    # add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

#se nao tem argumento para o programa, faz o treino e salva o model com o nome do script python
if len(sys.argv) == 1:
    cnn_n = base_model()
    cnn_n.summary()
    cnn = cnn_n.fit(x_train, y_train, batch_size=batch_size, epochs=epochz, validation_data=(x_test,y_test),shuffle=True)
    cnn_n.save_weights(model_name, overwrite=True)

    scores = cnn_n.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

#se tiver argumento, faz predicao do diretorio de imagem passado como parametro
else:
    if not os.path.isdir(sys.argv[1]):
        print("Directory not found. Exiting...")
        exit(0)

    #faz loop nos arquivos e limpa oa nomes pra nao ter LF
    directory = sys.argv[1]
    files     = os.listdir(directory)

    files = [w.replace("\n", "") for w in files]
    #print(files)

    #se for diretorio vazio, sai sem dar erro
    if len(files) < 1:
        print("There is no files in \'"+directory+"\' directory. Exiting...")
        exit(0)

    #...mas se tiver arquivos, deve fazer predicao
    cnn_n = base_model()
    cnn_n.load_weights(model_name)

    font = cv2.FONT_HERSHEY_SIMPLEX

    #ler imagens de um diretorio e redimensionar para o padrao do MNIST
    for filename in files:
        print(filename)
        img = cv2.imread(directory + "/" +filename)
        img = cv2.resize(img, (28, 28), 1)

        xxx = img_to_array(img)
        #print(str(xxx.shape))
        xxx = np.expand_dims(xxx, axis=0)
        #print(str(xxx))

        start_t = time.time()

        res = cnn_n.predict(xxx, verbose=0)

        print("res.argmax e res 0 argmax")
        print(res.argmax())
        print(res[0][res.argmax()])
        #print(res[0])

        detected = res.argmax()

        print(round(res[0][res.argmax()]*100))

        imgRead = cv2.imread(directory + "/" + filename, cv2.IMREAD_GRAYSCALE)
        cv2.putText(imgRead, str(res.argmax()), (2, 22), font, 1, (100, 100, 100), 2, cv2.LINE_AA)

        cv2.imshow("MINIST", imgRead)
        pressed = cv2.waitKey(0)
        if pressed == 27:
            exit(0)
        cv2.destroyAllWindows()