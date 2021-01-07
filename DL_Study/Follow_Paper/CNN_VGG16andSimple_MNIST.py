#MNIST dataset을 224x224로 늘려 input, output으로 사용하였습니다.
#VGG16 model은 실행완료까지 너무 오래걸려 Simple model을 만들어두었습니다.

import os
import numpy as np #numpy ver1.19.4
import tensorflow as tf #tensorflow ver2.4.0
import matplotlib.pyplot as plt #matplotlib ver3.3.3
from tensorflow.keras import layers, models #tensorflow ver2.4.0
from scipy.misc import imresize #scipy ver1.1.0


mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#이미지 값 0~1로 정규화
train_images, test_images = train_images/255. , test_images/255.
#라벨 One-Hot 인코딩
train_labels_oneHot = [[0]*10 for _ in train_labels]
for index, value in enumerate(train_labels):
    train_labels_oneHot[index][value] = 1

test_labels_oneHot = [[0]*10 for _ in test_labels]
for index, value in enumerate(test_labels):
    test_labels_oneHot[index][value] = 1

train_labels_oneHot = np.array(train_labels_oneHot)
test_labels_oneHot = np.array(test_labels_oneHot)

#data 섞어서 나누는 함수 train/validation split용
def data_split(input_data, output_data, split_rate):
    ninput = input_data.shape[0]
    randarray = np.random.permutation(ninput)
    splitted_ninput = int(round(ninput*split_rate))
    new_input_data = np.zeros(shape=(ninput-splitted_ninput,input_data.shape[1],input_data.shape[2],input_data.shape[3]))
    new_output_data = np.zeros(shape=(ninput-splitted_ninput,output_data.shape[1]))

    splitted_input_data = np.zeros(shape=(splitted_ninput, input_data.shape[1], input_data.shape[2], input_data.shape[3]))
    splitted_output_data = np.zeros(shape=(splitted_ninput, output_data.shape[1]))

    for index in range(ninput):
        if index < ninput-splitted_ninput:
            new_input_data[index] = input_data[randarray[index]]
            new_output_data[index] = output_data[randarray[index]]
        else:
            splitted_input_data[index-(ninput-splitted_ninput+1)] = input_data[randarray[index]]
            splitted_output_data[index-(ninput-splitted_ninput+1)] = output_data[randarray[index]]

    return new_input_data, new_output_data, splitted_input_data, splitted_output_data


#VGG16 model
def VGG16Model(train_images, test_images, train_labels_oneHot, test_labels_oneHot, is_train=True):
    #VGG16용 28x28 MNIST -> 224x224 MNIST 변환
    resized_train_images = np.zeros(shape=(train_images.shape[0], 224, 224))
    resized_test_images = np.zeros(shape=(test_images.shape[0], 224, 224))
    for index in range(train_images.shape[0]):
        resized_train_images[index] = imresize(train_images[index], [224, 224,1])
    for index in range(test_images.shape[0]):
        resized_test_images[index] = imresize(test_images[index], [224,224,1])

    #shape조정
    resized_train_images = resized_train_images.reshape(-1, 224, 224, 1)
    resized_test_images = resized_test_images.reshape(-1, 224, 224, 1)

    resized_train_images, train_labels_oneHot, validation_images, validation_labels_oneHot = data_split(input_data=resized_train_images,
                                                                                                        output_data=train_labels_oneHot,
                                                                                                        split_rate=5000. / 60000.)
    if is_train:
    # 저장된 모델 없을 경우 모델 생성
        if not os.path.isfile('./MNIST_VGG16_MODEL.h5'):
            model = models.Sequential()
            #conv1
            model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', input_shape=(224, 224,1)))
            model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
            #conv2
            model.add(layers.Conv2D(filters=128, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=128, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
            #conv3
            model.add(layers.Conv2D(filters=256, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=256, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=256, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
            #conv4
            model.add(layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1,),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
            #conv5
            model.add(layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
            model.add(layers.Flatten())
            #fully connected
            model.add(layers.Dense(units=4096, activation='relu'))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.Dense(units=4096, activation='relu'))
            model.add(layers.Dropout(rate=0.5))
            #10 classes
            model.add(layers.Dense(units=10, activation='softmax'))
        #저장된 모델 존재하면 이미 학습된 모델 학습
        else:
            model = models.load_model('MNIST_VGG16_MODEL.h5')
        #model.summary()

        #학습
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        history = model.fit(x=resized_train_images, y=train_labels_oneHot, epochs=1, batch_size=100, validation_data=(validation_images, validation_labels_oneHot))
        # 저장
        model.save('MNIST_VGG16_MODEL.h5')
    # is_train=False면 저장 데이터 있을 경우 모델 load 없으면 'No saved model'문구 출력
    else :
        if not os.path.isfile('./MNIST_VGG16_MODEL.h5'):
            print('No saved model')
        else:
            model = models.load_model('MNIST_VGG16_MODEL.h5')

    #평가
    test_loss, test_acc = model.evaluate(resized_test_images, test_labels_oneHot, verbose=1)



#Simple model
def SimpleModel(train_images, test_images, train_labels_oneHot, test_labels_oneHot, is_train=True):
    # shape 조정
    train_images = train_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)

    # train / validation data 분리
    train_images, train_labels_oneHot, validation_images, validation_labels_oneHot = data_split(input_data=train_images,
                                                                                                output_data=train_labels_oneHot,
                                                                                                split_rate=5000. / 60000.)
    if is_train:
        # 저장된 모델 없을 경우 모델 생성
        if not os.path.isfile('./MNIST_SIMPLE_MODEL.h5'):
            model = models.Sequential()
            model.add(layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', input_shape=(28, 28,1)))
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid'))
            model.add(layers.Flatten())
            model.add(layers.Dense(units=100, activation='relu'))
            model.add(layers.Dropout(rate=0.5))
            model.add(layers.Dense(units=10, activation='softmax'))
        else:
            model = models.load_model('MNIST_SIMPLE_MODEL.h5')
        #학습
        model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
        history = model.fit(x=train_images, y=train_labels_oneHot, epochs=1, batch_size=50,validation_data=(validation_images, validation_labels_oneHot))
        #저장
        model.save('MNIST_SIMPLE_MODEL.h5')
    #is_train=False면 저장 데이터 있을 경우 모델 load 없으면 'No saved model'문구 출력
    else :
        if not os.path.isfile('./MNIST_SIMPLE_MODEL.h5'):
            print('No saved model')
        else:
            model = models.load_model('MNIST_SIMPLE_MODEL.h5')
    #평가
    test_loss, test_acc = model.evaluate(test_images, test_labels_oneHot, verbose=50)
    print("test_loss : ", test_loss)
    print("test_acc : ", test_acc)


SimpleModel(train_images, test_images, train_labels_oneHot, test_labels_oneHot, is_train=True)
#VGG16Model(train_images, test_images, train_labels_oneHot, test_labels_oneHot, is_train=False)