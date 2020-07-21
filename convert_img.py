
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import  cv2

def convert_mnist_img(data, save_path):
    for i in range(data.images.shape[0]):
        img = data.images[i].reshape([28, 28, 1])
        img = (img * 255).astype(np.uint8)
        label = data.labels[i]
        savepath = save_path + '/{}'.format(label)

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        # cv2.imshow('image', img)
        # cv2.waitKey(10)
        filename = savepath + '/{}_{}.jpg'.format(label, i)
        cv2.imwrite(filename, img)
        print("save filename:",filename)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("./mnist/") #读取 lenet数据
    convert_mnist_img(mnist.train, 'img_train')
    print('convert training data to image complete')
    convert_mnist_img(mnist.test, 'img_test')
    print('convert test data to image complete')
    convert_mnist_img(mnist.validation, 'img_validation')
    print('convert validation data to image complete')
