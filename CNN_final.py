import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np

LR = 1e-3
MODEL_NAME = 'shapes'.format(LR,'shapes_identi_shapes')
#train_data = np.load('train_data.npy')
#print(train_data.shape)
##X=np.load("Xshape_nn.npy")
##y=np.load("y_nn.npy")

X= np.load ('drive/X_new.npy',encoding = 'latin1')
np.random.shuffle(X)


print("data loaded")
convnet = input_data(shape=[None, 64, 64, 1], name='input')

convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 512, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 512, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 512, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

#convnet = conv_2d(convnet, 128, 5, activation='relu')
#convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 4096, activation='relu')
convnet = dropout(convnet, 0.8)

#convnet = fully_connected(convnet, 128, activation='softmax')
convnet = fully_connected(convnet, 13, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log',tensorboard_verbose=3)
#file_writer = tf.summary.FileWriter('/Users/swati/Desktop/train_shapes/log', sess.graph)


    

#train = train_data[:-10]
#test = train_data[-10:]
X1=X[:-62800]
test=X[-62800:]
#X_train = np.array(X[:-10]).reshape(-1,32,32,1)
Y_train = np.array([i[1] for i in X1])

X_train = np.array([i[0] for i in X1]).reshape(-1,64,64,1)   
    

#X = tf.reshape(X,shape = [-1,50,50,1])
#Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,64,64,1)
#print(test_x.shape)
#text_x = tf.reshape(text_x,shape = [-1,50,50,1])
#test_y = [i[1] for i in test]
#test_x = np.array(X[-10:]).reshape(-1,32,32,1)
test_y = np.array([i[1] for i in test])
model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=25, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)
