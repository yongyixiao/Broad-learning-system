from cnn_model_test import *
import tensorflow as tf
import numpy as np
import numpy.linalg as nlg
import os
from scipy.linalg import  orth
from total_function import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 100
LR = 0.001 #learning rate


test_x = mnist.test.images[:1000]
test_new = test_x/255
test_y = mnist.test.labels[:1000]

print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape) # (55000, 10)

n_input = 784
n_class = 10

# tf GRAPH input
#x = tf.placeholder(tf.float32,[None,n_input])
#y = tf.placeholder(tf.float32,[None,n_class])

# def conv2d(x,W):
#     return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#
# def max_pool(x):
#     return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
#
# def init_parameter(filter):
#     #tf.reset_default_graph()
#     weights = {
#         'conv' : tf.Variable(tf.truncated_normal([5,5,1,filter],stddev=0.1)),
#         'out' : tf.Variable(tf.truncated_normal([14*14*filter,10],stddev=0.1))
#     }
#     biase = {
#         'conv_bias' : tf.Variable(tf.constant(0.1,shape=[filter])),
#         'out_bias' : tf.Variable(tf.constant(0.1,shape=[10]))
#     }
#     return weights,biase
# def initdouble_parameter(filter1,filter2):
#     #tf.reset_default_graph()
#     weights = {
#         'conv1' : tf.Variable(tf.truncated_normal([5,5,1,filter1],stddev=0.1)),
#         'conv2' : tf.Variable(tf.truncated_normal([5,5,filter1,filter2],stddev=0.1)),
#         'out' : tf.Variable(tf.truncated_normal([7*7*filter2,10],stddev=0.1))
#     }
#     biase = {
#         'conv1_bias' : tf.Variable(tf.constant(0.1,shape=[filter1])),
#         'conv2_bias' : tf.Variable(tf.constant(0.1,shape=[filter2])),
#         'out_bias' : tf.Variable(tf.constant(0.1,shape=[10]))
#     }
#     return weights,biase
#
# #create model
# def multilayer_forword(x,weights,biase,filter):
#     x_image = tf.reshape(x,[-1,28,28,1])
#     h_conv = tf.nn.relu(conv2d(x_image,weights['conv']) + biase['conv_bias'])
#     h_pool = tf.nn.relu(max_pool(h_conv))
#     h_pool_flat = tf.reshape(h_pool,[-1,14*14*filter])
#     h_out  = tf.nn.relu(tf.matmul(h_pool_flat,weights['out'])+biase['out_bias'])
#     return  h_out
#
# #create double layer model
# def doublelayer_forward(x,weights,biases,filter1,filter2):
#     x_image = tf.reshape(x,[-1,28,28,1])
#     h_conv1 = tf.nn.relu(conv2d(x_image,weights['conv1'])+biases['conv1_bias'])
#     h_pool1 = max_pool(h_conv1)
#     h_conv2 = tf.nn.relu(conv2d(h_pool1,weights['conv2'])+biases['conv2_bias'])
#     h_pool2 = max_pool(h_conv2)
#     h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*filter2])
#     h_out = tf.nn.softmax(tf.matmul(h_pool2_flat,weights['out'])+biases['out_bias'])
#     return h_out
#
#
# # whole_W = []
# # whole_biase = []
# # whole_wfc = []
# # whole_fcbiase = []
# # ave_whole_W = []
# # ave_whole_biase = []
# # ave_whole_wfc = []
# # ave_wfcbiase = []
#
#
#
#
# def conv_single(b_x,filter):
#     # W = weight_variable([5,5,1,16])
#     # biase = bias_variable([16])
#     # w_fc = weight_variable([14*14*16,10])
#     # w_bias = bias_variable([10])
#     # whole_W.append(W)
#     # whole_biase.append(biase)
#     # whole_wfc.append(w_fc)
#     # whole_fcbiase.append(w_bias)
#     # h_conv = tf.nn.relu(conv2d(image,W)+biase)
#     # h_pool = max_pool_2x2(h_conv)
#     # h_pool_new = tf.reshape(h_pool,[-1,14*14*16])
#     # h_fc = tf.nn.relu(tf.matmul(h_pool_new,w_fc)+w_bias)
#     # with tf.Session() as sess:
#     #     init = tf.global_variables_initializer()
#     #     sess.run(init)
#     #     #sess.run(h_fc)
#     #     h_fc_new = sess.run(h_fc)
#     #tf.reset_default_graph()
#     weights,biase = init_parameter(filter)
#     ret = multilayer_forword(b_x,weights,biase,filter)
#     with tf.Session() as sess:
#         #with g1.as_default():
#             model_dir = "mnist"
#             model_name = np.str(filter)+"single"+"ckpt"
#             model_path = os.path.join(model_dir,model_name)
#             #x_image = tf.reshape(b_x,[-1,28,28,1])
#             # h_conv = tf.nn.relu(conv2d(x_image,weights['conv1']) + biase['conv_bias'])
#             # h_pool = tf.nn.relu(max_pool(h_conv))
#             # h_pool_flat = tf.reshape(h_pool,[-1,14*14*16])
#             # h_out  = tf.nn.relu(tf.matmul(h_pool_flat,weights['out'])+biase['out_bias'])
#             model_saver = tf.train.Saver()
#             model_saver.restore(sess,model_path)
#             pret = sess.run(ret,feed_dict={x : b_x})
#         #print(pret)
#         #print(np.shape(pret))
#     return pret
#
# def conv_double(b_x,filter1,filter2):
#     #tf.reset_default_graph()
#     weights,biases = initdouble_parameter(filter1,filter2)
#     ret = doublelayer_forward(
# ,weights,biases,filter1,filter2)
#     with tf.Session() as sess:
#         model_dir = "mnist"
#         model_name = np.str(filter2)+"double"+"ckpt"
#         model_path = os.path.join(model_dir,model_name)
#         model_saver = tf.train.Saver()
#         model_saver.restore(sess,model_path)
#         pret = sess.run(ret,feed_dict={x:b_x})
#     print(pret)

# def test_singleforward(x,filter):
#     weights,biases = init_parameter(filter)
#     pred = multilayer_forword(x,weights,biases)
#     with tf.Session() as sess:
#             # init = tf.global_variables_initializer()
#             # sess.run(init)
#             # #sess.run(h_fc)
#             # TT1 = sess.run(h_fc)
#            model_dir = "mnist"
#            model_name = np.str(filter)+"single"+"ckpt"
#            model_path = os.path.join(model_dir,model_name)
#            model_saver.restore(sess,model_path)
#            TT1 = sess.run(pred,feed_dict={x:x,y:y})


def cnn_bls(x,y,test_x,test_y,s,C,N1,N2,N3):

    ynew = np.zeros((np.shape(x)[0],N2*N1))
    Betal = []
    We = []
    for i in range(N2):
        if i==0:
           featuremap = init_feature1(x,8)
        elif i==1:
           featuremap = init_feature2(x,16)
           print(np.shape(featuremap))
        elif i==2:
           featuremap = init_feature3(x,32)
        elif i==3:
           featuremap = initdouble_feature1(x,8,16)
        elif i==4:
           featuremap = initdouble_feature2(x,16,32)
        elif i==5:
           featuremap = initdouble_feature3(x,32,64)

        #Betal.append(featuremap)
        print('feature nodes in window %d:Max Val of output %f Min val %f' % (i ,getMax(featuremap), getMin(featuremap)))
        #print(np.shape(y[:,i*N1:(i+1)*N1] ))
        ynew[:,i*N1:(i+1)*N1] = featuremap

     # enhancement node
    H2 = np.hstack((ynew, 0.1 * np.ones((np.shape(ynew)[0],1))))
    if N1 * N2 >=N3:
        wh = orth((2*np.random.random((N1*N2+1, N3))-1))
    else:
        wh = orth((2*np.random.random((N1*N2+1, N3)).T-1)).T
    T2 = np.dot(H2,wh)
    l2 = np.max(np.max(T2))
    l2 = s/l2
    print('enhancement nodes:max val of output %f min val %f' % (l2, getMin(T2)))

    T2 = tansig(T2 * l2)
    T3 = np.hstack((ynew, T2))
    #print(np.shape(np.dot(T3.T,T3)))
    #print(nlg.matrix_rank(np.dot(T3.T,T3)))
    WA = np.dot(T3.T,T3)+np.eye(np.shape(T3.T)[0]) * C
    #print(nlg.matrix_rank(WA))
    WA_inv = nlg.inv(WA)
    T3_train = np.dot(T3.T,y)
    beta = np.dot(WA_inv,T3_train) # 权重值
    #endtime = datetime.datetime.now()
    #time = (endtime - starttime).seconds
    print('Training has been finished!')
    #print('the total training time is : %d seconds'% time)

    # Training Accuracy
    xx = np.dot(T3, beta)
    yy = result(xx)
    train_yy = result(y)
    total = 0
    for j in range(np.shape(yy)[0]):
        if(yy[j] == train_yy[j]):
            total += 1
    print('total:%d' % total)
    trainingAccuracy = float(total/np.shape(yy)[0])
    print('Training Accuracy is:%d%%' % (trainingAccuracy*100))

    #numer = k
    #print(k)
    #if(np.equal(k , 4)):
    #HH1 = np.hstack((test_x, 0.1 * np.ones((np.shape(test_x)[0],1))))
    yy1 = np.zeros((np.shape(test_x)[0],N2*N1))
    text_x_new = test_x/255
    #test_x_new = tf.reshape(test_x/255,[-1,28,28,1])
    for i in range(N2):
        if i==0:
           TT1 = init_feature1(test_x,8)
        elif i==1:
           TT1 = init_feature2(test_x,16)
        elif i==2:
            TT1 = init_feature3(test_x,32)
        elif i==3:
            TT1 = initdouble_feature1(test_x,8,16)
        elif i==4:
            TT1 = initdouble_feature2(test_x,16,32)
        elif i==5:
            TT1 = initdouble_feature3(test_x,32,64)

        yy1[:,N1*i:N1*(i+1)] = TT1
    HH2 = np.hstack((yy1,0.1*np.ones((np.shape(yy1)[0],1))))
    TT2 = tansig(np.dot(HH2,wh)*l2)
    TT3 = np.hstack((yy1,TT2))

    # testing accuracy
    x_test = np.dot(TT3,beta)
    print(np.shape(x_test))
    y_test = result(x_test)
    test_yy = result(test_y)
    test_total = 0
    for j in range(np.shape(y_test)[0]):
        if(y_test[j] == test_yy[j]):
            test_total += 1
    print('total:%d' % test_total)
    testingAccuracy = float(test_total/np.shape(y_test)[0])
    #test_endtime = datetime.datetime.now()
    #test_time = (test_endtime - test_starttime).seconds
    print('testing has benn finished')
   #print('test time is : %d seonds' %test_time)
    print('testing accuracy is:%d%%' % (testingAccuracy*100))



# b_x = mnist.train.images[:]
# b_y = mnist.train.labels[:]
b_x,b_y = mnist.train.next_batch(3000)
print(np.shape(b_x))
print(np.shape(b_y))
for i in range(1):
     cnn_bls(b_x,b_y,test_x,test_y,0.8,2e-30,10,6,150)