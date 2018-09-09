import tensorflow as tf
import numpy as np
from scipy.linalg import  orth
from total_function import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 100
LR = 0.001 #learning rate


test_x = mnist.test.images[:2000]
test_new = test_x/255
test_y = mnist.test.labels[:2000]

print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape) # (55000, 10)

b_x,b_y = mnist.train.next_batch(8000)
tf_x = b_x/255
#tf_x = tf.placeholder(tf.float32,[None,28*28])/255
image = tf.reshape(tf_x,[-1,28,28,1])
tf_y = b_y
#tf_y = tf.placeholder(tf.float32,[None,10])

# 定义权重和偏置的初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#stddev表示标准差0.1
    return  tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

# 定义卷积层函数
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
# 定义池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

whole_W = []
whole_biase = []
whole_wfc = []
whole_fcbiase = []
ave_whole_W = []
ave_whole_biase = []
ave_whole_wfc = []
ave_wfcbiase = []
def conv(image):
    W = weight_variable([5,5,1,16])
    biase = bias_variable([16])
    w_fc = weight_variable([14*14*16,10])
    w_bias = bias_variable([10])
    whole_W.append(W)
    whole_biase.append(biase)
    whole_wfc.append(w_fc)
    whole_fcbiase.append(w_bias)
    h_conv = tf.nn.relu(conv2d(image,W)+biase)
    h_pool = max_pool_2x2(h_conv)
    h_pool_new = tf.reshape(h_pool,[-1,14*14*16])
    h_fc = tf.nn.relu(tf.matmul(h_pool_new,w_fc)+w_bias)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #sess.run(h_fc)
        h_fc_new = sess.run(h_fc)
    return h_fc_new


def cnn_bls(image,tf_x,tf_y,test_x,test_y,s,C,N1,N2,N3):

    y = np.zeros((np.shape(tf_x)[0],N2*N1))
    Betal = []
    We = []
    for i in range(N2):
        featuremap = conv(image)
        #Betal.append(featuremap)
        print('feature nodes in window %d:Max Val of output %f Min val %f' % (i ,getMax(conv(image)), getMin(conv(image))))
        #print(np.shape(y[:,i*N1:(i+1)*N1] ))
        y[:,i*N1:(i+1)*N1] = featuremap

     # enhancement node
    H2 = np.hstack((y, 0.1 * np.ones((np.shape(y)[0],1))))
    if N1 * N2 >=N3:
        wh = orth((2*np.random.random((N1*N2+1, N3))-1))
    else:
        wh = orth((2*np.random.random((N1*N2+1, N3)).T-1)).T
    T2 = np.dot(H2,wh)
    l2 = np.max(np.max(T2))
    l2 = s/l2
    print('enhancement nodes:max val of output %f min val %f' % (l2, getMin(T2)))

    T2 = tansig(T2 * l2)
    T3 = np.hstack((y, T2))
    #print(np.shape(np.dot(T3.T,T3)))
    #print(nlg.matrix_rank(np.dot(T3.T,T3)))
    WA = np.dot(T3.T,T3)+np.eye(np.shape(T3.T)[0]) * C
    #print(nlg.matrix_rank(WA))
    WA_inv = nlg.inv(WA)
    T3_train = np.dot(T3.T,tf_y)
    beta = np.dot(WA_inv,T3_train) # 权重值
    #endtime = datetime.datetime.now()
    #time = (endtime - starttime).seconds
    print('Training has been finished!')
    #print('the total training time is : %d seconds'% time)

    # Training Accuracy
    xx = np.dot(T3, beta)
    yy = result(xx)
    train_yy = result(tf_y)
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
    test_x_new = tf.reshape(test_x/255,[-1,28,28,1])
    # for i in range(N2):
    #     ave_whole_W.append((whole_W[i]+whole_W[N2+i]+whole_W[2*N2+i]+whole_W[3*N2+1]+whole_W[4*N2+1])/5)
    #     ave_whole_biase.append((whole_biase[i]+whole_biase[N2+i]+whole_biase[2*N2+i]+whole_biase[3*N2+1]+whole_biase[4*N2+1])/5)
    #     ave_whole_wfc.append((whole_wfc[i]+whole_wfc[N2+i]+whole_wfc[2*N2+i]+whole_wfc[3*N2+1]+whole_wfc[4*N2+1])/5)
    #     ave_wfcbiase.append(whole_fcbiase[i]+whole_fcbiase[N2+i]+whole_fcbiase[2*N2+i]+whole_fcbiase[3*N2+1]+whole_fcbiase[4*N2+1])
    for i in range(N2):
        #beta_test = Betal[i]
        #TT1 = np.dot(HH1,beta_test)
        #TT1 = mapminmax_onezero(TT1)
        w_feature = whole_W[i]
        w_biase = whole_biase[i]
        w_fc = whole_wfc[i]
        w_fcbiase = whole_fcbiase[i]
        h_conv = tf.nn.relu(conv2d(test_x_new,w_feature)+w_biase)
        h_pool = max_pool_2x2(h_conv)
        h_pool_new = tf.reshape(h_pool,[-1,14*14*16])
        h_fc = tf.nn.relu(tf.matmul(h_pool_new,w_fc)+w_fcbiase)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            #sess.run(h_fc)
            TT1 = sess.run(h_fc)
        #TT1 = conv(test_x_new)
        yy1[:,N1*i:N1*(i+1)] = TT1
    HH2 = np.hstack((yy1,0.1*np.ones((np.shape(yy1)[0],1))))
    TT2 = tansig(np.dot(HH2,wh)*l2)
    TT3 = np.hstack((yy1,TT2))

    # testing accuracy
    x = np.dot(TT3,beta)
    y = result(x)
    test_yy = result(test_y)
    test_total = 0
    for j in range(np.shape(test_yy)[0]):
        if(y[j] == test_yy[j]):
            test_total += 1
    print('total:%d' % test_total)
    testingAccuracy = float(test_total/np.shape(test_yy)[0])
    #test_endtime = datetime.datetime.now()
    #test_time = (test_endtime - test_starttime).seconds
    print('testing has benn finished')
   #print('test time is : %d seonds' %test_time)
    print('testing accuracy is:%d%%' % (testingAccuracy*100))



#b_x = mnist.train.images[:]
#b_y = mnist.train.labels[:]
for i in range(1):

     cnn_bls(image,b_x,b_y,test_x,test_y,0.8,2e-30,10,6,150)