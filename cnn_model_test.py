import tensorflow as tf
import numpy as np
from scipy.linalg import  orth
from total_function import *
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

BATCH_SIZE = 100
LR = 0.001 #learning rate

n_input = 784
n_class = 10

# tf GRAPH input

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def init_parameter(filter):
    weights = {
        'conv' : tf.Variable(tf.truncated_normal([5,5,1,filter],stddev=0.1),name="singleweight"+np.str(filter)),
        'out' : tf.Variable(tf.truncated_normal([14*14*filter,10],stddev=0.1),name="singleout"+np.str(filter))
    }
    biase = {
        'conv_bias' : tf.Variable(tf.constant(0.1,shape=[filter]),name="singlebias"+np.str(filter)),
        'out_bias' : tf.Variable(tf.constant(0.1,shape=[10]),name="singlebias"+np.str(filter))
    }
    return weights,biase

def initdouble_parameter(filter1,filter2):
    weights = {
        'conv1' : tf.Variable(tf.truncated_normal([5,5,1,filter1],stddev=0.1),name="conv1weight"+np.str(filter2)),
        'conv2' : tf.Variable(tf.truncated_normal([5,5,filter1,filter2],stddev=0.1),name="conv2weight"+np.str(filter2)),
        'out' : tf.Variable(tf.truncated_normal([7*7*filter2,10],stddev=0.1),name="outweight"+np.str(filter2))
    }
    biase = {
        'conv1_bias' : tf.Variable(tf.constant(0.1,shape=[filter1]),name="conv1bias"+np.str(filter2)),
        'conv2_bias' : tf.Variable(tf.constant(0.1,shape=[filter2]),name="conv1bias"+np.str(filter2)),
        'out_bias' : tf.Variable(tf.constant(0.1,shape=[10]),name="conv1bias"+np.str(filter2))
    }
    return weights,biase
#create model
def multilayer_forword(x,weights,biases,filter):
    x_image = tf.reshape(x,[-1,28,28,1])
    h_conv = tf.nn.relu(conv2d(x_image,weights['conv']) + biases['conv_bias'])
    h_pool = tf.nn.relu(max_pool(h_conv))
    h_pool_flat = tf.reshape(h_pool,[-1,14*14*filter])
    h_out  = tf.nn.relu(tf.matmul(h_pool_flat,weights['out'])+biases['out_bias'])
    return  h_out

#create double layer model
def doublelayer_forward(x,weights,biases,filter1,filter2):
    x_image = tf.reshape(x,[-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image,weights['conv1'])+biases['conv1_bias'])
    h_pool1 = max_pool(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1,weights['conv2'])+biases['conv2_bias'])
    h_pool2 = max_pool(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*filter2])
    h_out = tf.nn.softmax(tf.matmul(h_pool2_flat,weights['out'])+biases['out_bias'])
    return h_out

def init_feature1(b_x,filter):


    g1 = tf.Graph()
    gsess = tf.Session(graph=g1)
    with g1.as_default():
        x = tf.placeholder(tf.float32,[None,n_input])/255
        y = tf.placeholder(tf.float32,[None,n_class])
        weights,biase = init_parameter(filter)
        pred = multilayer_forword(x,weights,biase,filter)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()
        gsess.run(init)
        model_dir = "mnist"
        model_name = np.str(filter)+"singleckpt"
        model_path=os.path.join(model_dir,model_name)
        model_saver = tf.train.Saver()
        model_saver.restore(gsess,model_path)
        #img = mnist.test.images[:20].reshape(-1,784)
        img_label = gsess.run(tf.argmax(mnist.test.labels[:20],1))
        ret = gsess.run(pred,feed_dict={x:b_x})
        #num = gsess.run(tf.argmax(ret,1))
        #print("model saved successfully")
        # print( num)
        #print(img_label)
        print("model1 end")
    return ret

def init_feature2(b_x,filter):


    g2 = tf.Graph()
    gsess = tf.Session(graph=g2)
    with g2.as_default():
        x = tf.placeholder(tf.float32,[None,n_input])
        y = tf.placeholder(tf.float32,[None,n_class])
        weights,biase = init_parameter(filter)
        pred = multilayer_forword(x,weights,biase,filter)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()
        gsess.run(init)
        model_dir = "mnist"
        model_name = np.str(filter)+"singleckpt"
        model_path=os.path.join(model_dir,model_name)
        model_saver = tf.train.Saver()
        model_saver.restore(gsess,model_path)
        #img = mnist.test.images[:20].reshape(-1,784)
        img_label = gsess.run(tf.argmax(mnist.test.labels[:20],1))
        ret = gsess.run(pred,feed_dict={x:b_x})
        num = gsess.run(tf.argmax(ret,1))
        #print("model saved successfully")
        # print( num)
        # print(img_label)
        print("model2 end")
    return ret

def init_feature3(b_x,filter):


    g3 = tf.Graph()
    gsess = tf.Session(graph=g3)
    with g3.as_default():
        x = tf.placeholder(tf.float32,[None,n_input])
        y = tf.placeholder(tf.float32,[None,n_class])
        weights,biase = init_parameter(filter)
        pred = multilayer_forword(x,weights,biase,filter)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()
        gsess.run(init)
        model_dir = "mnist"
        model_name = np.str(filter)+"singleckpt"
        model_path=os.path.join(model_dir,model_name)
        model_saver = tf.train.Saver()
        model_saver.restore(gsess,model_path)
        #img = mnist.test.images[:20].reshape(-1,784)
        img_label = gsess.run(tf.argmax(mnist.test.labels[:20],1))
        ret = gsess.run(pred,feed_dict={x:b_x})
        num = gsess.run(tf.argmax(ret,1))
        #print("model saved successfully")
        # print( num)
        # print(img_label)
        print("model3 end")
    return ret

def initdouble_feature1(b_x,filter1,filter2):

    g4 = tf.Graph()
    gsess = tf.Session(graph=g4)
    with g4.as_default():
        x = tf.placeholder(tf.float32,[None,n_input])
        y = tf.placeholder(tf.float32,[None,n_class])
        weights,biase = initdouble_parameter(filter1,filter2)
        pred = doublelayer_forward(x,weights,biase,filter1,filter2)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()


        gsess.run(init)
        model_dir = "mnist"
        model_name = np.str(filter2)+"doubleckpt"
        model_path=os.path.join(model_dir,model_name)
        moder_saver = tf.train.Saver()
        moder_saver.restore(gsess,model_path)
        img = mnist.test.images[:20].reshape(-1,784)
        img_label = gsess.run(tf.argmax(mnist.test.labels[:20],1))
        ret = gsess.run(pred,feed_dict={x:b_x})
        num = gsess.run(tf.argmax(ret,1))
        #print("model saved successfully")
        # print( num)
        # print(img_label)
        print("model4 end")
    return ret

def initdouble_feature2(b_x,filter1,filter2):

    g5 = tf.Graph()
    gsess = tf.Session(graph=g5)
    with g5.as_default():
        x = tf.placeholder(tf.float32,[None,n_input])
        y = tf.placeholder(tf.float32,[None,n_class])
        weights,biase = initdouble_parameter(filter1,filter2)
        pred = doublelayer_forward(x,weights,biase,filter1,filter2)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()


        gsess.run(init)
        model_dir = "mnist"
        model_name = np.str(filter2)+"doubleckpt"
        model_path=os.path.join(model_dir,model_name)
        moder_saver = tf.train.Saver()
        moder_saver.restore(gsess,model_path)
        #img = mnist.test.images[:20].reshape(-1,784)
        img_label = gsess.run(tf.argmax(mnist.test.labels[:20],1))
        ret = gsess.run(pred,feed_dict={x:b_x})
        num = gsess.run(tf.argmax(ret,1))
        #print("model saved successfully")
        # print( num)
        # print(img_label)
        print("model5 end")
    return ret

def initdouble_feature3(b_x,filter1,filter2):

    g6 = tf.Graph()
    gsess = tf.Session(graph=g6)
    with g6.as_default():
        x = tf.placeholder(tf.float32,[None,n_input])
        y = tf.placeholder(tf.float32,[None,n_class])
        weights,biase = initdouble_parameter(filter1,filter2)
        pred = doublelayer_forward(x,weights,biase,filter1,filter2)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)


        correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        init = tf.global_variables_initializer()


        gsess.run(init)
        model_dir = "mnist"
        model_name = np.str(filter2)+"doubleckpt"
        model_path=os.path.join(model_dir,model_name)
        moder_saver = tf.train.Saver()
        moder_saver.restore(gsess,model_path)
        img = mnist.test.images[:20].reshape(-1,784)
        img_label = gsess.run(tf.argmax(mnist.test.labels[:20],1))
        ret = gsess.run(pred,feed_dict={x:b_x})
        num = gsess.run(tf.argmax(ret,1))
        #print("model saved successfully")
        # print( num)
        # print(img_label)
        print("model6 end")
    return ret
# init_feature(16)
#init_feature(8)
#init_feature1(8,16)
# b_x,b_y = mnist.train.next_batch(1000)
# init_feature1(b_x,8)
# init_feature2(b_x,16)
# init_feature3(b_x,32)
# initdouble_feature1(b_x,8,16)
# initdouble_feature2(b_x,16,32)
# initdouble_feature3(b_x,32,64)
