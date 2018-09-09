import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

BATCH_SIZE = 100
LR = 0.001 #learning rate

#tf.set_random_seed(1)
# np.random.seed(1)

n_input = 784
n_class = 10
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]
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

#create singal model
def multilayer_forword(x,weights,biases,filter):
    x_image = tf.reshape(x,[-1,28,28,1])
    h_conv = tf.nn.relu(conv2d(x_image,weights['conv']) + biases['conv_bias'])
    h_pool = max_pool(h_conv)
    h_pool_flat = tf.reshape(h_pool,[-1,14*14*filter])
    h_out  = tf.nn.softmax(tf.matmul(h_pool_flat,weights['out'])+biases['out_bias'])
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


def init_feature(filter):
    g3 = tf.Graph()
    gsess = tf.Session(graph=g3)
    with g3.as_default():
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
        for epoch in range(1):
            total = int(mnist.train.num_examples/BATCH_SIZE)
            for i in range(total):
                batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
                gsess.run([optimizer,loss],feed_dict={x:batch_x, y:batch_y})
                if(i%50 == 0):
                   acc = gsess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
                   print("in the step %d ,the accuracy is %f"%(i,acc))
        print("Optimal finished")
        test_output = gsess.run(pred, {x: test_x[:10]})
        pred_y = np.argmax(test_output, 1)
        print(pred_y, 'prediction number')
        print(np.argmax(test_y[:10], 1), 'real number')
        model_dir = "mnist"
        model_name = np.str(filter)+"single"+"ckpt"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        moder_saver = tf.train.Saver()
        moder_saver.save(gsess,os.path.join(model_dir,model_name))
        print("model saved successfully")

def initdouble_feature(filter1,filter2):

    g5 = tf.Graph()
    gsess = tf.Session(graph=g5)
    with g5.as_default():

        x = tf.placeholder(tf.float32,[None,n_input])/255
        y = tf.placeholder(tf.float32,[None,n_class])

        weights,biases = initdouble_parameter(filter1,filter2)
        pred = doublelayer_forward(x,weights,biases,filter1,filter2)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y,logits=pred)
        optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        init = tf.global_variables_initializer()
        gsess.run(init)

        total = int(mnist.train.num_examples/BATCH_SIZE)
        for i in range(total):
            batch_x,batch_y = mnist.train.next_batch(BATCH_SIZE)
            gsess.run([optimizer,loss],feed_dict={x:batch_x, y:batch_y})
            if(i%50 == 0):
               acc = gsess.run(accuracy,feed_dict={x:test_x,y:test_y})
               print("in the step %d ,the accuracy is %f"%(i,acc))
        print("Optimal finished")
        test_output = gsess.run(pred, {x: test_x[:10]})
        pred_y = np.argmax(test_output, 1)
        print(pred_y, 'prediction number')
        print(np.argmax(test_y[:10], 1), 'real number')
        model_dir = "mnist"
        model_firname = np.str(filter2)
        model_name = model_firname+"double"+"ckpt"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_saver = tf.train.Saver()
        model_saver.save(gsess,os.path.join(model_dir,model_name))
        print("model saved successfully")

initdouble_feature(16,32)
#init_feature(32)