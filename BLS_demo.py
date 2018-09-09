from loadMinist import *
from bls_train import *
from bls_train_enhance import *
from bls_train_enhancefeature import *
from bls_train_input import *
from bls_train_inputenhance import *

#加载训练集和测试集
train_x = load_train_images()
train_y = load_train_labels()

test_x = load_test_images()
test_y = load_test_labels()

# 对样本进行归一化
train_x = (train_x/255)
#train_x = float(train_x)
train_y = (train_y)
test_x = (test_x/255)
test_y = (test_y)

train_y = (train_y - 1)*2 + 1
print(train_y[2])
test_y = (test_y - 1)*2 + 1

# one shot solution using pseudo-inverse
s = 0.8   #增强节点的收缩比例
C = 2e-30  # L2正则化参数
N11 = 10  # feature nodes per window
N2 = 6 # number of windows of feature nodes
N33 = 150 # number of enhancement nodes
epochs = 1 # number of epochs
train_err = np.zeros((1,epochs))
test_err = np.zeros((1,epochs))
train_time = np.zeros((1,epochs))
test_time = np.zeros((1,epochs))

def bls_no():
    N1 = N11
    N3 = N33
    j = 1
    for j in range(epochs):
        bls_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3)

def incre_enh():
    # The following is incremental learning by adding m enhancement nodes
    m = 50 # number of Enh nodes in each incremental learning
    l = 5  # steps of incremental learning
    epochs = 1
    N1 = N11
    N3 = N33
    bls_train_enhance(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,epochs,m,l)


def incre_feature():
    #The following is the increment of m2+m3 enhancement nodes and m1 feature nodes
    epochs = 1
    m1 = 10 # number of feature nodes per increment step
    m2 = 15 # number of enhancement nodes related to the incremental feature nodes per increment step
    m3 = 35 # number of enhancement nodes in each incremental learning
    l = 5
    N1 = N11
    N3 = N33
    bls_train_enhancefeature(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,m1,m2,m3,l)



train_xf = train_x
train_yf = train_y
train_x1 = train_xf[0:10000,:]
train_y1 = train_yf[0:10000,:]# the selected input patterns of int incremental learning

def incre_input():
    epochs = 1
    m = 10000 # number of added input paterns per increment step
    l = 6
    N1 = N11
    N3 = N33
    bls_train_input(train_x1,train_y1,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,l)


def incre_inputenhance():
    #This is a demo of the increment of m input patterns and m2 enhancement nodes
    epochs = 1
    m = 10000 # number of added input patterns per incremental step
    m2 = 60  # number of added enhancement nodes per incremental step
    l = 4  # steps of increment learning
    N1 = N11
    N3 = N33
    bls_train_inputenhance(train_x1,train_y1,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,m2,l)

incre_enh()





