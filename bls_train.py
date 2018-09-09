import numpy as np
import numpy.linalg as nlg
from scipy.linalg import  orth
import datetime,time
from scipy import stats
from total_function import *
from sklearn import preprocessing




def bls_train(train_x,train_y,test_x,test_y,s,C,N1,N2,N3):
    # zscore标准化处理 ,当X是一个矩阵是，采用zscore方法仍然是一个矩阵，在计算的过程中使用的均值及标准差使用的是每一列的均值与方差。
    # feature nodes
    starttime = datetime.datetime.now()
    #train_x_invex = np.transpose(train_x)
    #mean = np.average(train_x_invex, axis=0)# axis=0 表示每一列；axis=1 表示每一行
    #sigma = np.std(train_x_invex, axis=0)
   # train_x_nor = (train_x_invex-mean)/sigma
    #train_x = np.transpose(train_x_nor)
    train_x_inv = stats.zscore(train_x.T,axis=0)
    train_x = train_x_inv.T
    Hl = np.hstack((train_x,0.1 * np.ones((np.shape(train_x)[0],1))))
    y = np.zeros((np.shape(train_x)[0],N2*N1))
    Betal = []
    We = []
    for i in range(N2):
        we = 2 * np.random.rand(np.shape(train_x)[1]+1, N1)-1
        We.append(we)
        Al = np.dot(Hl, we)
        Al = mapminmax(Al)
        betal_invex = sparse_bls(Al,Hl,1e-3,50)
        betal = np.transpose(betal_invex)
        Betal.append(betal)
        T1 = np.dot(Hl , betal)
        print('feature nodes in window %d:Max Val of output %f Min val %f' % (i ,getMax(T1), getMin(T1)))
        T1 = mapminmax_onezero(T1)
        #print(np.shape(Tl))
        #print(np.shape(y[:,N1*i+1:N1*(i+1)+1]))
        y[:,N1*i:N1*(i+1)] = T1



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
    T3_train = np.dot(T3.T,train_y)
    beta = np.dot(WA_inv,T3_train) # 权重值
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print('Training has been finished!')
    print('the total training time is : %d seconds'% time)

    # Training Accuracy
    xx = np.dot(T3, beta)
    yy = result(xx)
    train_yy = result(train_y)
    total = 0
    for j in range(np.shape(yy)[0]):
        if(yy[j] == train_yy[j]):
            total += 1
    print('total:%d' % total)
    trainingAccuracy = float(total/np.shape(yy)[0])
    print('Training Accuracy is:%f%%' % (trainingAccuracy*100))

    # testing process
    test_starttime = datetime.datetime.now()
    # 对测试集进行zscore标准化处理
    test_x_invex = stats.zscore(test_x.T)
    #test_mean = np.average(test_x_invex,axis=0)
    #test_sigma = np.std(test_x_invex,axis=0)
    #test_x_nor = (test_x_invex-test_mean)/test_sigma
    test_x = np.transpose(test_x_invex)
    HH1 = np.hstack((test_x, 0.1 * np.ones((np.shape(test_x)[0],1))))
    yy1 = np.zeros((np.shape(test_x)[0],N2*N1))
    for i in range(N2):
        beta_test = Betal[i]
        TT1 = np.dot(HH1,beta_test)
        TT1 = mapminmax_onezero(TT1)
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
    test_endtime = datetime.datetime.now()
    test_time = (test_endtime - test_starttime).seconds
    print('testing has benn finished')
    print('test time is : %d seonds' %test_time)
    print('testing accuracy is:%f%%' % (testingAccuracy*100))




