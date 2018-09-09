import numpy as np
import numpy.linalg as nlg
import datetime,time
from scipy import stats
from scipy.linalg import  orth
from total_function import *
from  bls_train import  *


def bls_train_input(train_x1,train_y1,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,l):
    starttime = datetime.datetime.now()

    train_x_inv = stats.zscore(train_x1.T,axis=0)
    train_x = train_x_inv.T
    Hl = np.hstack((train_x,0.1 * np.ones((np.shape(train_x)[0],1))))
    y = np.zeros((np.shape(train_x)[0],N2*N1))
    Betal = []
    We = []
    for i in range(N2):
        we = 2 * np.random.rand(np.shape(train_x)[1]+1, N1)-1
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


    Wh = []
    # enhancement node
    H2 = np.hstack((y, 0.1 * np.ones((np.shape(y)[0],1))))
    if N1 * N2 >=N3:
        wh = orth((2*np.random.random((N1*N2+1, N3))-1))
    else:
        wh = orth((2*np.random.random((N1*N2+1, N3)).T-1)).T
    Wh.append(wh)
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
    beta = np.dot(WA_inv,T3.T)
    beta2 = np.dot(beta,train_y1)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print('Training has been finished!')
    print('the total training time is : %d seconds'% time)

    # Training Accuracy
    xx = np.dot(T3, beta2)
    yy = result(xx)
    train_yy = result(train_y1)
    total = 0
    for j in range(np.shape(yy)[0]):
        if(yy[j] == train_yy[j]):
            total += 1
    print('total:%d' % total)
    trainingAccuracy = float(total/np.shape(yy)[0])
    print('Training Accuracy is:%d%%' % (trainingAccuracy*100))

    # testing process
    test_starttime = datetime.datetime.now()
    # 对测试集进行zscore标准化处理
    test_x_invex = stats.zscore(test_x.T)
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
    x = np.dot(TT3,beta2)
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
    print('testing accuracy is:%d%%' % (testingAccuracy*100))

    # incremental training steps
    for e in range(l-1):
        train_xx_inv = stats.zscore(train_xf[(10000+e*m):(10000+(e+1)*m),:].T,axis=0)
        train_xx = train_xx_inv.T
        train_yx = train_yf[(10000+e*m):(10000+(e+1)*m),:]
        train_y1 = train_yf[0:10000+(e+1)*m,:]
        Hx1 = np.hstack((train_xx,0.1*np.ones((np.shape(train_xx)[0],1))))
        yx = []
        for i in range(N2):
            beta1 = Betal[i]
            Tx1 = np.dot(Hx1,beta1)
            Tx1 = mapminmax_onezero(Tx1)
            if(i==0):
                yx[:] = Tx1
            else:
                yx = np.hstack((yx,Tx1))
        Hx2 = np.hstack((yx,0.1*np.ones((np.shape(yx)[0],1))))
        wh = Wh[0]
        t2 = np.dot(Hx2,wh)
        #l2 = np.max(np.max(t2))
        #l2 = s/l2
        print('Enhancement nodes in incremental setp %d: Max Val of Output %f Min Val %f'%(e,l2,getMin(t2)))
        t2 = tansig(t2 * l2)
        t2 = np.hstack((yx,t2))
        betat_f = np.dot(t2.T,t2) + np.eye(np.shape(t2.T)[0]) * C
        betat = np.dot(nlg.inv(betat_f),t2.T)
        print(np.shape(betat))
        print(np.shape(beta))
        beta = np.hstack((beta,betat))
        beta2 = np.dot(beta,train_y1)
        T3 = np.vstack((T3,t2))
        xx = np.dot(T3,beta2)
        yy = result(xx)
        train_yy = result(train_y1)
        inc_total = 0
        for i in range(np.shape(yy)[0]):
            if(yy[i] == train_yy[i]):
                inc_total += 1
        print('total:%d'%inc_total)
        trainAccuracy = inc_total/np.shape(train_yy)[0]
        print('training accuracy is:%f %%'% (trainAccuracy*100))

        x = np.dot(TT3,beta2)
        y1 = result(x)
        test_yy = result(test_y)
        inc_test_total = 0
        for j in range(np.shape(y1)[0]):
            if(y1[j] == test_yy[j]):
                inc_test_total += 1
        print('total:%d'%inc_test_total)
        testAccuracy = inc_test_total/np.shape(test_yy)[0]
        print('testAccuracy accuracy is:%f %%'% (testAccuracy*100))





    return 1