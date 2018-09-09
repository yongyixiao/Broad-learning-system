import numpy as np
import numpy.linalg as nlg
import  datetime,time
from scipy.linalg import orth
from scipy import stats
from total_function import *



def bls_train_inputenhance(train_x,train_y,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,m,m2,l):

    train_x_inv = stats.zscore(train_x.T,axis=0)
    train_x = train_x_inv.T
    H1 = np.hstack((train_x,0.1*np.ones((np.shape(train_x)[0],1))))
    y = np.zeros((np.shape(train_x)[0],N2*N1))
    Beta1 = []
    for i in range(N2):
        we = 2 * np.random.rand(np.shape(train_x)[1]+1,N1)-1
        A1 = np.dot(H1,we)
        A1 = mapminmax(A1)
        beta1_inv = sparse_bls(A1,H1,1e-3,50)
        beta1 = beta1_inv.T
        Beta1.append(beta1)
        T1 = np.dot(H1,beta1)
        print('feature nodes in window %d :max val is %f,min val is %f'%(i,getMax(T1),getMin(T1)))
        T1 = mapminmax_onezero(T1)
        y[:,N1*i:N1*(i+1)] = T1

    H2 = np.hstack((y,0.1 * np.ones((np.shape(y)[0],1))))
    Wh = []
    if N1 * N2 >= N3:
        wh = orth(2 * np.random.rand(N1*N2+1,N3)-1)
    else:
        wh = orth(2 * np.random.rand(N1*N2+1,N3).T-1).T
    Wh.append(wh)
    T2 = np.dot(H2,wh)
    l2 = np.max(np.max(T2))
    l2 = s/l2
    print('enhancement node:max val of output is %f ,min val is %f'%(l2,getMin(T2)))
    T2 = tansig(T2 * l2)
    T3 = np.hstack((y,T2))
    beta_f = np.dot(T3.T,T3) + np.eye(np.shape(T3.T)[0]) * C
    beta = np.dot(nlg.inv(beta_f),T3.T)
    beta2 = np.dot(beta,train_y)
    print('training has been finished!')

    xx = np.dot(T3,beta2)
    yy = result(xx)
    train_yy = result(train_y)
    train_total = 0
    for i in range(np.shape(yy)[0]):
        if(yy[i] == train_yy[i]):
            train_total += 1
    print('total:%d'%train_total)
    trainaccuracy = train_total/np.shape(train_yy)[0]
    print('training accuracy is %f %%'%(trainaccuracy*100))

    test_x_inv = stats.zscore(test_x.T)
    test_x = test_x_inv.T
    HH1 = np.hstack((test_x,0.1*np.ones((np.shape(test_x)[0],1))))
    yy1 = np.zeros((np.shape(test_x)[0],N2*N1))
    for i in range(N2):
        beta1 = Beta1[i]
        TT1 = np.dot(HH1,beta1)
        TT1 = mapminmax_onezero(TT1)
        yy1[:,i*N1:(i+1)*N1] = TT1
    HH2 = np.hstack((yy1,0.1*np.ones((np.shape(yy1)[0],1))))
    TT2 = tansig(np.dot(HH2,wh)*l2)
    TT3 = np.hstack((yy1,TT2))
    x = np.dot(TT3,beta2)
    y1 = result(x)
    test_yy = result(test_y)
    test_total = 0
    for j in range(np.shape(test_yy)[0]):
        if(y1[j] == test_yy[j]):
            test_total += 1
    print('test_total:%d'% test_total)
    testaccuracy = test_total/np.shape(test_yy)[0]
    print('testing accuracy is %f %%'%(testaccuracy*100))

    #incremental training steps


    for e in range(l-1):
        train_xx_inv = stats.zscore(train_xf[10000+e*m:(10000+(e+1)*m),:].T,axis=0)
        train_xx = train_xx_inv.T
        train_yx = train_yf[(10000+e*m):(10000+e*m),:]
        train_y1 = train_yf[0:10000+(e+1)*m,:]
        Hx1 = np.hstack((train_xx,0.1*np.ones((np.shape(train_xx)[0],1))))
        yx = []
        j = 1
        for i in range(N2):
            beta1 = Beta1[i]
            Tx1 = np.dot(Hx1,beta1)
            Tx1 = mapminmax_onezero(Tx1)
            if(j == 1):
                yx[:] = Tx1
                j += 1
            else:
                yx = np.hstack((yx,Tx1))
        Hx2 = np.hstack((yx,0.1*np.ones((np.shape(yx)[0],1))))
        tx22 = []
        p = 1
        for k in range(e+1):
            wh = Wh[k]
            tx2 = np.dot(Hx2,wh)
            print('enhancement nodes update by input patterns %d: Max Val of Output %f Min Val %f'%(k,getMax(tx2),getMax(tx2)))
            tx2 = tansig(tx2 * l2)
            if(p == 1):
                tx22[:] = tx2
                p += 1
            else:
                tx22 = np.hstack((tx22,tx2))
        tx2x = np.hstack((yx,tx22))
        betat_f_inv = np.dot(tx2x.T,tx2x) + np.eye(np.shape(tx2x.T)[0]) * C
        betat_f = nlg.inv(betat_f_inv)
        betat = np.dot(betat_f,tx2x.T)
        beta = np.hstack((beta,betat))
        T3 = np.vstack((T3,tx2x))
        y = np.vstack((y,yx))
        H2 = np.hstack((y,np.ones((np.shape(y)[0],1))))
        print(np.shape(H2))
        if N1 * N2 >= m2:
            wh = orth(2 * np.random.rand(N2*N1+1,m2)-1)
        else:
            wh = orth(2 * np.random.rand(N2*N1+1,m2).T-1).T
        Wh.append(wh)
        t2 = np.dot(H2,wh)
        l3 = np.max(np.max(t2))
        l3 = s/l3
        print('addititional enhancement nodes in incremental setp %d: Max Val of Output %f Min Val %f'%(e,l3,getMin(t2)))
        t2 = tansig(t2 * l3)
        print(np.shape(t2))
        print(np.shape(T3))
        T3_temp = np.hstack((T3,t2))
        d = np.dot(beta,t2)
        c = t2 - np.dot(T3,d)
        if not c.all():
            q,w = np.shape(d)
            b_f = nlg.inv((np.eye(w) + np.dot(d.T,d)))
            b_s = np.dot(d.T,beta)
            b = np.dot(b_f,b_s)
        else:
            b = np.dot(nlg.inv(np.dot(c.T,c)+ np.eye(np.shape(c.T)[0]) * C),c.T)
        beta = np.vstack((beta-np.dot(d,b),b))
        beta2 = np.dot(beta,train_y1)
        T3 = T3_temp

        xx = np.dot(T3,beta2)
        yy = result(xx)
        train_yy = result(train_y1)
        inc_train_total = 0
        for i in range(np.shape(yy)[0]):
            if(yy[i] == train_yy[i]):
                inc_train_total += 1
        print('increment total:%d'%inc_train_total)
        TrainingAccuracy = inc_train_total/np.shape(train_yy)[0]
        print('training accuracy is %f %%'%(TrainingAccuracy*100))

        wh = Wh[e+1]
        tt2 = tansig(np.dot(HH2,wh) * l3)
        TT3 = np.hstack((TT3,tt2))
        x = np.dot(TT3,beta2)
        y1 = result(x)
        test_yy = result(test_y)
        inc_test_total = 0
        for j in range(np.shape(test_yy)[0]):
            if(y1[j] == test_yy[j]):
                inc_test_total += 1
        print("total is %d"%inc_test_total)
        testaccuracy_inc = inc_test_total/np.shape(y1)[0]
        print("testaccuracy is %f %%"%(testaccuracy_inc*100))



