import numpy as np
import numpy.linalg as nlg
import datetime,time
from scipy import stats
from scipy.linalg import  orth
from total_function import *

def bls_train_enhance(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,epochs,m,l):
    N11 = N1
    for i in range(epochs):
        train_starttime = datetime.datetime.now()
        train_x_inv = stats.zscore(train_x.T,axis=0)
        train_x = train_x_inv.T
        H1 = np.hstack((train_x,0.1*np.ones((np.shape(train_x)[0],1))))
        y = np.zeros((np.shape(train_x)[0], N2*N11))
        Betal = []
        for i in range(N2):
            we = 2*np.random.rand(np.shape(train_x)[1]+1,N1)-1
            A1 = np.dot(H1,we)
            A1 = mapminmax(A1)
            betal_inv = sparse_bls(A1,H1,1e-3,50)
            betal1 = np.transpose(betal_inv)
            Betal.append(betal1)
            T1 = np.dot(H1,betal1)
            print('feature nodes in window %d:Max Val of output %f Min val %f' % (i ,getMax(T1), getMin(T1)))
            T1 = mapminmax_onezero(T1)
            y[:,N1*i:N1*(i+1)] = T1

        H2 = np.hstack((y,0.1 * np.ones((np.shape(y)[0],1))))
        if N1 * N2 >=N3:
            wh = orth((2*np.random.random((N1*N2+1, N3))-1))
        else:
            wh = orth((2*np.random.random((N1*N2+1, N3)).T-1)).T
        T2 = np.dot(H2,wh)
        l2 = np.max(np.max(T2))
        l2 = s/l2
        print('enhancement nodes:max val of output %f min val %f' % (l2, getMin(T2)))
        T2 = tansig(T2 * l2)
        T3 = np.hstack((y,T2))
        Wa = np.dot(T3.T,T3) + np.eye(np.shape(T3.T)[0]) * C
        Wa_inv = nlg.inv(Wa)
        beta = np.dot(Wa_inv,T3.T)
        beta2 = np.dot(beta,train_y)
        train_endtime = datetime.datetime.now()
        time = (train_endtime - train_starttime).seconds
        print("training has been finished")
        print('the total training time is : %d seconds'% time)
        xx = np.dot(T3,beta2)

        yy = result(xx)
        train_yy = result(train_y)
        total = 0
        for i in range(np.shape(yy)[0]):
            if(yy[i] == train_yy[i]):
                total += 1
        print('total:%d' % total)
        trainingAccuracy = float(total/60000)
        print('Training Accuracy is:%d%%' % (trainingAccuracy*100))

        test_starttime = datetime.datetime.now()
        test_x_inv = stats.zscore(test_x.T,axis=0)
        test_x = test_x_inv.T
        HH1 = np.hstack((test_x,0.1 * np.ones((np.shape(test_x)[0],1))))
        yy1 = np.zeros((np.shape(test_x)[0],N2*N1))
        for i in range(N2):
            betal_test = Betal[i]
            TT1 = np.dot(HH1,betal_test)
            TT1 = mapminmax_onezero(TT1)
            yy1[:,N11*i:N11*(i+1)] = TT1
        HH2 = np.hstack((yy1,0.1 * np.ones((np.shape(yy1)[0],1))))
        TT2 = tansig(np.dot(HH2,wh) * l2)
        TT3 = np.hstack((yy1,TT2))
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
        #print(np.shape(T3))
       # print(np.shape(beta))
        # incremental training steps
        WH = []
        train_time = []
        for j in range(l-1):
            increment_start = datetime.datetime.now()
            if N1*N2 >=m:
               wh_inc = orth((2*np.random.random((N1*N2+1, m))-1))
            else:
               wh_inc = orth((2*np.random.random((N1*N2+1, m)).T-1)).T

            WH.append(wh_inc)
            t2 = np.dot(H2,wh_inc)
            l2_inc = (np.max(np.max(t2)))
            l2_inc = s/l2_inc
            print("enhancement nodes in incremental step %d: max val of output %f min val %f" %(j,l2_inc,getMin(t2)))
            t2 = tansig(t2 * l2_inc)
            T3_temp = np.hstack((T3,t2))
            d = np.dot(beta,t2)
            c = t2 - np.dot(T3,d)
            if not c.all():
                q,w = np.shape(d)
                dd_inv = np.eye(w)-np.dot(d.T,d)
                dd = nlg.inv(dd_inv)
                d_beta = np.dot(d.T,beta)
                b = np.dot(dd,d_beta)
            else:
                cc_inv = np.dot(c.T,c) + np.eye(np.shape(c.T)[0])*C
                cc = nlg.inv(cc_inv)
                b = np.dot(cc,c.T)
            beta_inc = np.vstack((beta-np.dot(d,b),b))
            beta2_inc = np.dot(beta_inc,train_y)
            T3_inc = T3_temp
            increment_end = datetime.datetime.now()
            time = (increment_end-increment_start).seconds
            train_time.append(time)
            xx_inc = np.dot(T3_inc,beta2_inc)
            yy_inc = result(xx_inc)
            train_yy = result(train_y)
            incre_total = 0
            for i in range(np.shape(yy_inc)[0]):
                if(yy_inc[i] == train_yy[i]):
                    incre_total += 1
            print('total:%d '% incre_total)
            trainingAccuracy_incre = incre_total/60000
            print(' increment training accuracy is:%d%%' % (trainingAccuracy_incre*100))

            TT2_inc = tansig(np.dot(HH2,wh_inc) * l2_inc)
            TT3_inc = np.hstack((TT3,TT2_inc))

            # incremental testing steps
            x_inc = np.dot(TT3_inc,beta2_inc)
            y_inc = result(x_inc)
            test_yy_inc = result(test_y)
            total_inc_test = 0
            for k in range(np.shape(y_inc)[0]):
                if(y_inc[k] == test_yy_inc[k]):
                    total_inc_test += 1
            print('inc_total:%d'% total_inc_test)
            testingAccuracy_inc = float(total_inc_test/np.shape(y_inc)[0])
            print("testing has been finished")
            print('testing Accuracy is : %d %%' % (testingAccuracy_inc*100))












