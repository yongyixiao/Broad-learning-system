import numpy as np
import numpy.linalg as nlg
import datetime,time
from scipy.linalg import  orth
from scipy import stats
from total_function import  *

def bls_train_enhancefeature(train_x,train_y,test_x,test_y,s,C,N1,N2,N3,m1,m2,m3,l):
    N11 = N1
    for i in range(1):
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
        y1 = result(x)
        test_yy = result(test_y)
        test_total = 0
        for j in range(np.shape(test_yy)[0]):
            if(y1[j] == test_yy[j]):
                test_total += 1
        print('total:%d' % test_total)
        testingAccuracy = float(test_total/np.shape(test_yy)[0])
        test_endtime = datetime.datetime.now()
        test_time = (test_endtime - test_starttime).seconds
        print('testing has benn finished')
        print('test time is : %d seonds' %test_time)
        print('testing accuracy is:%d%%' % (testingAccuracy*100))


        # incremental training steps
        B3 = []
        Wh = []
        for e in range(l-1):
            inc_starttime = datetime.datetime.now()
            we = 2 * np.random.rand(np.shape(train_x)[1]+1,m1) - 1
            A1 = np.dot(H1,we)
            A1 = mapminmax(A1)
            beta1_inv = sparse_bls(A1,H1,1e-3,50)
            beta1 = beta1_inv.T
            Betal.append(beta1)
            T1 = np.dot(H1,beta1)
            print('feature nodes in window %d:Max Val of output %f Min val %f' % (e ,getMax(T1), getMin(T1)))
            T1 = mapminmax_onezero(T1)
            y = np.hstack((y,T1))
            H2 = np.hstack((y,0.1 * np.ones((np.shape(y)[0],1))))
            h2 = np.hstack((T1, 0.1 * np.ones((np.shape(T1)[0],1))))
            if m1>=m2:
                wh = orth(2 * np.random.rand(m1+1,m2)-1)
            else:
                wh = orth(2 * np.random.rand(m1+1,m2).T-1).T
            B3.append(wh)
            t22 = np.dot(h2,wh)
            l1 = np.max(np.max(t22))
            l1 = s/l1
            t22 = tansig(t22 * l1)
            if N2*N1+(e+1)*m1>=m3:
                wh = orth(2 * np.random.rand(N2*N1+(e+1)*m1+1,m3)-1)
            else:
                wh = orth(2 * np.random.rand(N2*N1+(e+1)*m1+1,m3).T-1).T
            Wh.append(wh)
            t2 = np.dot(H2,wh)
            l2 = np.max(np.max(t2))
            l2 = s/l2
            print("additinal enhancement nodes in incremental step %d: max val is %f ,min val is %f"%(e,getMax(t2),getMin(t2)))
            t2 = tansig(t2*l2)
            t2_f = np.hstack((T1,t22))
            t2 = np.hstack((t2_f,t2))
            T3_temp = np.hstack((T3,t2))
            d = np.dot(beta,t2)
            c = t2 - np.dot(T3,d)
            if not c.all():
                q,w = np.shape(d)
                d_new = nlg.inv((np.eye(w) - np.dot(d.T,d)))
                d_beta = np.dot(d.T,beta)
                b = np.dot(d_new,d_beta)
            else:
                c_new = np.dot(c.T,c)+np.eye(np.shape(c.T)[0])*C
                b = np.dot(nlg.inv(c_new),c.T)
            beta = np.vstack((beta-np.dot(d,b),b))
            beta2 = np.dot(beta,train_y)
            T3 = T3_temp

            xx = np.dot(T3,beta2)
            yy11 = result(xx)
            train_yy = result(train_y)
            inc_total_train = 0
            for i in range(np.shape(yy)[0]):
                if(yy11[i] == train_yy[i]):
                    inc_total_train += 1
            print('inc_total_train:%d'%inc_total_train)
            trainAccuracy_inc_feature = float(inc_total_train/np.shape(train_x)[0])
            print('training accuracy is %f %%'%(trainAccuracy_inc_feature*100))

            beta1 = Betal[N2+e]
            TT1 = np.dot(HH1,beta1)
            TT1 = mapminmax_onezero(TT1)
            yy1 = np.hstack((yy1,TT1))
            HH2 = np.hstack((yy1,0.1*np.ones((np.shape(yy1)[0],1))))
            hh2 = np.hstack((TT1,0.1*np.ones((np.shape(TT1)[0],1))))
            wh = B3[e]
            tt22 = tansig(np.dot(hh2,wh)*l1)
            wh = Wh[e]
            tt2 = tansig(np.dot(HH2,wh)*l2)
            TT3_f = np.hstack((TT3,TT1))
            TT3_s = np.hstack((TT3_f,tt22))
            TT3 = np.hstack((TT3_s,tt2))
            x = np.dot(TT3,beta2)
            y1 = result(x)
            test_yy = result(test_y)
            inrfeature_total = 0
            for i in range(np.shape(y1)[0]):
                if(y1[i] == test_yy[i]):
                    inrfeature_total += 1
            print('total number:%d'%inrfeature_total)
            testingAccuracy_incfeature = inrfeature_total/np.shape(y1)[0]
            print('testing accuracy is %f %%'%(testingAccuracy_incfeature*100) )











