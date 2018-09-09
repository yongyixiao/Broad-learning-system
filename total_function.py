import numpy as np
import numpy.linalg as nlg

def mapminmax(Al):
    for i in range(np.shape(Al)[0]):
        for j in range(np.shape(Al)[1]):
            xmin = np.min(Al[i])
            xmax = np.max(Al[i])
            if(xmin == xmax):
                Al[i][j] == xmin
            else:
                Al[i][j] = 2 * (Al[i][j]-xmin)/(xmax-xmin) - 1
    return Al

def mapminmax_onezero(Al):
    for i in range(np.shape(Al)[0]):
        for j in range(np.shape(Al)[1]):
            xmin = np.min(Al[i])
            xmax = np.max(Al[i])
            if(xmin == xmax):
                Al[i][j] == xmin
            else:
                Al[i][j] = (Al[i][j] -xmin)/(xmax-xmin)
    return Al

def getMin(Tl):
    mindata = []
    for i in range(len(Tl)):
        mindata.append(np.min(Tl[i]))
    return np.min(mindata)

def getMax(Tl):
   maxdata = []
   for i in range(len(Tl)):
        maxdata.append(np.max(Tl[i]))
   return np.max(maxdata)

def sparse_bls(Al,Hl,lam,itrs):
    AA = np.dot(Al.T,Al)
    m = np.shape(Al)[1]
    n = np.shape(Hl)[1]
    x = np.zeros((m,n))
    wk = x
    ok = x
    uk = x
    AA_inv = nlg.inv(AA+np.eye(m)) # 求逆
    L1 = np.dot(np.eye(m),AA_inv)
    L2_ = np.dot(L1 , Al.T)
    L2 = np.dot(L2_ , Hl)

    for i in range(itrs):
        tempc = ok - uk
        ck = L2 + np.dot(L1,tempc)
        ok = change_max(ck+uk, lam)
        uk = uk + (ck-ok)
        wk = ok
    return  wk


def tansig(net):
    tansig_value = 2/(1+np.exp(-2*net))-1
    return tansig_value

def change_max(x,kappa):
    aa = x - kappa
    aa[aa<0] = 0
    bb = -x - kappa
    bb[bb<0] = 0
    z = aa - bb
    return z

def result(x):
    new_result = []
    new_result.append(np.argmax(x,axis=1))
    new_result = np.transpose(new_result)
    #print(np.shape(new_result))
    return new_result