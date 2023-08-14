import numpy as np
import numpy.matlib as mtlb
def col_sum(arr):
    return np.sum(arr, axis=0)

def sqdist_mod(p, q=None, A=None):
   
    
    d, pn = p.shape
    p=np.reshape(p,(d,pn))
    #print("Shape for p and q)")
    #print(p.shape)

    if q is None:
        qn = pn
    else:
        d, qn = q.shape
        q = np.reshape(q,(d,qn))
    
        #print(q.shape)

    if pn == 0 or qn == 0:
        m = np.zeros((pn, qn))
        return m

    if q is None:
        pmag = col_sum(p * p)
        pmag=np.reshape(pmag,(1,pn))
        qmag = pmag
        m = np.abs(mtlb.repmat(qmag, pn, 1) + mtlb.repmat(pmag.T, 1, qn) - 2 * np.dot(p.T,p))
        np.fill_diagonal(m, 0)
    elif A is None:
        pmag = col_sum(p * p)
        qmag = col_sum(q * q)
        pmag=np.reshape(pmag,(1,pn))
        qmag=np.reshape(qmag,(1,qn))
        #print(pmag.shape)
        #print(qmag.shape)
        m = np.abs(mtlb.repmat(qmag, pn, 1) + mtlb.repmat(pmag.T, 1, qn) - 2 * np.dot(p.T,q))
    else:
        Ap =np.multiply(A,p)
        Aq = np.multiply(A,q)
        pmag = col_sum(np.multiply(p , Ap))
        qmag = col_sum(np.multiply(q , Aq))
        m = np.abs(mtlb.repmat(qmag, pn, 1) + mtlb.repmat(pmag.T, 1,qn) - 2 * np.dot(p.T,Aq))
    #print(m)
    return m