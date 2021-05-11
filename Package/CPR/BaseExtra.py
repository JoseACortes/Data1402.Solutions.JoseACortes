import random
import numpy as np

def randomvector(size, maximum):
    randomlist = []
    for i in range(0,size):
        n = random.randint(1,maximum)
        randomlist.append(n)
    return randomlist

def testcheck(x, vset):
    for v in vset.values:
        if np.equal(x, v).all().all():
            return True
    return False

def closestvector(target, A):
    newv = [np.linalg.norm(np.array(np.array(target)-np.array(a))) for a in A]
    return np.argmin(newv)

def farthestvector(target, A):
    newv = [np.linalg.norm(np.array(np.array(target)-np.array(a))) for a in A]
    return np.argmax(newv)

def move(x, a, v):
    n = abs(x-a)*v
    if a>x:
        return x+n
    elif a<x:
        return x-n
    else:
        return x

def multimove(X, A, v):
    newvec = list()
    for n, x in enumerate(X):
        newvec.append(move(x, A[n], v))
    return newvec