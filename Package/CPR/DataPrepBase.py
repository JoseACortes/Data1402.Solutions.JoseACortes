import numpy as np
import random

def randomVector(size, maximum):
    randomlist = []
    for i in range(0,size):
        n = random.randint(1,maximum)
        randomlist.append(n)
    return randomlist

def crush(vector):
    return [(v)/max(vector) for v in vector]

def reduce(vector):
    return (np.array(vector)/np.sum(vector)).tolist()