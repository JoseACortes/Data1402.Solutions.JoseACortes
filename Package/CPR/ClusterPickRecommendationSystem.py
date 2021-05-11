#Third Party Packages
import pandas as pd
import numpy as np
import math
from collections import Counter
import random
from sklearn import cluster
import matplotlib.pyplot as plt
import time
import progressbar

#cluster function
from sklearn.cluster import KMeans

#Modules of Packacge
import BaseExtra
import Prep
import Filter
import Decision
import Run
import DeepRun
import Diagnose
import Graph



def quick(data_set, 
    decision_function = Decision.SimpleAveragePick, 
    cluster_function = KMeans, 
    filter_function = 'nearest', 
    max_epochs = 10, 
    cycles_per_epoch = 10, 
    n_clusters=2, 
    minimum_cluster_size=10, 
    maximum_cluster_size=40):
    curset = data_set
    cfunc = cluster_function(n_clusters = n_clusters)
    cfunc.fit(curset)
    echeck = 0
    for e in range(max_epochs):
        if len(curset)<maximum_cluster_size:
            return curset
        else:
            if len(curset)>= n_clusters:
                if min([len(curset[cfunc.predict(curset)==n]) for n in range(0, n_clusters)])<3:
                    pass
                else:
                    if len(curset)<maximum_cluster_size:
                        pass
                    else:
                        echeck+=1
                        o = decision_function([curset[cfunc.predict(curset)==n] for n in range(0, n_clusters)], cycles_per_epoch)
                        curset = Run.dataFilter(filter_function = filter_function, cluster_function=cfunc, pick = o, data_set = curset)
                        lastcurset = curset
                        try:
                            cfunc.fit(curset)
                        except:
                            return curset
                        if len(curset)<minimum_cluster_size:
                            return lastcurset
    return curset, echeck