import Filter
def dataFilter(filter_function, pick, data_set, cluster_function = 0, cut = 0.50):
    if filter_function == 'hard':
        return Filter.hardFilterFunction(cluster_function, pick, data_set)
    elif filter_function == 'nearest':
        return Filter.nearestFilterFunction(pick, data_set, cut)

def clusterChoiceFiltering(decision_function, data_set, cluster_function, filter_function, max_epochs, cycles_per_epoch, target_array = 0, n_clusters=2, minimum_cluster_size=10, maximum_cluster_size=40):
    curset = data_set
    cfunc = cluster_function(n_clusters = n_clusters)
    cfunc.fit(curset)
    echeck = 0
    for e in range(max_epochs):
        if len(curset)<maximum_cluster_size:
            return curset, echeck
        else:
            if len(curset)>= n_clusters:
                if min([len(curset[cfunc.predict(curset)==n]) for n in range(0, n_clusters)])<3:
                    pass
                else:
                    if len(curset)<maximum_cluster_size:
                        pass
                    else:
                        echeck+=1
                        o = decision_function(target = target_array, sets = [curset[cfunc.predict(curset)==n] for n in range(0, n_clusters)], cycles = cycles_per_epoch)
                        curset = dataFilter(filter_function = filter_function, cluster_function=cfunc, pick = o, data_set = curset)
                        lastcurset = curset
                        try:
                            cfunc.fit(curset)
                        except:
                            return curset, echeck
                        if len(curset)<minimum_cluster_size:
                            return lastcurset, echeck
    return curset, echeck
