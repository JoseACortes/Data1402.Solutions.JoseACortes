import Run
import pandas as pd
import progressbar

def clusterChoiceFiltering(decision_functions, 
    data_set, 
    cluster_function_list, 
    filter_function_list, 
    max_epochs, 
    cycles_per_epoch_list, 
    n_clusters_list, 
    minimum_cluster_size=10, 
    maximum_cluster_size=40,
    showprogress = True):
    check = list()
    step = 0
    with progressbar.ProgressBar(max_value=(len(data_set)*len(decision_functions)*len(cluster_function_list)*len(filter_function_list)*len(n_clusters_list)*len(cycles_per_epoch_list))) as bar:
        for dfun in decision_functions:
            for cfun in cluster_function_list:
                for ffun in filter_function_list:
                    for nclu in n_clusters_list:
                        for ccyc in cycles_per_epoch_list:
                            print([dfun, cfun, ffun, nclu, ccyc])
                            for g in range(0,len(data_set)):
                                if showprogress==True:
                                    step+=1
                                    bar.update(step)
                                clust = Run.clusterChoiceFiltering(decision_function = dfun,
                                    data_set = data_set, 
                                    target_array = data_set[g:g+1], 
                                    cluster_function = cfun, 
                                    filter_function = ffun, 
                                    max_epochs = max_epochs, 
                                    cycles_per_epoch = ccyc, 
                                    n_clusters = nclu, 
                                    minimum_cluster_size = minimum_cluster_size,
                                    maximum_cluster_size = maximum_cluster_size)
                                check.append([str(dfun), str(cfun), str(ffun), str(nclu), str(ccyc), data_set[g:g+1], clust[1], clust[0]])
    return pd.DataFrame(check, columns=['Decision_Function', 'Cluster_Function', 'Filter', 'N_Clusters', 'Cycles', 'Item', 'Epochs', 'Cluster'])