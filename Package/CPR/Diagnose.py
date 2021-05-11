import Graph
import BaseExtra
import pandas as pd
def deepVitals(DeepFull):
    check = list()
    for d in range(len(DeepFull)):
        check.append([BaseExtra.testcheck(DeepFull[d:d+1]['Item'].iloc[0], DeepFull[d:d+1]['Cluster'].iloc[0]), DeepFull[d:d+1]['Epochs'].iloc[0], len(DeepFull[d:d+1]['Cluster'].iloc[0])])
    checkframe = pd.DataFrame(check, columns=['Sucessful', 'Epochs', 'Cluster_Size'])
    Graph.diagnosePlot(checkframe, ['Epochs', 'Cluster_Size'])
    print(str(len(checkframe[checkframe['Sucessful']==True]))+' Sucessful out of '+str(len(checkframe))+" ("+str(round(len(checkframe[checkframe['Sucessful']==True])/len(checkframe), 3))+")")
    return checkframe