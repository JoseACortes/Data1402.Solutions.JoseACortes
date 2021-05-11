import matplotlib.pyplot as plt
import numpy as np

def alternatespace(x, y):
    group = list()
    for a in range(y):
        for b in range(x):
            group.append([b, a])
    return group

def multiclustgraph(dataset, size=50):
    a = len(dataset.columns)
    fig, axs = plt.subplots(a, a, figsize=(size, size))
    for c, t in enumerate(alternatespace(a, a)):
        if t[0] == t[1]:
            axs[t[0]][t[1]].hist(dataset[dataset.columns[[t[0]]]], color ='tab:blue', histtype = 'step', bins=100)
            axs[t[0]][t[1]].title.set_text(dataset.columns[t[0]]+' vs. '+dataset.columns[t[1]])
        else:
            if t[0]>t[1]:
                axs[t[0]][t[1]].scatter(dataset[dataset.columns[[t[0]]]],dataset[dataset.columns[t[1]]], color ='tab:blue')
                axs[t[0]][t[1]].title.set_text(dataset.columns[t[0]]+' vs. '+dataset.columns[t[1]])
                axs[t[0]][t[1]].set_xlabel(dataset.columns[t[0]])
                axs[t[0]][t[1]].set_ylabel(dataset.columns[t[1]])
            if t[0]<t[1]:
                axs[t[0]][t[1]].scatter(dataset[dataset.columns[[t[0]]]],dataset[dataset.columns[t[1]]], color ='tab:orange')
                axs[t[0]][t[1]].title.set_text(dataset.columns[t[0]]+' vs. '+dataset.columns[t[1]])
                axs[t[0]][t[1]].set_xlabel(dataset.columns[t[0]])
                axs[t[0]][t[1]].set_ylabel(dataset.columns[t[1]])
    fig.show()

def diagnosePlot(frame, columns):
    for c in columns:
        m = list(range(1, max(frame[c])+2))
        plt.figure(figsize=(8, 6))
        plt.title('Distribution of Total '+c, fontsize=20)
        plt.xticks(m)
        plt.xlabel('Total '+c)
        plt.hist(frame[frame['Sucessful']==True][c], bins = m, align = 'left', stacked=True, label='Sucessful Test', color = 'blue')
        plt.hist(frame[frame['Sucessful']==False][c], bins = m, align = 'left', stacked=True, label='Un-Sucessful Test', color = 'orange')
        plt.legend()
        plt.text(0, -20, 'Minimum '+c+': '+str(np.min(frame[c]))+' | Average '+c+': '+str(round(np.average(frame[c]), 1))+' | Maximum '+c+': '+str(np.max(frame[c])), fontsize=15)
        plt.text(0, -40, 'Average '+c+': '+str(round(np.average(frame[c]), 1)), fontsize=15)
        plt.text(0, -60, 'Maximum '+c+': '+str(np.max(frame[c])), fontsize=15)
        plt.show()