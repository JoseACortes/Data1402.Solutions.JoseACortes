import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import tabulate
import math

#L3E1

def alternatespace(x, y):
    group = list()
    for a in range(y):
        for b in range(x):
            group.append([b, a])
    return group

def pairplots(dataset, size = 60):
    a = len(dataset.columns)
    fig, axs = plt.subplots(a-1, a-1, figsize=(size, size))
    for c, t in enumerate(alternatespace(a, a)):
        if dataset.columns[t[0]] == 'signal':
            pass
        elif dataset.columns[t[1]] == 'signal':
            pass
        else:
            if t[0] == t[1]:
                axs[t[0]-1][t[1]-1].hist(dataset.loc[dataset['signal'] == 0][dataset.columns[[t[0]]]], color ='tab:blue', histtype = 'step', bins=100)
                axs[t[0]-1][t[1]-1].hist(dataset.loc[dataset['signal'] == 1][dataset.columns[[t[0]]]], color ='tab:orange', histtype = 'step', bins=100)
                axs[t[0]-1][t[1]-1].title.set_text(dataset.columns[t[0]]+' vs. '+dataset.columns[t[1]])
            else:
                if t[0]>t[1]:
                    axs[t[0]-1][t[1]-1].scatter(dataset.loc[dataset['signal'] == 0][dataset.columns[[t[0]]]],dataset.loc[dataset['signal'] == 0][dataset.columns[t[1]]], color ='tab:blue')
                    axs[t[0]-1][t[1]-1].scatter(dataset.loc[dataset['signal'] == 1][dataset.columns[[t[0]]]],dataset.loc[dataset['signal'] == 1][dataset.columns[t[1]]], color ='tab:orange')
                    axs[t[0]-1][t[1]-1].title.set_text(dataset.columns[t[0]]+' vs. '+dataset.columns[t[1]])
                    axs[t[0]-1][t[1]-1].set_xlabel(dataset.columns[t[0]])
                    axs[t[0]-1][t[1]-1].set_ylabel(dataset.columns[t[1]])
                if t[0]<t[1]:
                    axs[t[0]-1][t[1]-1].scatter(dataset.loc[dataset['signal'] == 1][dataset.columns[[t[0]]]],dataset.loc[dataset['signal'] == 1][dataset.columns[t[1]]], color ='tab:orange')
                    axs[t[0]-1][t[1]-1].scatter(dataset.loc[dataset['signal'] == 0][dataset.columns[[t[0]]]],dataset.loc[dataset['signal'] == 0][dataset.columns[t[1]]], color ='tab:blue')
                    axs[t[0]-1][t[1]-1].title.set_text(dataset.columns[t[0]]+' vs. '+dataset.columns[t[1]])
                    axs[t[0]-1][t[1]-1].set_xlabel(dataset.columns[t[0]])
                    axs[t[0]-1][t[1]-1].set_ylabel(dataset.columns[t[1]])


#L3E2

def tab_everything(dataframe):
    seperate_matricies = [dataframe.iloc[:, 1:], dataframe.loc[dataframe['signal'] == 0].iloc[:, 1:], dataframe.loc[dataframe['signal'] == 1].iloc[:, 1:]]
    mat_titles = ['Full Data', 'Signal 0', 'Signal 1']
    for n, mat in enumerate(seperate_matricies):
        print(mat_titles[n]+' Correlation Coeficent')
        head = mat.columns.tolist()
        table = np.round(np.corrcoef(np.rot90(mat)), decimals = 1).tolist()
        for i, t in enumerate(table):
            t.insert(0, head[i])
        display(HTML(tabulate.tabulate(table, tablefmt='html', headers=head)))
        print(mat_titles[n]+' Covariance')
        table = np.round(np.cov(np.rot90(mat)), decimals = 1).tolist()
        for i, t in enumerate(table):
            t.insert(0, head[i])
        display(HTML(tabulate.tabulate(table, tablefmt='html', headers=head)))

#L3E3 / L4E5
def crit_1(v, c):
    return v > c

def crit_2(v, c):
    return v < c

def crit_3(v, c):
    return abs(v) > c

def crit_4(v, c):
    return abs(v) < c

def CheckRate(dataset, field, criteria, c):
    dd = dataset.loc[criteria(dataset[field], c)]
    positive_count = len(dd.loc[dataset['signal']==1])
    negative_count = len(dd.loc[dataset['signal']==0])
    total = len(dataset)
    return positive_count/total, negative_count/total

def makerange(a, b, step):
    rangelist = list()
    c = a
    while c < b:
        rangelist.append(c)
        c += step
    return rangelist

def ChartTry(dataset, field, criteria, checkrange):
    z = list()
    y = list()
    x = list()
    for c in checkrange:
        yz = CheckRate(dataset, field, criteria, c)
        y.append(yz[0])
        z.append(yz[1])
        x.append(c)
    return x, y, z

def integrateunordered(x, y):
    return np.trapz(np.sort([x]).tolist(), np.sort([y]).tolist())

def significance(tpr_list, fpr_list):
    o_s = list()
    N_s = [10, 100, 1000, 10000]
    N_b = [100, 1000, 10000, 100000]
    
    for t, n in enumerate(N_s):
        oss = list()
        N_sa = np.multiply(tpr_list, n)
        N_ba = np.multiply(fpr_list, N_b[t])
        for v, r in enumerate(N_sa):
            if r+N_ba[v]==0:
                oss.append(0)
            else:
                oss.append(r/(np.sqrt(r+N_ba[v])))
        o_s.append(oss)
    return o_s

def testall(data, testlist):
    xclist = list()
    etlist = list()
    eblist = list()
    for t in testlist:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 6))
        tempchart = ChartTry(data, t['item'], t['crit'], t['range'])
        ###
        ax1.set_title('Selection vs TPR, FPR for '+str(t['item']))
        ax1.plot(tempchart[0], tempchart[1], label = 'TPR', color = 'blue')
        ax1.plot(tempchart[0], tempchart[2], label = 'FPR', color = 'orange')
        ax1.set_xlabel('Selection')
        ax1.set_ylabel('TPR, FPR')
        ax1.legend()
        ###
        ax2.set_title('ROC curve for '+str(t['item']))
        auc = integrateunordered(tempchart[1], tempchart[2])
        ax2.plot(tempchart[2], tempchart[1], label = 'roc ('+str(auc)+')', color = 'blue')
        ax2.set_xlabel('FPR')
        ax2.set_ylabel('TPR')
        ax2.legend()
        ###
        sigcalc = significance(tempchart[1], tempchart[2])
        ax3.set_title('Selection vs Significance for '+str(t['item']))
        ax3.plot(tempchart[0], sigcalc[3], label = 'Sig Senario 4', color = 'red')
        ax3.plot(tempchart[0], sigcalc[2], label = 'Sig Senario 3', color = 'green')
        ax3.plot(tempchart[0], sigcalc[1], label = 'Sig Senario 2', color = 'orange')
        ax3.plot(tempchart[0], sigcalc[0], label = 'Sig Senario 1', color = 'blue')
        ax3.set_xlabel('Selection')
        ax3.set_ylabel('TPR - FPR')
        ax3.legend()
        ###
        v = sigcalc[3].index(max(sigcalc[3]))
        xclist.append(tempchart[0][v])
        etlist.append(tempchart[1][v])
        eblist.append(tempchart[2][v])
    return [xclist, etlist, eblist]

#L3E4

def test_significance(elist, tested):
    o_s = list()
    N_sa = list()
    N_ba = list()
    E_s = elist[1]
    E_b = elist[2]
    N_s = [10, 100, 1000, 10000]
    N_b = [100, 1000, 10000, 100000]
    plt.figure(figsize=(5, 20))
    for q, e in enumerate(tested):
        templist = list()
        v = e
        Nsaa = list()
        Nsbb = list()
        oss = list()
        for t, n in enumerate(N_s):
            Nis = E_s[q]*n
            Nib = E_b[q]*N_b[t]
            templist.append(Nis/(np.sqrt(Nis+Nib)))
        plt.scatter([1, 2, 3, 4], templist)
        for c, t in enumerate(templist):
            plt.text([1, 2, 3, 4][c], t, v)
        plt.title('Significance Comparisons')
        plt.xticks([1, 2, 3, 4])
        plt.xlabel('Senario')
        plt.ylabel('Significance')
    col = list()
    for t, n in enumerate(N_s):
        print('senario '+str(t))
        row = list()
        Nsaa = list()
        Nsbb = list()
        oss = list()
        for q, e in enumerate(tested):
            Nis = E_s[q]*n
            Nib = E_b[q]*N_b[t]
            print(str(e)+' significance: '+str(Nis/(np.sqrt(Nis+Nib))))
            row.append(Nis/(np.sqrt(Nis+Nib)))
            Nsbb.append(Nib)
            Nsaa.append(Nis)
            oss.append(Nis/(np.sqrt(Nis+Nib)))
        col.append(row)
        N_sa.append(Nsaa)
        N_ba.append(Nsbb)
        o_s.append(oss)
        print('=======')
    full = list()
    for n, c in enumerate(tested):
        temp = list()
        temp.append(E_s[n])
        temp.append(E_b[n])
        for t in range(len(N_sa)):
            temp.append(N_sa[t][n])
            temp.append(N_ba[t][n])
            temp.append(o_s[t][n])
        full.append(temp)
    return np.round(pd.DataFrame(full, tested, columns = ['e_s', 'e_b', "N's_1", "N'b_1", "o's_1", "N's_2", "N'b_2", "o's_2", "N's_3", "N'b_3", "o's_3", "N's_4", "N'b_4", "o's_4"]), decimals = 2)

def cross_set(data, cross):
    dd = data
    for c in cross:
        dd = dd.loc[c['crit'](dd[c['item']], c['selection'])].loc[dd['signal']==1]
    return dd

def cross_rate(data, cross):
    dd = cross_set(data, cross)
    tpr = len(dd)
    total = len(data.loc[data['signal']==1])
    return tpr/total

def AUC(TPR,FPR):
    n = FPR.shape[0]
    return np.sum((FPR[0:n-1]-FPR[1:]) * (TPR[0:n-1]+TPR[1:])/2)

def compute_rate(d,bins=1000):
    hist,bins_=np.histogram(d,bins=bins,density=True)
    R = np.cumsum(hist[::-1])[::-1] * (bins_[1]-bins_[0])
    return R,bins_

def lin_disc(dataset):
    dataset_sig=dataset[dataset.signal==1]
    dataset_bkg=dataset[dataset.signal==0]
    dataset_sig_0 = dataset_sig.drop("signal",axis=1)
    dataset_bkg_0 = dataset_bkg.drop("signal",axis=1)
    m_s= np.mean(dataset_sig_0,axis=0)
    m_b= np.mean(dataset_bkg_0,axis=0)
    delta = np.matrix(m_s-m_b).transpose()
    S_B= delta*delta.transpose()
    delta_s=np.matrix(dataset_sig_0-m_s).transpose()
    S_W_s= delta_s*delta_s.transpose()
    delta_b=np.matrix(dataset_bkg_0-m_b).transpose()
    S_W_b= delta_b*delta_b.transpose()
    S_W=S_W_s-S_W_b
    S_W_inv = np.linalg.inv(S_W)
    w = S_W_inv * np.matrix(m_b - m_s).transpose()
    return w

def testcoefficients(w, dataset):
    dataset_sig=dataset[dataset.signal==1]
    dataset_bkg=dataset[dataset.signal==0]
    dataset_sig_0 = dataset_sig.drop("signal",axis=1)
    dataset_bkg_0 = dataset_bkg.drop("signal",axis=1)
    output_s=np.matrix(dataset_sig_0)*w
    output_b=np.matrix(dataset_bkg_0)*w
    plt.title('Histogram Comparison')
    h_s,bins,_=plt.hist(output_s,label="signal",alpha=0.5,bins=100)
    h_b,bins,_=plt.hist(output_b,bins=bins,alpha=0.5,label="background")
    plt.legend()
    plt.show()
    tpr,bins=compute_rate(output_s)
    fpr,bins=compute_rate(output_b,bins=bins)
    plt.title('TPR and FPR')
    plt.plot(bins[:-1],tpr,label="TPR")
    plt.plot(bins[:-1],fpr,label="FPR")
    plt.legend()
    plt.show()
    roc_auc = AUC(tpr, fpr)
    plt.title('ROC Curve')
    plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return tpr, fpr