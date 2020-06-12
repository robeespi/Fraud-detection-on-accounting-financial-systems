#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.externals.joblib import Memory
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from numpy import savetxt

mem = Memory("/Users/roberto/DevNet-master copy 3/dataset/svm_data")


@mem.cache
def get_data_from_svmlight_file(path):
    data = load_svmlight_file(path)
    return data[0], data[1]


def dataLoading():
    # loading data
    #df = pd.read_csv(path)

    labels = np.load('/Users/roberto/DevNet-master copy 3/dataset/ytrain_NSL-KDD.npy')
    #labels = np.load('/Users/roberto/DevNet-master copy 3/dataset/y_train_kddcup.npy')
    #x_df = df.drop(['class'], axis=1)

    #x = x_df.values
    
    x = np.load('/Users/roberto/DevNet-master copy 3/dataset/trainX_NSL-KDD.npy')
    #x = np.load('/Users/roberto/DevNet-master copy 3/dataset/X_train_kddcup.npy')
    
    print("Data shape: (%d, %d)" % x.shape)

    return x, labels


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def prec(scores, y_test):
    prec, rec, thr = precision_recall_curve(y_test, scores)
    figura = plt.figure()
    plt.plot(thr, prec[:-1], 'b--', label='precision')
    plt.plot(thr, rec[:-1], 'g--', label = 'recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper right')
    plt.ylim([0,1])
    #pyplot.show()
    figura.savefig('figura.png')
        
    figura2 = plt.figure()
    #precision, recall, _ = precision_recall_curve(scores, y_test)
    # plot the model precision-recall curve
    plt.plot(rec, prec)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    #plt.legend(loc="upper right")
    # show the plot
    figura2.savefig('figura2.png')
    
    f1 = 2 * ((prec * rec) / (prec + rec))
    #thresholds.append(thresh[np.argmax(f1_score)])
    max_prec = prec[np.argmax(f1)]
    max_rec = rec[np.argmax(f1)]
    print('max precision', max_prec)
    print('max recall', max_rec)
    #print('threshold', thresholds)
    f1_ = 2 * ((max_prec * max_rec) / (max_prec + max_rec ))
    print('f1', f1_)
    #savetxt('/content/gdrive/My Drive/devnet/DevNet/DevNet-master copy 2/results/precision_test.csv', precision)
    #savetxt('/content/gdrive/My Drive/devnet/DevNet/DevNet-master copy 2/results/recall_test.csv', recall)
    #savetxt('/content/gdrive/My Drive/devnet/DevNet/DevNet-master copy 2/results/thresh_test.csv', thresh)
    return max_prec, max_rec, f1_


def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, depth, rauc, ap, std_auc, std_ap,
                 train_time, test_time, path="/Users/roberto/DevNet-master copy 3/results/auc_performance_cl0.5.csv"):
    csv_file = open(path, 'a')
    row = name + "," + str(n_samples) + "," + str(dim) + ',' + str(n_samples_trn) + ',' + str(
        n_outliers_trn) + ',' + str(n_outliers) + ',' + str(depth) + "," + str(rauc) + "," + str(std_auc) + "," + str(
        ap) + "," + str(std_ap) + "," + str(train_time) + "," + str(test_time) + "\n"
    csv_file.write(row)
