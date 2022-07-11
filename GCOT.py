import heapq
import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize,minmax_scale
import ot
import matplotlib.pyplot as plt
from gen_a import adjacent_a
import time
import copy
import time
import signal

class GCOT:
    def __init__(self, n_clusters):
        """
        :param n_clusters: number of clusters
        """
        self.n_clusters = n_clusters

    def __adj_mat(self, C, k):
        n = C.shape[0]
        ridx = np.argpartition(C, -k, axis = 1)[:, -k:]
        cidx = np.argpartition(C, -k, axis = 0)[-k:, :]
        row = np.zeros((n,n))
        col = np.zeros((n,n))
        I = np.identity(n)
        for i in range(ridx.shape[0]):
            row[i, ridx[i, :]] = 1
            col[cidx[:, i], i] = 1
        A = I + row * col
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return  normlized_A


    def fit_base(self, X, eps):
        n = X.shape[0]
        a, b = np.ones((n,)) / n, np.ones((n,)) / n
        M = ot.dist(X, X)
        M /= M.max()
        C = ot.sinkhorn(a, b, M, eps, verbose=True)
        return C

    def fit_gcot(self, X, eps, C, k):
        n = X.shape[0]
        a, b = np.ones((n,)) / n, np.ones((n,)) / n
        A = self.__adj_mat(C, k) 
        E = np.dot(A, X)
        M = ot.dist(E, E)
        M /= M.max()
        C = ot.sinkhorn(a, b, M, eps, verbose=True)
        return C

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L
    def cluster_accuracy(self, y_true, y_pre):
        Label1 = np.unique(y_true)
        nClass1 = len(Label1)
        Label2 = np.unique(y_pre)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1, nClass2)
        G = np.zeros((nClass, nClass))
        for i in range(nClass1):
            ind_cla1 = y_true == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = y_pre == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i, j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:, 1]
        y_best = np.zeros(y_pre.shape)
        for i in range(nClass2):
            y_best[y_pre == Label2[i]] = Label1[c[i]]

        # # calculate accuracy
        err_x = np.sum(y_true[:] != y_best[:])
        missrate = err_x.astype(float) / (y_true.shape[0])
        acc = 1. - missrate
        nmi = normalized_mutual_info_score(y_true, y_pre)
        kappa = cohen_kappa_score(y_true, y_best)
        ca = self.class_acc(y_true, y_best)
        return acc, nmi, kappa, ca

    def class_acc(self, y_true, y_pre):
        """
        calculate each class's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca

    def call_acc(self, C, y, ro):
        C = self.thrC(C, ro)
        y_pre, C_final = self.post_proC(C, self.n_clusters, 8, 18)

        acc = self.cluster_accuracy(y, y_pre)
        return acc
