# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 2017

@author: carius

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
from sklearn import cluster
from class_clusteringV2 import clustering

from sklearn import mixture
from scipy.spatial.distance import cdist
import time
import random
from numpy.matlib import repmat

class MaxSilhouette (object):
    
    __slots__ = ['data', 'algorithm' 'min_clusters', 'max_clusters', 'connectivity', 'title', 'path', '__dict__']
    
    def __init__ (self, data=None, algorithm='KMeans', min_clusters=1, max_clusters=20, connectivity=None, title=None, path='./'):
        self.data = data
        self.algorithm = algorithm
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.connectivity = connectivity
        self.title = title
        self.path = path
        
        self.k_number = self.__maxmization()
        
        #return self
    
    def __maxmization (self):
    
        #silhouette = []
        K = list(range(self.min_clusters, self.max_clusters+1))
        
        #CM = Parallel(n_jobs=-1) (delayed(PAMCluster) (data_matrix=data, num_clusters=i) for i in xrange(1, 5))
        CM = [clustering(data=self.data, algorithm=self.algorithm, n_clusters=k, connectivity=self.connectivity).fit() for k in K]
        
        silhouette = [c.quality()['silhouette'] for c in CM]
        
        print(silhouette)
        print(K)
        
        indice = np.where(silhouette == np.max(silhouette))[0][0]
        maxS =  K[indice]
        #print maxS
        
        fig = pl.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        ax.plot(K, silhouette, 'b*-')
        ax.plot(K[indice], silhouette[indice], marker='o', markersize=12, 
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        #pl.grid(True)
        pl.xlim(self.min_clusters, self.max_clusters)
        pl.xlabel('Number of clusters')
        pl.ylabel('Silhouette score')
        pl.title(self.title)
        fp=self.path+self.title+'.png'
        pl.savefig(fp, bbox_inches='tight')
        pl.close('all')
        
        del CM
        
        return maxS

class BIC (object):
    
    __slots__ = ['data', 'algorithm', 'min_clusters', 'max_clusters', 'connectivity', 'title', 'path', '__dict__']
    
    def __init__(self, data=None, algorithm='KMeans', min_clusters=1, max_clusters=20, connectivity=None, title=None, path='./'):
         self.data = data
         self.algorithm = algorithm
         self.min_clusters =min_clusters
         self.max_clusters = max_clusters
         self.connectivity = connectivity
         self.title = title
         self.path = path
         self.k_number = self.__calc_BIC()
         
         #return self
     
    def __angle_data(self, x1, y1, x2, y2):
            u = np.array([x1,y1])
            v = np.array([x2,y2])
            cos = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
            angle =  np.arccos(np.clip(cos, -1, 1))
            return ((angle*180)/np.pi)
    
    def __BestPoint(curve):
            nPoints = len(curve)
            allCoord = np.vstack((list(range(nPoints)), curve)).T
            np.array([list(range(nPoints)), curve])
            firstPoint = allCoord[0]
            lineVec = allCoord[-1] - allCoord[0]
            lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
            vecFromFirst = allCoord - firstPoint
            scalarProduct = np.sum(vecFromFirst * repmat(lineVecNorm, nPoints, 1), axis=1)
            vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
            vecToLine = vecFromFirst - vecFromFirstParallel
            distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
            idxOfBestPoint = np.argmax(distToLine)
            
            return idxOfBestPoint
    def __compute_bic(self, clusters, X):
        """
        Computes the BIC metric for a given clusters
    
        Parameters:
        -----------------------------------------
        clusters:  object from clustering class
    
        X     :  multidimension np array of data points
    
        Returns:
        -----------------------------------------
        BIC value
        """
        # assign centers and labels
        centers = [clusters.cluster_centers_]
        labels  = clusters.labels
        #number of clusters
        m = clusters.n_clusters
        # size of the clusters
        n = np.bincount(labels)
        #size of data set
        N, d = X.shape
    
        #compute variance for all clusters beforehand
        cl_var = (1.0 / (N - m) / d) * sum([sum(cdist(X[np.where(labels == i)], [centers[0][i]], 
                 'sqeuclidean')**2) for i in range(m)])
    
        const_term = 0.5 * m * np.log(N) * (d+1)
    
        BIC = np.sum([n[i] * np.log(n[i]) -
                   n[i] * np.log(N) -
                 ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
                 ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    
        return(BIC)
    
    def __calc_BIC(self):
    
        K = list(range(self.min_clusters, self.max_clusters+1))
        clust = [clustering(data=self.data, algorithm=self.algorithm, n_clusters=k, connectivity=self.connectivity).fit() for k in K]
    
        # now run for each cluster the BIC computation
        valBIC = [self.__compute_bic(cl,self.data) for cl in clust]
        
        max_angle=0.0
        kIdx=0
        t=0
        '''
        for t in xrange(1, len(valBIC)-1):
            angle= self.__angle_data(valBIC[t-1], valBIC[t], valBIC[t],valBIC[t+1])
            if max_angle<angle:
                 max_angle=angle
                 kIdx=t+1
        ''' 
        '''
        while t < (len(valBIC)-2):
             angle= self.__angle_data(valBIC[t], valBIC[t+1], valBIC[t+1],valBIC[t+2])
    
             if max_angle<angle:
                 max_angle=angle
                 kIdx=t+1
             t=t+1
        '''
        kIdx = BestPoint(valBIC)
        
        # BIC curve
        fig = pl.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        ax.plot(K, valBIC, 'b*-')
        ax.plot(K[kIdx], valBIC[kIdx], marker='o', markersize=12, 
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        #pl.grid(True)
        pl.xlabel('Number of clusters')
        pl.ylabel('BIC values')
        pl.xlim(self.min_clusters, self.max_clusters)
        pl.title(self.title)
        fp=self.path+self.title+'.png'
        pl.savefig(fp, bbox_inches='tight')
        pl.close('all')
        
        df = pd.DataFrame(data=valBIC)
        df.to_csv(self.path+'valBIC.csv', sep=' ', header=False, index=False)
        
        del df
        del clust
        
        return K[kIdx]

class GAP_statistic(object):
    
    __slots__ = ['data', 'min_clusters', 'max_clusters', 'path', '__dict__']
    
    def __init__(self, data=None, algorithm='KMeans', min_clusters=1, max_clusters=20, connectivity=None, path='./'):
        
        self.data = data
        self.algorithm = algorithm
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.connectivity = connectivity
        self.path = path
        
        self.results = self._fit()
        
        #return self
        
    
    def __short_pair_wise_D(self, each_cluster):
        '''
        this function computes the sum of the pairwise distance(repeatedly) of all points in one cluster;
        each pair be counting twice here; using the short formula below instead of the original meaning of pairwise distance of all points
        each_cluster: np.array, with containing all points' info within the array
        '''
        mu = each_cluster.mean(axis = 0)
        total = sum(sum((each_cluster - mu)**2)) * 2.0 * each_cluster.shape[0]
        return total

    def __compute_Wk(self, data, classfication_result):
        '''
        this function computes the Wk after each clustering
        data:np.array, containing all the data
        classfication_result: np.array, containing all the clustering results for all the data
        '''
        Wk = 0
        label_set = set(classfication_result.tolist())
        for label in label_set:
            each_cluster = data[classfication_result == label, :]
            D = self.__short_pair_wise_D(each_cluster)
            Wk = Wk + D/(2.0*each_cluster.shape[0])
        return Wk
     
    def __bounding_box(self):
            xmin, xmax = min(self.data,key=lambda a:a[0])[0], max(self.data,key=lambda a:a[0])[0]
            ymin, ymax = min(self.data,key=lambda a:a[1])[1], max(self.data,key=lambda a:a[1])[1]
            return (xmin,xmax), (ymin,ymax)
    
    def _fit(self):
        
        #shape = data.shape
        B = self.max_clusters+1
        (xmin,xmax), (ymin,ymax) = self.__bounding_box()
        
        K= list(range(self.min_clusters,self.max_clusters+2))
        
        gaps = np.zeros(len(K))
        Wks = np.zeros(len(K))
        Wkbs = np.zeros((len(K),B))
    
        for indk, k in enumerate(K):
            cl = clustering(data=self.data, algorithm=self.algorithm, n_clusters=k, connectivity=self.connectivity)
            cl.fit()
            classfication_result = cl.labels
            #compute the Wk for the classification result
            Wks[indk] = self.__compute_Wk(self.data, classfication_result)
            
            # clustering on B reference datasets for each 'k' 
            for i in range(B):
                Xb = []
                for n in range(len(self.data)):
                    Xb.append([np.random.uniform(xmin,xmax),
                              np.random.uniform(ymin,ymax)])
                Xb = np.array(Xb)
                cl = clustering(data=Xb, algorithm=self.algorithm, n_clusters=k, connectivity=self.connectivity)
                cl.fit()
                classfication_result_b = cl.labels
                Wkbs[indk,i] = self.__compute_Wk(Xb,classfication_result_b)
    
        #compute gaps and sk
        gaps = (np.log(Wkbs)).mean(axis = 1) - np.log(Wks)        
        sd_ks = np.std(np.log(Wkbs), axis=1)
        sk = sd_ks*np.sqrt(1+1.0/B)
        
        G = []
        for i in range(1, len(gaps)):
            G.append(gaps[i-1] - (gaps[i]-sk[i]))
        G = np.array(G)
        foundG= np.where(G > 0)[0][0]
        self.k_value = K[foundG]
        fig = pl.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        ax.bar(K[:-1], G, alpha=0.5, color='b', align='center')
        ax.set_xlabel('Number of clusters K', fontsize=14)
        ax.set_ylabel(r'$Gap(K)-Gap(K+1)-S_{(K+1)}$', fontsize=14)
        if self.k_value == 1:
            title = 'Gap statistic found %s cluster' % (self.k_value)
        else:
            title = 'Gap statistic found %s clusters' % (self.k_value)
        ax.set_title(title, fontsize=14)
        ax.xaxis.set_ticks(list(range(self.min_clusters, self.max_clusters+1)))
        pl.xlim(self.min_clusters-1, self.max_clusters+1)
        #pl.ylim(min(G), max(G))
        #pl.grid(False)
        pl.title(title)
        fp=self.path+'detK_N%s' % (str(len(self.data)))+'.png'
        pl.savefig(fp, bbox_inches='tight')
        pl.close('all')
        
        return(K, Wks, Wkbs, sk, self.k_value, gaps, G)
    
    
class Elbow(object):
    
    __slots__ = ['data', 'algorithm', 'method', 'linkage',
                 'min_clusters', 'max_clusters', 'connectivity', 'path', 'title', '__dict__']
        
    def __init__(self, data=None, algorithm='KMeans', method='k-means++', linkage='ward', #data=None,
                 min_clusters=1, max_clusters=20, connectivity=None, path='./', title=None):
        self.data=data
        self.algorithm=algorithm
        self.method=method
        self.linkage=linkage
        self.min_clusters=min_clusters
        self.max_clusters=max_clusters
        self.path = path
        self.connectivity = connectivity
        self.title=title

    
    def fit(self):
        
        def angle_data_elbow(x1, y1, x2, y2):
            u = np.array([x1,y1])
            v = np.array([x2,y2])
            cos = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
            angle =  np.arccos(np.clip(cos, -1, 1))
            return ((angle*180)/np.pi)
        
        def BestPoint(curve):
            nPoints = len(curve)
            allCoord = np.vstack((list(range(nPoints)), curve)).T
            np.array([list(range(nPoints)), curve])
            firstPoint = allCoord[0]
            lineVec = allCoord[-1] - allCoord[0]
            lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
            vecFromFirst = allCoord - firstPoint
            scalarProduct = np.sum(vecFromFirst * repmat(lineVecNorm, nPoints, 1), axis=1)
            vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
            vecToLine = vecFromFirst - vecFromFirstParallel
            distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
            idxOfBestPoint = np.argmax(distToLine)
            
            return idxOfBestPoint
        
        if (self.title == None):
            self.title=self.algorithm

        start=time.time()
        
        K = list(range(self.min_clusters, self.max_clusters+1))
        if (self.algorithm == 'KMeans'):
            CM = [cluster.KMeans(n_clusters=k, init=self.method).fit(self.data) for k in K]
        
        elif (self.algorithm=='SpectralClustering'):
            CM = [cluster.SpectralClustering(n_clusters=k, assign_labels='discretize', affinity='nearest_neighbors').fit(self.data) for k in K]
            
            labels = []
            for c in CM:
                labels.append(c.labels_)
                c.cluster_centers_ = []
                for lb in set(c.labels_):
                    c.cluster_centers_.append(self.data[c.labels_==lb].mean(axis=0))
                    
#        elif (self.algorithm == 'BayesianGaussianMixture'):
#            CM = [mixture.BayesianGaussianMixture(n_components=k, covariance_type='spherical').fit(data) for k in K]
#            
#            labels = []
#            for c in CM:
#                c.labels_ = c.predict(data)
#                labels.append(c.labels_)
#                c.cluster_centers_ = []
#                for lb in set(c.labels_):
#                    c.cluster_centers_.append(data[c.labels_==lb].mean(axis=0))
        
        elif (self.algorithm == 'PAMCluster'):
            CM = [clustering(self.data, algorithm=self.algorithm, n_clusters=k).fit() for k in K]
        
        else:
            CM = [cluster.AgglomerativeClustering(n_clusters=k, connectivity=self.connectivity,
                                          linkage=self.linkage).fit(self.data) for k in K]
            labels = []
            for c in CM:
                labels.append(c.labels_)
                c.cluster_centers_ = []
                for lb in set(c.labels_):
                    c.cluster_centers_.append(self.data[c.labels_==lb].mean(axis=0))
        
        Z_k = [cdist(self.data, c.cluster_centers_, 'sqeuclidean') for c in CM]
        dist = [np.min(z,axis=1) for z in Z_k]
        avgWithinSS = [sum(d)/self.data.shape[0] for d in dist]
        
        kIdx = BestPoint(avgWithinSS)
        
        '''
        max_angle=0.0
        kIdx=0
        t=0
        while t < (len(avgWithinSS)-2):
             angle=angle_data_elbow(avgWithinSS[t], avgWithinSS[t+1], avgWithinSS[t+1],avgWithinSS[t+2])
    #         print angle, t
             if max_angle<angle:
                 max_angle=angle
                 kIdx=t+1
             t=t+1
        '''
        
        # elbow curve
        fig = pl.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        ax.plot(K, avgWithinSS, 'b*-')
        ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        #pl.grid(True)
        pl.xlabel('Number of clusters')
        pl.ylabel('Average within-cluster sum of squares')
        pl.xlim(self.min_clusters, self.max_clusters)
        pl.title(self.title)
        fp=self.path+self.title+'.png'
        pl.savefig(fp, bbox_inches='tight')
        pl.close('all')
        stop = time.time()
        self.time=stop-start
        #print ("Elbow time: %0.3f" % (self.time))
        
        del dist
        del c
        del CM

        return K[kIdx]      