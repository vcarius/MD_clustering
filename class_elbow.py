# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 2017

@author: carius
"""

"""
Class elbow
"""
class Elbow(object):
        
    def __init__(self, algorithm='KMeans', #method='k-means++', #data=None,
                 min_cluster=2, max_cluster=20, connectivity=None):
        self.algorithm=algorithm
        #self.method=method
        self.min_cluster=min_cluster
        self.max_cluster=max_cluster
#        self.data = data
        self.connectivity = connectivity

    def __angle_data_elbow(self, x1, y1, x2, y2):
        import numpy as np
        u = np.array([x1,y1])
        v = np.array([x2,y2])
        cos = np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))
        angle =  np.arccos(np.clip(cos, -1, 1))
        return ((angle*180)/np.pi)
    
    def fit(self, data=None, path='./', title=None):

        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pl
        from sklearn import cluster
        from sklearn import mixture
        from scipy.spatial.distance import cdist
        import time
        
        if (title == None):
            title=self.algorithm

        start=time.time()
        
        K = list(range(self.min_cluster, self.max_cluster+1))
        if (self.algorithm == 'KMeans'):
            CM = [cluster.KMeans(n_clusters=k, init='k-means++').fit(data) for k in K]
        
        elif (self.algorithm=='SpectralClustering'):
            CM = [cluster.SpectralClustering(n_clusters=k, assign_labels='discretize', affinity='nearest_neighbors').fit(data) for k in K]
            
            labels = []
            for c in CM:
                labels.append(c.labels_)
                c.cluster_centers_ = []
                for lb in set(c.labels_):
                    c.cluster_centers_.append(data[c.labels_==lb].mean(axis=0))
                    
        elif (self.algorithm == 'BayesianGaussianMixture'):
            CM = [mixture.BayesianGaussianMixture(n_components=k, covariance_type='spherical').fit(data) for k in K]
            
            labels = []
            for c in CM:
                c.labels_ = c.predict(data)
                labels.append(c.labels_)
                c.cluster_centers_ = []
                for lb in set(c.labels_):
                    c.cluster_centers_.append(data[c.labels_==lb].mean(axis=0))
        
        elif (self.algorithm == 'Agglomerative'):
            CM = [cluster.AgglomerativeClustering(n_clusters=k, connectivity=self.connectivity,
                                          linkage='ward').fit(data) for k in K]
            labels = []
            for c in CM:
                labels.append(c.labels_)
                c.cluster_centers_ = []
                for lb in set(c.labels_):
                    c.cluster_centers_.append(data[c.labels_==lb].mean(axis=0))
        
        Z_k = [cdist(data, c.cluster_centers_, 'euclidean') for c in CM]
        dist = [np.min(z,axis=1) for z in Z_k]
        avgWithinSS = [sum(d)/data.shape[0] for d in dist]
        max_angle=0.0
        kIdx=0
        t=0
        while t < (len(avgWithinSS)-2):
             angle=self.__angle_data_elbow(avgWithinSS[t], avgWithinSS[t+1], avgWithinSS[t+1],avgWithinSS[t+2])
    #         print angle, t
             if max_angle<angle:
                 max_angle=angle
                 kIdx=t+1
             t=t+1
        # elbow curve
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(K, avgWithinSS, 'b*-')
        ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, 
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
        pl.grid(True)
        pl.xlabel('Number of clusters')
        pl.ylabel('Average within-cluster sum of squares')
        pl.title(title)
        fp=path+title+'.png'
        pl.savefig(fp, bbox_inches='tight')
        pl.close('all')
        stop = time.time()
        self.time=stop-start
        #print ("Elbow time: %0.3f" % (self.time))
        return K[kIdx]      