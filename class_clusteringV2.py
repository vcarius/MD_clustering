# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:18:22 2017

@author: carius
"""

from scipy.spatial import distance
import numpy as np
import random
from sklearn import metrics
from sklearn import cluster
from sklearn import mixture
from itertools import combinations as comb
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.style.use(['seaborn-paper'])
#plt.rcParams['ytick.labelsize'] = 'large'
#plt.rcParams['xtick.labelsize'] = 'large'

from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


sns.set_context('paper')
sns.set_color_codes()

random.seed()


#class BaseCluster(object):
#        
#    """
#    BaseCluster
#        Common variables and methods to each clustering method
#        Properties
#            distance        - distance function
#            num_clusters    - K value
#            distance_matrix - Distance matrix (Optional in all cases)
#            data_matrix     - "Current" sample data.
#    """
#
#    __slots__ = ['data_matrix', 'num_clusters', 'distance_matrix', '__dict__']
#    
#    def __init__(self, data_matrix, num_clusters, distance_matrix=None):
#
#        #self.__dict__ = {'data_matrix': data_matrix, 'distance': distance_metric, 'num_clusters': num_clusters, 
#        #                  'distance_matrix': distance_matrix, 'num_samples': data_matrix.shape[0]}
#        self.data_matrix = data_matrix
#        self.num_clusters = num_clusters
#        self.distance_matrix = distance_matrix
#        self.num_samples = data_matrix.shape[0]
#        
#    def _gen_distance_matrix(self, data_matrix, num_samples):
#        """Generate a distance matrix so that we can use indices to specify distance rather than data vectors"""
#
#        distance_matrix = np.zeros((num_samples, num_samples), dtype = float)
#
#        for i, j in comb(range(num_samples), 2):
#            distance_matrix[i][j] = distance.euclidean(data_matrix[i], data_matrix[j])
#
#        return distance_matrix + distance_matrix.T
#
#
#class PAMCluster(BaseCluster):
#    """
#    PAMCluster
#        Performs Partition Around Medoids clustering algorithm
#        Usage:
#            PAMCluster(datapoints, data_matrix, num_clusters, distance_metric, distance_matrix)
#                data_matrix     - Current sample data
#                num_clusters    - K
#                distance_matrix - Optional distance matrix
#    """
#
#    #__slots__ = ['data_matrix', 'num_clusters', 'distance_matrix']
#    
#    def __init__(self, data_matrix, num_clusters, distance_matrix=None):
#
#        BaseCluster.__init__(self, data_matrix, num_clusters, distance_matrix)
#
#        num_samples = data_matrix.shape[0]
#
#        if distance_matrix is None:
#            distance_matrix = self._gen_distance_matrix(data_matrix, num_samples)
#
#        #Work
#        medoids = self._cluster_data(random.sample(range(num_samples), num_clusters), distance_matrix, num_samples)
#        
#        #Assign cluster ids
#        labels = []
#        for j in range(num_samples):
#            medoid = min([ (distance_matrix[x][j], x) for x in medoids ])[1]
#            labels.append(medoid)
#        labels = np.array(labels)
#        
#        for y, z in enumerate(set(labels)):
#            labels[np.where(labels==z)]=y
#            
#        self.labels_ = labels
#        self.medois_ = medoids
#        
#        self.cluster_centers_ = []
#        for lb in set(self.labels_):
#            self.cluster_centers_.append(data_matrix[self.labels_==lb].mean(axis=0))
#        
#        
#    def _cluster_data(self, medoids, distance_matrix, num_samples):
#        """Swap a medoid within a cluster if it reduces the total cost"""
#
#        swapped  = 1
#        old_cost = sum([ min([ distance_matrix[x][j] for x in medoids ]) for j in range(num_samples) ])
#
#        while swapped:
#            swapped = 0
#
#            for i in range(self.num_clusters):
#                new_medoids = list(medoids)
#                old_medoid = medoids[i]
#
#                #I really hope this is faster
#                search_space = list(range(num_samples))
#                not_search = new_medoids
#                not_search.sort(reverse=True)
#
#                for idx in not_search:
#                    del search_space[idx]
#    
#                for sample_idx in search_space:
#                    new_medoids[i] = sample_idx
#                    
#                    #Objective function
#                    new_cost = sum([ min([ distance_matrix[x][j] for x in new_medoids ]) for j in range(num_samples) ])
#
#                    if new_cost < old_cost:
#                        swapped = 1
#                        old_cost = new_cost
#                        medoids = list(new_medoids)
#
#        return medoids

class clustering(object):
    
    __slots__ = ['data', 'algorithm', 'k_method', 'linkage', 'affinity', 'n_clusters', 'connectivity', '__dict__']
        
    def __init__(self, data=None, algorithm='AgglomerativeClustering', k_method=None, linkage=None, affinity=None, n_clusters=2, connectivity=None):
        self.data = data
        self.algorithm=algorithm
        self.k_method=k_method
        self.linkage=linkage
        self.affinity=affinity
        self.n_clusters=n_clusters
        self.connectivity = connectivity
    
    def __writeLog(self,path='./', name= None, info=None):
        logfile = open(path+name, 'a')    
        logfile.write(info)
        logfile.close()

    def __compute_s(self, data=None, labels=None, center=None, medoid=None):
        s = 0
        cluster = data[labels==labels[medoid]]
        for c in cluster:
            s += distance.euclidean(c, center)
        return np.sqrt(s/len(cluster))

    def __compute_Rij(self, i=None, j=None, data=None, labels=None, cluster_centers=None, cluster_medoids=None, nc=None):
        Rij = 0
        try:
            d = distance.euclidean(cluster_centers[i],cluster_centers[j])
            Rij = (self.__compute_s(data, labels, cluster_centers[i], cluster_medoids[i]) + self.__compute_s(data, labels, cluster_centers[j], cluster_medoids[j]))/d
        except:
            Rij = 0	
        return Rij

    def __compute_R(self, data=None, labels=None, cluster_centers=None, cluster_medoids=None, nc=None): 
        list_r = []
        list_r.append(0.0)
        for i in range(nc):
            for j in range(nc):
                if(i!=j):
                    temp = self.__compute_Rij(i, j, data, labels, cluster_centers, cluster_medoids, nc)
                    list_r.append(temp)
        
        return max(list_r)

    def compute_DB_index(self, data=None, labels=None, cluster_centers=None, cluster_medoids=None):
        sigma_R = 0.0
        nc = len(cluster_centers)
        for i in range(nc):
            sigma_R = sigma_R + self.__compute_R(data, labels, cluster_centers, cluster_medoids, nc)

        DB_index = float(sigma_R)/float(nc)
        
        return DB_index

    def quality(self, manual_labels=None):        
        if 1 < len(set(self.labels)):
            self.__silhouette = metrics.silhouette_score(self.data, self.labels)
            self.__CHI = metrics.calinski_harabaz_score(self.data, self.labels)
        else:
            self.__silhouette = 0.
            self.__CHI = 0.
        self.__DB = self.compute_DB_index(data=self.data, labels=self.labels, cluster_centers=self.cluster_centers_, cluster_medoids=self.medoids)
        
        if np.all(manual_labels !=None):
            self.__homogeneity = metrics.homogeneity_score(manual_labels, self.labels)
            self.__completeness = metrics.completeness_score(manual_labels, self.labels)
            self.__vmeasure = metrics.v_measure_score(manual_labels, self.labels)
            self.__ARS = metrics.adjusted_rand_score(manual_labels, self.labels)
            self.__AMI =  metrics.adjusted_mutual_info_score(manual_labels, self.labels)
            self.__FMI = metrics.fowlkes_mallows_score(manual_labels, self.labels)
        else:
            self.__homogeneity = '-'
            self.__completeness = '-'
            self.__vmeasure = '-'
            self.__ARS = '-'
            self.__AMI='-'
            self.__FMI = '-'
        
        name_alg = self.algorithm
        
        if name_alg is 'AgglomerativeClustering':
            name_alg = self.algorithm+'_'+self.linkage
        
        if name_alg is 'KMeans':
            name_alg = self.algorithm+'_'+self.k_method
        
        qualities = {'algorithm':name_alg, 'n_clusters':self.n_clusters, 'silhouette':self.__silhouette, 'Davies-Bouldin':self.__DB, 'homogeneity':self.__homogeneity,
                   'completeness':self.__completeness, 'vmeasure':self.__vmeasure, 'ARS':self.__ARS, 'AMI':self.__AMI, 'FMI':self.__FMI, 'Calinski-Harabaz Index': self.__CHI}
        
        return qualities
        
    
    def plot (self, data=None, title=None, path='./'):
        
        if data is None:
            data = self.data
        
        index=np.arange(len(self.data))
        nzeros = len(str(abs(len(self.data))))  
        ####plot cluster---------------------------------------------------------------------------------------
        yint=np.asarray(list(range(-1,max(self.labels)+1)))
        plt.figure(figsize=(14, 8)); plt.title(title);
        plt.xlim([0,len(self.data)]);plt.yticks(yint)
        ix=index.argsort()
        plt.scatter(index[ix], self.labels[ix], c=self.labels[ix], s=40, linewidths=0)
        plt.ylabel('Cluster Label');plt.xlabel('Conformation')
        fp=path+title+'-1D.png'
        plt.savefig(fp, bbox_inches='tight')
        plt.close('all')
        #####plot clusters colors -----------------------------------------------------------------------------
        #x = [self.labels]*(10**(nzeros-2))
        x = [self.labels]*(len(self.data)/5)
        fig = plt.figure(figsize=(14, 8)); plt.title(title); ax = fig.add_subplot(111); 
        values = np.unique(self.labels); m= ax.matshow(x); 
        ax.xaxis.set_ticks_position('bottom'); ax.axes.get_yaxis().set_visible(False)
        colors = [ m.cmap(m.norm(value)) for value in values];
        patches = [ mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=values[i]) ) for i in range(len(values)) ]; 
        ax.legend(handles=patches, bbox_to_anchor=(1., 1.15), loc=2, borderaxespad=2., ncol = int(len(set(self.labels))/10.)+1);plt.xlabel('Conformation')
        
        fp=path+title+'-colors.png'
        plt.savefig(fp, bbox_inches='tight')
        plt.close('all')
        
        """
        true_medians = []
        for i in set(self.labels):
            a = np.array(np.where(self.labels==i))
            true_medians.append(a.flat[np.abs(a - self.medoids[i]).argmin()])
        true_medians=np.array(true_medians)
        """
        ####write logfile----
        for l,m in zip(set(self.labels), self.medoids):
            lst = ([str(j).rjust(nzeros, '0') for j in index[self.labels==l] ])
            lst = list(map(str, lst))
            line = ",".join(lst)
            st= '\n'+self.algorithm+'- cluster '+str(l)+'\t['+str(m).rjust(nzeros, '0')+']\t'+line
            self.__writeLog(path=path, name = title, info=st)
    
    def plot2D (self, data=None, medoids = None, title=None, path='./'):
        
        font = FontProperties()
        font.set_weight('bold')
        font.set_size('x-large')
         
        if data is None:
            data = self.data
        index=np.arange(len(self.data))
        ####plot 2D--------------------------------------------------------------------------------------------
        plt.figure(figsize=(14, 8)); plt.title(title)
        plt.xlabel('Component 1');plt.ylabel('Component 2')
        '''
        vmin1 = min(data[:, 0]) + min(data[:, 0])*0.5
        vmax1 = max(data[:, 0]) + max(data[:, 0])*0.5
        vmin2 = min(data[:, 1]) + min(data[:, 1])*0.5
        vmax2 = max(data[:, 1]) + max(data[:, 1])*0.5
        '''
        #plt.xlim([vmin1, vmax1]); plt.ylim([vmin2, vmax2])
        plt.scatter(data[:, 0], data[:, 1], c=self.labels.astype(np.float), s=70, linewidths=0)
        for (k,u) in zip(index, data):
            plt.text(u[0], u[1], str(k).split('-')[0], fontsize=10)  
        # Plot the centroids-------------
        #medians, _ = metrics.pairwise_distances_argmin_min(self.cluster_centers_, self.data)
        if medoids is None:
            medoids = self.medoids
        for m in medoids:
            plt.scatter(data[m, 0], data[m, 1], marker='o', s=100, linewidths=3., color='grey', zorder=10)
            #plt.text(data[m, 0], data[m, 1], str(m), fontsize=16, color='r', fontproperties=font) 
            plt.text(data[m, 0], data[m, 1], str(m), color='r', fontproperties=font) 
        fp=path+title+'-2D.png'
        plt.savefig(fp, bbox_inches='tight')
        plt.close('all')
        ####plot 2D_2-------------------------------------------------------------------------------------------
        plt.figure(figsize=(14, 8)); plt.title(title)
        plt.xlabel('Component 1');plt.ylabel('Component 2')
        #plt.xlim([vmin1, vmax1]); plt.ylim([vmin2, vmax2])
        plt.scatter(data[:, 0], data[:, 1], c=self.labels.astype(np.float), s=70, linewidths=0)
        for m in medoids:
            plt.scatter(data[m, 0], data[m, 1], marker='o', s=100, linewidths=3., color='grey', zorder=10)
            plt.annotate(str(m), (data[m, 0], data[m, 1]), color='r', fontproperties=font) 
        
        fp=path+title+'-2D_2.png'
        plt.savefig(fp, bbox_inches='tight')
        plt.close('all')
        
    
    def plot3D (self, data=None, medoids = None, title=None, path='./'):
        
        if data is None:
            data = self.data
        
        index=np.arange(len(self.data))
        """
        true_medians = []
        for i in set(self.labels):
            a = np.array(np.where(self.labels==i))
            true_medians.append(a.flat[np.abs(a - self.medoids[i]).argmin()])
        true_medians=np.array(true_medians)
        """
        ####plot 3D--------------------------------------------------------------------------------------------
        plt.title(title)
        fig=plt.figure(figsize=(14, 8))
        ax = fig.gca(projection='3d')
        ax.set_title(title)
        ax.set_xlabel('Component 1');ax.set_ylabel('Component 2');ax.set_zlabel('Component 3')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=self.labels.astype(np.float), s=70, linewidths=0)
        ax.grid(False)
        for (k,u) in zip(index, data):
          ax.text(u[0], u[1], u[2], str(k).split('-')[0], fontsize=10)
        
    #####Plot the centroids-------------
        if medoids is None:
            medoids = self.medoids
        
        for m in medoids:
          ax.scatter(data[m, 0], data[m, 1], data[m, 2], marker='o', s=100, linewidths=3., color='grey', zorder=10)
          ax.text(data[m, 0], data[m, 1], data[m, 2], str(m), fontsize=16, color='r')  
         
        fp=path+title+'3D.png'
        plt.savefig(fp, bbox_inches='tight')
        plt.close('all')
    ####plot 3D_2-------------------------------------------------------------------------------------------
        plt.title(title)
        fig=plt.figure(figsize=(14, 8))
        ax = fig.gca(projection='3d')
        ax.set_title(title)
        ax.set_xlabel('Component 1');ax.set_ylabel('Component 2');ax.set_zlabel('Component 3')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=self.labels.astype(np.float), s=70, linewidths=0)
        ax.grid(False)
    ####Plot the centroids-------------  
        for m in medoids:
          ax.scatter(data[m, 0], data[m, 1], data[m, 2], marker='o', s=100, linewidths=3., color='grey', zorder=10)
          ax.text(data[m, 0], data[m, 1], data[m, 2], str(m), fontsize=16, color='r')  
         
        fp=path+title+'3D_2.png'
        plt.savefig(fp, bbox_inches='tight')
        plt.close('all')
           
    def fit (self):
                
        if (self.algorithm=='KMeans'):
            if self.k_method is None:
                self.k_method = 'k-means++'
            cl = cluster.KMeans(n_clusters=self.n_clusters, init=self.k_method).fit(self.data)
            self.cluster_centers_ = cl.cluster_centers_
        
        elif (self.algorithm=='Affinity'):
            if self.affinity is None:
                self.affinity = 'euclidean'
            cl = cluster.AffinityPropagation(damping=.75).fit(self.data)
            self.cluster_centers_ = []
            for lb in set(cl.labels_):
                self.cluster_centers_.append(self.data[cl.labels_==lb].mean(axis=0))            
        elif (self.algorithm=='Meanshift'):
            bandwidth = cluster.estimate_bandwidth(self.data, quantile=.2)
            #bandwidth = cluster.estimate_bandwidth(data) 
            cl = cluster.MeanShift(bandwidth = bandwidth).fit(self.data)
            self.cluster_centers_ = []
            for lb in set(cl.labels_):
                self.cluster_centers_.append(self.data[cl.labels_==lb].mean(axis=0))            
    
        elif (self.algorithm=='SpectralClustering'):
            cl = cluster.SpectralClustering(n_clusters=self.n_clusters, eigen_solver='arpack', assign_labels='discretize', affinity='nearest_neighbors').fit(self.data)
            self.cluster_centers_ = []
            for lb in set(cl.labels_):
                self.cluster_centers_.append(self.data[cl.labels_==lb].mean(axis=0)) 
        elif (self.algorithm == 'BayesianGaussianMixture'):
            cl = mixture.BayesianGaussianMixture(n_components=self.n_clusters, covariance_type='spherical').fit(self.data)
            cl.labels_ = cl.predict(self.data)
            self.cluster_centers_ = []
            for lb in set(cl.labels_):
                self.cluster_centers_.append(self.data[cl.labels_==lb].mean(axis=0))
        
        elif (self.algorithm == 'PAMCluster'):
            cl = PAMCluster(data_matrix=self.data, num_clusters=self.n_clusters, distance_matrix=None)
            self.cluster_centers_ = []
            for lb in set(cl.labels_):
                self.cluster_centers_.append(self.data[cl.labels_==lb].mean(axis=0))
        
        else:
            self.algorithm='AgglomerativeClustering'
            if self.linkage is None:
                self.linkage = 'ward'
            cl = cluster.AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage).fit(self.data)
            self.cluster_centers_ = []
            for lb in set(cl.labels_):
                self.cluster_centers_.append(self.data[cl.labels_==lb].mean(axis=0))            
        
        self.labels=cl.labels_
        self.n_clusters=len(set(cl.labels_))
        medians, _ = metrics.pairwise_distances_argmin_min(self.cluster_centers_, self.data)
        
        medoids = []
        for i, j in zip(set(self.labels), list(range(len(medians)))):
            a = np.array(np.where(self.labels==i))
            medoids.append(a.flat[np.abs(a - medians[j]).argmin()])
        self.medoids=np.array(medoids)
        self.cluster_centers_  = self.data[self.medoids]
        #print "medoids "
        #print self.medoids
    
        return self
