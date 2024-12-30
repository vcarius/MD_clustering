#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: viniciuscarius
"""

import warnings; warnings.simplefilter('ignore')

import time
import os 
import random
import h5py

import numpy as np
import pandas as pd
import scipy as sp
from scipy.spatial import distance
from scipy import stats
from itertools import combinations as comb

from class_KpredictionV2 import *
from class_clusteringV2 import clustering
from class_energy import Energy


from sklearn import metrics, manifold
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.style.use(['seaborn-paper'])
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import csv
sns.set_context('paper')
sns.set_color_codes()

import pytraj as pt

random.seed(100)
    
def plot_rmsf(data=None, step=15., path=None):
    
    plt.figure(figsize=(12.5, 8))
    '''
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(9))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    '''
    def computeTicks (x, step):
        """
        Computes domain with given step encompassing series x
        @ params
        x    - Required - A list-like object of integers or floats
        step - Optional - Tick frequency
        """
        #print step
        import math as Math
        xMax, xMin = Math.ceil(max(x)), Math.floor(min(x))
        dMax, dMin = xMax + abs((xMax % step) - step) + (step if (xMax % step != 0) else 0), xMin - abs((xMin % step))
        
        return np.arange(dMin, dMax, step)
    
    x = computeTicks(x=np.arange(1, len(data.T[0])+1, 1), step=15.)
    y = computeTicks(x=np.arange(0, int(max(data.T[1])+1), 1), step=1.0)
    plt.xticks(x)
    plt.yticks(y)
    plt.xlim(min(data[:, 0]), max(data[:, 0]))
    plt.ylim(0, int(max(data[:, 1]))+1)
    
    plt.xlabel('residues', color='black', fontsize=18); plt.ylabel(r'RMSF ($\AA$)', color='black', fontsize=18)
    plt.plot(data[:, 0], data[:, 1])
    plt.savefig(path+'RMSF.eps', bbox_inches='tight', format='eps', dpi=600)
    plt.close()

def compute_RMSD_Variation(traj=None, rmsd_matrix= None, labels=None):
    
    extra_disp = 0.
    intra_disp = 0.
    
    grand_mean = np.mean(rmsd_matrix, axis=0)
    stat=[]
    for i in set(labels):
        n_conf = np.where(labels==i)[0]
        print(n_conf)
        rmsd_k = rmsd_matrix[n_conf]
        mean_k = np.mean(rmsd_k, axis=0)
        extra_disp += len(n_conf)*np.sum((mean_k - grand_mean)**2)
        intra_disp += np.sum((rmsd_k - mean_k) ** 2)
        
        if len(set(labels)) == 1:
            var_extra_disp = 0
            var_intra_disp = 0
        else:
            var_intra_disp = intra_disp/(len(labels) - len(set(labels)))
            var_extra_disp = extra_disp/(len(set(labels))-1)
        if var_intra_disp == 0 :
            F_test = 1.
        else:
            F_test = var_extra_disp/var_intra_disp
            
    stat.append({'intra_disp':intra_disp, 'var_intra_disp': var_intra_disp, 'extra_disp': extra_disp, 'var_extra_disp': var_extra_disp, 'F_test': F_test})
    return stat

def compute_RMSD_Medoids(traj=None, rmsd_matrix=None, medoids=None, labels=None):
    
    import heapq
    
    result = []
    stat = []
    for m in medoids:
        indices = np.where(labels ==labels[m] )
        m_rmsd = rmsd_matrix[m][indices]
        
        df = pd.DataFrame({'Frame':indices[0], 'Label': labels[indices], 'RMSD': m_rmsd})
        
        result.append(df)
        
        average = np.mean(m_rmsd)
        MIN = heapq.nsmallest(2, m_rmsd)[-1]
        MAX = heapq.nlargest(2, m_rmsd)[-1]
        median = np.median(m_rmsd)
        std = np.std(m_rmsd)
        var = np.var(m_rmsd)
        count = len(m_rmsd)
        stat.append({'CLUSTER': labels[m], 'SIZE':count, 'MIN': MIN, 'MAX': MAX, 'AVERAGE': average, 'MEDIAN':median, 'STD':std, 'VAR':var})
        
    result = pd.concat(result)
    stat = pd.DataFrame(stat)
    
    return result, stat

def Matrices_Distance (Matrices=None, medoids=None, labels=None, ):

    Ndecimals = 2
    decade = 10**Ndecimals
    ml = 0
    distance = []
    if medoids is None:
        num_frames = Matrices.shape[0]
        distance = np.zeros(shape=(num_frames, num_frames), dtype=np.float64)
        np.fill_diagonal(distance, 0.0)
        k = 0
        for i in range(num_frames):
            for j in range(num_frames):
                j = j + k
                if i != j and j <num_frames:
                    m = Matrices[i] - Matrices[j]
                    d = np.linalg.norm(m, ord='fro')
                    distance[i][j] = np.trunc(d*decade)/decade
                    
            k += 1
    else:
        
        while ml < len(medoids):
            fl = 0
            frm = medoids[ml]
            while fl < len(labels):
                if ml == labels[fl]:
                    m = Matrices[fl] - Matrices[frm]
                    d = np.linalg.norm(m, ord='fro')
                    r1 = np.trunc(d*decade)/decade
                    distance.append([ml,fl,r1])
                fl += 1
            ml += 1
        
    
    return distance

def K_prediction (data=None, method=None, algorithm=None, connectivity=None, title=None, min_clusters=2, max_clusters=20, path='./'):
    t1 = time.time()
    if method == 'elbow':
        n_clusters = Elbow(data=data, algorithm=algorithm, connectivity=connectivity, path=path, title='Elbow-'+title).fit()
    elif method == 'BIC':
        n_clusters = BIC(data=data, algorithm=algorithm, title='BIC-'+algorithm, path=path).k_number
    elif method == 'GAP':
        Gap = GAP_statistic(data, min_clusters=min_clusters, max_clusters=max_clusters, path=path)
        n_clusters = Gap.k_value
    else:
        n_clusters = MaxSilhouette(data=data, algorithm=algorithm, connectivity=connectivity, path=path, title='Max silhouette-'+title).k_number
    print(("total time to run "+method+" method to "+algorithm+" %4f\n"%(time.time() - t1)))
    return n_clusters

def Clustering_data(data=None, reduced_dimension=None, traj=None, rmsd_matrix= None, algorithm=None, n_clusters=None, connectivity=None, manual_labels = None, title=None, path='./'):
    #t1 = time.time()
    cl = clustering(data=data, algorithm=algorithm, n_clusters=n_clusters, connectivity=connectivity)
    cl.fit()
    qual = cl.quality(manual_labels=manual_labels)
    #print(("total time to run "+algorithm+" algorithm %4f\n"%(time.time() - t1)))
    #t1 = time.time()
        
    cl.plot(data=reduced_dimension, title=title, path=path)
    cl.plot2D(data=reduced_dimension, title=title, path=path)
    
    medoids = cl.medoids
    if reduced_dimension.shape[1] > 2:
        cl.plot3D(data=reduced_dimension, title=title, path=path)
    #min_energy = plot_map(data=data, bins=50, title=title, cluster_centers=cl.medoids, labels=cl.labels, path=path)
    #print(("total time to plot %4f\n"%(time.time() - t1)))
    #t1 = time.time()
    if rmsd_matrix is not None:
        rmsd, stat = compute_RMSD_Medoids(traj=traj,  rmsd_matrix=rmsd_matrix, medoids=cl.medoids, labels=cl.labels )
        rmsd.to_csv(path+algorithm+'-RMSD.csv', index=False)
        stat.to_csv(path+algorithm+'-RMSD_stats.csv', index=False)
        var = compute_RMSD_Variation(traj=traj, rmsd_matrix=rmsd_matrix, labels=cl.labels)
        C = pd.DataFrame(var)
        C.to_csv(path+algorithm+'-RMSD_var.csv', index=False)
    #print(("total time to calculate stats %4f\n"%(time.time() - t1)))
    
    del cl
  
    return qual, medoids

def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Script for molecular dynamic simulations analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trajectory',type=str, help='', required=True, default=None)
    parser.add_argument('--topology', type=str, help='', required=True, default=None)
    parser.add_argument('--selection', type=str, help='Atoms or residues selection', default='@CA')
    parser.add_argument('--option', type=str, help='Information used to clusterization of dataset', default='EDM')
    parser.add_argument('--residues', nargs='+', type=str, help='', default=None)    
    parser.add_argument('--rmsf_cutoff', type=float, help='', default=None)
    parser.add_argument('--rmsf_cutoff_range', nargs='+', type=int, help='', default=None)
    parser.add_argument('--contact_cutoff', type=float, help='', default=7.0)
    parser.add_argument('--paramDist', type=str, help='Function used to calculte the distance between atoms', default='euclidean')
    parser.add_argument('--pkl_file',type=str, help='Pickle file with distances between atoms', default=None)
    parser.add_argument('--contact_matrix', type=str, \
                        help='',
                        default=None)    
    parser.add_argument('--RMSD_matrix', type=str, \
                        help='',
                        default=None)
    parser.add_argument('--components', type=str, \
                        help='',
                        default=None)
    parser.add_argument('--reduction', type=str, help='method used for dimensionality reduction of data', default=None)
    parser.add_argument('--n_dims', type=int, help='number of dimensions after reduction', default=2)
    parser.add_argument('--n_neighbors', type=int, \
                        help='number of neighbors to consider for each point, this value is used by manifold reduction methods, e.g. Isomap. If n_neighbors is None, the value is redefined to max(int(data.shape[0] / 10), 1)', default=20)
    parser.add_argument('--algorithm', type=str, help='Clustering algorithm', default='KMeans')
    parser.add_argument('--KMeans_method',type=str, help='Method for initialization. ‘k-means++’ : uses a smart way to speed up convergence.  \
                        ‘random’: choose k observations (rows) at random from data for the initial centroids.', \
                        default='k_means++')
    parser.add_argument('--linkage_method',type=str, help='', default='ward')
    parser.add_argument('--connectivity', type=str, \
                        help='File with connectivity matrix. Defines for each sample the neighboring samples following a given structure of the data.',
                        default=None)
    parser.add_argument('--Knumber', type=int, help='Number of clusters for dataset.', default=None)
    parser.add_argument('--Kdetection', type=str, help='Method used to detect the number of cluster in dataset.', default='elbow')
    parser.add_argument('--Kvalues', nargs='+', type=int, help='Min and Max of expected clusters for data', default=[1, 20])
    parser.add_argument('--manualLabels',type=str, help='Labels of classes previously detected', default=None)
    parser.add_argument('--bins', type=int, help='', default=50)
    parser.add_argument('--weights', type=int, help='', default=None)
    parser.add_argument('--rangevals', nargs='+', type=int, help='', default=None)
    parser.add_argument('--step', type=int, help='', default=15.) 
    parser.add_argument('--out_name', type=str, help='', default=None)    
    parser.add_argument('--save_dir', type=str, help='directory to save results', default='results')
    args = parser.parse_args()
    
    print(args)
    
    traj_name = args.trajectory.split('/')[-1].split('.')[0]
    out = args.trajectory.split(traj_name)[0]+args.save_dir+'/'
    
    if os.path.exists(out) == False:
        os.makedirs(out, mode=00777)
    
    open_traj = time.time()
    
    traj = pt.load(args.trajectory, top=args.topology)
    traj = pt.superpose(traj, ref=0)
    select = args.selection
    print(traj)
    print(select)
    
    traj = traj[select]#traj.atom_slice(atom_indices=traj.topology.select(select))

    print(traj)
    
    print("Time to open the trajectories: %4f" %(time.time() - open_traj))
    
    if args.components is None:
        
        new_dimension = None
        
        if args.option == "eigenvalues" or args.option == "EDM":           
            
            if args.pkl_file is None:
                build_edm_matrix = time.time()
                EDM = []
                for i in range(traj.n_frames):
                    dist = distance.cdist(traj.xyz[i], traj.xyz[i], args.paramDist)
                    
                    dist = np.array(dist[np.triu_indices(dist.shape[0])], copy=False).round(decimals=2)
                    
                    EDM.append(dist)
                
                del dist
                EDM = np.asarray(EDM)
         
                
                EDM = EDM[(EDM !=0)].reshape(traj.n_frames, -1)
                
                print("Time to obtain the EDM %4f" %(time.time() - build_edm_matrix))
                
            else:
                print("Opening pkl")
                    
                EDM=pd.read_pickle(args.pkl_file)
                EDM = EDM.values
                
            if args.option == "eigenvalues":
                data = []
                for d in EDM:
                    m = sp.spatial.distance.squareform(d)
                    eigenvals, eigenvec = np.linalg.eig(m)
                    data.append(eigenvals[:3])
                    
                dfEigVals = pd.DataFrame(data)
                dfEigVals.to_csv(out+traj_name+'_eigvals_EDM.csv', sep=',', header=False,
                                 float_format='%.4f', index=False)
                
                del dfEigVals
            
            else:
            
                data = EDM
            
        elif args.option == 'RMSF':
            print('Running RMSF')
            start_r = time.time()
            pt.superpose(traj, ref=0)
    
            # compute rmsf
            rmsf_data = pt.rmsf(traj, 'byres')
            rmsf_data = np.array(rmsf_data)
            
            plot_rmsf(data=rmsf_data, step=args.step, path=out)            
            
            residues = []
            
            if args.contact_matrix is not None:

                h5f = h5py.File(args.contact_matrix, "r")
                data =  h5f["ContactMatrix"][:]
            
            else:
                
                if args.residues is not None:
                    array = args.residues[0].split(",")
                    for res in array:
                        try:
                            start, end = res.split("-")
                            if start > end:
                                aux = start
                                start =end
                                end = aux
                            residues.extend(range(int(start), int(end)+1))
                        except:
                            residues.extend([int(res)])
                    residues = np.asarray(residues) - 1
                     
                elif args.rmsf_cutoff is not None:
                    residues = np.asanyarray(sorted(rmsf_data[np.where(rmsf_data.T[1][:]>=float(args.rmsf_cutoff))], key=lambda x:x[1], reverse=True)).T[0] - 1
                    #np.array(np.where(rmsf_data.T[1]>=args.rmsf_cutoff))
                elif args.rmsf_cutoff_range is not None:
                    #print args.rmsf_cutoff_range[0]
                    #print args.rmsf_cutoff_range[1]
                    residues = np.where(np.logical_and(rmsf_data.T[1][:] >= args.rmsf_cutoff_range[0], rmsf_data.T[1][:] <= args.rmsf_cutoff_range[1]))[0][:] # media ou media
                else:
                    rmsf_data_normalized = rmsf_data.copy()
                    v = rmsf_data_normalized[:, 1]   # foo[:, -1] for the last column
                    rmsf_data_normalized[:, 1] = (v - v.min()) / (v.max() - v.min())
                    plot_rmsf(data=rmsf_data_normalized, step=args.step, path=out+"normalized_")
                    # Compute a histogram of the sample
                    bins = np.linspace(-5, 5, 30)
                    histogram, bins = np.histogram(rmsf_data_normalized[:, 1], bins=bins, normed=True)
                    
                    bin_centers = 0.5*(bins[1:] + bins[:-1])
                    
                    # Compute the PDF on the bin centers from scipy distribution object
                    pdf = stats.norm.pdf(bin_centers)
                    
                    plt.figure(figsize=(12.5, 8))
                    plt.plot(bin_centers, histogram, label="Histogram of samples")
                    plt.plot(bin_centers, pdf, label="PDF")
                    plt.legend()
                    plt.savefig(out+'normalized_RMSF_histogram.eps', bbox_inches='tight', format='eps', dpi=600)
                    plt.close()
                    a = rmsf_data_normalized.T[1][:].mean() - rmsf_data_normalized.T[1][:].std()
                    b = rmsf_data_normalized.T[1][:].mean() + rmsf_data_normalized.T[1][:].std()
                    
                    residues = np.where(np.logical_and(rmsf_data_normalized.T[1][:] >= a, rmsf_data_normalized.T[1][:] <= b))[0][:] # media ou media
        
                residues = residues.astype(int)
                print residues
                #with open('residues_rmsf-5A.txt', 'w') as f:
                
                np.savetxt('residues_rmsf.txt', rmsf_data[residues])
                contacts = {}        
                for resid in residues:
                    in_contact = []
                    for i in xrange(traj.n_frames):
                        dist = distance.cdist(traj[i].xyz[resid][np.newaxis, :], traj[i].xyz, "euclidean")
                        #print dist[0]
                        aux = np.where(dist[0]<=args.contact_cutoff)[0][:]
                        in_contact.extend(aux)
                        
                    contacts[resid]= list(set(in_contact)) #colocar dentro de dicionário!!!!
                    
                    del aux                
                    del in_contact
                
                print ("Writing contacts of residues!!")
                
                w = csv.writer(open("contacts_residues.csv", "w"))
                for key, val in contacts.items():
                    w.writerow([key, val])
    
                contact_matrix = [] 
    
                print ("Obtaining contact matrix!!")            
                
                for i in xrange(traj.n_frames):
                    aux = []
                    for resid in contacts.keys():
                        aux.extend(distance.cdist(traj[i].xyz[resid][np.newaxis, :], traj[i].xyz, "euclidean")[0][contacts[resid]])
                    contact_matrix.append(aux)
                
                data= np.array(contact_matrix)
                           
                del aux
                del contacts
                del contact_matrix            
                
                print(('time of RMSF calculate: %.4f'%(time.time() - start_r)))
                
                print ('Writing contact matrix !!')
                h5f = h5py.File(out+"ContactMatrix.h5", "w") 
                h5f.create_dataset("ContactMatrix", data=data)
                h5f.close()
                
                del h5f
            
        elif args.option == 'RMSD':
            if args.RMSD_matrix is None or os.path.exists(args.RMSD_matrix) == False:
                print("Calculating RMSD matrix!")
                
                build_rmsd_matrix = time.time()
                RMSD = np.round(pt.pairwise_rmsd(traj), 4)
                print("Time to obtain the RMSD matrix: %4f" %(time.time() - build_rmsd_matrix))
                    
                h5f = h5py.File(out+"RMSD.h5", "w") 
                h5f.create_dataset("rmsd", data=RMSD)
                args.RMSD_matrix = out+"RMSD.h5"
                
                data =  h5f["rmsd"][:]
                h5f.close()
                
                del RMSD
                del h5f

            else:
                h5f = h5py.File(args.RMSD_matrix, "r")
                data =  h5f["rmsd"][:]
                    
        else:
            build_coord_matrix = time.time()
            data = []
            
            for i in range(traj.n_frames):
                data.append(traj.xyz[i].flatten())
    
            data = np.asarray(data)
            
            print("Time to obtain the coordinates XYZ: %4f" %(time.time() - build_coord_matrix))
            
            print ('Writing contact matrix !!')
            
            h5f = h5py.File(out+"CoordinatetMatrix.h5", "w") 
            h5f.create_dataset("ContactMatrix", data=data)
            h5f.close()
            
            del h5f
        
        
        n_neighbors_ = (args.n_neighbors
                                         if args.n_neighbors is not None
                                         else max(int(data.shape[0] / 10), 1))
        
        #print ("n_neighbors is %d"%n_neighbors_)
        
        if args.option != "eigenvalues":
           
            data = MinMaxScaler().fit_transform(data)
            start_r = time.time()
            if args.reduction=='Isomap':
                print('Running Isomap')    
                data = manifold.Isomap(n_neighbors=n_neighbors_, \
                    n_components=args.n_dims, n_jobs=-1).fit_transform(data)
            
            elif args.reduction=='TSNE':
                print('Running TSNE')
                data = manifold.TSNE(n_components=args.n_dims, init='random', \
                    random_state=100, perplexity=20).fit_transform(data)
            
            elif args.reduction=='MDS':
                
                print('Running MDS')
                
                """
                if args.option=='RMSD':
                    data_triu = np.triu(data, k=0)
                    data_diag = np.diag(np.diag(data))
                    data = data_triu + data_triu.T - data_diag
                """
                #else:
                num_samples = len(data)
                distance_matrix = np.zeros((num_samples, num_samples), dtype = float)
            
                for i, j in comb(range(num_samples), 2):
                    distance_matrix[i][j] = distance.euclidean(data[i], data[j])
                
                data = distance_matrix + distance_matrix.T
            
                data=manifold.MDS(n_components=args.n_dims, n_jobs=-1, \
                    random_state=100, dissimilarity='precomputed').fit_transform(data)
            
            elif args.reduction=='Spectral':
                print('Running Spectral')
                data=(manifold.SpectralEmbedding(n_neighbors=n_neighbors_, eigen_solver='arpack', \
                    n_components=args.n_dims, n_jobs=-1, random_state=0 ).fit_transform(data))*100
                
            elif args.reduction == 'AutoEncoder':
                print('Running AutoEncoder')
                from class_autoencoder import AutoEncoder
                
                AE = AutoEncoder(n_features=data.shape[1], n_neurons=[data.shape[1], 8192, 512, 32, 2], \
                    n_epochs=200, batch_size=500,verbose=0)
                result = AE.fit(data)
                data = result.reduced
                plt.plot(result.history.history['loss']); plt.xlabel('Epochs'); plt.ylabel('Training Loss Values')
                fp=out+'Loss plot.png'
                plt.savefig(fp, bbox_inches='tight')
                plt.close('all')
                plt.plot(result.history.history['val_loss']); plt.xlabel('Epochs'); plt.ylabel('Validation Loss Values')
                fp=out+'Val_loss.png'
                plt.savefig(fp, bbox_inches='tight')
                plt.close('all')
            
            else:
                print('Running PCA')
                #X_std = StandardScaler().fit_transform(data)
                '''
                ######## DESCOMENTAR NO FUTURO ###########
                for num_comp in range(data.shape[1]):
                    pca = PCA(n_components=num_comp, copy=False)
                    pca.fit_transform(data)
                    if np.sum(pca.explained_variance_ratio_) >= 0.7:
                        break
                '''
                pca = PCA(n_components=args.n_dims, copy=False)
                
                if args.reduction is not None:
                    data = pca.fit_transform(data)
                else:
                    print ("The reduction option is None ... So, PCA will only be used for plotting !")
                    args.reduction = "PCA"                    
                    new_dimension = pca.fit_transform(data)
                            
                cum_var_exp = np.cumsum(pca.explained_variance_ratio_)
                print(('cumulative explained variance:' + str(cum_var_exp)))
                var_exp = pca.explained_variance_ratio_
                with plt.style.context('seaborn-whitegrid'):
                
                    plt.bar(list(range(1, pca.n_components_+1)), var_exp, alpha=0.8, align='center',
                            label='individual explained variance')
                    plt.step(list(range(1, pca.n_components_+1)), cum_var_exp, where='mid',
                             label='cumulative explained variance')
                    plt.grid(False)
                    plt.ylabel('Explained variance ratio')
                    plt.xlabel('Principal components')
                    plt.legend(loc='best')
                    plt.tight_layout()
                    fp=out+'cumulative explained variance.png'
                    plt.savefig(fp, bbox_inches='tight')
                    plt.close('all')
        
            componets = pd.DataFrame(data=data.astype(float))
            componets.to_csv(out+traj_name+'_'+args.reduction+'_components.csv', header=False, float_format='%.4f', index=False)
            
            print(('time of reduction: %.4f'%(time.time() - start_r)))


    else:
        data = pd.read_csv(args.components, sep=',', header=0, index_col=False).values
        
    manual_labels=args.manualLabels
    
    if manual_labels is not None and os.path.exists(manual_labels) == True:
        manual_labels = np.array(pd.read_csv(args.manual_labels)['Label'], copy=False)
        
    connect=args.connectivity
    
    if connect is not None and os.path.exists(connect) == True:
        connect = pd.read_csv(args.connectivity).values
        connect = kneighbors_graph(connect, n_neighbors=n_neighbors_, include_self=False)
        connect = 0.5 * (connect + connect.T)   
    
    if args.Knumber is None:
        n_clusters = K_prediction(data=data, method=args.Kdetection, algorithm=args.algorithm, connectivity=connect, \
                              title=args.algorithm, min_clusters=args.Kvalues[0], max_clusters=args.Kvalues[-1], \
                              path=out)    
    else:
        n_clusters = args.Knumber
        
    try:
        h5f = h5py.File(args.RMSD_matrix, "r")
        rmsd_matrix =  h5f["rmsd"][:]
        h5f.close()
    except:
        rmsd_matrix=None
    
    if new_dimension is None:
        new_dimension = data    

    quality, medoids = Clustering_data(data=data, reduced_dimension=new_dimension, traj=traj, rmsd_matrix=rmsd_matrix, algorithm=args.algorithm,\
                              n_clusters=n_clusters, connectivity=connect, manual_labels = manual_labels,\
                              title=args.algorithm, path=out)
    
    log = pd.DataFrame(quality, index=[0])
    log.to_csv(out+'evaluation.log', index=False)
    
    energy = Energy(data=new_dimension, weights=args.weights, bins=args.bins, cluster_centers=medoids, rangevals=args.rangevals, path=out)    
    dat0, CS, minima_frames, global_minimum = energy.plot_map(title=args.out_name)
    
    print("The medois are: ", medoids)
    print("The minimum of energy is: ", global_minimum[0])
    print("The minima of energy is: ", minima_frames)
    
    energy.plot_map3d(title=args.out_name)

if __name__ == "__main__":
    
    main()
    
    exit()
