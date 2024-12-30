# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 2017

@author: carius
"""

"""
Class energy
"""

import warnings; warnings.simplefilter('ignore')

import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
plt.style.use(['seaborn-paper'])
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set_context('paper')
sns.set_color_codes()

class Energy(object):
    
    def __init__(self, data=None, weights=None, bins=50, title=None, cluster_centers=None, density=None, rangevals=None, level_curves=1, labels=None, path='./'):
        
        self.data = data
        self.weights=weights
        self.bins=bins
        self.cluster_centers = cluster_centers
        self.path=path
        self.rangevals=rangevals
        self.level_curves = level_curves
        
    def plot_map(self, title=None):
        """
        Mudar esta função para melhor gerar os gráficos de energia
        """
    
        font = FontProperties()
        font.set_weight('bold')
        font.set_size('large')    
        
        Y1 = self.data[:, 0]
        Xmin = np.min(Y1)
        Xmax = np.max(Y1)
        Y2 = self.data[:, 1]
        Ymin = np.min(Y2)
        Ymax = np.max(Y2)

        if self.rangevals is None:
            self.rangevals=[[Xmin*1.15, Xmax*1.15],[Ymin*1.15, Ymax*1.15]]
        # figure
        
        plt.figure(figsize=(14, 8))
        
        # histogram data

        z,x,y = np.histogram2d(Y1, Y2, bins=self.bins, range=np.asarray(self.rangevals), weights=self.weights)
        z += 0.1

        # compute free energies

        F = -np.log(z)
        F = (F - F.min())/(F.max() - F.min())

        # contour plot
     
        extent = [x[0], x[-1], y[0], y[-1]]
        
        CS = plt.contourf(F.T, 100, cmap=plt.cm.nipy_spectral, extent=extent)
        cbar = plt.colorbar(CS)
        cbar.ax.set_ylabel('free energy (kT)')
        
        
        dat0= CS.allsegs[:self.level_curves]
    
        self.minima_frames = {}
        
        for i in xrange(len(dat0)):
            for j in xrange (len(dat0[i])):
                
                minima= metrics.pairwise_distances_argmin_min(dat0[i][j], self.data[:,:2])
    
                m = np.where(minima[1]==min(minima[1]))[0][0]
                mf = minima[0][m]
    
                me=minima[1][m]
    
                #print(mf, me)
                plt.scatter(self.data[mf, 0], self.data[mf, 1], marker='o', c="red", alpha=1., s=400, edgecolor='k')
                plt.scatter(self.data[mf, 0], self.data[mf, 1], marker='$%d$' % mf, alpha=1., s=200, edgecolor='k', linewidths=1.)
                
                self.minima_frames[mf] = me
                

        m = np.where(self.minima_frames.values() == min(self.minima_frames.values()))[0][0]
        self.global_minimum = [self.minima_frames.keys()[m], self.minima_frames.values()[m]]
        
        if self.cluster_centers is not None:
    
            for m in self.cluster_centers:
                #plt.scatter(data[m, 0], data[m, 1], marker='o', s=20, linewidths=3., color='white', zorder=10)
                plt.scatter(self.data[m, 0], self.data[m, 1], marker='o', c="white", alpha=1., s=400, edgecolor='k')
                plt.scatter(self.data[m, 0], self.data[m, 1], marker='$%d$' % m, alpha=1., s=200, edgecolor='k', linewidths=1.)
                #plt.scatter(data[m, 0], data[m, 1], marker='o', s=30, linewidths=3., color='white', zorder=10)
        #plt.annotate(str(m), (data[m, 0], data[m, 1]), color='black', fontproperties=font)
        
        plt.xlim(x[0], x[-1])
        plt.ylim(y[0], y[-1])
    
        plt.xlabel('Component1'); plt.ylabel('Component2')        
    
        if title is not None:
            plt.title(title)
            fp=self.path+title+'-ENERGY-2D.eps'
        else:
            fp=self.path+'ENERGY-2D.eps'
    
        plt.savefig(fp, bbox_inches='tight', dpi=300, format="eps")
        #plt.show()
        plt.close('all')
        
        return dat0, CS, self.minima_frames, self.global_minimum
    
    
    def plot_map3d(self, title):
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        h, xedges, yedges = np.histogram2d(self.data[:,0],self.data[:,1], bins=self.bins)
    
        h += 0.1
        hist = -np.log(h)
    
        hist = (hist - hist.min())/(hist.max() - hist.min())
    
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        dz = hist.ravel()
        
        surf = ax.plot_trisurf(xpos, ypos, dz, cmap=plt.cm.nipy_spectral, linewidth=1, alpha=1, antialiased=True, shade=True)
        ax.grid(False)
        
        cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.ax.set_ylabel('free energy (kT)')
    
        me = self.global_minimum[1]
    
        m=self.global_minimum[0]
        
        ax.scatter(self.data[m, 0], self.data[m, 1], zs=me, zdir='z', marker='o', s=20, linewidths=3., color='black', zorder=10)
        ax.text(self.data[m, 0], self.data[m, 1], me, str(m).split('-')[0], fontsize=10)
        
        ax.set_xlabel('Component1'); ax.set_ylabel('Component2'); ax.set_zlabel("Energy")
        
        if title is not None:
            fp=self.path+title+'-ENERGY-3D.eps'
        else:
            fp=self.path+'ENERGY-3D.eps'
        plt.savefig(fp, bbox_inches='tight', dpi=300, format="eps")
    
        #plt.show()        
        
        plt.close()