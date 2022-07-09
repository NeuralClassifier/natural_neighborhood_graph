#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn
from scipy.sparse.csgraph import minimum_spanning_tree
import time
import sklearn
from math import sqrt


# In[35]:


def get_data(df_name):

    
    df_path = 'C:\\Research\\AlbertaResearch\\RankAggregation\\real_life_dfs\\df_to_consider\\'
    df_train = pd.read_csv(df_path+df_name+'_train.csv')
    df_train.drop(['Unnamed: 0'],axis=1,inplace=True)
    df_test = pd.read_csv(df_path+df_name+'_test.csv')
    df_test.drop(['Unnamed: 0'],axis=1,inplace=True)
    df = pd.concat([df_train,df_test],axis=0)
    df.reset_index(drop=True,inplace=True)
    
    return df



class NaturalNeighborhoodG():
    
    def __init__(self, df,max_mpts):
        self.df = df
        self.K = max_mpts #minimum can be 2
        self.kmax_NearestNeighbourG = None
        self.pairwise_distances = None

        self.nearest_neighbors = None
        self.kmax_DistWeightdNearestNeighbourG = None
        self.NaN_Edge = None
        self.r = 1
        self.NaN_Num = {}
        self.NaNE = 0

    
    def k_max_NNG(self):
        dist = {}
        euclidean_pairwise_dist = sklearn.metrics.pairwise.euclidean_distances(self.df.iloc[:,:-1], Y=None)
        self.pairwise_distances = euclidean_pairwise_dist
        kmax_NNG = np.zeros(euclidean_pairwise_dist.shape)
        kmax_NNG_mrd = np.zeros(euclidean_pairwise_dist.shape)
        core_dist = []
        nearest_negh = []

        for can in range(len(euclidean_pairwise_dist)):
            candidate = euclidean_pairwise_dist[can]
            neighbors = sorted(range(len(candidate)), key = lambda sub: candidate[sub])[:self.K]
        
            nearest_negh.append(np.array(neighbors[1:]))
            
            for nn in neighbors:
                if can!=nn:
                    kmax_NNG[can,nn] = 1
                    #kmax_NNG_mrd[can,nn] = self.find_core_distance()
                    
        
        self.nearest_neighbors = np.array(nearest_negh)
        self.kmax_NearestNeighbourG = kmax_NNG
    
        
    def create_dist_weighted_kmax_NNG(self):
        weightedkmax_NNG = np.zeros(self.kmax_NearestNeighbourG.shape)
        for row in range(self.kmax_NearestNeighbourG.shape[0]):
            for col in range(self.kmax_NearestNeighbourG.shape[1]):
                if self.kmax_NearestNeighbourG[row,col] == 1:
                    weightedkmax_NNG[row,col] = max(self.find_core_distance(row,self.K),self.find_core_distance(col,self.K))
        
        self.kmax_DistWeightdNearestNeighbourG = weightedkmax_NNG    
    
        
    def KNN_r(self,candidate):
        return candidate[:self.r]
    
    def count(self):
        no_NaN_count = 0
        for candidate in list(self.NaN_Num.keys()):
            if self.NaN_Num[candidate] == 0:
                no_NaN_count += 1
        
        return no_NaN_count
       
    def repeat(self,init_c,c):
        if init_c == c:
            
            return 1
        
        else:
            return 0
            
    
    def compute_naturalN_g(self):
        start_time = time.time()
        #self.small_k = k
        self.k_max_NNG()
        #self.reverse_k_max_NNG(k)
        self.NaN_Edge = np.zeros(self.pairwise_distances.shape)
        flag=0
        init_count = 0
        rep = 0
        while flag == 0:
            for can in range(self.nearest_neighbors.shape[0]):
                r_neighbors = self.KNN_r(self.nearest_neighbors[can])
                
                natural_neighb_count = 0
                for neighb in r_neighbors:
                    if can in self.nearest_neighbors[neighb][:self.r]:
                        natural_neighb_count +=1

                        if self.NaN_Edge[can,neighb] != 1:
                            self.NaN_Edge[can,neighb] = 1
                            self.NaN_Edge[neighb,can] = 1
                

                
                self.NaN_Num[can] = natural_neighb_count
            
            cnt = self.count()
            rep = rep + self.repeat(init_count,cnt)
            init_count = cnt
            
            if cnt == 0 or rep >= sqrt(self.r-rep):
                flag = 1
            
            self.r +=1
        
        self.NaNE = self.r - 1 
        

        print("--- %s seconds ---" % (time.time() - start_time))
            
        

if __name__ == "__main__":
    
    df_names = ['Glass','Ionosphere','ann_thyroid','arrhythmia','pendigits16']

    for dataset in df_names:   
        
        print("--- Dataset: ",dataset," ---")

        df = get_data(dataset)

        nan_g = NaturalNeighborhoodG(df,120)
        nan_g.compute_naturalN_g()

        print("NaNE: ")
        print()
        print(nan_g.NaNE)
        print()
        #print("NaN_Edge: ")
        #print()
        #print(nan_g.NaN_Edge)
        #print()
        #print("NaN_Num: ")
        #print()
        #print(nan_g.NaN_Num)
        #print()


