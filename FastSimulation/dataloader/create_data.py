import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
import torch

def unscale(x,max_,min_):
    return x*0.5*(max_ - min_) + min_ + (max_-min_)/2


def scale_data(hits,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.0,"time_min":0.0}):
    x = hits[:,0]
    y = hits[:,1]
    time = hits[:,2]

    x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
    y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
    time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0

    return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)


class CherenkovPhotons(Dataset):
    def __init__(self,kaon_path=None,pion_path=None,mode=None,combined=False,inference=False,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0}):
        if mode is None:
            print("Please select one of the following modes:")
            print("1. Pion")
            print("2. Kaon")
            print("3. Combined")
            exit()
        self.inference = inference
        self.combined = combined
        self.stats = stats
        self.kaon_path = kaon_path
        self.pion_path = pion_path
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max'],self.stats['phi_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min'],self.stats['phi_min']])

        if mode == "Kaon":
            columns=["x","y","time","P","theta","phi"]
            if not self.inference:
                print('Using training mode.')
                df = pd.read_csv(kaon_path,sep=',',index_col=None)
                self.data = df[df.time < self.stats['time_max']].to_numpy()
                print('Max Time: ',self.stats['time_max'])
                print(self.data.max(0))
                print(self.data.min(0))

            else:
                print("Using inference mode.")
                df = pd.read_csv(kaon_path,sep=',',index_col=None)
                self.data = df[df.time < self.stats['time_max']].to_numpy()

            self.data = np.concatenate([self.data,np.c_[np.ones_like(self.data[:,0])]],axis=1)

        elif mode == "Pion":
            if not self.inference:
                print('Using training mode.')
                df = pd.read_csv(pion_path,sep=',',index_col=None)
                self.data = df[df.time < self.stats['time_max']].to_numpy()
                print('Max Time: ',self.stats['time_max'])
                print(self.data.max(0))
                print(self.data.min(0))
                print('Max Time: ',self.stats['time_max'])
            else:
                print("Using inference mode.")
                df = pd.read_csv(pion_path,sep=',',index_col=None)
                self.data = df[df.time < self.stats['time_max']].to_numpy()

            self.data = np.concatenate([self.data,np.c_[np.zeros_like(self.data[:,0])]],axis=1)

        elif mode == "Combined":
            pions = pd.read_csv(pion_path,sep=',',index_col=None)
            pions['PID'] = np.zeros(len(pions))
            kaons = pd.read_csv(kaon_path,sep=',',index_col=None)
            kaons['PID'] = np.ones(len(kaons))
            self.data = pd.concat([pions,kaons],axis=0).to_numpy()
            random.shuffle(self.data)
            del pions,kaons
        else:
            print("Error in dataset creation. Exiting")
            exit()

        self.stats = stats

    def __len__(self):
        return len(self.data)

    def scale_data(self,hits,stats):
        x = hits[0]
        y = hits[1]
        time = hits[2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.array([x,y,time])

    def __getitem__(self, idx):
        # Get the sample
        data = self.data[idx]
        hits = data[:3]

        hits = self.scale_data(hits,self.stats)
        conds = data[3:6]

        unscaled_conds = conds.copy()
        metadata = data[6:-1]

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)

        PID = data[-1]

        return hits,conds,PID,metadata,unscaled_conds


class DLL_Dataset(Dataset):
    def __init__(self,file_path,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0},time_cuts=None):
        self.data = np.load(file_path,allow_pickle=True)#[:10000] # Useful for testing
        self.n_photons = 250
        self.stats = stats
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max'],self.stats['phi_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min'],self.stats['phi_min']])
        self.time_cuts = time_cuts
        if self.time_cuts is not None:
            print('Rejecting photons with time > {0}'.format(self.time_cuts))

    def __len__(self):
        return len(self.data)

    def scale_data(self,hits,stats):
        x = hits[:,0]
        y = hits[:,1]
        time = hits[:,2]
        x = 2.0 * (x - stats['x_min'])/(stats['x_max'] - stats['x_min']) - 1.0
        y = 2.0 * (y - stats['y_min'])/(stats['y_max'] - stats['y_min']) - 1.0
        time = 2.0 * (time - stats['time_min'])/(stats['time_max'] - stats['time_min']) - 1.0
        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)

    def __getitem__(self, idx):

        particle = self.data[idx]
        LL_k = particle['LikelihoodKaon']
        LL_pi = particle['LikelihoodPion']
        pmtID = np.array(particle['pmtID'])
        o_box = pmtID//108
        if o_box[0] == 1:
            pmtID -= 108

        pixelID = np.array(particle['pixelID'])

        row = (pmtID//18) * 8 + pixelID//8
        col = (pmtID%18) * 8 + pixelID%8

        time = np.array(particle['leadTime'])

        pos_time = np.where((time > 0) & (time < self.stats['time_max']))[0]
        row = row[pos_time]
        col = col[pos_time]
        time = time[pos_time]

        pmtID = pmtID[pos_time]

        assert len(row) == len(time)

        x = col * 6. + (pmtID % 18) * 2. + 3.
        y = row * 6. + (pmtID // 18) * 2. + 3.

        hits = np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)
        conds = np.array([particle['P'],particle['Theta'],particle['Phi']])

        if self.time_cuts is not None:
            idx_ = np.where(hits[:,2] < self.time_cuts)[0]
            hits = hits[idx_]

        PID = np.array(particle['PDG'])
        n_hits = len(hits)

        hits = self.scale_data(hits,self.stats)

        conds = conds.reshape(1,-1).repeat(len(hits[:,0]),0)
        unscaled_conds = conds.copy()

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)


        if len(hits) > self.n_photons:
            #usually argsort in time
            hits = hits[np.argsort(time)]
            hits = hits[:self.n_photons]
            conds = conds[:self.n_photons]
            unscaled_conds = unscaled_conds[:self.n_photons]
            time = time[np.argsort(time)]
            time = time[:self.n_photons]

        elif len(hits) < self.n_photons:
            n_needed = self.n_photons - len(hits)
            hits = np.pad(hits,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
            conds = np.pad(conds,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
            unscaled_conds = np.pad(unscaled_conds,((0,n_needed),(0,0)),mode='constant',constant_values=-np.inf)
        else: # Already taken care of
            pass
    
        return hits,conds,PID,n_hits,unscaled_conds,LL_k,LL_pi


def transform(pmtID,pixelID,time):
        o_box = pmtID//108
        if o_box[0] == 1:
            pmtID -= 108

        pixelID = np.array(particle['pixelID'])

        row = (pmtID//18) * 8 + pixelID//8
        col = (pmtID%18) * 8 + pixelID%8
        pmtID = pmtID[pos_time]

        assert len(row) == len(time)

        x = col * 6. + (pmtID % 18) * 2. + 3.
        y = row * 6. + (pmtID // 18) * 2. + 3.

        return np.concatenate([np.c_[x],np.c_[y],np.c_[time]],axis=1)