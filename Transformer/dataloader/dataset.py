from torch.utils.data import Dataset
import numpy as np
import os
import torch
import random
import collections

class DIRC_Dataset(Dataset):
    def __init__(self,kaon_path,pion_path,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":500.00,"time_min":0.0},nf_generated=False):
        kaons = np.load(kaon_path,allow_pickle=True)#[:10000]
        pions = np.load(pion_path,allow_pickle=True)#[:10000]
        self.nf_generated = nf_generated
        self.time_cut = stats['time_max']
        print("Max Time: ",self.time_cut)

        print(" ")
        print("Number of Kaons: ",len(kaons))
        print("Number of Pions: ",len(pions))
        print("Total: ",len(kaons) + len(pions))
        print(" ")
        data = pions + kaons
        random.shuffle(data)
        self.data = data
        self.stats = stats
        self.conditional_maxes = np.array([self.stats['P_max'],self.stats['theta_max'],self.stats['phi_max']])
        self.conditional_mins = np.array([self.stats['P_min'],self.stats['theta_min'],self.stats['phi_min']])
        

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
        pmtID = np.array(particle['pmtID'])
        time = np.array(particle['leadTime'])
        if not self.nf_generated:
           dll = np.array(particle['LikelihoodKaon']) - np.array(particle['LikelihoodPion'])
            
        else:
           dll = np.array([0.0])

        o_box = pmtID//108
        if o_box[0] == 1:
            pmtID -= 108

        if not self.nf_generated:
            pixelID = np.array(particle['pixelID'])
            row = (pmtID//18) * 8 + pixelID//8
            col = (pmtID%18) * 8 + pixelID%8
            
            x = col * 6. + (pmtID % 18) * 2. + 3.
            y = row * 6. + (pmtID // 18) * 2. + 3.

            row_col = np.concatenate((np.c_[col],np.c_[row],np.c_[time]),axis=1)
            hits = np.concatenate((np.c_[x],np.c_[y],np.c_[time]),axis=1)
            assert row_col.shape == hits.shape
            col = row_col[:,0].astype('int')
            row = row_col[:,1].astype('int')
            time = row_col[:,2]

        else:
            row = particle['row'].astype('int')
            col = particle['column'].astype('int')

        pos_time = np.where((time > 0) & (time < self.time_cut))[0]
        time = time[pos_time]
        pmtID = pmtID[pos_time]
        row = row[pos_time]
        col = col[pos_time]

        sorted_indices = np.argsort(time)

        # Reverse the sorted indices to process the earliest times first
        # Latest times will be inserted first, and overwritten with earliest times - FIFO
        reversed_sorted_indices = sorted_indices[::-1]
        sorted_row = row[reversed_sorted_indices]
        sorted_col = col[reversed_sorted_indices]
        sorted_time = time[reversed_sorted_indices]

        sorted_time = (sorted_time - self.stats['time_min'])/(self.stats['time_max'] - self.stats['time_min'])

        assert len(row) == len(time)
        assert len(col) == len(time)
        assert len(pmtID) == len(time)

        conds = np.array([particle['P'],particle['Theta'],particle['Phi']])
        unscaled_conds = conds.copy()
        if not self.nf_generated:
            PID = np.array(particle['PDG'])
        else:
            PID = np.array(particle['PDG'])

        conds = (conds - self.conditional_maxes) / (self.conditional_maxes - self.conditional_mins)

        a = np.zeros((1,48,144))
        a[0,sorted_row,sorted_col] = 1.0
        b = np.zeros((1,48,144))
        b[0,sorted_row,sorted_col] = sorted_time

        optical_box = np.concatenate([a,b],axis=0)

        if abs(PID) == 211: # 211 is Pion from rho decay
            PID = 0
        elif abs(PID) == 321: # 321 is Kaon from phi decay
            PID = 1
        else:
            print("Unknown PID!")


        return optical_box,PID,conds,unscaled_conds,dll



def create_dataset(config):
    rho = np.load(config['dataset']['rho_filename'],allow_pickle=True)
    phi = np.load(config['dataset']['phi_filename'],allow_pickle=True)

    data = rho + phi

    random.shuffle(data)

    return DIRC_Dataset(data)
