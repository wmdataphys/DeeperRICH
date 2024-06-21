import torch.nn as nn
from FrEIA.modules.base import InvertibleModule
import warnings
from typing import Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from typing import Union, Iterable, Tuple
import scipy


class FreiaNet(nn.Module):
    def __init__(self,input_shape,layers,context_shape,embedding=False,hidden_units=512,num_blocks=2,stats={"x_max": 898,"x_min":0,"y_max":298,"y_min":0,"time_max":380.00,"time_min":0.0,
            "P_max":8.5 ,"P_min":0.95 , "theta_max": 11.63,"theta_min": 0.90,"phi_max": 175.5, "phi_min":-176.0 },device='cuda'):
        super(FreiaNet, self).__init__()
        self.input_shape = input_shape
        self.layers = layers
        self.context_shape = context_shape
        self.embedding = embedding
        self.hidden_units = hidden_units
        self.num_blocks = num_blocks
        self.photons_generated = 0
        self.photons_resampled = 0
        self.device = device

        self._allowed_x = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,   # 0
                                                    53.,  59.,  65., 71.,  77.,  83.,  89.,  95.,  # 1
                                                    103., 109., 115., 121., 127., 133., 139., 145.,# 2
                                                    153., 159., 165., 171., 177., 183., 189., 195.,# 3
                                                    203., 209., 215., 221., 227., 233., 239., 245.,# 4 
                                                    253., 259., 265., 271.,277., 283., 289.,  295.,# 5
                                                    303., 309., 315., 321., 327., 333., 339., 345.,# 6
                                                    353., 359., 365., 371., 377., 383., 389., 395.,# 7 
                                                    403., 409., 415., 421., 427., 433., 439., 445.,# 8
                                                    453., 459., 465., 471., 477., 483., 489., 495.,# 9 
                                                    503., 509., 515., 521., 527., 533., 539., 545.,# 10
                                                    553., 559., 565., 571., 577., 583., 589., 595.,# 11
                                                    603., 609., 615., 621., 627., 633., 639., 645.,# 12 
                                                    653., 659., 665., 671., 677., 683., 689., 695.,# 13
                                                    703., 709., 715., 721., 727., 733., 739., 745.,# 14 
                                                    753., 759., 765., 771., 777., 783., 789., 795.,# 15 
                                                    803., 809., 815., 821., 827., 833., 839., 845.,# 16
                                                    853., 859., 865., 871., 877., 883., 889., 895.])).to(self.device) # 17
        self._allowed_y = torch.tensor(np.array([  3.,   9.,  15.,  21.,  27.,  33.,  39.,  45.,  # 0
                                                   53.,  59.,  65.,71.,  77.,  83.,  89.,  95.,   # 1
                                                   103., 109., 115., 121., 127., 133.,139., 145., # 2
                                                   153., 159., 165., 171., 177., 183., 189., 195.,# 3  
                                                   203., 209., 215., 221., 227., 233., 239., 245.,# 4 
                                                   253., 259., 265., 271.,277., 283., 289., 295.])).to(self.device) # 5
        self.stats_ = stats

        if self.embedding:
            self.context_embedding = nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape)])

        context_encoder =  nn.Sequential(*[nn.Linear(context_shape,16),nn.ReLU(),nn.Linear(16,input_shape*2)])

        self.distribution = ConditionalDiagonalNormal(shape=[input_shape],context_encoder=context_encoder)

        def create_freai(input_shape,layer,cond_shape):
            inn = Ff.SequenceINN(input_shape)
            #inn.append(InvertibleTanh)
            for k in range(layers):
                inn.append(Fm.AllInOneBlock,cond=0,cond_shape=(cond_shape,),subnet_constructor=subnet_fc, permute_soft=True)

            return inn

        def block(hidden_units):
            return [nn.Linear(hidden_units,hidden_units),nn.ReLU(),nn.Linear(hidden_units,hidden_units),nn.ReLU()]

        def subnet_fc(c_in, c_out):
            blks = [nn.Linear(c_in,hidden_units)]
            for _ in range(num_blocks):
                blks += block(hidden_units)

            blks += [nn.Linear(hidden_units,c_out)]
            return nn.Sequential(*blks)

        self.sequence = create_freai(self.input_shape,self.layers,self.context_shape)

    def log_prob(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise,logabsdet = self.sequence.forward(inputs,rev=False,c=[embedded_context])
        log_prob = self.distribution.log_prob(noise,context=embedded_context)

        return log_prob + logabsdet

    def __sample(self,num_samples,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context

        noise = self.distribution.sample(num_samples,context=embedded_context)

        if embedded_context is not None:
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self.sequence.forward(noise,rev=True,c=[embedded_context])

        return samples

    def unscale(self,x,max_,min_):
        return x*0.5*(max_ - min_) + min_ + (max_-min_)/2

    def unscale_conditions(self,x,max_,min_):
        return x * (max_ - min_) + max_

    def set_to_closest(self, x, allowed):
        x = x.unsqueeze(1)  # Adding a dimension to x for broadcasting
        diffs = torch.abs(x - allowed.to(self.device).float())
        closest_indices = torch.argmin(diffs, dim=1)
        closest_values = allowed[closest_indices]
        return closest_values
            
    def _sample(self,num_samples,context):
        samples = self.__sample(num_samples,context)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min'])
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min'])
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        x = self.set_to_closest(x,self._allowed_x)
        y = self.set_to_closest(y,self._allowed_y)
        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1).detach().cpu().numpy()

    def __get_track(self,num_samples,context):
        samples = self.__sample(num_samples,context)
        x = self.unscale(samples[:,0].flatten(),self.stats_['x_max'],self.stats_['x_min']).round()
        y = self.unscale(samples[:,1].flatten(),self.stats_['y_max'],self.stats_['y_min']).round()
        t = self.unscale(samples[:,2].flatten(),self.stats_['time_max'],self.stats_['time_min'])

        return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def _apply_mask(self, hits):
        # Time > 0 and < maximum time seen
        mask = torch.where((hits[:,2] > 0) & (hits[:,2] < self.stats_['time_max']))
        hits = hits[mask]
        # Outter bounds
        mask = torch.where((hits[:, 0] > 0) & (hits[:, 0] < 898) & (hits[:, 1] > 0) & (hits[:, 1] < 298))[0] # Acceptance mask
        hits = hits[mask]
        # PMTs OFF
        top_row_mask = torch.where(~((hits[:, 1] > 249) & (hits[:, 0] < 350)))[0] # rejection mask (keep everything not identified)
        hits = hits[top_row_mask]
        # PMTs OFF
        bottom_row_mask = torch.where(~((hits[:, 1] < 50) & (hits[:, 0] < 550)))[0] # rejection mask (keep everything not identified)
        hits = hits[bottom_row_mask]

        return hits

    def create_tracks(self,num_samples,context,plotting=False):
        hits = self.__get_track(num_samples,context)
        updated_hits = self._apply_mask(hits)
        n_resample = int(num_samples - len(updated_hits))
        

        self.photons_generated += len(hits)
        self.photons_resampled += n_resample
        while n_resample != 0:
            resampled_hits = self.__get_track(n_resample,context)
            updated_hits = torch.concat((updated_hits,resampled_hits),0)
            updated_hits = self._apply_mask(updated_hits)
            n_resample = int(num_samples - len(updated_hits))
            self.photons_resampled += n_resample
            self.photons_generated += len(resampled_hits)
            

        x = self.set_to_closest(updated_hits[:,0],self._allowed_x).detach().cpu()
        y = self.set_to_closest(updated_hits[:,1],self._allowed_y).detach().cpu()
        t = updated_hits[:,2].detach().cpu()

        pmtID = torch.div(x,torch.tensor(50,dtype=torch.int),rounding_mode='floor') + torch.div(y, torch.tensor(50,dtype=torch.int),rounding_mode='floor') * 18
        row = (1.0/6.0) * ( y - 3 - 2* torch.div(pmtID,torch.tensor(18,dtype=torch.int),rounding_mode='floor'))
        col = (1.0/6.0) * ( x - 3 - 2*(pmtID % 18))

        assert(len(row) == num_samples)
        assert(len(col) == num_samples)
        assert(len(pmtID) == num_samples)

        P = self.unscale_conditions(context[0][0].detach().cpu().numpy(),self.stats_['P_max'],self.stats_['P_min'])
        Theta = self.unscale_conditions(context[0][1].detach().cpu().numpy(),self.stats_['theta_max'],self.stats_['theta_min'])
        Phi = self.unscale_conditions(context[0][2].detach().cpu().numpy(),self.stats_['phi_max'],self.stats_['phi_min'])

        if not plotting:
            return {"NHits":num_samples,"P":P,"Theta":Theta,"Phi":Phi,"row":row.numpy(),"column":col.numpy(),"leadTime":t.numpy(),"pmtID":pmtID.numpy()}
        else:
            return torch.concat((x.unsqueeze(1),y.unsqueeze(1),t.unsqueeze(1)),1)

    def to_noise(self,inputs,context):
        if self.embedding:
            embedded_context = self.context_embedding(context)
        else:
            embedded_context = context
        noise,_ = self.sequence.forward(inputs,rev=False,c=[embedded_context])

        return noise

    def sample_and_log_prob(self,num_samples,context):
        if self.embedding:
            embedded_context = self._embedding_net(context)
        else:
            embedded_context = context

        noise, log_prob = self.distribution.sample_and_log_prob(
            num_samples, context=embedded_context
        )
 
        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )
        samples, logabsdet = self.sequence.forward(noise,rev=True,c=[embedded_context])

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

