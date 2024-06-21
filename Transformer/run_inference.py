import os
import json
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataloader.dataloader import InferenceLoader
from dataloader.dataset import DIRC_Dataset
from pickle import dump
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import pandas as pd
from pickle import load
from models.swin_classifier import Classifier
import torch.nn.functional as F
from torch.utils.data import Subset
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from models.lydia_model import CNN
import time
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.nn.parallel import DataParallel
import pickle
import scipy.integrate

def compute_efficiency_rejection_DLL(delta_log_likelihood, true_labels):
    thresholds = np.linspace(-4000.0, 4000.0, 20000)
    thresholds_broadcasted = np.expand_dims(thresholds, axis=1)
    predicted_labels = delta_log_likelihood > thresholds_broadcasted
    TP = np.sum((predicted_labels == 1) & (true_labels == 1), axis=1)
    FP = np.sum((predicted_labels == 1) & (true_labels == 0), axis=1)
    TN = np.sum((predicted_labels == 0) & (true_labels == 0), axis=1)
    FN = np.sum((predicted_labels == 0) & (true_labels == 1), axis=1)

    efficiencies = TP / (TP + FN)  # Efficiency (True Positive Rate)
    rejections = TN / (TN + FP)  # Rejection (True Negative Rate)
    auc = np.trapz(y=np.flip(rejections),x=np.flip(efficiencies))

    return efficiencies,rejections,auc


def compute_efficiency_rejection_prob(probs, true_labels):
    thresholds = np.linspace(0.0, 1.0, 10000)
    thresholds_broadcasted = np.expand_dims(thresholds, axis=1)
    predicted_labels = probs > thresholds_broadcasted
    TP = np.sum((predicted_labels == 1) & (true_labels == 1), axis=1)
    FP = np.sum((predicted_labels == 1) & (true_labels == 0), axis=1)
    TN = np.sum((predicted_labels == 0) & (true_labels == 0), axis=1)
    FN = np.sum((predicted_labels == 0) & (true_labels == 1), axis=1)

    efficiencies = TP / (TP + FN)  # Efficiency (True Positive Rate)
    rejections = TN / (TN + FP)  # Rejection (True Negative Rate)
    auc = np.trapz(y=np.flip(rejections),x=np.flip(efficiencies))

    return efficiencies,rejections,auc


def main(config,trained_as_distributed):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    if not os.path.exists("Figures"):
        os.makedirs("Figures")

    fig_dir = config['Inference']['fig_dir']
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        print("Figures can be found in ",fig_dir)

    # Load the dataset
    print('Creating Loaders.')
    test_dataset = DIRC_Dataset(kaon_path=config['dataset']['testing']['kaon_data_path'],pion_path=config['dataset']['testing']['pion_data_path'],stats=config['stats'],nf_generated=bool(config['nf_generated']))
    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = True
    test_loader = InferenceLoader(test_dataset,config)
    print("Testing Size: {0}".format(len(test_loader.dataset)))

    patch_size = config['model']['patch_size']
    channels = config['model']['channels']
    embed_dim = config['model']['embed_dim']
    drop_rates = config['model']['drop_rates']
    num_heads = config['model']['num_heads']
    depths = config['model']['depths']
    window_size = config['model']['window_size']

    # Use single GPU but model is saved as DataParallel type 
    net = Classifier(patch_size=patch_size,in_chans=channels,embed_dim=embed_dim,drop_rates=drop_rates,num_heads=num_heads,depths=depths,window_size=window_size)
    if trained_as_distributed:
        net = DataParallel(net)

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')
    model_path = config['Inference']['model_path']
    dicte = torch.load(model_path)
    net.load_state_dict(dicte['net_state_dict'])

    kbar = pkbar.Kbar(target=len(test_loader), width=20, always_stateful=False)
    net.eval()
    predictions = []
    truth = []
    conditions = []
    dll_geom = []
    start = time.time()
    tts = []
    for i, data in enumerate(test_loader):
        x  = data[0].to('cuda').float()
        y  = data[1].to('cuda').float()
        uc = data[3].numpy()
        truth.append(data[1].numpy())
        conditions.append(uc)
        c  = data[2].to('cuda').float()
        dll_geom.append(data[4].numpy())

        with torch.set_grad_enabled(False):
            tt = time.time()
            p = net(x,c)
            tts.append((time.time() - tt)/len(p))
        predictions.append(p.detach().cpu().numpy())
        kbar.update(i)
        
    print(" ")
    print("Average GPU time: ",np.average(tts))

    end = time.time()

    print(" ")
    print("Elapsed Time: ", end - start)

    predictions = np.concatenate(predictions).astype('float32')
    truth = np.concatenate(truth).astype('float32')
    dll_geom = np.concatenate(dll_geom).astype('float32')
    print("Is NaN" ,np.isnan(dll_geom))
    print(dll_geom.max(),dll_geom.min())
    conditions = np.concatenate(conditions)
    print("Time / event: ",(end - start) / len(predictions))
    print(" ")
    sim_type = config['sim_type']

    if sim_type == 'decays':
        idx_ = np.where((conditions[:,0] > 2.0) & (conditions[:,0] < 8.0))
        predictions = predictions[idx_]
        truth = truth[idx_]
        dll_geom = dll_geom[idx_]
        conditions = conditions[idx_]


    efficiencies, rejections,auc = compute_efficiency_rejection_prob(predictions, truth)
    efficiencies_geom, rejections_geom, auc_geom = compute_efficiency_rejection_DLL(dll_geom, truth)

    #to_dump = {"efficiencies":efficiencies,"rejections":rejections,"auc":auc}
    #with open("swin_results.pkl","wb") as file:
    #    pickle.dump(to_dump,file)

    #to_dump = {"efficiencies":efficiencies_geom,"rejections":rejections_geom,"auc":auc_geom}
    #with open("geom_results.pkl","wb") as file:
    #    pickle.dump(to_dump,file)
    # ROC Curve
    #swin = np.load("swin_results_2x_NF.pkl",allow_pickle=True)
    fig = plt.figure()
    plt.plot(rejections_geom,efficiencies_geom, color='blue', lw=2, label=r'Classical Method. Area = {0:.3f}'.format(auc_geom))
    plt.plot(rejections,efficiencies,color='red', lw=2, label=r'Swin. Area = {0:.3f}'.format(auc))
    #plt.plot(swin['rejections'],swin['efficiencies'],color='k',label=r'Swin$._{2x \; Fast Sim}$'+ ' Area = {0:.3f}'.format(swin['auc']))
    plt.plot([0, 1], [1, 0], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(r'kaon efficiency',fontsize=25)
    plt.ylabel(r'pion rejection',fontsize=25) 
    plt.legend(loc="lower left",fontsize=14)
    plt.ylim(0,1)
    plt.xticks(fontsize=18)  # adjust fontsize as needed
    plt.yticks(fontsize=18)  # adjust fontsize as needed
    out_path = os.path.join(fig_dir,"ROC_Swin.pdf")
    plt.savefig(out_path,bbox_inches='tight')
    plt.close(fig)

    acc = accuracy_score(predictions.round(),truth)
    unc_acc = np.sqrt(acc*(1.0 - acc)/ len(predictions))

    print("Accuracy: ",acc,"+-",unc_acc)
    print("AUC: ",auc)

    # ROC as a function of momentum
    
    if sim_type == 'pgun':
        mom_ranges = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5]
    elif sim_type == 'decays':
        mom_ranges = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
    else:
        print("")
        print("Please ensure sim_type is correctly set in the config file.")
        print("1. pgun")
        print("2. decays")
        exit()

    centers = [mr+0.25 for mr in mom_ranges[:-1]]
    aucs = []
    aucs_upper = []
    aucs_lower = []
    aucs_geom = []
    aucs_geom_upper = []
    aucs_geom_lower = []
    lengths = []
    n_kaons = []
    n_pions = []
    for i in range(len(mom_ranges) - 1):
        mom_low = mom_ranges[i]
        mom_high = mom_ranges[i+1]
        idx = np.where((conditions[:,0] > mom_low) & (conditions[:,0] < mom_high))[0]
        p = predictions[idx]
        p_geom = dll_geom[idx]
        t = truth[idx]
        print("Momentum Range: ",mom_low,"-",mom_high)
        print("# Kaons: ",len(t[t==1]))
        n_kaons.append(len(t[t==1]))
        n_pions.append(len(t[t==0]))
        print("# Pions: ",len(t[t==0]))
        lengths.append(len(p))
        eff,rej,_ = compute_efficiency_rejection_prob(p,t)#roc_curve(t,p)
        eff_geom,rej_geom,_= compute_efficiency_rejection_DLL(p_geom,t)#roc_curve(t_geom,p_geom)
        AUC = []
        AUC_geom = []
        sigma_eff = np.sqrt(eff * (1.0 - eff) / len(t[t == 1]))
        sigma_rej = np.sqrt(rej * (1.0 - rej) / len(t[t == 0]))
        sigma_eff_geom = np.sqrt(eff_geom * (1.0 - eff_geom) / len(t[t == 1]))
        sigma_rej_geom = np.sqrt(rej_geom * (1.0 - rej_geom) / len(t[t == 0]))
        #print('FPR: ',fpr,'+-',sigma_fpr, " TPR: ",tpr,"+-",sigma_tpr)

        for _ in range(1000):
            eff_ = np.random.normal(eff,sigma_eff)
            rej_ = np.random.normal(rej,sigma_rej)
            eff_geom_ = np.random.normal(eff_geom,sigma_eff_geom)
            rej_geom_ = np.random.normal(rej_geom,sigma_rej_geom)

            AUC.append(np.trapz(y=np.flip(rej_),x=np.flip(eff_)))
            AUC_geom.append(np.trapz(y=np.flip(rej_geom_),x=np.flip(eff_geom_)))


        aucs.append(np.mean(AUC))
        aucs_geom.append(np.mean(AUC_geom))

        aucs_upper.append(np.percentile(AUC,97.5))
        aucs_lower.append(np.percentile(AUC,2.5))

        aucs_geom_upper.append(np.percentile(AUC_geom,97.5))
        aucs_geom_lower.append(np.percentile(AUC_geom,2.5))
        print("Swin. -> Mean AUC: ",np.mean(AUC)," 95%",np.percentile(AUC,2.5),"-",np.percentile(AUC,97.5))
        print("Geom. -> Mean AUC: ",np.mean(AUC_geom)," 95%",np.percentile(AUC_geom,2.5),"-",np.percentile(AUC_geom,97.5))

    fig = plt.figure(figsize=(10,10))
    to_dump = {"aucs":aucs,"uppers":aucs_upper,"lowers":aucs_lower}
    #with open("auc_func_p_swin.pkl","wb") as file:
    #    pickle.dump(to_dump,file)

    to_dump = {"aucs":aucs_geom,"uppers":aucs_geom_upper,"lowers":aucs_geom_lower}
    #with open("auc_func_p_geom.pkl","wb") as file:
    #    pickle.dump(to_dump,file)

    #swin = np.load("auc_func_p_swin_2x_NF.pkl",allow_pickle=True)
    plt.errorbar(centers,aucs_geom,yerr=[np.array(aucs_geom) - np.array(aucs_geom_lower),np.array(aucs_geom_upper) - np.array(aucs_geom)],label=r"$AUC_{Classical.}$",color='blue',marker='o',capsize=5)
    plt.errorbar(centers,aucs,yerr=[np.array(aucs) - np.array(aucs_lower),np.array(aucs_upper) - np.array(aucs)],label=r"$AUC_{Swin.}$",color='red',marker='o',capsize=5)
    #plt.errorbar(centers,swin['aucs'],yerr=[np.array(swin['aucs']) - np.array(swin['lowers']),np.array(swin['uppers']) - np.array(swin['aucs'])],label=r"$AUC_{Swin._{2x \; Fast Sim.}}$",color='k',marker='o',capsize=5)
    #plt.legend(loc=(0.602,0.67),fontsize=20)
    plt.legend(loc=(0.654,0.74),fontsize=20)
    plt.xlabel("momentum [GeV/c]",fontsize=30,labelpad=10)
    plt.ylabel("AUC",fontsize=30,labelpad=10)
    plt.xticks(fontsize=22)  # adjust fontsize as needed
    plt.yticks(fontsize=22)  # adjust fontsize as needed
    plt.title("AUC as function of momentum",fontsize=32)
    if np.min(aucs) < np.min(aucs_geom):
        min_aucs = np.min(aucs)
    else:
        min_aucs = np.min(aucs_geom)
    if np.max(aucs) > np.max(aucs_geom):
        max_aucs = np.max(aucs)
    else:
        max_aucs = np.max(aucs_geom)

    plt.ylim(min_aucs - 0.05,max_aucs + 0.05)

    ax2 = plt.twinx()

    # Plot bars for pions and kaons
    ax2.bar(np.array(centers) - 0.1, n_pions, width=0.2, label='Pions', color='blue', alpha=0.25)
    ax2.bar(np.array(centers) + 0.1, n_kaons, width=0.2, label='Kaons', color='green', alpha=0.25)
    ax2.set_ylabel('Counts', fontsize=30,labelpad=10)
    ax2.tick_params(axis='y', labelsize=20)
    ax2.legend(loc='upper right', fontsize=20)
    out_path = os.path.join(fig_dir,"Swin_AUC_func_P.pdf")
    plt.savefig(out_path,bbox_inches='tight')
    plt.close()

if __name__=='__main__':
	# PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Swin Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-d', '--trained_as_distributed', default=1, type=int,
                        help='Trainined on multiple GPUs.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,bool(args.trained_as_distributed))
