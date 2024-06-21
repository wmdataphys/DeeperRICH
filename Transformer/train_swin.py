import os
import json
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataloader.dataloader import CreateLoaders
from dataloader.dataset import create_dataset
from pickle import dump
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import pandas as pd
from pickle import load
from models.swin_classifier import Classifier
from dataloader.dataset import DIRC_Dataset
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.nn.parallel import DataParallel

def main(config,resume,distributed):

	# Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

	# Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

	# Create directory structure
    output_folder = config['output']['dir']
    os.mkdir(os.path.join(output_folder,exp_name))
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)

    nf_generated = bool(config['nf_generated'])

    if nf_generated:
        print("Using FastSim generations for training.")
    else:
        print("Using traditional input from Monte Carlo / Data.")
	# Load the dataset
    print('Creating Loaders.')
    train_dataset = DIRC_Dataset(kaon_path=config['dataset']['training']['kaon_data_path'],pion_path=config['dataset']['training']['pion_data_path'],stats=config['stats'],nf_generated=nf_generated)
    val_dataset = DIRC_Dataset(kaon_path=config['dataset']['validation']['kaon_data_path'],pion_path=config['dataset']['validation']['pion_data_path'],stats=config['stats'],nf_generated=nf_generated)
    history = {'train_loss':[],'val_loss':[],'lr':[]}
    run_val = True
    train_loader,val_loader = CreateLoaders(train_dataset,val_dataset,config)

    # Model params for Transformer.
    patch_size = config['model']['patch_size']
    channels = config['model']['channels']
    embed_dim = config['model']['embed_dim']
    drop_rates = config['model']['drop_rates']
    num_heads = config['model']['num_heads']
    depths = config['model']['depths']
    window_size = config['model']['window_size']

    if not distributed:
        print("Using single GPU.")
        net = Classifier(patch_size=patch_size,in_chans=channels,embed_dim=embed_dim,drop_rates=drop_rates,num_heads=num_heads,depths=depths,window_size=window_size)
    else:
        print("Using {0} GPUs.".format(torch.cuda.device_count()))
        print(" ")
        net = Classifier(patch_size=patch_size,in_chans=channels,embed_dim=embed_dim,drop_rates=drop_rates,num_heads=num_heads,depths=depths,window_size=window_size)
        net = DataParallel(net)
        #net = DDP(net)
    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ",t_params)
    net.to('cuda')

	# Optimizer
    num_epochs = int(config['num_epochs'])
    lr = float(config['optimizer']['lr'])

    optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=lr)
    num_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
	                                                       eta_min=0)

    startEpoch = 0
    global_step = 0

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)
    else:
        print("========= Starting Training ================:")

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')

	# Train


	# Define your loss function
    loss_fn = nn.BCELoss()

    for epoch in range(startEpoch,num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

		###################
		## Training loop ##
		###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            x  = data[0].to('cuda').float()
            y  = data[1].to('cuda').float()
            c  = data[2].to('cuda').float()

            optimizer.zero_grad()

			# forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                p = net(x,c)

            loss = loss_fn(p,y)
            train_acc = (torch.sum(torch.round(p) == y)).item() / len(y)
            loss.backward()
            optimizer.step()
            scheduler.step()

			# statistics
            running_loss += loss.item() * y.shape[0]

            kbar.update(i, values=[("loss", loss.item()),("Train_Acc",train_acc)])

            global_step += 1

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


		######################
		## validation phase ##
		######################
        if run_val:
            net.eval()
            val_loss = 0.0
            val_acc = 0.0
            for i, data in enumerate(val_loader):
                x  = data[0].to('cuda').float()
                y  = data[1].to('cuda').float()
                c  = data[2].to('cuda').float()
				
                with torch.no_grad():
                    p = net(x,c)

                val_acc += (torch.sum(torch.round(p) == y)).item() / len(y)
                val_loss += loss_fn(p,y)

            val_acc = val_acc/len(val_loader)
            val_loss = val_loss/len(val_loader)

            kbar.add(1, values=[("Val_loss", val_loss.item()),("Val_Acc",val_acc)])

			# Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = config['name']+'_epoch{:02d}_val_loss_{:.6f}_val_acc_{:.6f}.pth'.format(epoch, val_loss,val_acc)

        else:
            kbar.add(1,values=[('val_loss',0.)])
            name_output_file = config['name']+'_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

        filename = os.path.join(output_folder , exp_name , name_output_file)

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')




if __name__=='__main__':
	# PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Swin Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--distributed', default=0, type=int,
	                    help='Training on multiple GPUs.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.resume,bool(args.distributed))
