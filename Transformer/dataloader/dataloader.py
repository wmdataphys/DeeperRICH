import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import numpy as np
import torch

def DIRC_collate(batch):
    opt_boxes = []
    conditions = []
    PIDs = []
    unscaled_conds = []
    dlls = []
    for opt_box,PID,cond,uc,dll in batch:
        opt_boxes.append(torch.tensor(opt_box))
        conditions.append(torch.tensor(cond))
        PIDs.append(torch.tensor(PID))
        unscaled_conds.append(torch.tensor(uc))
        dlls.append(torch.tensor(dll))

    return torch.stack(opt_boxes),torch.tensor(PIDs),torch.stack(conditions),torch.stack(unscaled_conds),torch.tensor(dlls)

# Create dataloaders to iterate.
def CreateLoaders(train_dataset,val_dataset,config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,collate_fn=DIRC_collate,num_workers=config['dataloader']['train']['num_workers'],
                            pin_memory=True)
    val_loader =  DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,collate_fn=DIRC_collate,num_workers=config['dataloader']['val']['num_workers'],
                            pin_memory=True)

    return train_loader,val_loader


def InferenceLoader(test_dataset,config):
    return DataLoader(test_dataset,
                            batch_size=config['dataloader']['test']['batch_size'],
                            shuffle=False,collate_fn=DIRC_collate,num_workers=0,
                            pin_memory=False)