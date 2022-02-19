import torch
from torch.utils.data import DataLoader

class WADSLoader(DataLoader):
    def __init__(self, **hparams):
        self.hparams = hparams
        
    def __init__(self, **hparams):
        self.hparams = hparams
        
    def collate_fn(self, samples):
        proj_xyz = torch.stack([sample[0] for sample in samples])
        proj_range = torch.stack([sample[1] for sample in samples])
        proj_remission = torch.stack([sample[2] for sample in samples])
        # label = torch.stack([sample[3] for sample in samples])
        # label sizes are different. Using semlasaerscan?!
        return proj_xyz, proj_range, proj_remission
        
    def train_dataloader(self):
        ds = self.hparams['train_ds']
        dataloader = DataLoader(ds,
                                batch_size=self.hparams['batch_size'],
                                shuffle=True,
                                num_workers=self.hparams['num_workers'],
                                collate_fn = self.collate_fn,
                                drop_last=True
                               )
        return dataloader
    
    def validation_dataloader(self):
        ds = self.hparams['valid_ds']
        dataloader = DataLoader(ds,
                                batch_size=self.hparams['batch_size'],
                                shuffle=True,
                                num_workers=self.hparams['num_workers'],
                                collate_fn = self.collate_fn,
                                drop_last=True
                               )
        return dataloader
    
    def test_dataloader(self):
        ds = self.hparams['test_ds']
        dataloader = DataLoader(ds,
                                batch_size=self.hparams['batch_size'],
                                shuffle=True,
                                num_workers=self.hparams['num_workers'],
                                collate_fn = self.collate_fn,
                                drop_last=True
                               )
        return dataloader