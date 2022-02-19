import os
import numpy as np

import torch
from torch.utils.data import Dataset

from laserscan import LaserScan, SemLaserScan


class WADS(Dataset):
    def __init__(self, root, do_projection=False):
        self.root = root
        self.scans = []
        self.labels = []
        self.do_projection = do_projection
        
        # Expand the dataset directory
        for folder in os.walk(self.root):
            # Add .label files to labels list
            if "labels" in folder[0]:
                for file in sorted(folder[-1]):
                    self.labels.append(os.path.join(folder[0], file))
            # Add .bin files to scans list
            elif "velodyne" in folder[0]:
                for file in sorted(folder[-1]):
                    self.scans.append(os.path.join(folder[0], file))
                
    def __getitem__(self, idx):
        if self.do_projection:
            laser_scan = LaserScan(project=True)
            # open and project scan at index
            laser_scan.open_scan(self.scans[idx])
            proj_xyz = torch.tensor(laser_scan.proj_xyz).permute(2, 0, 1)
            proj_range = torch.tensor(np.expand_dims(laser_scan.proj_range, axis=0))
            proj_remission = torch.tensor(np.expand_dims(laser_scan.proj_remission, axis=0))
            
            label = torch.tensor(np.fromfile(self.labels[idx], dtype=np.int16).reshape(-1,2))
            
            return proj_xyz, proj_range, proj_remission, label
            
        else:
            scan = torch.tensor(np.fromfile(self.scans[idx], dtype=np.float32).reshape(-1,4))
            label = torch.tensor(np.fromfile(self.labels[idx], dtype=np.int16).reshape(-1,2))
            return scan, label
        
    def __len__(self):
        return len(self.labels)