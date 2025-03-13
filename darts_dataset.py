import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
import cv2
from score import get_score, fields
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, labels_filename, transform=None):
        """
        Args:
            root_dir (str): Root directory containing image folders.
            labels_filename (str): Path to the pickle file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.labels = pd.read_pickle(labels_filename)
        # labels = labels[labels["img_folder"].str.startswith(dataset)]
        # length = len(labels)
        # if train:
        #     self.labels = labels[:length//2]
        # else:
        #     self.labels = labels[length//2:]
        print(self.labels.head())
        self.transform = transform
        
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_folder = self.labels.iloc[idx]['img_folder']
        img_name = self.labels.iloc[idx]['img_name']
        try: 
            img_path = os.path.join(self.root_dir, img_folder, img_name)
        except:
            print(self.root_dir, img_folder, img_name)
            print(idx, self.labels.iloc[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        
        if self.transform:
            image = self.transform(image)
        else: 
            image = transforms.ToTensor()(image)
            
        xy = torch.tensor(self.labels.iloc[idx]['xy'], dtype=torch.float32)
            
        # 1. Random Rotation
        angle = random.uniform(-10, 10)  # Max 15-degree rotation
        image = TF.rotate(image, angle)

        # Rotate dart positions around the center of the image
        angle_rad = torch.deg2rad(torch.tensor(angle))
        center = torch.tensor([0.5, 0.5])

        R = torch.tensor([[torch.cos(-angle_rad), -torch.sin(-angle_rad)],
                            [torch.sin(-angle_rad),  torch.cos(-angle_rad)]])

        transformed_xy = torch.mm(xy - center, R.T) + center  # Apply rotation
        
        points, sectors = get_score(xy[:4], xy[4:])
        
        labels = torch.cat((torch.arange(4, dtype=torch.long), torch.from_numpy(sectors).long() + 4))
        targets = {"points": transformed_xy, "labels": labels, "score": torch.from_numpy(points).int()}
        return image, targets
    
    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)  # Stack images (assuming they're the same size)

        # # Handling variable-sized "darts"
        # batch_targets = {
        #     "corners": torch.stack([t["corners"] for t in targets]),
        #     "darts": [t["darts"] for t in targets]  # Keep as a list
        # }

        return images, targets
    
    
class RealWorldDataset(Dataset):
    def __init__(self, root_dir, labels_filename, transform=None):
        """
        Args:
            root_dir (str): Root directory containing image folders.
            labels_filename (str): Path to the pickle file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.df = pd.read_csv(labels_filename) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dir = self.df.iloc[idx]['img_folder']
        name = self.df.iloc[idx]['img_name']
        try: 
            img_path = os.path.join(self.root_dir, dir, name)
        except:
            print(self.root_dir, dir, name)
            print(idx, self.filenames.iloc[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    