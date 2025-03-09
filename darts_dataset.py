import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
import cv2

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
            

        xy = torch.tensor(self.labels.iloc[idx]['xy'], dtype=torch.float32)
        targets = {"corners": xy[:4], "darts": xy[4:]}
        
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
        
        df_unlabeled = pd.read_csv(labels_filename) 
        # Step 1: Calculate the occurrence of each value in 'filename'
        filename_counts = df_unlabeled['filename'].value_counts()

        # Step 2: Identify filenames that occur exactly once
        unique_filenames = filename_counts[filename_counts == 1].index

        # Step 3: Filter the DataFrame to include only rows with unique filenames
        self.filenames = df_unlabeled[df_unlabeled['filename'].isin(unique_filenames)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames.iloc[idx]['filename']
        try: 
            img_path = os.path.join(self.root_dir, filename)
        except:
            print(self.root_dir, filename)
            print(idx, self.filenames.iloc[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    