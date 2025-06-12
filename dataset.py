import pandas as pd
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np

class DartsDataset(Dataset):
    def __init__(self, dirs, bbox_size=10):
        """
        Args:
            root_dir (str): Root directory containing image folders.
            labels_filename (str): Path to the pickle file with labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        dfs = [pd.read_pickle(os.path.join(d, 'labels.pkl')) for d in dirs]
        for i, d in enumerate(dirs):
            df = dfs[i]
            df['filename'] = df['filename'].apply(lambda x: d + '/' + x)
        self.labels = pd.concat(dfs, ignore_index=True)
        self.bbox_size = bbox_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename, locations, num_darts = self.labels.iloc[idx][['filename', 'locations', 'num_darts']]
        image = Image.open(filename)
        width, height = image.size
        bbox_size = self.bbox_size / max(width, height) * 768  
        bbox_width = bbox_size * 2
            
        locations = torch.tensor(locations, dtype=torch.float32)
        locations[:, 1] = 1 - locations[:, 1]  # Convert y-coordinates to match image coordinates (0 at top)
        # 1. Random Rotation
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)

        # Rotate dart positions around the center of the image
        angle_rad = torch.deg2rad(torch.tensor(angle))
        center = torch.tensor([0.5, 0.5])

        R = torch.tensor([[torch.cos(-angle_rad), -torch.sin(-angle_rad)],
                            [torch.sin(-angle_rad),  torch.cos(-angle_rad)]])

        transformed_locations = torch.mm(locations - center, R.T) + center  # Apply rotation
        assert transformed_locations.shape == (4 + num_darts, 2)
        coords = transformed_locations * torch.tensor([width, height])  # Convert to pixel coordinates
        coords = coords.round()
        
        boxes = torch.cat((coords-bbox_size, torch.ones_like(coords)*bbox_width), dim=1)
        labels = torch.cat((torch.arange(4, dtype=torch.long), torch.ones(num_darts, dtype=torch.long)*4))
        annotations = [{"bbox": boxes[i], "category_id": labels[i], "area": bbox_width * bbox_width} for i in range(boxes.shape[0])]
        targets = {
            "image_id": idx,
            "annotations": annotations,
            "filename": filename,
        }
        return image, targets