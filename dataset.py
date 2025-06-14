import pandas as pd
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np

class DartsDataset(Dataset):
    def __init__(self, dirs, bbox_size=13, random_rotation=True, random_rescale=True, resize_to=None):
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
        self.random_rotation = random_rotation
        self.random_rescale = random_rescale
        self.resize_to = resize_to

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        filename, locations, num_darts = self.labels.iloc[idx][['filename', 'locations', 'num_darts']]
        image = Image.open(filename)
        width, height = image.size            
        
            
        locations = torch.tensor(locations, dtype=torch.float32)
        locations[:, 1] = 1 - locations[:, 1]  # Convert y-coordinates to match image coordinates (0 at top)
        pixel_coords = locations * torch.tensor([width, height])  # Convert to pixel coordinates
        
        # 1. Random Rotation
        if self.random_rotation:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)

            # Rotate dart positions around the center of the image
            angle_rad = torch.deg2rad(torch.tensor(angle))
            center = torch.tensor([width/2, height/2])

            R = torch.tensor([[torch.cos(-angle_rad), -torch.sin(-angle_rad)],
                                [torch.sin(-angle_rad),  torch.cos(-angle_rad)]])

            
            pixel_coords = torch.mm(pixel_coords - center, R.T) + center  # Apply rotation
            
            
        # 2. Resize to fixed size
        if self.resize_to is not None:
            if width > height:
                new_size = (self.resize_to, int(height * self.resize_to / width))
            else:
                new_size = (int(width * self.resize_to / height), self.resize_to)
            image = image.resize(new_size, Image.BILINEAR)
            pixel_coords *= torch.tensor([new_size[0] / width, new_size[1] / height])
            width, height = image.size
            
            
        # 3. Random Rescale
        if self.random_rescale:
            scale_factor = random.uniform(0.625, 1)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.BILINEAR)
            pixel_coords *= torch.tensor([new_size[0] / width, new_size[1] / height])
            width, height = image.size
            
        # Update pixel coordinates after resizing
        pixel_coords = pixel_coords.round()
        
        # rescale bbox size according to the size of the image: 768px -> 13px bbox size    
        bbox_size = self.bbox_size * (max(width, height) / 768) 
        
        # rescale boxes according to the size of the dartboard in the image
        d = max((pixel_coords[0, 0] - pixel_coords[2, 0]) / width, (pixel_coords[3, 1] - pixel_coords[1, 1]) / height) 
        bbox_size *= d
        
        bbox_width = bbox_size * 2
        
        boxes = torch.cat((pixel_coords-bbox_size, torch.ones_like(pixel_coords)*bbox_width), dim=1)
        labels = torch.cat((torch.arange(4, dtype=torch.long), torch.ones(num_darts, dtype=torch.long)*4))
        annotations = [{"bbox": boxes[i], "category_id": labels[i], "area": bbox_width * bbox_width} for i in range(boxes.shape[0])]
        targets = {
            "image_id": idx,
            "annotations": annotations,
            "filename": filename,
        }
        return image, targets