import pandas as pd
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import cv2
import random
import numpy as np

class DartsDataset(Dataset):
    def __init__(self, dirs, bbox_size=13, random_rotation=True, random_rescale=False, resize_to=None, is_synthetic=True):
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
        self.is_synthetic = is_synthetic

    def __len__(self):
        return len(self.labels)
    
    def rotate(self, image, pixel_coords, angle):
        angle_rad = torch.deg2rad(torch.tensor(angle))
        center = torch.tensor([image.width / 2, image.height / 2])
        R = torch.tensor([[torch.cos(-angle_rad), -torch.sin(-angle_rad)],
                          [torch.sin(-angle_rad),  torch.cos(-angle_rad)]])
        return TF.rotate(image, angle), torch.mm(pixel_coords - center, R.T) + center
    
    def resize(self, image, pixel_coords, new_size):
        width, height = image.size
        if width > height:
            new_size = (new_size, int(height * new_size / width))
        else:
            new_size = (int(width * new_size / height), new_size)
        image = image.resize(new_size, Image.BILINEAR)
        pixel_coords *= torch.tensor([new_size[0] / width, new_size[1] / height])
        return image, pixel_coords
    
    def transform(self, image, pixel_coords, size=768):
        src = np.array(pixel_coords[:4])
        tgt = np.array([[0.88, 0.5], [0.5, 0.12], [0.12, 0.5], [0.5, 0.88]]) * size
        transform, _ = cv2.findHomography(src, tgt)
        transformed_coords = torch.from_numpy(cv2.perspectiveTransform(np.array(pixel_coords)[np.newaxis], transform)[0])
        transformed_image = Image.fromarray(cv2.warpPerspective(np.array(image), transform, (size, size)))
        return transformed_image, transformed_coords
    
    def get_annotations(self, image, coords, id):
        coords = coords.round()
        
        # rescale bbox size according to the size of the image: 768px -> 13px bbox size    
        bbox_size = self.bbox_size * (max(image.size) / 768) 
        
        # rescale boxes according to the size of the dartboard in the image
        d = np.array([(coords[0, 0] - coords[2, 0]) / image.width, (coords[3, 1] - coords[1, 1]) / image.height])
        bbox_size *= d
        
        bbox_width = bbox_size * 2
        
        boxes = torch.cat((coords-bbox_size, torch.ones_like(coords)*bbox_width), dim=1)
        labels = torch.cat((torch.arange(4, dtype=torch.long), torch.ones(coords.shape[0]-4, dtype=torch.long)*4))
        annotations = [{"bbox": boxes[i], "category_id": labels[i], "area": bbox_width * bbox_width} for i in range(boxes.shape[0])]
        targets = {
            "annotations": annotations,
            "image_id": id,
        }
        return image, targets

    def __getitem__(self, idx):
        filename, locations, num_darts = self.labels.iloc[idx][['filename', 'locations', 'num_darts']]
        image = Image.open(filename)
        
        locations = torch.tensor(locations, dtype=torch.float32)
        locations[:, 1] = 1 - locations[:, 1]  # Convert y-coordinates to match image coordinates (0 at top)
        pixel_coords = locations * torch.tensor(image.size)  # Convert to pixel coordinates
        
        # 1. Random Rotation
        if self.random_rotation:
            angle = random.uniform(-10, 10)
            image, pixel_coords = self.rotate(image, pixel_coords, angle)
            
        new_size = None
        # 2. Resize to fixed size
        if self.resize_to is not None:
            image, pixel_coords = self.resize(image, pixel_coords, self.resize_to)
            
            
        # 3. Random Rescale
        if self.random_rescale:
            resize_to = np.random.choice([480, 512, 544, 576, 608, 640, 672, 704, 736, 768])
            image, pixel_coords = self.resize(image, pixel_coords, resize_to)
            
        
        transformed_image, transformed_coords = self.transform(image, pixel_coords, size=768)
        
        # Update pixel coordinates after resizing
        
            
        return *self.get_annotations(image, pixel_coords, id=2*idx), *self.get_annotations(transformed_image, transformed_coords, id=2*idx+1)