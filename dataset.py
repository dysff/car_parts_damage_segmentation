import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class Segm_Dataset(Dataset):
  def __init__(self, image_dir, mask_dir, color_map):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.image_files = os.listdir(self.image_dir)
    self.mask_files = os.listdir(self.mask_dir)
    self.color_map = color_map
    
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    image_path = os.path.join(self.image_dir, self.image_files[idx])
    mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('RGB'), dtype=np.float32)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)

    for color, label in self.color_map.items():
      color_array = np.array(color, dtype=np.float32)
      mask_area = np.all(mask == color_array, axis=-1)
      label_mask[mask_area] = label

    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    label_mask = torch.tensor(label_mask, dtype=torch.long)

    return image, label_mask