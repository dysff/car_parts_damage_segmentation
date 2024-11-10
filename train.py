from model import UNET
from tqdm import tqdm
from dataset import Segm_Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

LEARNING_RATE = 1e-4
BATCH_SIZE = 5
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 400
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r'data\train\images'
TRAIN_MASK_DIR = r'data\train\masks'
VAL_IMG_DIR = r'data\val\images'
VAL_MASK_DIR = r'data\val\masks'

color_map = {
  (19, 164, 201): 0,   # Missing part: #13A4C9
  (166, 255, 71): 1,    # Broken part: #A6FF47
  (180, 45, 56): 2,     # Scratch: #B42D38
  (225, 150, 96): 3,    # Cracked: #E19660
  (144, 60, 89): 4,     # Dent: #903C59
  (167, 116, 27): 5,    # Flaking: #A7741B
  (180, 14, 19): 6,     # Paint chip: #B40E13
  (115, 194, 206): 7,   # Corrosion: #73C2CE
}

dataset = Segm_Dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, color_map)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNET(in_channels=3, out_channels=len(color_map))
model = model.cuda() if torch.cuda.is_available() else model

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
  
  for batch_index, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):    
    #forward
    scores = model(data)
    loss = criterion(scores, targets)

    #backward
    optimizer.zero_grad()
    loss.backward()

    #gradient descent or optimizer step
    optimizer.step()