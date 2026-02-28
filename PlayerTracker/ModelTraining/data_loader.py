import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

class FootballDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        images = cv2.imread(img_path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt'))
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                labels.append([class_id, x_center, y_center, width, height])
            
        if self.transform:
            images = self.transform(image=images)["image"]
            
        return images, labels

def get_dataloader(img_dir, label_dir, batch_size=8):
    dataset = FootballDataset(img_dir, label_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)