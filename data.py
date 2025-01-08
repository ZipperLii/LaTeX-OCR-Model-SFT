import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class Im2Latex(Dataset):
    def __init__(self, root_dir, csv_file, processor, max_length=200, trans=None):
        self.root_dir = root_dir
        self.csv_file = os.path.join(self.root_dir, csv_file)
        self.trans = trans
        self.max_length = max_length
        self.processor = processor
        self.data = pd.read_csv(self.csv_file)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'formula_images_processed', self.data.iloc[idx]['image'])
        image = Image.open(img_path).convert("RGB")
        
        if self.trans:
            image = self.trans(image)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        label = self.data.iloc[idx]['formula']
        encoding = self.processor.tokenizer(
            label,
            truncation=True,
            padding = 'max_length',
            max_length = self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return pixel_values, input_ids, attention_mask


