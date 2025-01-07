import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class Im2Latex(Dataset):
    def __init__(self, root_dir, csv_file, processor, max_length=1000 ,trans=None):
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
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        label = self.data.iloc[idx]['formula']
        encoding = self.processor.tokenizer(
            label,
            truncation=True,
            padding='max_length',
            # max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {
            'image' : pixel_values,
            'labels': input_ids,
            'attention_mask': attention_mask
        }
        # return self.data.iloc[idx]['formula'], image
        


# def data_preprocess(example, processor):
#     formulas = example['formula']
#     image = example['image']
#     label_ids = processor.tokenizer(formulas, add_special_tokens=False, return_tensors="pt").input_ids
#     inputs = processor(image, return_tensors="pt").pixel_values
#     return (inputs, label_ids)
    
#     # torch.utils.data.Dataset

# def dataset_process(dataset, processor):
#     preprocess_with_args = partial(data_preprocess, processor=processor)
#     train_dataset = dataset['train'].map(preprocess_with_args, batched=False)
#     test_dataset = dataset['test'].map(preprocess_with_args, batched=False)
#     validation_dataset = dataset['validation'].map(preprocess_with_args, batched=False)
#     return train_dataset, test_dataset, validation_dataset



