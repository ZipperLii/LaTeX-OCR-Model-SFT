import os
from PIL import Image
from functools import partial
from datasets import load_dataset

# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
def data_preprocess(example, processor, image_dir):
    image_paths = [os.path.join(image_dir, image) for image in example['image']]
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    formulas = example['formula']
    inputs = processor(images=images, text=formulas, return_tensors="pt", padding="max_length", truncation=True)
    return inputs

def dataset_process(dataset, processor, image_dir):
    preprocess_with_args = partial(data_preprocess, processor=processor, image_dir=image_dir)
    train_dataset = dataset['train'].map(preprocess_with_args, batched=True)
    test_dataset = dataset['test'].map(preprocess_with_args, batched=True)
    validation_dataset = dataset['validation'].map(preprocess_with_args, batched=True)
    
    return train_dataset, test_dataset, validation_dataset



