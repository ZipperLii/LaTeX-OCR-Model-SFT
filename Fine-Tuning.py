import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel
from datasets import load_dataset, DatasetDict
from data import dataset_process
from eval import compute_metrics
from transformers import TrainingArguments, Trainer
from datasets import load_dataset


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# train_file = "./data/im2latex_train.csv"
# test_file = "./data/im2latex_test.csv"
# validation_file = "./data/im2latex_validate.csv"
# image_dir = './data/formula_images_processed'

# dataset = DatasetDict({
#     'train': load_dataset('csv', data_files=train_file, split='train'),
#     'test': load_dataset('csv', data_files=test_file, split='train'),
#     'validation': load_dataset('csv', data_files=validation_file, split='train')
# })

processor = NougatProcessor.from_pretrained("facebook/nougat-base")

model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

device = try_gpu()
model.to(device)

dataset = load_dataset("yuntian-deng/im2latex-100k")

task_prompt = "<s_rvlcdip>"

decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
image = dataset['train'][0]["image"]
print(decoder_input_ids)
pixel_values = processor(image, return_tensors="pt").pixel_values
# training_args = TrainingArguments(output_dir="test_trainer")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset['train'],
#     eval_dataset=dataset['val'],
#     compute_metrics=compute_metrics,
# )
# trainer.train()