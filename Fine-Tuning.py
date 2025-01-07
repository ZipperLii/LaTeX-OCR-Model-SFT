import torch
from transformers import VisionEncoderDecoderModel
from data import Im2Latex
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer
from eval import compute_metrics

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

batch_size = 32

root = "./data"
train_data = Im2Latex(root, 'im2latex_train.csv')
validate_data = Im2Latex(root, 'im2latex_validate.csv')
test_data = Im2Latex(root, 'im2latex_test.csv')

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()


num_epoch = 10
training_args = TrainingArguments(
    output_dir="./checkpoints/mt0-large-lora",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=num_epoch,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validate_data,
)

trainer.train()

model.save_pretrained("output_dir")

