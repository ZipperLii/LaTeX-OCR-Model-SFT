import torch
from transformers import NougatProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from data import Im2Latex
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from eval import compute_metrics

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def main():
    device = try_gpu(i=0)


    # load model and processor
    config = VisionEncoderDecoderConfig.from_pretrained("facebook/nougat-base")
    processor = NougatProcessor.from_pretrained("facebook/nougat-base")
    # model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base", config=config)
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # model.to(device)
    
    root = "./data"
    train_data = Im2Latex(root, 'im2latex_train.csv', processor)
    validate_data = Im2Latex(root, 'im2latex_validate.csv', processor)
    test_data = Im2Latex(root, 'im2latex_test.csv', processor)
    
    batch_size = 8
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    validate_dataloader = DataLoader(validate_data, batch_size=batch_size, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    aa = 0
    model.to(device)
    
    for i, X in enumerate(train_dataloader):
        imgs, labels, attn_masks = X
        imgs, labels, attn_masks = imgs.to(device), labels.to(device), attn_masks.to(device)
        
        optputs = model(imgs, labels)
        print(optputs.logits)
        aa += 1
        if aa == 4:
            break
    outputs = model(pixel_values=train_data[0]['image'], labels=train_data[0]['labels'])


    # lora_config = LoraConfig(
    #     r=16,
    #     target_modules=["q_proj", "v_proj", "k_proj"],
    #     task_type=TaskType.CAUSAL_LM,
    #     lora_alpha=32,
    #     lora_dropout=0.05
    # )

    # lora_model = get_peft_model(model, lora_config, "default")


    # lora_model.print_trainable_parameters()


    # outputs = lora_model(pixel_values=pixel_values, labels=labels)

    # loss = outputs.loss


if __name__ == '__main__':
    main()
    



