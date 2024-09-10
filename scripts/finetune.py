
import os

import torch
import argparse

from accelerate import Accelerator
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AdamW, default_data_collator, get_linear_schedule_with_warmup
from transformers import LlamaForCausalLM
from torch.utils.tensorboard import SummaryWriter

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(model_name_or_path, train_file, valid_file=None, valid_split=0.1, batch_size=8, text_column="text",
         max_length=4096, lr=1e-3, weight_decay=0.1, num_epochs=1, use_flash_attn=False, torch_dtype=torch.float16, output_dir=""):
    accelerator = Accelerator()
    
    # Initialize TensorBoard writer
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
        
    label_column = text_column # For CausalLM, the target is the original string

    init_random_seed(42)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        use_flash_attention_2=use_flash_attn
    )
    model.gradient_checkpointing_enable()

    dataset = DatasetDict()
    
    train_ext = os.path.splitext(train_file)[1]
    if valid_file:
        assert train_ext == os.path.splitext(valid_file)[1], "train and valid file must be of same type"
    
    if train_ext == '.jsonl':
        data_files = {"train": train_file}
        if valid_file:
            data_files["validation"] = valid_file
        dataset = load_dataset(
            "json",
            data_files=data_files
        )
    elif not train_ext:
        dataset = load_dataset(train_file, "default")
        if valid_file:
            dataset["validation"] = load_dataset(valid_file)["train"]
    else:
        raise NotImplementedError(f"Do not support .{train_ext} extension")
        
    if "validation" not in dataset.keys():
        dataset = dataset["train"].train_test_split(test_size=valid_split)
        dataset["validation"] = dataset.pop("test")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Add pad token if it does not exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(
            inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    # When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer
    model = accelerator.prepare(model)

    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # When using FSDP, it is efficient and recommended to call prepare for the model before creating the optimizer
    train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )

    accelerator.print(model)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Log training loss
            if accelerator.is_main_process and step % 100 == 0:  # Log every 100 steps
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + step)

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            # preds = accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
            # eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        
        # Log epoch metrics
        if accelerator.is_main_process:
            writer.add_scalar('Loss/train_epoch', train_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/train', train_ppl.item(), epoch)
            writer.add_scalar('Loss/eval', eval_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/eval', eval_ppl.item(), epoch)

        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        accelerator.wait_for_everyone()

    # Close the TensorBoard writer
    if accelerator.is_main_process:
        writer.close()
    
    # Save our final checkpoint!
    if output_dir:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        print(f"Saved model at {output_dir}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a Causal LLM with FSDP.")
    parser.add_argument(
        "--model-name-or-path", type=str, default="meta-llama/Llama-2-7b-hf", help="model path"
    )
    parser.add_argument(
        "--model_config", type=str, default="configs/llama2_7b.json"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="togethercomputer/llama-instruct",
        help="Train dataset file (jsonl) or huggingface hub datset"
    )
    parser.add_argument(
        "--valid-file", type=str,
        default=None,
        help="Validation dataset file (jsonl) or huggingface hub datset. If None, uses a split of train."
    )
    parser.add_argument(
        "--valid-split", type=float,
        default=0.1,
        help="Percent of train to split in to validation if dedicated validation file not specified."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch Size"
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column for the text in the datsets"
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max sequence length."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning Rate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs"
    )
    parser.add_argument(
        "--use-flash-attn", action="store_true", help="Use flash attention v2 (not supported by all models)"
    )
    parser.add_argument(
        "--torch-dtype", type=str, default="bf16", help="Torch data type", choices=["bf16", "fp16", "fp32"]
    )
    parser.add_argument(
        "--output-dir", type=str, default="", help="Output dir to save model"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1)

    args = parser.parse_args()

    torch_types = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    main(
        args.model_name_or_path,
        args.train_file,
        args.valid_file,
        args.valid_split,
        args.batch_size, 
        args.text_column,
        args.max_length,
        args.learning_rate,
        args.weight_decay,
        args.num_epochs,
        args.use_flash_attn,
        torch_types[args.torch_dtype],
        args.output_dir,
    )
