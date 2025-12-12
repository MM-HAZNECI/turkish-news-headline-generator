"""
Model Training Module
"""

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch
import json
import os
from typing import Tuple


class HeadlineGeneratorTrainer:
    def __init__(self, model_name: str = "unsloth/Llama-3.2-1B-Instruct", max_seq_length: int = 512, load_in_4bit: bool = True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Tuple:
        print("="*70)
        print("LOADING BASE MODEL")
        print("="*70)
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        
        print("Adding LoRA adapters...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        
        print("Model loaded!")
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer
    
    def prepare_training_data(self, data_path: str) -> Dataset:
        print("Loading training data...")
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"Loaded {len(data)} samples")
        
        def formatting_prompts_func(examples):
            texts = []
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
                text = self.alpaca_prompt.format(inst, inp, out) + self.tokenizer.eos_token
                texts.append(text)
            return {"text": texts}
        
        dataset = Dataset.from_dict({
            "instruction": [d["instruction"] for d in data],
            "input": [d["input"] for d in data],
            "output": [d["output"] for d in data],
        })
        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset
    
    def train(self, dataset: Dataset, output_dir: str = "outputs", max_steps: int = 800):
        print("="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=50,
                max_steps=max_steps,
                learning_rate=1e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=20,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=output_dir,
            ),
        )
        
        return trainer.train()
    
    def save_model(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving model to: {save_dir}")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print("Model saved!")
