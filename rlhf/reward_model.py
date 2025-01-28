from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import transformers
import torch
from transformers import DataCollatorForSeq2Seq, Trainer, PreTrainedTokenizerBase, DataCollatorWithPadding
from trl import RewardTrainer
import os
from torch import nn

os.environ["WANDB_PROJECT"]="policy-model"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "meta-llama/Llama-3.2-1B"
#model_name = "distilroberta-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="cuda:0", trust_remote_code=False,
        revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model.train()
model.gradient_checkpointing_enable()
#model = prepare_model_for_kbit_training(model)

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

# LoRA trainable version of model
#model = get_peft_model(model, config)

# trainable parameter count
#model.print_trainable_parameters()

data = load_dataset("CarperAI/openai_summarize_comparisons")

def filter_empty_labels(example):
    return len(example["prompt"].strip()) > 0

filtered_data = data.filter(filter_empty_labels)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
# Tokenize training and validation datasets
def formatting_func(examples):
    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": 512,
        "return_tensors": "pt"
    }
    
    # Procesar cada ejemplo en el batch
    input_ids_chosen = []
    attention_mask_chosen = []
    input_ids_rejected = []
    attention_mask_rejected = []
    
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        prompt_plus_chosen_response = prompt + "\n" + chosen
        prompt_plus_rejected_response = prompt + "\n" + rejected

        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        input_ids_chosen.append(tokens_chosen["input_ids"])
        attention_mask_chosen.append(tokens_chosen["attention_mask"])
        input_ids_rejected.append(tokens_rejected["input_ids"])
        attention_mask_rejected.append(tokens_rejected["attention_mask"])


    return {
        "input_ids_chosen": input_ids_chosen,
        "attention_mask_chosen": attention_mask_chosen,
        "input_ids_rejected": input_ids_rejected,
        "attention_mask_rejected": attention_mask_rejected
    }
tokenized_data = filtered_data.map(formatting_func, batched=True,remove_columns=["prompt", "chosen", "rejected"])
#print(tokenized_data["train"]["input_ids_chosen"])
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

def custom_data_collator(batch):
    input_ids_chosen = torch.stack([torch.tensor(example["input_ids_chosen"]) for example in batch])
    attention_mask_chosen = torch.stack([torch.tensor(example["attention_mask_chosen"]) for example in batch])
    input_ids_rejected = torch.stack([torch.tensor(example["input_ids_rejected"]) for example in batch])
    attention_mask_rejected = torch.stack([torch.tensor(example["attention_mask_rejected"]) for example in batch])
    
    # Concatenar para entrada única al modelo
    input_ids = torch.cat([input_ids_chosen, input_ids_rejected], dim=0)
    attention_mask = torch.cat([attention_mask_chosen, attention_mask_rejected], dim=0)
    
    print(f"Dimensiones de input_ids final: {input_ids.shape}")
    print(f"Dimensiones de attention_mask final: {attention_mask.shape}")
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
#data_collator = custom_data_collator(tokenizer)

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "full-reward-tldr-distilroberta-base",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="wandb",
    logging_steps=1,
    #requires_grad=True

)
# configure trainer
trainer = RewardTrainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["valid1"],
    args=training_args,
    #peft_config=config,
    #processing_class=CustomRewardProcessing(tokenizer),
    #processing_class=tokenizer,
    #data_collator=custom_data_collator,
    #compute_loss=custom_compute_loss
)


# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
print("Entrenamiento finalizado")
# renable warnings
model.config.use_cache = True
print("Guardando modelo")
hf_name = 'rvergara2017' # your hf username or org name
model_id = hf_name + "/" + "full-reward-tldr-distilroberta-base"

model.push_to_hub(model_id)
print("Modelo guardado")
trainer.push_to_hub(model_id)
