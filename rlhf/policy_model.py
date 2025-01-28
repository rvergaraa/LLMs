from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import torch
from transformers import DataCollatorForSeq2Seq
import os

os.environ["WANDB_PROJECT"]="policy-model"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0", trust_remote_code=False,
        revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

model.train()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model = get_peft_model(model, config)

# trainable parameter count
model.print_trainable_parameters()

data = load_dataset("CarperAI/openai_summarize_tldr")

def filter_empty_labels(example):
    return len(example["label"].strip()) > 0

filtered_data = data.filter(filter_empty_labels)
tokenizer.pad_token = tokenizer.eos_token
def tokenize_function(example):
        txt = example["prompt"]
        label = example["label"]
        tokenizer.pad_token = tokenizer.eos_token
        encodings_dict = tokenizer(txt, truncation=True, max_length=256, padding="max_length")
        encodings_dict_label = tokenizer(label,truncation=True, max_length=256, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])
        labels_ids = torch.tensor(encodings_dict_label["input_ids"])
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": labels_ids,
        }
# Tokenize training and validation datasets
tokenized_data = filtered_data.map(tokenize_function, batched=True,remove_columns=["prompt", "label"])
print(tokenized_data)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "policy-tldr-llama3.1-1b",
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

)

# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["valid"],
    args=training_args,
    data_collator=data_collator
)


# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
print("Entrenamiento finalizado")
# renable warnings
model.config.use_cache = True
print("Guardando modelo")
hf_name = 'rvergara2017' # your hf username or org name
model_id = hf_name + "/" + "policy-tldr-llama3.1-1b"

model.push_to_hub(model_id)
print("Modelo guardado")
trainer.push_to_hub(model_id)
