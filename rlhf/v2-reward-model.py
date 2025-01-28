from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import transformers
import torch
from transformers import DataCollatorForSeq2Seq, TrainingArguments
import os
from trl import RewardTrainer, RewardConfig
import pandas as pd

os.environ["WANDB_PROJECT"]="policy-model"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
model_name = "bigcode/tiny_starcoder_py"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=False,
        revision="main")
DATA_PATH = "data/train.parquet"
MODEL_PATH = "bigcode/tiny_starcoder_py"
df = pd.read_parquet(DATA_PATH)
df2 = pd.read_parquet("data/valid.parquet")
raw_dataset = Dataset.from_pandas(df)
raw_vdataset = Dataset.from_pandas(df2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id
def formatting_func(examples):
    kwargs = {"padding": "max_length",
              "truncation": True,
              "max_length": 256,
              "return_tensors": "pt"
              }

    # Prepend the prompt and a line break to the original_response and response-1 fields.
    #prompt_plus_chosen_response = examples["prompt"] + "\n" + examples["chosen"]
    #prompt_plus_rejected_response = examples["prompt"] + "\n" + examples["rejected"]
    prompt_plus_chosen_response = [prompt + "\n" + chosen for prompt, chosen in zip(examples["prompt"], examples["chosen"])]
    prompt_plus_rejected_response = [prompt + "\n" + rejected for prompt, rejected in zip(examples["prompt"], examples["rejected"])]

    #print("Prompt + Chosen Response:", prompt_plus_chosen_response)
    #print("Prompt + Rejected Response:", prompt_plus_rejected_response)

    # Then tokenize these modified fields.
    try:
        tokens_chosen = tokenizer(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer(prompt_plus_rejected_response, **kwargs)
    except Exception as e:
        print(e)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
    }
formatted_dataset = raw_dataset.map(formatting_func, remove_columns=['prompt', 'chosen', 'rejected'])
#vformated_dataset = raw_vdataset.map(formatting_func, remove_columns=['prompt', 'chosen', 'rejected'])
formatted_dataset = formatted_dataset.train_test_split()
model.config
#max_index = model.config.vocab_size - 1
#for ids in formatted_dataset["train"]["input_ids_chosen"]:
#    if any(idx > max_index for idx in ids):
#        print(f"Índice fuera de rango detectado: {ids}")

# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10
training_args = RewardConfig(
        output_dir= "full-reward-tldr-tiny_starcoder",
        num_train_epochs=1,
        logging_steps=10,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        eval_strategy="steps",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=500,
        save_steps=500,
        warmup_steps=100,
        learning_rate=1e-5,
        save_total_limit=1,
        #use_cpu=True,
        report_to="wandb",
        remove_unused_columns=False
    )
trainer = RewardTrainer(model=model,
                        tokenizer=tokenizer,
                        train_dataset=formatted_dataset['train'],
                        eval_dataset=formatted_dataset['test'],
                        args= training_args
                        )

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
