from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name,
        device_map="cuda:0", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main", load_in_8bit=True) # which version of model to use in repo
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#model = model.to(device)
#model.eval() # model in evaluation mode (dropout modules are deactivated)

# craft prompt
intstructions_string = f"""
Shinji Ikari is the Third Child, the main protagonist of the Neon Genesis Evangelion franchise, and the designated pilot of Evangelion Unit-01. He is the son of Gehirn bioengineer Yui Ikari and NERV Commander (formerly Chief of Gehirn) Gendo Ikari. After his mother's death, he was abandoned by his father and lived for 11 years with his sensei, until he was summoned to Tokyo-3 to pilot Unit-01 against the Angels. He lives initially just with Misato Katsuragi; they are later joined by Asuka Langley Soryu.
Your role is to act like Shinji Ikari.
Shinji shows a great fear of emotional pain and of being hated or left behind, likely due to his perception of being abandoned in his youth and, subsequently, blaming himself for not being good enough to make his father stick around.
"""
comment = "Hello Shinji Ikari, how have you been?"

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

prompt = prompt_template(comment)

# tokenize input
#inputs = tokenizer(prompt, return_tensors="pt")

# generate output
#outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"))

#print(tokenizer.batch_decode(outputs)[0])
model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
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

data = load_dataset("rvergara2017/shinjitxt")

print(data)

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples['text']

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_data = data.map(tokenize_function, batched=True)

# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "shin-ft",
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

)
# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    args=training_args,
    data_collator=data_collator
)


# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# renable warnings
model.config.use_cache = True
hf_name = 'rvergara2017' # your hf username or org name
model_id = hf_name + "/" + "shin02-ft"
model.push_to_hub(model_id)
trainer.push_to_hub(model_id)
