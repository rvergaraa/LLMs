from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name,
        device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model.eval() # model in evaluation mode (dropout modules are deactivated)

# craft prompt
comment = "Hello Shinji Ikari, how have you been?"
prompt=f'''[INST] {comment} [/INST]'''

# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)

print(tokenizer.batch_decode(outputs)[0])
