from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name,
        device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#model = model.to(device)
model.eval() # model in evaluation mode (dropout modules are deactivated)

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
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"))

print(tokenizer.batch_decode(outputs)[0])
