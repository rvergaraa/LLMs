from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
import transformers
import torch
from transformers import DataCollatorForSeq2Seq
import os

os.environ["WANDB_PROJECT"]="policy-model"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = LoraConfig(
    r=8,
    lora_alpha=32,
    #target_modules=["q_proj"],
    target_modules=['k_proj', 'v_proj', 'q_proj', 'dense'],

    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
quant = BitsAndBytesConfig(
        load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
#model_name = "rvergara2017/full-policy-tldr-llama3.1-1b"
model_name = "./dpo_llama-3.2-1B-tldr/checkpoint-92534"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=False,
        revision="main")#, quantization_config=quant, torch_dtype=torch.float16)
#model.load_adapter(config, adapter_name="reference")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

#model.load_adapter("./policy-tldr-llama3.1-1b", adapter_name="reference")
#model.load_adapter("./policy-tldr-llama3.1-1b", adapter_name="training2")
#model = prepare_model_for_kbit_training(model)
#model.config.use_cache=False
#model = get_peft_model(model, config, adapter_name="train")
#model.gradient_checkpointing_enable()

#model.train()
#model = prepare_model_for_kbit_training(model)

# LoRA config

# LoRA trainable version of model
#model = get_peft_model(model, config)

# trainable parameter count
#model.print_trainable_parameters()

data = load_dataset("CarperAI/openai_summarize_comparisons")

training_args = DPOConfig(output_dir="dpo_llama-3.2-1B-tldr", logging_steps=10,
                          logging_strategy="epoch", eval_strategy="epoch", save_strategy="epoch",
                          num_train_epochs=3,  per_device_train_batch_size=2, per_device_eval_batch_size=2,)
                          #model_adapter_name="training2", ref_adapter_name="reference")
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=data['train'], eval_dataset=data['valid1'],)
                    # peft_config = config, beta=0.2)
trainer.train()
print("Entrenamiento finalizado")
# renable warnings
model.config.use_cache = True
print("Guardando modelo")
hf_name = 'rvergara2017' # your hf username or org name
model_id = hf_name + "/" + "dpo-tldr-llama3.1-1b"

model.push_to_hub(model_id)
print("Modelo guardado")
trainer.push_to_hub(model_id)
