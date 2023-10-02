################################
### OPEN-ENDED QUESTIONS
################################


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import math
from transformers import GenerationConfig

df = pd.read_csv("truthfulqa_6SS_all_scored_assessed_CV=10.csv")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)

output_lis = []
dic = {'llama-2-7b':"meta-llama/Llama-2-7b-hf", 'llama-2-7b-chat':"meta-llama/Llama-2-7b-chat-hf", 
 'llama-2-13b':"meta-llama/Llama-2-13b-hf", 'llama-2-13b-chat':"meta-llama/Llama-2-13b-chat-hf",
 'llama-2-70b':"meta-llama/Llama-2-70b-hf", 'llama-2-70b-chat':"meta-llama/Llama-2-70b-chat-hf"}

df["model"] = df["model"].map(dic)
for index, row in df.iterrows():
  if "70b" not in row["model"]:
      model_id = row["model"]
      token="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"
      tokenizer = AutoTokenizer.from_pretrained(model_id,  use_auth_token=token)
      model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_auth_token=token)
        
      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.pad_token_id = tokenizer.eos_token_id
    
      PROMPT = row["prompt_with_feedback"]
    
      inputs = tokenizer(
          PROMPT,
          return_tensors="pt",
      )
      input_ids = inputs["input_ids"].cuda()
    
      generation_config = GenerationConfig(
          temperature=0,
      )
      print("Generating...")
      generation_output = model.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          return_dict_in_generate=True,
          output_scores=True,
          max_new_tokens=256,
      )
    
      string = ''
    
      for s in generation_output.sequences:
          s_dec = tokenizer.decode(s[len(input_ids[0]):])  # remove input IDs before decoding
          string += s_dec
      print(row["question"])
      print(string)
      output_lis.append(string)
  else:
      output_lis.append("Model not runable in Falco. Switching to TogetherAI API.")
    
dic_inv = {v: k for k, v in dic.items()}
df["model"] = df["model"].map(dic_inv)
df["output_phase2"] = output_lis
df.to_csv("truthfulqa_6SS_all_CV=10_phase2_rerun.csv", index=False)

import time
while True:
    print("This program is continuously running")
    time.sleep(5)  # sleep for 5 seconds