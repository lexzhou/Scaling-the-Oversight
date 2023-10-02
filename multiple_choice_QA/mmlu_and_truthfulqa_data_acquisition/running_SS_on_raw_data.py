# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 21:55:54 2023

@author: 17245
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import math
from transformers import GenerationConfig
import pandas as pd
from datasets import Dataset
import ast
def num_options(string):
 return len(ast.literal_eval(string))


for model_id in ["meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-7b-hf", 
                 "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"]:

    ### remove quantization_config=bnb_config when downloading the model if the 4-bit quantisation is not needed.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    token="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"
    tokenizer = AutoTokenizer.from_pretrained(model_id,  use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_auth_token=token)
    
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    # Read the csv file
    df = pd.read_csv('raw_data.csv') 
    df["num_options"] = df["options"].map(num_options)
    
    output_lis = []
    confidence_lis = []
    for index, row in df.iterrows():
      PROMPT = row["input"]
    
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
          max_new_tokens=1,
      )
    
      string = ''
    
      for s in generation_output.sequences:
          s_dec = tokenizer.decode(s[len(input_ids[0]):])  # remove input IDs before decoding
          string += s_dec
      print("\n\n\n\n", row["input"])
      print(row["correct_option"], "  ", string)
      output_lis.append(string)
    
      # Print log probabilities and corresponding tokens
      for score, token_id in zip(generation_output.scores[0], generation_output.sequences[0][-1:]):
          a = [float(i) for i in score]
          prob_arr = torch.softmax(torch.tensor(a), dim=0)
    
          # Get the probabilities for the specific tokens
          options = list('ABCDEFGHIJKLM')[:row["num_options"]]
          option_ids = [tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
          option_probs = prob_arr[option_ids]
    
          option_prob_dict = {option: float(prob) for option, prob in zip(options, option_probs)}
          print(f"Option probabilities: {option_prob_dict}")
      confidence_lis.append(str(option_prob_dict))
    df["output"] = output_lis
    df["confidence"] = confidence_lis

    df.to_csv(f"{model_id.split('/')[-1]}-subject_system_data.csv")