# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:32:28 2023
"""

import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from transformers import IntervalStrategy, AdamW, get_linear_schedule_with_warmup
import time
from transformers import  GenerationConfig
import math
from sklearn.utils import resample

def uncontaminated_undersampling(df):
    ### undersampling while ensuring all models share the same instances.
    # Select one model
    one_model = df['model'].unique()[0]
    
    # Filter the dataframe for that model
    df_one_model = df[df['model'] == one_model]
    
    # Get the count of the least frequent class for that model
    min_class_count = df_one_model['task'].value_counts().min()
    
    # Perform undersampling for that model
    df_one_model_undersampled = pd.DataFrame()
    
    for task in df_one_model['task'].unique():
        df_task = df_one_model[df_one_model['task'] == task]
        df_task_undersampled = resample(df_task, replace=False, n_samples=min_class_count, random_state=42)
        df_one_model_undersampled = pd.concat([df_one_model_undersampled, df_task_undersampled])
    
    # Extract the same data instances for other models
    df_undersampled = pd.DataFrame()
    
    for model in df['model'].unique():
        if model != one_model:
            for task in df_one_model_undersampled['task'].unique():
                ids = df_one_model_undersampled[df_one_model_undersampled['task'] == task]['id']
                df_model_task = df[(df['model'] == model) & (df['task'] == task) & (df['id'].isin(ids))]
                df_undersampled = pd.concat([df_undersampled, df_model_task])
    
    # Concatenate the undersampled data of the one model with the extracted data of other models
    df_undersampled = pd.concat([df_one_model_undersampled, df_undersampled])

    return df_undersampled


for model_id in ["meta-llama/Llama-2-13b-chat-hf"]:
    
    ### remove quantization_config=bnb_config when downloading the model if the 4-bit quantisation is not needed.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    token="YOUR_TOKEN_HERE"
    tokenizer = AutoTokenizer.from_pretrained(model_id,  use_auth_token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, use_auth_token=token)
    
    
    # Read the csv file
    df = pd.read_csv('assessor_train_data_id.csv')
    # df = uncontaminated_undersampling(df)
    df = df[["assessor_prompt"]]
    
    # Convert the dataframe to a Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Tokenize the "assessor_prompt" column
    dataset = dataset.map(lambda x: tokenizer(x['assessor_prompt']), batched=True)
    
    # Prepare the data for training
    columns = ['input_ids', 'attention_mask', 'assessor_prompt']
    dataset.set_format(type='torch', columns=columns)
    
    
    
    # Setting for A100 - For 3090
    MICRO_BATCH_SIZE = 1  # change to 4 for 3090 (for less VRAM requirement)
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 1  # paper uses 3
    LEARNING_RATE = 2e-6
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 4
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    
    for i in range(1, 3): # Do 2 epochs
        try:
          ### Adaptive learning rate
          # Create the optimizer
          optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        
          # Create a learning rate scheduler
          total_steps = (len(dataset) // BATCH_SIZE) * EPOCHS
          scheduler = get_linear_schedule_with_warmup(
              optimizer, num_warmup_steps=100, num_training_steps=total_steps
          )
        
          trainer = transformers.Trainer(
              model=model,
              train_dataset=dataset,
              args=transformers.TrainingArguments(
                  per_device_train_batch_size=MICRO_BATCH_SIZE,
                  gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                  warmup_steps=100,
                  num_train_epochs=EPOCHS,
                  learning_rate=LEARNING_RATE,
                  logging_steps=1,
                  output_dir="output",
                  optim="paged_adamw_8bit",
              ),
              data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
              optimizers=(optimizer, scheduler),
          )
        
          model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        
          trainer.train()
          
          
          ### Do the inference on the test data for each epoch
          # Read the test data
          df = pd.read_csv('assessor_test_data_id.csv')
          output_lis = []
          confidence_lis_top1 = []
          confidence_lis_top2 = []
          for index, row in df.iterrows():
              PROMPT = row["assessor_prompt"]
        
              inputs = tokenizer(
                  PROMPT,
                  return_tensors="pt",
              )
              input_ids = inputs["input_ids"].cuda()
        
              generation_config = GenerationConfig(
                  temperature=0,
              )
              # print("Generating...")
              generation_output = model.generate(
                  input_ids=input_ids,
                  generation_config=generation_config,
                  return_dict_in_generate=True,
                  output_scores=True,
                  max_new_tokens=1,
              )
        
              string = ''
        
              for s in generation_output.sequences:
                  s_dec = tokenizer.decode(s)
                  string += s_dec
              print(row["correct"], string.split()[-1])
              output_lis.append(string)
        
              # Print log probabilities and corresponding tokens
              for score, token_id in zip(generation_output.scores[0], generation_output.sequences[0][-1:]):
                  a = [float(i) for i in score]
                  prob_arr = torch.softmax(torch.tensor(a), dim=0)
                  top_probs, top_idxs = torch.topk(prob_arr, 3)
                  top_tokens = [tokenizer.decode([idx]) for idx in top_idxs]
        
                  for token, prob in zip(top_tokens, top_probs):
                      print(f"Token: {token}, Prob: {prob}")
              confidence_lis_top1.append(top_probs[0])
              confidence_lis_top2.append(top_probs[1])
          df["assessor_output"] = output_lis
          df["confidence_top1"] = [float(i) for i in confidence_lis_top1]
          df["confidence_top2"] = [float(i) for i in confidence_lis_top2]
          df["correct"] = (df["correct_option"]==df["output"])*1
          df.to_csv(f"{model_id.split('/')[-1]}_assessor_id_epoch={i}.csv")


          ### Do the inference on the OOD data for each epoch
          # Read the test data
          df = pd.read_csv('assessor_test_data_ood.csv')
          output_lis = []
          confidence_lis_top1 = []
          confidence_lis_top2 = []
          for index, row in df.iterrows():
              PROMPT = row["assessor_prompt"]
        
              inputs = tokenizer(
                  PROMPT,
                  return_tensors="pt",
              )
              input_ids = inputs["input_ids"].cuda()
        
              generation_config = GenerationConfig(
                  temperature=0,
              )
              # print("Generating...")
              generation_output = model.generate(
                  input_ids=input_ids,
                  generation_config=generation_config,
                  return_dict_in_generate=True,
                  output_scores=True,
                  max_new_tokens=1,
              )
        
              string = ''
        
              for s in generation_output.sequences:
                  s_dec = tokenizer.decode(s)
                  string += s_dec
              print(row["correct"], string.split()[-1])
              output_lis.append(string)
        
              # Print log probabilities and corresponding tokens
              for score, token_id in zip(generation_output.scores[0], generation_output.sequences[0][-1:]):
                  a = [float(i) for i in score]
                  prob_arr = torch.softmax(torch.tensor(a), dim=0)
                  top_probs, top_idxs = torch.topk(prob_arr, 3)
                  top_tokens = [tokenizer.decode([idx]) for idx in top_idxs]
        
                  for token, prob in zip(top_tokens, top_probs):
                      print(f"Token: {token}, Prob: {prob}")
              confidence_lis_top1.append(top_probs[0])
              confidence_lis_top2.append(top_probs[1])
          df["assessor_output"] = output_lis
          df["confidence_top1"] = [float(i) for i in confidence_lis_top1]
          df["confidence_top2"] = [float(i) for i in confidence_lis_top2]
          df["correct"] = (df["correct_option"]==df["output"])*1
          df.to_csv(f"{model_id.split('/')[-1]}_assessor_ood_epoch={i}.csv")
          
          
          
        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero
    
