################################
### OPEN-ENDED QUESTIONS
################################
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import math
from transformers import GenerationConfig

import requests
import json
import os
import pip

import numpy as np
import re
import csv
import time

def togetherai_http_request(prompt, model, new_max_tokens):
    ### CHANGE TOKEN AND USER-AGENT TO JOSE'S ACCOUNT
    # Define the API endpoint

    token = "bafec30861a64d4727408e699b491324e9d703f59314f33f85e0900d6988fb46"  # "8b49f8a950d54a8093e3bb087fa838c20a46f4cb75ee57ba98ceee71a5be0d68"
    url = "https://api.together.xyz/inference"
    # Define the headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "gabesmlmodels@gmail.com"  ### Change this
    }

    # Define the data for the API request
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.001,  # a temperature of 0 raises error...
        "max_tokens": new_max_tokens,
    }
    
    # Make the API request
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=300) # timeout = 5 mins
    response_text = response.json()
    return response_text




# Load the dataset
import pandas as pd
from datasets import load_dataset
dataset = load_dataset("competition_math")
df = dataset["test"].to_pandas()

# Number of unique classes
num_classes = len(df['type'].unique())

"""
Create a balanced sample by taking an equal number of samples from each class in the "type" column. 
If there are remaining samples to reach 1000, it will randomly select the remaining samples from the entire dataframe.
"""

# Minimum samples per class
min_samples_per_class = 1000// num_classes

# Remaining samples
remaining_samples = 1000 % num_classes

# Create a balanced sample
df_sample = df.groupby('type').apply(lambda x: x.sample(n=min_samples_per_class, random_state=42))

# Flatten the multi-index dataframe
df_sample.reset_index(drop=True, inplace=True)

# If there are remaining samples, randomly select from the dataframe
if remaining_samples > 0:
    df_remaining = df.sample(n=remaining_samples, random_state=42)
    df = pd.concat([df_sample, df_remaining]).reset_index(drop=True)
    
    
for model_id in ["togethercomputer/llama-2-7b",
                 "togethercomputer/llama-2-13b",
                 "togethercomputer/llama-2-70b"]: # "togethercomputer/llama-2-70b"

    output_lis = []
    for index, row in df.iterrows():
        ok = False
        while not ok:
            try:
                print("INDEX: ", index)
                PROMPT = row["problem"]
                
                print("PROMPT:", PROMPT, "\n")
                response = togetherai_http_request(prompt=PROMPT, 
                                                   model=model_id, 
                                                   new_max_tokens=1000)
                print("\n\n", response)
                response_text = response["output"]["choices"][0]["text"]
              
                output_lis.append(response_text)
                ok = True
            except Exception as e:
                print(e)
                time.sleep(10)
                
    df["output"] = output_lis
    df.to_csv(f"MATH_SS_{model_id.split('/')[-1]}.csv")
    
