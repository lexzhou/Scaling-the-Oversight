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
        "temperature": 0.01,  # instead of 0 bc a temperature of 0 raises error...
        "max_tokens": new_max_tokens,
    }
    
    # Make the API request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_text = response.json()
    return response_text



df = pd.read_csv("MATH_SS_all_scored_assessed.csv")

model_id_mapping = {'llama-2-7b':"togethercomputer/llama-2-7b", 'llama-2-7b-chat':"togethercomputer/llama-2-7b-chat", 
 'llama-2-13b':"togethercomputer/llama-2-13b", 'llama-2-13b-chat':"togethercomputer/llama-2-13b-chat",
 'llama-2-70b':"togethercomputer/llama-2-70b", 'llama-2-70b-chat':"togethercomputer/llama-2-70b-chat"}

# for model_id in ["togethercomputer/llama-2-70b-chat", "togethercomputer/llama-2-70b"]: # "togethercomputer/llama-2-70b"
output_lis = []
for index, row in df.iterrows():
    model_id = model_id_mapping[row["model"]]
    ok = False
    while not ok:
        try:
            PROMPT = row["prompt_with_feedback"] + f"\n My apologies for my previous wrong answer. The suggestion from the expert is indeed true. Instead, the right ANSWER to the QUESTION '{row['question']}' should be:"
            print(11111, "INDEX: ", index)
            response = togetherai_http_request(prompt=PROMPT, model=model_id, new_max_tokens=1000)
            print("\n\n", PROMPT)

            response_text = response["output"]["choices"][0]["text"]
            print("\n\n", response_text)
            output_lis.append(response_text)
            ok = True
        except Exception as e:
            print(e)
            time.sleep(10)

                
df["output_phase2"] = output_lis
df.to_csv(f"MATH_SS_all_phase2_rerun.csv", index=False)

