################################
### OPEN-ENDED QUESTIONS
################################


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
import math
from transformers import GenerationConfig
# Load the dataset
dataset = load_dataset("truthful_qa", "generation")

# Convert it to a pandas dataframe
df = pd.DataFrame()

for split in dataset.keys():   # Loop over 'train','test','validation'
    temp_df = dataset[split].to_pandas()
    temp_df['split'] = split
    df = df.append(temp_df,ignore_index=True)

for model_id in ["meta-llama/Llama-2-13b-chat-hf"]:
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

    output_lis = []
    confidence_lis = []
    for index, row in df.iterrows():
      PROMPT = row["question"]

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
          max_new_tokens=64,
      )

      string = ''
      token_probs = {"token": [], "prob": []}

      for s in generation_output.sequences:
          s_dec = tokenizer.decode(s[len(input_ids[0]):])  # remove input IDs before decoding
          string += s_dec
      print(row["question"])
      print(string)
      output_lis.append(string)

      # Print log probabilities and corresponding tokens
      for k in range(len(generation_output.scores)):
        for score in generation_output.scores[k]:
          a = [float(i) for i in score]
          prob_arr = torch.softmax(torch.tensor(a), dim=0)

          # Get the probability for the top 1 token
          top1_token_id = torch.argmax(prob_arr).item()
          top1_token = tokenizer.decode([top1_token_id])
          top1_prob = prob_arr[top1_token_id].item()

          token_probs["token"].append(top1_token)
          token_probs["prob"].append(top1_prob)

      print(f"Top 1 token and its probability: {token_probs}")
      confidence_lis.append(token_probs)
    df["output"] = output_lis
    df["confidence"] = confidence_lis
    df.to_csv(f"truthfulqa_sub_sys={model_id}")

import time
while True:
    print("This program is continuously running")
    time.sleep(5)  # sleep for 5 seconds