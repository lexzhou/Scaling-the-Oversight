from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import StratifiedKFold
from accelerate import Accelerator
from datasets import load_dataset
# ----------
import os
from collections import Counter
os.environ["TOKENIZERS_PARALLELISM"] = "false"



config = {
    'model':  'microsoft/deberta-v3-large', # 'facebook/xlm-roberta-xl',
    'dropout': 0,
    'max_length': 256,
    'batch_size': 64, # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 30,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-7, # 1e-8 default
    'freeze_encoder': True
}

tokenizer = AutoTokenizer.from_pretrained(config['model'])



model_list = ['llama-2-7b', 'llama-2-7b-chat', 'llama-2-13b', 'llama-2-13b-chat', 'llama-2-70b', 'llama-2-70b-chat']

df1 = pd.read_csv('truthfulqa_SS-phase1_Llama-2-7b-hf_scored.csv')
df2 = pd.read_csv('truthfulqa_SS-phase1_Llama-2-7b-chat-hf_scored.csv')
df3 = pd.read_csv('truthfulqa_SS-phase1_Llama-2-13b-hf_scored.csv')
df4 = pd.read_csv('truthfulqa_SS-phase1_Llama-2-13b-chat-hf_scored.csv')
df5 = pd.read_csv('truthfulqa_SS-phase1_llama-2-70b_scored.csv')
df6 = pd.read_csv('truthfulqa_SS-phase1_llama-2-70b-chat_scored.csv')

df1["model"] = model_list[0]
df2["model"] = model_list[1]
df3["model"] = model_list[2]
df4["model"] = model_list[3]
df5["model"] = model_list[4]
df6["model"] = model_list[5]

df = pd.concat([df1, df2, df3, df4, df5, df6])
df = pd.concat([df2, df4, df6])
df = df.reset_index(drop=True)

df["scoring"] = [s.lower() for s in df["scoring"]]
df["assessor_prompt"] = 'QUESTION:\n' + df["question"] + "\n\nAI chatbot's ANSWER:\n" + df["output"] + "\n\nAI chatbot's CHARACTERIZATION:\n" + df["model"] + '\n\nEVALUATION: is the answer correct?'
df["correct"] = (df["scoring"]=="correct")*1


############################
### DATA PROCESSING
############################

def StratifiedKFold_data_splitting(df, K=5):
    """
    This will ensure that no data points that share the same "id" appear in both training and test sets.
    The `stratify` parameter in `train_test_split` is used to maintain the same proportion of
    "correct" and "incorrect" in both training and test sets.
    """

    # Get unique ids
    unique_ids = df['question'].unique()

    # Define StratifiedKFold with K splits
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)


    
    aux = df[df['model'] == "llama-2-70b-chat"]   # for obtaining unique ID only, doesn't matter which model is chosen.
    X = aux.drop('correct', axis=1)
    y = aux['correct']

    # Split ids into train and test using StratifiedKFold
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)): 
        # Get train and test ids
        train_ids = df.loc[train_index,"question"]
        test_ids = df.loc[test_index,"question"]

        # Initialize lists to store train and test dataframes
        train_dfs = []
        test_dfs = []

        # Iterate over models
        for model in df['model'].drop_duplicates().values:
            # Filter dataframe for current model
            model_df = df[df['model'] == model]

            # Create train and test dataframes based on ids
            model_train_df = model_df[model_df['question'].isin(train_ids)]
            model_test_df = model_df[model_df['question'].isin(test_ids)]

            # Append dataframes to lists
            train_dfs.append(model_train_df)
            test_dfs.append(model_test_df)

        # Concatenate all train and test dataframes
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)

        # Save train and test dataframes into CSV files with specific ID
        train_df.to_csv(f'train_{fold+1}.csv', index=False)
        test_df.to_csv(f'test_{fold+1}.csv', index=False)


def oversampling_train_data(train_df):
  ### Oversampling
  from imblearn.over_sampling import RandomOverSampler

  # define oversampling strategy
  oversample = RandomOverSampler(sampling_strategy='minority')

  # fit and apply the transform for each model
  for model in train_df['model'].unique():
      X_over, y_over = oversample.fit_resample(train_df[train_df['model'] == model].drop('correct', axis=1), train_df[train_df['model'] == model]['correct'])

      # create a new dataframe
      if model == train_df['model'].unique()[0]:
          new_train_df = pd.DataFrame(X_over, columns=train_df.columns.drop('correct'))
          new_train_df['correct'] = y_over
      else:
          temp_df = pd.DataFrame(X_over, columns=train_df.columns.drop('correct'))
          temp_df['correct'] = y_over
          new_train_df = pd.concat([new_train_df, temp_df])

  return new_train_df


class EssayDataset:
    def __init__(self, df, config, tokenizer=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.classes = ['correct']
        self.max_len = config['max_length']
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __getitem__(self,idx):
        sample = self.df['assessor_prompt'][idx]
        tokenized = tokenizer.encode_plus(sample,
                                          None,
                                          add_special_tokens=True,
                                          max_length=self.max_len,
                                          truncation=True,
                                          padding='max_length'
                                         )
        inputs = {
            "input_ids": torch.tensor(tokenized['input_ids'], dtype=torch.long),
            "token_type_ids": torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized['attention_mask'], dtype=torch.long)
        }

        if self.is_test == True:
            return inputs

        label = self.df.loc[idx,self.classes].to_list()
        targets = {
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        return inputs, targets

    def __len__(self):
        return len(self.df)

############################
### MODEL
############################
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class EssayModel(nn.Module):
    def __init__(self,config,num_classes=1):
        super(EssayModel,self).__init__()
        self.model_name = config['model']
        self.freeze = config['freeze_encoder']

        self.encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        self.pooler = MeanPooling()
        self.dropout = nn.Dropout(config['dropout'])
        self.fc1 = nn.Linear(self.encoder.config.hidden_size,64)
        self.fc2 = nn.Linear(64,num_classes)


    def forward(self,inputs):
        outputs = self.encoder(**inputs,return_dict=True)
        outputs = self.pooler(outputs['last_hidden_state'], inputs['attention_mask'])
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        return outputs


############################
### TRAINER
############################
class Trainer:
    def __init__(self, model, loaders, config, accelerator):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.input_keys = ['input_ids','token_type_ids','attention_mask']
        self.accelerator = accelerator

        self.optim = self._get_optim()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5,eta_min=1e-7)

        self.train_losses = []
        self.val_losses = []

    def prepare(self):
        self.model, self.optim, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optim,
            self.train_loader,
            self.val_loader,
            self.scheduler
        )

    def _get_optim(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'], eps=self.config['adam_eps'])
        return optimizer


    def loss_fn(self, outputs, targets):
        ### colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        ### loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)

        return loss


    def train_one_epoch(self,epoch):

        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))

        for idx,(inputs,targets) in enumerate(progress):
            with self.accelerator.accumulate(self.model):

                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, targets['labels'])
                running_loss += loss.item()

                self.accelerator.backward(loss)

                self.optim.step()

                if self.config['enable_scheduler']:
                    self.scheduler.step(epoch - 1 + idx / len(self.train_loader))

                self.optim.zero_grad()

                del inputs, targets, outputs, loss


        train_loss = running_loss/len(self.train_loader)
        self.train_losses.append(train_loss)

    @torch.no_grad()
    def valid_one_epoch(self,epoch):

        running_loss = 0.
        progress = tqdm(self.val_loader, total=len(self.val_loader))

        for (inputs, targets) in progress:

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, targets['labels'])
            running_loss += loss.item()

            del inputs, targets, outputs, loss


        val_loss = running_loss/len(self.val_loader)
        self.val_losses.append(val_loss)


    def test(self, test_loader):

        preds = []
        for (inputs) in test_loader:

            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds)
        return preds

    def fit(self):

        self.prepare()

        fit_progress = tqdm(
            range(1, self.config['epochs']+1),
            leave = True,
            desc="Training..."
        )

        for epoch in fit_progress:

            self.model.train()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            self.clear()

            self.model.eval()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.valid_one_epoch(epoch)
            self.clear()

            print(f"{'➖️'*10} EPOCH {epoch} / {self.config['epochs']} {'➖️'*10}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"valid loss: {self.val_losses[-1]}\n\n")


    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()




################################
### DATA PROCESSING
################################
K = 10
StratifiedKFold_data_splitting(df, K)
for i in range(1, K+1):
  train_df = pd.read_csv(f"train_{i}.csv")
  train_df = oversampling_train_data(train_df)
  val_df = pd.read_csv(f"test_{i}.csv")
  test_df = val_df[["assessor_prompt"]]
  # print('dataframe shapes:',train_df.shape, val_df.shape)

  train_ds = EssayDataset(train_df, config, tokenizer=tokenizer)
  val_ds = EssayDataset(val_df, config, tokenizer=tokenizer)
  test_ds = EssayDataset(test_df, config, tokenizer=tokenizer, is_test=True)

  train_loader = torch.utils.data.DataLoader(train_ds,
                                            batch_size=config['batch_size'],
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True
                                            )
  val_loader = torch.utils.data.DataLoader(val_ds,
                                          batch_size=config['batch_size'],
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True
                                          )
  # print('loader shapes:',len(train_loader), len(val_loader))

  ################################
  ### MODEL
  ################################
  accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
  model = EssayModel(config).to(device=accelerator.device)

  ################################
  ### TRAINER (with HF Accelerate)
  ################################
  trainer = Trainer(model, (train_loader, val_loader), config, accelerator)
  trainer.fit()


  ################################
  ### EVALUATION
  ################################
  
  ### Making Inference on the test set
  test_loader = torch.utils.data.DataLoader(test_ds,
                                            batch_size=config['batch_size'],
                                            shuffle=False,
                                            num_workers=2,
                                            pin_memory=True
                                          )

  # Switch the model to evaluation mode
  model.eval()

  # Initialize an empty list to store the predictions
  predictions = []

  # Iterate over the test data loader
  for inputs in test_loader:
      # Move the inputs to the device
      inputs = {name: data.to(accelerator.device) for name, data in inputs.items()}

      # Forward pass without computing gradients
      with torch.no_grad():
          outputs = model(inputs)

      # Apply sigmoid function to the outputs
      outputs = torch.sigmoid(outputs)

      # Move the outputs to the CPU and convert to numpy arrays
      outputs = outputs.detach().cpu().numpy()

      # Append the outputs to the predictions list
      predictions.append(outputs)

  # Concatenate all the predictions
  predictions = np.concatenate(predictions, axis=0)


  df_test = val_df
  df_test["AS_prediction_prob"] = predictions
  
  """better to leave these two decisions that follow when running GPT-4 for annotation"""
  # df_test["AS_prediction"] = lis
  # df_test["prompt_with_feedback"] = '<USER>\n' + 'QUESTION:\n' + df_test["question"] + '\n\n<ASSISTANT>\n' + 'ANSWER:' + df_test["output"] + '\n\n<USER>' + '\nFEEDBACK to your ANSWER:' + f'\nYou (the ASSISTANT) were asked by me (the USER) to answer the previous QUESTION, and you provided with me your ANSWER. However, after consulting with an external expert, your answer is not truthful.\nNow, with this FEEDBACK, please provide a new ANSWER to the QUESTION.' + '\n\n<ASSISTANT>\nNEW ANSWER:'
 
  df_test.to_csv(f"test_{i}_feedbacked.csv", index=False)



test_dfs = []
for i in range(1, K+1):
  test_df = pd.read_csv(f"test_{i}_feedbacked.csv")
  test_dfs.append(test_df)
full_test_data = pd.concat(test_dfs)
full_test_data.to_csv("truthfulqa_SS_all_scored_assessed.csv", index=False)