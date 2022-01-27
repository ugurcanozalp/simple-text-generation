
import json
from typing import Any, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random 
import pandas as pd

lenp1 = lambda x: len(x)+1

def change_case(str):
    return ''.join(['_'+i.lower() if i.isupper()
               else i for i in str]).lstrip('_')
     
class WebNLG(Dataset):

	def __init__(self, tokenizer: Any,
			data_path: str,
			max_tokens = 128,
			phase="train",
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.is_training = phase == "train"
		self.df = pd.read_csv(data_path)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, i):
		sample = self.df.iloc[i]
		input_ = self.tokenizer.encode_plus(sample["input_text"], padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		target_ = self.tokenizer.encode_plus(sample["target_text"], padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]		
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
	data_path = "../data/webnlg/webNLG2020_dev.csv"
	ds = WebNLG(tokenizer, data_path)
	input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = ds[0]
