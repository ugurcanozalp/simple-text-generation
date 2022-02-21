
import json
from typing import Any, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random 
import json

def change_case(x):
	x = x.replace("_", " ")
	x = "".join([(' '+char.lower() if (char.isupper() and x[max(0,i-1)].islower()) else char) for i, char in enumerate(x)])
	return x
	 
class WebNLG(Dataset):

	def __init__(self, tokenizer: Any,
			data_path: str,
			max_tokens = 128,
			phase="train",
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.is_training = phase == "train"
		with open(data_path) as f:
			self.data = json.load(f)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		input_triplets = [change_case(triplet) for triplet in sample["triple_set"]]
		if self.is_training:
			random.shuffle(input_triplets)
		input_text = " </s> ".join(input_triplets)
		input_ = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		target_ = self.tokenizer.encode_plus(sample["text"], padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]		
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
	data_path = "../data/webnlg/train.json"
	ds = WebNLG(tokenizer, data_path)
	input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = ds[0]
