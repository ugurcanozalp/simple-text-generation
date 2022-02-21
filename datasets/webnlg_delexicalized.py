import json
from typing import Any, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random 
import pandas as pd
import json

def triplet_to_text(triplet, augment=False):
	return " <extra_id_1> ".join(triplet)

def triplet_texts_to_input(triplet_texts, augment=False):
	if augment:
		random.shuffle(triplet_texts)
	return " <extra_id_0> ".join(triplet_texts)

class WebNLGDelexicalized(Dataset):

	def __init__(self, tokenizer: Any,
			data_path: str,
			max_tokens = 50,
			phase="train",
			max_entity=8
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.entity_token_ids = list(range(max_entity))
		self.is_training = phase == "train"
		with open(data_path) as f:
			self.data = json.load(f)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		node_names = list(sample["node_info"].keys())
		num_entities = len(node_names)
		graph = sample["graph"]
		if self.is_training:
			random.shuffle(graph)
		input_text = " </s> ".join([" ".join(triple) for triple in graph])
		output_text = sample["text"]
		# create random map from node info to special tokens.
		random.shuffle(self.entity_token_ids)
		token_ids = self.entity_token_ids[:num_entities]
		for node_name, token_id in zip(node_names, token_ids):
			token = f"<extra_id_{token_id}>"
			input_text = input_text.replace(node_name, token)
			output_text = output_text.replace(node_name, token)

		input_ = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		
		target_ = self.tokenizer.encode_plus(output_text, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		#, padding='max_length'
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]		
		
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
	data_path = "../data/webnlg_delexicalized/train_clean.json"
	ds = WebNLGDelexicalized(tokenizer, data_path)
	input_ids, attention_mask, lm_labels, decoder_attention_mask = ds[0]