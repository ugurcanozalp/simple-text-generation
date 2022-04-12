import json
from typing import Any, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random 
import json
import os

def change_case(x):
	x = x.replace("_", " ")
	x = "".join([(' '+char.lower() if (char.isupper() and x[max(0,i-1)].islower()) else char) for i, char in enumerate(x)])
	return x

reserved_keywords = ['!=', '%', '&', "'", "''", '(', ')', '*', '+', ',', '-', '.', '/', '1', 
	'<', '<=', '<>', '=', '>', '>=', '^', '``', 'all', 'and', 'any', 'asc', 'as', 'avg', 
	'between', 'by', 'case', 'count', 'desc', 'distinct', 'except', 'exists', 'from', 'group', 
	'having', 'intersect', 'into', 'in', 'is', 'join', 'left', 'like', 'limit', 'max', 'min', 
	'none', 'not', 'null', 'on', 'order', 'or', 'right', 'select', 'sum', 't1', 't2', 
	't3', 't4', 't5', 'union', 'where', '|' ]

removable_keywords = ["%", "'", "''", '(', ')', '``', '`', "\"", "\"\""]
tmap = {f"t{i}": f"<extra_id_{i}>" for i in range(6)}

class Spider(Dataset):

	def __init__(self, tokenizer: Any,
			data_folder: str,
			max_tokens = 128,
			phase="train",
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.is_training = phase.startswith("train")
		data_path = os.path.join(data_folder, phase+".json")
		with open(data_path) as f:
			self.data = json.load(f)
		table_info_path = os.path.join(data_folder, "tables_info.json")
		with open(table_info_path) as f:
			self.table_info = json.load(f)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		db_id = sample["db_id"]
		question = sample["question"]
		query = sample["query"]
		for reserved_keyword in reserved_keywords:
			query = query.replace(reserved_keyword.upper(), reserved_keyword)
		table_info = self.table_info[db_id]
		table_info_as_list = []
		for table, columns in table_info.items():
			if self.is_training:
				random.shuffle(columns)
			txt_table = table + " & " + " | ".join(columns)
			table_info_as_list.append(txt_table)
		if self.is_training:
			random.shuffle(table_info_as_list)
		table_info_as_text = " </s> ".join(table_info_as_list)
		input_text = question + " </s> " + table_info_as_text
		input_ = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		target_ = self.tokenizer.encode_plus(query, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]		
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

class SpiderDatabaseAgnostic(Dataset):

	def __init__(self, tokenizer: Any,
			data_folder: str,
			max_tokens = 128,
			phase="train",
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.is_training = phase.startswith("train")
		data_path = os.path.join(data_folder, phase+".json")
		with open(data_path) as f:
			self.data = json.load(f)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		question = sample["question"]
		query_toks = sample["query_toks_no_value"]
		query_template_tokens =[(token.lower() if token.lower() in reserved_keywords else "~") for token in query_toks if token not in removable_keywords]
		query_template_tokens = [(tmap[token] if token in tmap.keys() else token) for token in query_template_tokens] 
		query_template = " ".join(query_template_tokens)
		input_ = self.tokenizer.encode_plus(question, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		target_ = self.tokenizer.encode_plus(query_template, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]		
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
	data_path = "../data/spider"
	ds = SpiderDatabaseAgnostic(tokenizer, data_path, phase="dev")
	input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = ds[2]