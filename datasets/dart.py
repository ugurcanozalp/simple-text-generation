
import json
from typing import Any, List
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random 
import pandas as pd
import json

special_tokens = r"""[-\]\[.,;"'?():_`“”/°º‘’″…#$%()*+<>=@\\^_{}|~❑&§] """

def triplet_to_text(triplet, augment=False):
	triplet[1] = format_relation(triplet[1])
	triplet[0] = triplet[0].replace("[TABLECONTEXT]", "<extra_id_2>")
	triplet[0] = triplet[0].replace("[tablecontext]", "<extra_id_2>")
	return " <extra_id_1> ".join(triplet)

def triplet_texts_to_input(triplet_texts, augment=False):
	if augment:
		random.shuffle(triplet_texts)
	return " <extra_id_0> ".join(triplet_texts)

def format_relation(x):
	x = x.replace("_", " ")
	if all([c.isupper() or c in special_tokens for c in x]):
		x = x.lower()
	x = ''.join([' '+i.lower() if i.isupper()
			   else i for i in x]).lstrip(' ')
	x = x.lower()
	x = x.replace("[title]", "<extra_id_3>")
	return x

class Dart(Dataset):

	def __init__(self, tokenizer: Any,
			data_path: str,
			max_tokens = 128,
			phase="train",
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.is_training = phase == "train"
		with open(data_path) as f:
			data = json.load(f)
		self.input_triplets_as_texts = []
		self.input_triplets = []
		self.output_texts = []
		for sample in data:
			for annot in sample["annotations"]:
				output_text = annot["text"]
				self.output_texts.append(output_text)
				self.input_triplets.append(sample["tripleset"])

	def __len__(self):
		return len(self.output_texts)

	def __getitem__(self, i):
		triplets = self.input_triplets[i]
		triplet_texts = [triplet_to_text(triplet, self.is_training) for triplet in triplets]
		input_text = triplet_texts_to_input(triplet_texts, self.is_training)
		output_text = self.output_texts[i]
		input_ = self.tokenizer.encode_plus(input_text, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		target_ = self.tokenizer.encode_plus(output_text, padding='max_length', truncation=True, max_length=self.max_tokens, return_tensors="pt")
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]		
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
	data_path = "../data/dart/dart-v1.1.1-full-train.json"
	ds = Dart(tokenizer, data_path)
	input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = ds[0]
