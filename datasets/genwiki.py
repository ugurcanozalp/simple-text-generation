
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

def number_to_special_token(i):
	return f"<extra_id_{i}>"

def entity_to_token(entity, entities, max_entities):
	entity = str(entity)
	try:
		i = entities.index(entity)
	except ValueError:
		i = max_entities
		entities.append(entity)
		max_entities += 1
	return number_to_special_token(i), entities, max_entities

class GenWiki(Dataset):

	def __init__(self, tokenizer: Any,
			data_path: str,
			max_tokens = 192,
			output_max_tokens = 128,
			phase="train",
		):
		self.tokenizer = tokenizer
		self.max_tokens = max_tokens
		self.output_max_tokens = output_max_tokens
		self.is_training = phase == "train"
		with open(data_path) as f:
			self.data = json.load(f)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		sample = self.data[i]
		entities = list(map(str, sample["entities"]))
		graph = sample["graph"]
		triplet_texts = []
		max_entities = len(entities)
		for triplet in graph:
			subj, rel, obj = triplet 
			new_rel = change_case(rel)
			new_subj, entities, max_entities = entity_to_token(subj, entities, max_entities)
			new_obj, entities, max_entities = entity_to_token(obj, entities, max_entities)
			triplet_text = new_subj + " " + new_rel + " " + new_obj
			triplet_texts.append(triplet_text)
		
		entity_texts = [e + f" <extra_id_{i}> " for i, e in enumerate(entities)]
		if self.is_training:
			random.shuffle(triplet_texts)
			random.shuffle(entity_texts)
		input_text = " </s> ".join(triplet_texts) + " </s> " + " </s> ".join(entity_texts) #+ " </s>"
		target_text = sample["text"].replace("<ENT_", "<extra_id_")
		input_ = self.tokenizer.encode_plus(input_text, padding="max_length", truncation=True, max_length=self.max_tokens, return_tensors="np")
		input_ids = input_["input_ids"][0]
		attention_mask = input_["attention_mask"][0]
		target_ = self.tokenizer.encode_plus(target_text, padding="max_length", truncation=True, max_length=self.output_max_tokens, return_tensors="np")
		lm_labels = target_["input_ids"][0]
		decoder_attention_mask = target_["attention_mask"][0]
		return input_ids, attention_mask, lm_labels, decoder_attention_mask

if __name__ == "__main__":
	import transformers
	tokenizer = transformers.T5Tokenizer.from_pretrained("t5-small")
	data_path = "../data/genwiki/train.json"
	ds = GenWiki(tokenizer, data_path)
	input_ids, attention_mask, lm_labels, decoder_attention_mask = ds[0]

	#inps = []
	#tars = [] 
	#import tqdm
	#for input_ids, attention_mask, lm_labels, decoder_attention_mask in tqdm.tqdm(ds):
	#	inps.append(len(input_ids))
	#	tars.append(len(lm_labels))
#
	#inps = np.array(inps)
	#tars = np.array(tars)
#
	#print(inps.min(), inps.max(), inps.mean(), inps.std())
	#print(tars.min(), tars.max(), tars.mean(), tars.std())
