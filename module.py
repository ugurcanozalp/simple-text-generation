
from argparse import ArgumentParser
from typing import List
import pprint

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import transformers
# from metric import TextGenerationMetric

class TextGeneration(pl.LightningModule):
	def __init__(self,
			arch: str = "t5-small",
			masking_ids = [],
			learning_rate: float = 1e-4,
			*args,
			**kwargs
		):

		super(TextGeneration, self).__init__()
		self.save_hyperparameters("learning_rate")
		# Construct the tokenizer
		self.tokenizer = transformers.T5Tokenizer.from_pretrained(arch, use_fast=True)
		# Construct the neural network here, with specific arch
		self.model = transformers.T5ForConditionalGeneration.from_pretrained(arch)
		
		for param in self.model.shared.parameters():
			param.requires_grad = False

	def configure_optimizers(self):
		optimizer_grouped_parameters = [
			{
				"params": self.model.parameters(),
				"lr": self.hparams.learning_rate,
			}
		]
		optimizer = torch.optim.Adam(optimizer_grouped_parameters)
		return optimizer

	def forward(self, input_ids, attention_mask, lm_labels, decoder_attention_mask):
		return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels, decoder_attention_mask=decoder_attention_mask)

	def training_step(self, batch, batch_idx):
		input_ids, attention_mask, lm_labels, decoder_attention_mask = batch
		output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels, decoder_attention_mask=decoder_attention_mask)
		loss = output.loss
		tensorboard_logs = {'train_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, lm_labels, decoder_attention_mask = batch
		output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels, decoder_attention_mask=decoder_attention_mask)
		loss = output.loss
		#preds = o.preds
		#self.metrics["basic"].update(preds, lm_labels, decoder_attention_mask)
		tensorboard_logs = {'val_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def test_step(self, batch, batch_idx):
		input_ids, attention_mask, lm_labels, decoder_attention_mask = batch
		output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=lm_labels, decoder_attention_mask=decoder_attention_mask)
		loss = output.loss
		#preds = o.preds
		#self.metrics["basic"].update(preds, lm_labels, decoder_attention_mask)
		tensorboard_logs = {'test_batch_loss': loss}
		for metric, value in tensorboard_logs.items():
			self.log(metric, value, prog_bar=False)
		return {"loss": loss}

	def validation_epoch_end(self, outputs):
		total_loss = sum(output["loss"].cpu().item() for output in outputs)
		self.log("total_val_loss", total_loss)
		#metrics = self.metrics["basic"].compute()
		#for name, value in metrics.items():
		#	self.log(name, value)
		#self.metrics["basic"].reset()

	def test_epoch_end(self, outputs):
		total_loss = sum(output["loss"].cpu().item() for output in outputs)
		self.log("total_test_loss", total_loss)
		#metrics = self.metrics["basic"].compute()
		#for name, value in metrics.items():
		#	self.log(name, value)
		#self.metrics["basic"].reset()

	@staticmethod
	def add_model_specific_args(parent_parser):
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--arch', type=str, default='t5-small')
		parser.add_argument('--learning_rate', type=float, default=2e-4)
		return parser

	@torch.no_grad()
	def predict(self, text: str):
		input_ = self.tokenizer.encode_plus(text, return_tensors="pt")
		input_ids = input_["input_ids"]
		attention_mask = input_["attention_mask"]
		generated = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, 
			max_length=64,
			early_stopping=False,
			num_beams=10,
			num_return_sequences=10,
			no_repeat_ngram_size=5)
		return self.tokenizer.decode(generated[0])


if __name__=="__main__":
	sd = torch.load("checkpoints/t5-small.pt.ckpt", map_location="cpu")["state_dict"]
	model = TextGeneration(arch='t5-small')
	model.load_state_dict(sd)
	model.eval()
	text = 'Korea <extra_id_1> gross spending amount <extra_id_1> $100 <extra_id_0> Croatia <extra_id_1> gross spending amount <extra_id_1> $200' # && <extra_id_2> | date range | 01.01.2020 - 31.12.2021
	result = model.predict(text)
	pprint.pprint(result)
