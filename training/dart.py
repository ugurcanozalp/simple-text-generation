
from argparse import ArgumentParser
import os
import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from module import TextGeneration
from datasets import Dart, numpy_collate_fn

parser = ArgumentParser()
parser.add_argument('--test', action="store_true")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--ckpt', type=str, default="checkpoints/t5-small.pt.ckpt")

parser = TextGeneration.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
dict_args = vars(args)
model = TextGeneration(**dict_args)
# define checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
	dirpath="./checkpoints",
	verbose=True,
	filename=args.arch+".pt",
	monitor="total_val_loss",
	save_top_k=1,
	save_weights_only=True,
	mode="min" # only pick min loss
)

trainer = pl.Trainer(
	gpus=args.gpus,
	callbacks=[checkpoint_callback],
	gradient_clip_val=args.gradient_clip_val)

train_ds = Dart(model.tokenizer, data_path="data/dart/dart-v1.1.1-full-train.json", phase="train") 

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
	num_workers=0)

val_ds = Dart(model.tokenizer, data_path="data/dart/dart-v1.1.1-full-dev.json", phase="val") 

val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
	num_workers=0)

test_ds = Dart(model.tokenizer, data_path="data/dart/dart-v1.1.1-full-test.json", phase="test") 

test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
	num_workers=0)

if args.test:
	model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["state_dict"])
	trainer.test(model, test_dataloaders=[test_dl])
else:
	trainer.fit(model, train_dataloader=train_dl, val_dataloaders=[val_dl])

# python dart.py --arch t5-small --gradient_clip_val 1.0 --max_epochs 10 --min_epochs 3 --gpus 1
