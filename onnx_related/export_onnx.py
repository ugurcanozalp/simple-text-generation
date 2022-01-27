from argparse import ArgumentParser
import json
import os
from tqdm import tqdm
from shutil import copyfile

import pytorch_lightning as pl
import torch

from module import TextGeneration

parser = ArgumentParser()
parser.add_argument('--arch', type=str, default='t5-small')
parser.add_argument('--ckpt', type=str, default="checkpoints/t5-small.ckpt")
parser.add_argument('--quantize', action='store_true')

args = parser.parse_args()
model = TextGeneration(arch = args.arch)

sd = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(sd["state_dict"])
model.eval()

text = "Russia | leader | Putin"
input_batches = model.encode_plus(text, return_tensors='pt')
inp = (input_batches['input_ids'], input_batches['attention_mask'])

deployment_path = os.path.join("deployment", args.arch)
os.mkdir(deployment_path)
onnx_path = os.path.join(deployment_path, "tuned-"+args.arch+".onnx")
model.to_onnx(
    onnx_path, inp, export_params=True,
    opset_version=11,
    input_names = ['input_ids', 'attention_mask'],   # the model's input names
    output_names = ['logits'], # the model's output names
    dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence_size'}, 'attention_mask': {0 : 'batch_size', 1: 'sequence_size'}, 'logits' : {0 : 'batch_size', 1: 'sequence_size'}}
)

if args.quantize:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quant_path = os.path.join(deployment_path, "tuned-"+args.arch+"-quantized.onnx")
    quantized_model = quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
