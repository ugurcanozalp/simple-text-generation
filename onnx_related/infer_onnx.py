import numpy as np
import pprint
import onnxruntime as ort
from tokenizers import Tokenizer 

class TextGenerationInference:

	def __init__(self, onnx_path, tokenizer_path, tags_path):
		self.tokenizer = Tokenizer.from_file(tokenizer_path)
		self.ort_session = ort.InferenceSession(onnx_path)
		self.decoder = None 

	def __call__(self, text):
		tokens = self.tokenizer.encode(text)
		input_ids = np.array(tokens.ids, dtype=np.int64)
		attention_mask = np.array(tokens.attention_mask, dtype=np.int64)
		# compute ONNX Runtime output prediction
		ort_inputs = {'input_ids': input_ids.reshape(1, -1), 'attention_mask': attention_mask.reshape(1, -1)}
		ort_outs = self.ort_session.run(None, ort_inputs)
		
		#output = ort_outs[0][0].argmax(-1).tolist()
		#return self.decoder(text, output, spans, self.tag_names)

if __name__ == '__main__':
	model = TextGenerationInference('deployment/t5-small/tuned-t5-small-quantized.onnx', 
		'deployment/t5-small/t5-small_tokenizer.json')
	
	text = 'Russia | leader | Putin'
	
	import time 
	t0 = time.perf_counter()
	for i in range(100):
	    result = model(text)
	dt = time.perf_counter() - t0
	
	pprint.pprint(result)
	print(f'Elapsed time: {dt} seconds..')

texts = [
	'Russia | leader | Putin',
	]

for text in texts:
	result = model(text)
	print(text)
	pprint.pprint(result)
