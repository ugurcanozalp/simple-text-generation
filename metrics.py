
from typing import List

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from seqeval.scheme import Token, Prefix, Tag, IOB2
import torch
from torchmetrics import Metric

class TextGenerationMetric(Metric):
	def __init__(self, dist_sync_on_step: bool = False):
		super().__init__(dist_sync_on_step=dist_sync_on_step)
		self.add_state("preds", default=[], dist_reduce_fx=None)
		self.add_state("targets", default=[], dist_reduce_fx=None)

	def update(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
		self.preds.append([self.tag_names[pred.item()] for pred in preds[mask]])
		self.targets.append([self.tag_names[target.item()] for target in targets[mask]])

	def compute(self):
		return {
			"bleu": accuracy,
		}
