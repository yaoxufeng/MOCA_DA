# coding:utf8

"""
dataloader of DA dataset
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import models, transforms


def generate_label(directory):
	label_name = [i for i in os.listdir(directory) if ".DS_Store" not in i]
	label_dict = {k: v for v, k in enumerate(label_name)}
	return label_dict


class Caltech256(Dataset):
	'''
	self data loader for caltech256
	'''
	
	def __init__(self, data_path, transformer=None):
		super(Caltech256, self).__init__()
		self.data_path = data_path
		self.transformer = transformer  # this transformer is generated by torchvision, for reproducing the paper result only
		self.label = generate_label(self.data_path)
		self._parse_directory()
		# self._targets()
	
	def _load_image(self, img_id):
		img = Image.open(img_id).convert("RGB")
		return img
	
	def _transform_label(self):
		self.label_path = [i.split(".")[0] for i in self.label_path]
		for index, item in enumerate(self.label_path):
			if item.startswith("00"):
				item = int(item[2:])
				self.label_path[index] = item - 1
			elif item.startswith("0"):
				item = int(item[1:])
				self.label_path[index] = item - 1
			else:
				item = int(item)
				self.label_path[index] = item - 1
	
	def _parse_directory(self):
		self.image_path = [os.path.join(root, file) for root, _, filenames in os.walk(self.data_path) for file in
		                   filenames]
		self.label_path = [root.split("/")[-1] for root, _, filenames in os.walk(self.data_path) for file in filenames]
		
		# sort image_path and label_path (attention! the reason why we can use sort is the path has the number order!)
		# now the path is ordered by label
		self.image_path = sorted(self.image_path)
		self.label_path = sorted(self.label_path)
		# self._transform_label()
		for index, item in enumerate(self.label_path):
			if "images" in item:
				print(self.label_path[index])
				print(self.image_path[index])
		print(len(self.image_path))
		print(len(self.label_path))
		self.label_path = [self.label[i] for i in self.label_path]
	
	def _targets(self):
		self.label_dict = {}
		for idx, key in enumerate(self.label_path):
			if key in self.label_dict.keys():
				self.label_dict[key].append(idx)
			else:
				self.label_dict[key] = [idx]
	
	def targets(self):
		return self.label_dict
	
	def __getitem__(self, idx):
		img_id = self.image_path[idx]
		label_id = self.label_path[idx]
		img = self._load_image(img_id)
		if self.transformer:
			img = self.transformer(img)
		else:
			raise ValueError("the img is supposed to be transformed to tensor")
		
		return (img, torch.tensor(label_id), idx)
	
	def __len__(self):
		return len(self.image_path)
	

