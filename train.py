# coding:utf-8

'''
Moca partialDA implementation
'''

import torch
from torch.autograd import Variable
from torchvision import models
import sys
import os
import numpy as np
import torchvision
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
import copy
from datasets import *
from operator import itemgetter
from models import ResNet
from utils import *
from torch.utils import model_zoo
from heapq import nsmallest
from tqdm import tqdm, trange
import logging
import math
import random
import log
import datetime
import operator
import copy

def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


# load pretrained model
def load_original(model, model_path, base):
	pretrained_model = torch.load(model_path)
	model_dict = model.state_dict()
	
	pretrained_model_dict = pretrained_model.state_dict()
	for k, v in pretrained_model_dict.items():
		model_dict[k] = pretrained_model_dict[k]
	model.load_state_dict(model_dict)
	
	if len(args.gpu_ids) > 1:
		# model.features = nn.DataParallel(model.features, device_ids=args.gpu_ids).cuda()
		# model.cuda()
		model = nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
	else:
		model.cuda()
	logging.info("Load original model done.")
	
	return model


def load_imagenet_pretrain(model, base):
	if base == 'vgg16':
		url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
	elif base == 'vgg16_bn':
		model_path = '../checkpoint/official/vgg16_bn-6c64b313.pth'
		url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
	elif base == 'resnet50':
		model_path = '../checkpoint/official/resnet50-19c8e357.pth'
		url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
	else:
		logging.error("Undefined network!")
		sys.exit()
	
	if os.path.exists(model_path):
		pretrained_dict = model_zoo.load_url(url, model_dir='../checkpoint/official/')
	else:
		pretrained_dict = model_zoo.load_url(url)
	model_dict = model.state_dict()
	
	if base == 'vgg16' or base == "vgg16_bn":
		model_dict_keys_list = []
		for item in list(model_dict.keys()):
			if 'mask' not in item:
				model_dict_keys_list.append(item)
		
		pretrained_dict_keys_list = list(pretrained_dict.keys())
		if base == "vgg16_bn":
			model_dict_keys_list = [i for i in model_dict_keys_list if 'num_batches_tracked' not in i]
			for idx, k in enumerate(model_dict_keys_list):
				if not "cls_fc" in k and not "num_batches_tracked" in k:
					model_dict[model_dict_keys_list[idx]] = pretrained_dict[pretrained_dict_keys_list[idx]]
		else:
			for a, b in zip(pretrained_dict_keys_list, model_dict_keys_list):
				print(a, b)
			for idx, k in enumerate(model_dict_keys_list):
				if not "cls_fc" in k:
					model_dict[model_dict_keys_list[idx]] = pretrained_dict[pretrained_dict_keys_list[idx]]
	
	else:
		# pretrained_dict = model_zoo.load_url(url)
		# model_dict = model.state_dict()
		for k, v in model_dict.items():
			if not "cls_fc" in k and not "num_batches_tracked" and not "metric_feature" in k:
				model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
	
	model.load_state_dict(model_dict)
	
	if len(args.gpu_ids) > 1:
		model = nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
	else:
		model.cuda()
	return model


class Moca_train(object):
	'''
	Moca partialDA implementation
	'''
	def __init__(self, model, args, num_class=256):
		super(Moca_train, self).__init__()
		self.args = args
		self.source_loader = source_loader
		self.target_train_loader = target_train_loader
		self.target_test_loader = target_test_loader
		self.len_source_loader = len(self.source_loader)
		self.len_target_loader = len(self.target_train_loader)
		self.len_source_dataset = len(self.source_loader.dataset)
		self.len_target_dataset = len(self.target_test_loader.dataset)
		if len(args.gpu_ids) > 1:
			self.source_model = nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
			self.target_model = copy.deepcopy(self.source_model)  # init target_model with source_model
		else:
			self.source_model = model.cuda()
			self.target_model = copy.deepcopy(self.source_model)  # init target_model with source_model
		self.criterion = torch.nn.CrossEntropyLoss()
		
		self.max_correct = 0
		self.littlemax_correct = 0
		self.cur_model = None
		self.queue = {}  # init the queue, update each step
		
	def test(self, mode='source', keep_feature=False):
		if keep_feature:
			test_data = self.target_train_loader
		else:
			test_data = self.target_test_loader
			
		if mode == "source":
			self.source_model.eval()
		elif mode == "target":
			self.target_model.eval()
		else:
			raise ValueError("wrong mode {}".format(mode))
		test_loss, correct, total_images = 0, 0, 0

		for _ in range(64):
			print("#", end="")
		print("\ntesting on target dataset!")
		
		for _, (data, target, target_idx) in tqdm(enumerate(test_data)):
			data, target = data.cuda(), target.cuda()
			if mode == "source":
				t_output, target_feature = self.source_model(data, data)
			else:
				t_output, target_feature = self.target_model(data, data)
			
			test_loss += F.nll_loss(F.log_softmax(t_output, dim=1), target,
			                        reduction='sum').item()  # sum up batch loss
			pred = t_output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()
			
			if keep_feature:
				target_feature = target_feature.data.cpu()
				target_idx = target_idx.data.cpu()
				for idx, feature in zip(target_idx, target_feature):
					idx = idx.item()
					if idx in self.queue.keys():
						self.queue.pop(idx)  # delete the last idx and its feature
					self.queue[idx] = feature.detach().data.cpu()  # update new idx and its feature
					
		print("queue length", len(self.queue.keys()))
		
		test_loss /= self.len_target_dataset
		logger.info('Test on target set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
			self.args.target, test_loss, correct, self.len_target_dataset,
			100. * correct / self.len_target_dataset))
		return correct

	def finetune_on_source(self, epoches=10, save_name=None):
		max_correct = 0
		self.source_model.cuda()
		
		for i in trange(epoches):
			logger.info("Epoch: {}".format(i + 1))
			
			if args.arc == "resnet50":
				LEARNING_RATE = 0.015 / math.pow((1 + 10 * i / epoches), 0.75)  # 10*
				optimizer = torch.optim.SGD([
					{'params': self.source_model.features.parameters()},
					{'params': self.source_model.metric_feature.parameters()},
					{'params': self.source_model.cls_fc.parameters(), 'lr': LEARNING_RATE},
				], lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=5e-4)
			
			for param_group in optimizer.param_groups:
				logger.info("Learning rate: {}".format(param_group['lr']))
				
			self.source_model.train()
			
			# source_idx = self.source_loader.dataset.targets()
			iter_source = iter(self.source_loader)
			iter_target = iter(self.target_train_loader)
			
			gamma = 2 / (1 + math.exp(-10 * (i * 0.2) / epoches)) - 1.0
			
			for b in tqdm(range(1, len(self.source_loader))):
				# for b in tqdm(range(1, 2)):  # FBI warning, for testing purpose
				data_source, label_source, source_idx = iter_source.next()
				data_target, _, target_idx = iter_target.next()
				if len(data_target < self.args.batch_size):
					iter_target = iter(self.target_train_loader)
					data_target, _, target_idx = iter_target.next()
				data_source, label_source = data_source.cuda(), label_source.cuda()
				data_target = data_target.cuda()
				
				optimizer.zero_grad()

				source_pred, target_pred, source_feature, target_feature = self.source_model(data_source, data_target)
				loss_cls = F.nll_loss(F.log_softmax(source_pred, dim=1), label_source)
				# loss_entropy_min = EntropyMinLoss(F.softmax(torch.cat([source_pred, target_pred], dim=0), dim=1))  # entropy minimization loss
				loss_entropy_min = EntropyMinLoss(F.softmax(target_pred, dim=1))
				total_loss = loss_cls

				total_loss.backward()
				optimizer.step()

				target_feature = target_feature.data.cpu()
				target_idx = target_idx.data.cpu()
				for idx, feature in zip(target_idx, target_feature):
					idx = idx.item()
					if idx in self.queue.keys():
						self.queue.pop(idx)  # delete the last idx and its feature
					self.queue[idx] = feature.detach().data.cpu()  # update new idx and its feature

			logger.info("train epoch {}".format(i + 1))
			logger.info("train loss_cls {}".format(loss_cls))
			logger.info("train loss_entropy_min {}".format(loss_entropy_min))
			
			cur_correct = self.test(mode="source")
			if cur_correct > max_correct:
				max_correct = cur_correct
				if save_name:
					# torch.save(self.model, save_name)
					logger.info("Model saved to {}".format(save_name))
			logger.info('Finetune on {} max correct: {} max accuracy{: .2f}%\n'.format(self.args.source, max_correct,
			                                                                           100.0 * max_correct / self.len_source_dataset))
		
		print("self.source_model.metric_fc", self.source_model.metric_feature.weight[0])
		# logger.info("Finished fine tuning.")
	

	def finetune_on_target(self, epoches=5, save_name=None):
		max_correct = 0
		self.target_model.cuda()
		
		for i in trange(epoches):
			logger.info("Epoch: {}".format(i + 1))
			
			if args.arc == "resnet50":
				LEARNING_RATE = 0.015 / math.pow((1 + 10 * i / epoches), 0.75)  # 10*
				optimizer = torch.optim.SGD([
					{'params': self.target_model.features.parameters()},
					{'params': self.target_model.metric_feature.parameters()},
					{'params': self.target_model.cls_fc.parameters(), 'lr': LEARNING_RATE},
				], lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=5e-4)
			
			for param_group in optimizer.param_groups:
				logger.info("Learning rate: {}".format(param_group['lr']))
			
			self.target_model.train()
			
			gamma = 2 / (1 + math.exp(-10 * (i * 0.2) / epoches)) - 1.0
			
			# print("self.queue.keys", sorted(self.queue.keys()))
			for data_target, _, target_idx in self.target_train_loader:
				data_target = data_target.cuda()
				optimizer.zero_grad()
				
				target_pred, target_pred, target_feature, target_feature = self.target_model(data_target, data_target)
				
				loss_contrastive = ContrastiveLoss(target_feature, target_idx, self.queue)
				total_loss = loss_contrastive
				
				total_loss.backward()
				optimizer.step()
			
			logger.info("train epoch {}".format(i + 1))
			logger.info("train loss_contrastive {}".format(loss_contrastive))
			
			cur_correct = self.test(mode="target")
			if cur_correct > max_correct:
				max_correct = cur_correct
				if save_name:
					# torch.save(self.model, save_name)
					logger.info("Model saved to {}".format(save_name))
			logger.info('Finetune on {} max correct: {} max accuracy{: .2f}%\n'.format(self.args.source, max_correct,
			                                                                           100.0 * max_correct / self.len_source_dataset))
		
		self.source_model = model_weights_update(self.source_model, self.target_model, m=0.95)
		# print("self.source_model.metric_feature", self.source_model.metric_feature.weight[0])
		# print("self.target_model.metric_feature", self.target_model.metric_feature.weight[0])
		# logger.info("Finished fine tuning.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--train_dir", type=str, default="")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--source", type=str, default="")
    parser.add_argument("--target", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--arc", type=str, default="resnet50")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--pretrain_model_dir", type=str, default="", help="trained model path")
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--pruned_model_dir", type=str, default="", help="pruned model path")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args


if __name__ == "__main__":
	args = get_args()
	
	if not os.path.exists('log'):
		os.mkdir('log')
	result_dir = os.path.join('log',
	                          args.source + "-" + args.target + '-' + args.arc + '-' + datetime.datetime.now().strftime(
		                          "%m%d-%H-%M-%S"))
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)
	logger = log.setup_global_logger('root', result_dir)
	args.result_dir = result_dir
	
	set_seed()
	for k, v in vars(args).items():
		logger.info("{}:{}".format(k, str(v)))
	
	cuda = args.use_cuda
	
	# set pretrain model path
	pretrain_model_path = os.path.join(args.pretrain_model_dir,
	                                   args.source + "_" + args.target + '_' + args.arc + "_pretrain.pth")
	
	if args.mode == 'train':
		logger.info("Finetune a model pretrained on ImageNet\n")
		if args.arc == "resnet50":
			model = ResNet()
			model = load_imagenet_pretrain(model, "resnet50")
		else:
			raise ValueError("the model arc {} is not provided yet".format(args.arc))

	
	elif args.mode == 'prune':
		logger.info("Prune a trained DA model\n")
		if os.path.exists(pretrain_model_path):
			model = torch.load(pretrain_model_path).cuda()
		elif args.arc == "resnet50":
			model = ResNet()
			model = load_imagenet_pretrain(model, "resnet50")
		else:
			raise ValueError("there's no arc {}".format(args.arc))
	
	# load target dataset for training, its label is not used for training default multi gpu
	
	transformer_train = model.train_augmentation()
	transformer_train_multi = model.train_multi_augmentation()
	transformer_test = model.test_augmentation()
	
	source_loader = DataLoader(
		OfficeDataSet(data_path=os.path.join(args.train_dir, args.source, "images"), transformer=transformer_train),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=True,
		drop_last=True,
		pin_memory=False,
	)
	
	target_train_loader = DataLoader(
		OfficeDataSet(data_path=os.path.join(args.test_dir, args.target, "images"), transformer=transformer_train),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=True,
		drop_last=False,
		pin_memory=False
	)
	
	# load target dataset for testing, its label is used for testing until the whole training process ends.
	target_test_loader = DataLoader(
		OfficeDataSet(data_path=os.path.join(args.test_dir, args.target, "images"), transformer=transformer_test),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=False,
		pin_memory=False
	)
	
	fine_tuner = Moca_train(model, args=args)
	fine_tuner.test(keep_feature=True)
	fine_tuner.finetune_on_source(epoches=10, save_name=pretrain_model_path)
	fine_tuner.finetune_on_target(epoches=1, save_name=pretrain_model_path)
	
	logger.info("\n++++Source:{} to target {} finish!++++".format(args.source, args.target))
	
	