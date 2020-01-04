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
			if not "cls_fc" in k and not "num_batches_tracked" in k and not "metric_feature" in k:
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
			self.source_model = model.cuda()
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
			data, target, target_idx = data[0], target[0], target_idx[0]
			data, target = data.cuda(), target.cuda()
			if mode == "source":
				if len(args.gpu_ids) > 1:
					t_output, target_feature = self.source_model.module(data, data)
				else:
					t_output, target_feature = self.source_model(data, data)
			else:
				if len(args.gpu_ids) > 1:
					t_output, target_feature = self.target_model.module(data, data)
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
						self.queue[idx] = 0.5 * feature.detach().data.cpu() + 0.5 * self.queue[idx] # update new idx and its feature
						# self.queue.pop(idx)  # delete the last idx and its feature
					else:
						self.queue[idx] = feature.detach().data.cpu()  # update new idx and its feature
					
		print("queue length", len(self.queue.keys()))
		
		test_loss /= self.len_target_dataset
		logger.info('Test on source set {} -> target set {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
			self.args.source, self.args.target, test_loss, correct, self.len_target_dataset,
			100. * correct / self.len_target_dataset))
	
		return correct

	def finetune_on_source(self, epoches=10, save_name=None, keep_feature=False, stage=None):
		max_correct = 0
		
		# if stage == "two":
		# 	self.source_model = model_fc_update(self.target_model, self.source_model, gpu_num=len(args.gpu_ids), m=0.1)
			
		self.source_model.cuda()
		
		for i in trange(epoches):
			for _ in range(64):
				print("#", end="")
			logger.info("fine tune on source Epoch: {}".format(i + 1))
			
			if args.arc == "resnet50":
				LEARNING_RATE = 0.015 / math.pow((1 + 10 * i / epoches), 0.75)  # 10*

				if len(args.gpu_ids) > 1:
					optimizer = torch.optim.SGD(self.source_model.module.parameters(), lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=5e-4)
				else:
					optimizer = torch.optim.SGD([
						{'params': self.source_model.features.parameters()},
						{'params': self.source_model.metric_feature.parameters()},
						{'params': self.source_model.cls_fc.parameters(), 'lr': LEARNING_RATE},
					], lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=5e-4)
			
			for param_group in optimizer.param_groups:
				logger.info("Learning rate: {}".format(param_group['lr']))
				
			self.source_model.train()
			
			iter_source = iter(self.source_loader)
			iter_target = iter(self.target_train_loader)
			
			gamma = 2 / (1 + math.exp(-10 * (i * 0.2) / epoches)) - 1.0
			
			for b in tqdm(range(1, len(self.source_loader))):
				data_source, label_source, source_idx = iter_source.next()
				data_source_k, label_source_k, source_idx = data_source[0], label_source[0], source_idx[0]
				data_source_q, _, _ = data_source[1], label_source[1], source_idx[1]
				
				data_target, _, target_idx = iter_target.next()
				data_target_k, target_idx = data_target[0], target_idx[0]
				data_target_q, _ = data_target[1], target_idx[1]
				
				if len(data_target_k < self.args.batch_size):
					iter_target = iter(self.target_train_loader)
					data_target, _, target_idx = iter_target.next()
					data_target_k, target_idx = data_target[0], target_idx[0]
					data_target_q, _ = data_target[1], target_idx[1]
					
				data_source_k, label_source = data_source_k.cuda(), label_source_k.cuda()
				data_source_q = data_source_q.cuda()
				
				data_target_k = data_target_k.cuda()
				data_target_q = data_target_q.cuda()
				
				optimizer.zero_grad()

				if len(args.gpu_ids) > 1:
					source_pred_k, target_pred_k, source_feature_k, target_feature_k = self.source_model.module(data_source_k, data_target_k)
					# source_pred_q, target_pred_q, source_feature_q, target_feature_q = self.source_model.module(data_source_q, data_target_q)
				else:
					source_pred_k, target_pred_k, source_feature_k, target_feature_k = self.source_model(data_source_k, data_target_k)
					# source_pred_q, target_pred_q, source_feature_q, target_feature_q = self.source_model(data_source_q, data_target_q)
				
				# source_feature = 1. * (source_feature_k + source_feature_q) / 2.
				# source_pred = 1. * (source_pred_k + source_pred_q) / 2.
				source_feature = source_feature_k
				source_pred = source_pred_k
				
				# target_feature = 1. * (target_feature_k + target_feature_q) / 2.
				# target_pred = 1. * (target_pred_k + target_pred_q) / 2.
				target_feature = target_feature_k
				target_pred = target_pred_k
				
				loss_cls = F.nll_loss(F.log_softmax(12*source_pred, dim=1), label_source)
				# loss_consistency = ConsistencyLoss(source_feature_k, source_feature_q, reverse=False, mode="l2")
				# loss_entropy_min = EntropyMinLoss(F.softmax(torch.cat([source_pred, target_pred], dim=0), dim=1))  # entropy minimization loss
				loss_entropy_min = EntropyMinLoss(F.softmax(target_pred, dim=1))
				if stage == "one":
					total_loss = loss_cls + 0.1 * loss_entropy_min
				# elif stage == "two":
				# 	total_loss = loss_cls + loss_entropy_min
				# else:
				# 	pass

				total_loss.backward()
				optimizer.step()

				if keep_feature:
					target_feature = target_feature.data.cpu()
					target_idx = target_idx.data.cpu()
					for idx, feature in zip(target_idx, target_feature):
						idx = idx.item()
						if idx in self.queue.keys():
							self.queue[idx] = 0.5 * self.queue[idx] + 0.5 * feature.detach().data.cpu()
							# self.queue.pop(idx)  # delete the last idx and its feature
						else:
							self.queue[idx] = feature.detach().data.cpu()  # update new idx and its feature

			logger.info("train epoch {}".format(i + 1))
			logger.info("train loss_cls {}".format(loss_cls))
			logger.info("train loss_entropy_min {}".format(loss_entropy_min))
			# logger.info("train loss_consistency {}".format(loss_consistency))
			
			cur_correct = self.test(mode="source")
			if cur_correct > max_correct:
				max_correct = cur_correct
				if save_name:
					# torch.save(self.model, save_name)
					logger.info("Model saved to {}".format(save_name))
					
		if stage == "one":
			self.target_model = model_weights_update(self.target_model, self.source_model, m=0.9, gpu_num=len(args.gpu_ids))
		# elif stage == "two":
		# 	self.target_model = model_weights_update(self.target_model, self.source_model, m=0.9)
		# else:
		# 	pass


	def finetune_on_target(self, epoches=5, save_name=None, keep_feature=True, reverse=False, stage=None):
		max_correct = 0
		if stage == "one":
			self.target_model = model_fc_update(self.source_model, self.target_model, gpu_num=len(args.gpu_ids))
			# freeze the weight of fc
			# for param in self.target_model.cls_fc.parameters():
			# 	param.requires_grad = False
				
		self.target_model.cuda()
		
		# self.target_model.cls_fc.require_grad = False
		
		for i in trange(epoches):
			for _ in range(64):
				print("#", end="")
			logger.info("fine tune on target stage {} Epoch: {}".format(stage, i + 1))
			
			if args.arc == "resnet50":
				LEARNING_RATE = 0.015 / math.pow((1 + 10 * i / epoches), 0.75)  # 10*
				if len(args.gpu_ids) > 1:
					optimizer = torch.optim.SGD(self.source_model.module.parameters(), lr=LEARNING_RATE / 10, momentum=0.9,
			                            weight_decay=5e-4)
				else:
					optimizer = torch.optim.SGD([
						{'params': self.target_model.features.parameters()},
						{'params': self.target_model.metric_feature.parameters()},
						# {'params': self.target_model.cls_fc.parameters(), 'lr': LEARNING_RATE},
					], lr=LEARNING_RATE / 10, momentum=0.9, weight_decay=5e-4)
					
			for param_group in optimizer.param_groups:
				logger.info("Learning rate: {}".format(param_group['lr']))
			
			self.target_model.train()
			
			gamma = 2 / (1 + math.exp(-10 * (i * 0.2) / epoches)) - 1.0
			
			for _, (data_target, _, target_idx) in tqdm(enumerate(self.target_train_loader)):
				
				data_q_target, target_q_idx = data_target[0], target_idx[0]
				data_k_target, target_k_idx = data_target[1], target_idx[1]
				
				data_q_target = data_q_target.cuda()
				data_k_target = data_k_target.cuda()
				
				optimizer.zero_grad()
				
				if len(args.gpu_ids) > 1:
					target_q_pred, target_k_pred, target_q_feature, target_k_feature = self.target_model.module(data_q_target, data_k_target)
				else:
					target_q_pred, target_k_pred, target_q_feature, target_k_feature = self.target_model(data_q_target, data_k_target)

				target_feature = 1. * (target_q_feature + target_k_feature) / 2.
				
				loss_discrepancy = DiscrepancyLoss(target_q_pred, target_k_pred, reverse=reverse, mode="l1")
				logits_q, label_q_target = ContrastiveLoss(target_q_feature, target_q_idx, self.queue)
				logits_k, label_k_target = ContrastiveLoss(target_k_feature, target_k_idx, self.queue)
				
				# logits, label_target = ContrastiveLoss(target_q_feature, target_k_feature, target_q_idx, self.queue)
				loss_q_cls = F.nll_loss(F.log_softmax(16*logits_q, dim=1), label_q_target)
				loss_k_cls = F.nll_loss(F.log_softmax(16*logits_k, dim=1), label_k_target)
				
				loss_consistency = ConsistencyLoss(target_q_feature, target_k_feature, reverse=reverse, mode="l2")
				# target_q_feature_queue = torch.tensor([self.queue[idx.item()] for idx in target_q_idx])
				# target_k_feature_queue = torch.tensor([self.queue[idx.item()] for idx in target_k_idx])
				
				# loss_consistency_k = ConsistencyLoss(target_k_feature, target_k_feature_queue, mode="l2")
				# loss_consistency_q = ConsistencyLoss(target_q_feature, target_q_feature_queue, mode="l2")
				
				loss_entropy_k = EntropyMinLoss(F.softmax(target_k_pred, dim=1))
				loss_entropy_q = EntropyMinLoss(F.softmax(target_q_pred, dim=1))
				loss_entropy = loss_entropy_k + loss_entropy_q
				
				if reverse:
					total_loss = loss_q_cls + loss_k_cls + loss_consistency
					# total_loss = loss_q_cls + loss_k_cls
					# total_loss = loss_consistency_k + loss_consistency_q + loss_consistency
				else:
					total_loss = loss_q_cls + loss_k_cls + 0.1 * loss_consistency + loss_entropy
					# total_loss = loss_consistency + 0.1 * (loss_entropy_k + loss_entropy_q)
				
				total_loss.backward()
				optimizer.step()
			
			logger.info("train epoch {}".format(i + 1))
			logger.info("train loss_cls_q {}".format(loss_q_cls))
			logger.info("train loss_cls_k {}".format(loss_k_cls))
			
			logger.info("train loss_consistency {}".format(loss_consistency))
			logger.info("train loss_discrepancy {}".format(loss_discrepancy))
			logger.info("train loss_entropy {}".format(loss_entropy))
			
			# logger.info("train loss_entropy_min {}".format(loss_entropy_min))
			
			if keep_feature:
				target_feature = target_feature.data.cpu()
				target_idx = target_q_idx.data.cpu()
				for idx, feature in zip(target_idx, target_feature):
					idx = idx.item()
					if idx in self.queue.keys():
						# self.queue[idx] = 0.5 * self.queue[idx] + 0.5 * feature.detach().data.cpu()
						self.queue[idx] = feature.detach().data.cpu()  # update new idx and its feature

					else:
						self.queue[idx] = feature.detach().data.cpu()  # update new idx and its feature
			
			cur_correct = self.test(mode="target")
			if cur_correct > max_correct:
				max_correct = cur_correct
				if save_name:
					# torch.save(self.model, save_name)
					logger.info("Model saved to {}".format(save_name))
		if stage == "one":
			self.source_model = model_weights_update(self.source_model, self.target_model, m=0.9, gpu_num=len(args.gpu_ids))
		elif stage == "two":
			self.source_model = model_weights_update(self.source_model, self.target_model, m=0.9, gpu_num=len(args.gpu_ids))
		else:
			pass
		# print("self.source_model.metric_feature", self.source_model.metric_fesature.weight[0])
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--arc", type=str, default="resnet50")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--dataset", type=str, default="office31")
    parser.add_argument("--pretrain_model_dir", type=str, default="", help="trained model path")
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--pruned_model_dir", type=str, default="", help="pruned model path")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
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
	
	if args.dataset == "office_31":
		num_classes = 31
	elif args.dataset == "office_home":
		num_classes = 65
	
	if args.mode == 'train':
		logger.info("Finetune a model pretrained on ImageNet\n")
		if args.arc == "resnet50":
			model = ResNet(num_classes=num_classes)
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
	
	if len(args.gpu_ids) > 1:
		transformer_train = model.module.train_augmentation()
		transformer_train_multi = model.module.train_multi_augmentation()
		transformer_test = model.module.test_augmentation()
	else:
		transformer_train = model.train_augmentation()
		transformer_train_multi = model.train_multi_augmentation()
		transformer_test = model.test_augmentation()

	source_loader = DataLoader(
		OfficeDataSet(data_path=os.path.join(args.train_dir, args.source, "images"), transformer=transformer_train, k=2, dataset=args.dataset),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=True,
		drop_last=True,
		pin_memory=False,
	)

	target_train_loader = DataLoader(
		OfficeDataSet(data_path=os.path.join(args.test_dir, args.target, "images"), transformer=transformer_train, k=2, dataset=args.dataset),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=True,
		drop_last=False,
		pin_memory=False
	)

	# load target dataset for testing, its label is used for testing until the whole training process ends.
	target_test_loader = DataLoader(
		OfficeDataSet(data_path=os.path.join(args.test_dir, args.target, "images"), transformer=transformer_test, k=1, dataset=args.dataset),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=False,
		pin_memory=False
	)

	fine_tuner = Moca_train(model, args=args)
	fine_tuner.test(keep_feature=True)

	# since every data size is different, this training epoch is designed for Office-31
	# however, a larger dataset is supposed to have a much smaller epoch
	for _ in trange(10):
		if _ <5:
			fine_tuner.finetune_on_source(epoches=5, save_name=pretrain_model_path, keep_feature=True, stage="one")
			fine_tuner.finetune_on_target(epoches=5, save_name=pretrain_model_path, keep_feature=True, reverse=True, stage="one")
		else:
			fine_tuner.finetune_on_source(epoches=5, save_name=pretrain_model_path, keep_feature=False, stage="two")
			fine_tuner.finetune_on_target(epoches=10, save_name=pretrain_model_path, keep_feature=True, reverse=False, stage="two")

	logger.info("\n++++Source:{} to target {} finish!++++".format(args.source, args.target))
	
	