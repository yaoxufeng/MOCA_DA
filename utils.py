# coding:utf-8

'''
some useful function
'''

import logging
import torch
import torch.nn as nn
import numpy as np
import os

from torch.autograd import Function

logger = logging.getLogger('root')


def get_source_class_weights(source_loader, num_classes, weight_file):
	logger.info("Getting class weights on source dataset...")
	class_weights = torch.zeros(num_classes)
	if os.path.exists(weight_file):
		with open(weight_file, 'r') as f:
			contents = f.read().splitlines()
		assert len(contents) == len(class_weights), "Length of class-weight tensor does not match the saved values."
		for i in range(len(contents)):
			class_weights[i] = float(contents[i].split(":")[1])
	else:
		num_samples = 0.0
		iter_source = iter(source_loader)
		for i in range(1, len(source_loader)):
			data_source, label_source = iter_source.next()
			num_samples += label_source.shape[0]
			for j in label_source:
				class_weights[j] += 1
		class_weights = class_weights / num_samples
		with open(weight_file, 'w') as f:
			for i in range(len(class_weights)):
				f.write("Class " + str(i) + ": " + str(class_weights[i].item()) + '\n')
	
	# for idx, val in enumerate(class_weights):
	#     logger.info("Weight of class {} in source: {}".format(idx, val))
	return class_weights


class Targeted_Dropout_Function(Function):
	'''
	Targeted Dropout from https://arxiv.org/abs/1905.13678
	we assume the block's arc is like conv->bn->relu (the most classical one)
	so we apply it after bn for channel pruning preparation
	'''
	
	@staticmethod
	def forward(ctx, input, target_index, dropout_rate, train_mode):
		"""
		:param input: bn layer
		:param targeted_index: the index of the targeted bn layer with lower weights magnitude
		:param dropout_rate: dropout rate for mask layer and default 0.5
		:return:
		"""
		
		mask = torch.ones(input.size(1)).cuda()  # init the mask
		if train_mode:
			if target_index is not None:
				target_index = target_index.cuda()
				target_prob = torch.FloatTensor(target_index.size(0)).uniform_(0,
				                                                               1).cuda()  # generate the uniform distribution to choose the drop index
				target_index = target_index[(target_prob < dropout_rate)].to(
					torch.long).cuda()  # get the index of the targeted bn layer to drop
				mask = mask.scatter_(0, target_index,
				                     0).cuda()  # get the mask of the input which is supposed to be equal to the size the channel of the input
			else:
				mask_prob = torch.FloatTensor(mask.size(0)).uniform_(0, 1).cuda()
				mask[mask_prob < dropout_rate] = 0.
			mask = mask.reshape(1, mask.size(0), 1, 1).cuda()  # reshape mask to do matrix multiplication
			
			# use scale rate to recover the capacity of the network
			if target_index is not None:
				keep_rate = 1. - target_index.size(0) / input.size(
					1)  # target_index is the index after dropout (total_index = remain_index)
			else:
				keep_rate = 1. - dropout_rate
			if keep_rate > 0.0:
				scale_rate = 1. / keep_rate
			else:
				scale_rate = 0.
			mask *= scale_rate
			output = mask * input
			ctx.save_for_backward(input, mask)
		else:
			output = input
		
		return output
	
	@staticmethod
	def backward(ctx, grad_output):
		"""
		:param grad_output: gradients of the last layers
		:return:
		"""
		input, mask = ctx.saved_tensors  # get the dropout value
		if mask is not None:
			grad_output *= mask
		return grad_output, None, None, None  # keep the scale of the output


class Targeted_Dropout(nn.Module):
	"""
	custom targeted dropout
	"""
	
	def __init__(self, targeted_index=None, dropout=0.5, inplace=True):
		super(Targeted_Dropout, self).__init__()
		self.dropout = dropout
		self.inplace = inplace
		self.targeted_index = targeted_index
	
	def forward(self, input):
		target_dropout = Targeted_Dropout_Function.apply
		return target_dropout(input, self.targeted_index, self.dropout, self.training)


def print_model_parm_nums(model):
	total = sum([param.nelement() for param in model.parameters()])
	return total


def print_model_parm_flops(model, list_mask_sparsity=None):
	prods = {}
	
	def save_hook(name):
		def hook_per(self, input, output):
			prods[name] = np.prod(input[0].shape)
		
		# prods.append(np.prod(input[0].shape))
		return hook_per
	
	list_1 = []
	
	def simple_hook(self, input, output):
		list_1.append(np.prod(input[0].shape))
	
	list_2 = {}
	
	def simple_hook2(self, input, output):
		list_2['names'] = np.prod(input[0].shape)
	
	multiply_adds = False
	list_conv = []
	
	def conv_hook(self, input, output):
		batch_size, input_channels, input_height, input_width = input[0].size()
		output_channels, output_height, output_width = output[0].size()
		
		kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
			2 if multiply_adds else 1)
		bias_ops = 1 if self.bias is not None else 0
		
		params = output_channels * (kernel_ops + bias_ops)
		flops = batch_size * params * output_height * output_width
		
		list_conv.append(flops)
	
	list_linear = []
	
	def linear_hook(self, input, output):
		batch_size = input[0].size(0) if input[0].dim() == 2 else 1
		weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
		bias_ops = self.bias.nelement()
		
		flops = batch_size * (weight_ops + bias_ops)
		list_linear.append(flops)
	
	list_bn = []
	
	def bn_hook(self, input, output):
		list_bn.append(input[0].nelement())
	
	list_relu = []
	
	def relu_hook(self, input, output):
		list_relu.append(input[0].nelement())
	
	list_pooling = []
	
	def pooling_hook(self, input, output):
		batch_size, input_channels, input_height, input_width = input[0].size()
		output_channels, output_height, output_width = output[0].size()
		
		kernel_ops = self.kernel_size * self.kernel_size
		bias_ops = 0
		params = output_channels * (kernel_ops + bias_ops)
		flops = batch_size * params * output_height * output_width
		list_pooling.append(flops)
	
	def foo(net):
		childrens = list(net.children())
		if not childrens:
			if isinstance(net, torch.nn.Conv2d):
				# net.register_forward_hook(save_hook(net.__class__.__name__))
				# net.register_forward_hook(simple_hook)
				# net.register_forward_hook(simple_hook2)
				net.register_forward_hook(conv_hook)
			if isinstance(net, torch.nn.Linear):
				net.register_forward_hook(linear_hook)
			if isinstance(net, torch.nn.BatchNorm2d):
				net.register_forward_hook(bn_hook)
			if isinstance(net, torch.nn.ReLU):
				net.register_forward_hook(relu_hook)
			if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
				net.register_forward_hook(pooling_hook)
			return
		for c in childrens:
			foo(c)
	
	foo(model)
	input_ = torch.rand(3, 224, 224).unsqueeze(0)
	model.eval()
	out = model(input_, input_)
	
	if list_mask_sparsity is None or len(list_mask_sparsity) < 1:
		total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
	else:
		assert len(list_conv) == len(list_mask_sparsity)
		total_flops = 0
		list_pooling_idx = [2, 5, 9, 13, 17]
		for i in range(len(list_conv)):
			total_flops += (list_conv[i] + list_bn[i] + list_relu[i]) * (1 - list_mask_sparsity[i])
			if i + 1 in list_pooling_idx:
				pos = list_pooling_idx.index(i + 1)
				total_flops += list_pooling[pos] * list_mask_sparsity[i]
		total_flops += sum(list_linear)
	
	# print(' + Number of FLOPs: %.2fG' % (total_flops / 1e9))
	return total_flops


# ========================= Entropy Minimization loss ==========================
def EntropyMinLoss(pred):
	"""
	:param pred: pred value of target
	:return:
	"""
	mask = pred.ge(1e-6)  # get rid of the trivial class
	mask_out = torch.masked_select(pred, mask)
	entropy_loss = -(torch.sum(mask_out * torch.log(mask_out)))
	
	return entropy_loss / pred.size(0)


# ========================= AM loss ==========================
def AMLoss(pred, pred_label, s=32, m=-0.1, num_class=256):
	costheta = torch.clamp(pred, -1, 1).cpu()  # for numerial steady
	phi = costheta - m
	pred_label = pred_label.cpu()
	pred_label_onehot = torch.zeros(pred_label.size(0), num_class).scatter_(1, pred_label.view(-1, 1), 1)
	adjust_theta = s * torch.where(torch.eq(pred_label_onehot, 1), phi, costheta)
	adjust_theta = adjust_theta.cuda()
	return adjust_theta


if __name__ == "__main__":
	# test targeted_dropout
	input = torch.randn(1, 16, 1, 1)
	target_index = torch.FloatTensor([i for i in range(16)])
	# # target_index = None
	dropout_rate = torch.FloatTensor([.5])
	input = input.cuda()
	target_dropout = Targeted_Dropout(targeted_index=target_index)
	# target_dropout.eval()
	output = target_dropout(input)
	print(output)
# print(input)

