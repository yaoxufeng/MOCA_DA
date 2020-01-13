import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import models, transforms
import torch
import logging

logger = logging.getLogger('root')

__all__ = ['ResNet', 'resnet50']

model_urls = {
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class Bottleneck(nn.Module):
	expansion = 4
	
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
		                       padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		
		out = self.conv3(out)
		out = self.bn3(out)
		
		if self.downsample is not None:
			residual = self.downsample(x)
		
		out += residual
		out = self.relu(out)
		
		return out


class ResNetFeature(nn.Module):
	
	def __init__(self, block, layers):
		self.inplanes = 64
		super(ResNetFeature, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
	
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		
		return x


class ResNet(nn.Module):
	
	def __init__(self, num_classes=31, targeted_dropout=None):
		super(ResNet, self).__init__()
		self.input_mean = [0.485, 0.456, 0.406]
		self.input_std = [0.229, 0.224, 0.225]
		self.features = resnet50(False)
		
		self.metric_feature = nn.Sequential(
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 128),
			nn.Dropout(p=0.2)
		)

		# self.metric_feature = nn.Linear(2048, 128)
		self.cls_fc = nn.Linear(128, num_classes)
	
		# self.cls_fc = nn.Parameter(torch.FloatTensor(num_classes, 128))
		# nn.init.xavier_uniform_(self.cls_fc)

	def forward(self, source, target):
		# ========================= normalize fc ==========================
		with torch.no_grad():
			self.cls_fc.weight.div_(torch.norm(self.cls_fc.weight, dim=1, keepdim=True))
			self.cls_fc.bias.data.fill_(0.0)
		# ========================= FBI warning !!! =======================
		source_feature = self.features(source)
		source_feature = source_feature.view(source_feature.size(0), -1)
		source_feature = self.metric_feature(source_feature)
		# print("source_feature.size", source_feature.shape)
		# ========================= normalize feature ==========================
		source_feature = F.normalize(source_feature, p=2, dim=1)
		# ========================= FBI warning !!! ============================
		source_cls = self.cls_fc(source_feature)
		# source_cls = F.linear(source_feature, F.normalize(self.cls_fc))

		if self.training:
			target_feature = self.features(target)
			target_feature = target_feature.view(target_feature.size(0), -1)
			target_feature = self.metric_feature(target_feature)
			# ========================= normalize feature ==========================
			target_feature = F.normalize(target_feature, p=2, dim=1)
			# ========================= FBI warning !!! ============================
			target_cls = self.cls_fc(target_feature)
			# target_cls = F.linear(target_feature, F.normalize(self.cls_fc))
			return source_cls, target_cls, source_feature, target_feature
		else:
			return source_cls, source_feature

	# we use the data augmentation method used in Kaiming He's paper
	def train_augmentation(self):
		return transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.7, 1.2),
			                             interpolation=3),
			transforms.RandomHorizontalFlip(),
			transforms.RandomGrayscale(p=0.25),
			transforms.RandomApply([
				transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.input_mean, std=self.input_std)
		])
	
	def train_multi_augmentation(self):
		return transforms.Compose([
			transforms.Resize(256),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation([-30, 30]),
			transforms.ColorJitter(),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.input_mean, std=self.input_std)
		])
	
	def test_augmentation(self):
		return transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.input_mean, std=self.input_std)
		])


class ResNet_Target(nn.Module):
	
	def __init__(self, num_classes=31, targeted_dropout=None):
		super(ResNet, self).__init__()
		self.input_mean = [0.485, 0.456, 0.406]
		self.input_std = [0.229, 0.224, 0.225]
		self.features = resnet50(False)
		
		self.metric_feature = nn.Sequential(
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 128)
			# nn.Dropout(p=0.4)
		)
		
		# self.metric_feature = nn.Linear(2048, 128)
		self.cls_fc = nn.Linear(128, num_classes)
	
	# self.cls_fc = nn.Parameter(torch.FloatTensor(num_classes, 128))
	# nn.init.xavier_uniform_(self.cls_fc)
	
	def forward(self, source, target):
		# ========================= normalize fc ==========================
		with torch.no_grad():
			self.cls_fc.weight.div_(torch.norm(self.cls_fc.weight, dim=1, keepdim=True))
			self.cls_fc.bias.data.fill_(0.0)
		# ========================= FBI warning !!! =======================
		source_feature = self.features(source)
		source_feature = source_feature.view(source_feature.size(0), -1)
		source_feature = self.metric_feature(source_feature)
		# print("source_feature.size", source_feature.shape)
		# ========================= normalize feature ==========================
		source_feature = F.normalize(source_feature, p=2, dim=1)
		# ========================= FBI warning !!! ============================
		source_cls = self.cls_fc(source_feature)
		# source_cls = F.linear(source_feature, F.normalize(self.cls_fc))
		
		if self.training:
			target_feature = self.features(target)
			target_feature = target_feature.view(target_feature.size(0), -1)
			target_feature = self.metric_feature(target_feature)
			# ========================= normalize feature ==========================
			target_feature = F.normalize(target_feature, p=2, dim=1)
			# ========================= FBI warning !!! ============================
			target_cls = self.cls_fc(target_feature)
			# target_cls = F.linear(target_feature, F.normalize(self.cls_fc))
			return source_cls, target_cls, source_feature, target_feature
		else:
			return source_cls, source_feature


def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNetFeature(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model


# test the resnet

cuda = True


def load_imagenet_pretrain(model, base):
	if base == 'vgg16':
		url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
	elif base == 'vgg16_bn':
		url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
	elif base == 'resnet50':
		url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
	elif base == 'resnet101':
		url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
	else:
		pass
	
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
		pretrained_dict = model_zoo.load_url(url)
		model_dict = model.state_dict()
		for k, v in model_dict.items():
			if not "cls_fc" in k and not "num_batches_tracked" and not "metric_feature" in k:
				model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
	
	model.load_state_dict(model_dict)
	if cuda:
		model.cuda()
	return model


if __name__ == "__main__":
	model = ResNet()
	model = load_imagenet_pretrain(model, "resnet101")
	for layer, module in model.features._modules.items():
		print(module)
	print(model.metric_feature._modules)
	print(model.features._modules['layer1'][0]._modules['conv1'])
# print(model.feature.layer4[0].bn1.weight)
