from torchvision import datasets, transforms
import torch
import os


def load_training(root_path, dir, batch_size, kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir, "images"), transform=transform)
    # data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)  # change it to fit the own dataset
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader


def load_testing(root_path, dir, batch_size, kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        # [transforms.Resize([224, 224]),
        [transforms.Resize([256, 256]),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir, "images"), transform=transform)
    # data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)  # change it to fit the own dataset
    test_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, **kwargs)
    return test_loader