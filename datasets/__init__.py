import torch
import torchvision
from .random_dataset import RandomDataset, GetLoader


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(data_dir, split='train+unlabeled' if train else 'test', transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(data_dir, split='train' if train == True else 'val', transform=transform, download=download)
    elif dataset == 'random':
        dataset = RandomDataset()
    elif dataset == 'logo2k':
        if train:
            dataset = GetLoader(data_root='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/Logo-2K+',
                                data_list='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/train.txt',
                                label_dict='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/logo2k_labeldict.pkl',
                                transform=transform)
        else:
            dataset = GetLoader(data_root='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/Logo-2K+',
                                data_list='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/test.txt',
                                label_dict='/home/l/liny/ruofan/PhishIntention/src/siamese_retrain/logo2k/logo2k_labeldict.pkl',
                                transform=transform)
            
    elif dataset == 'targetlist':
        if train:
            dataset = GetLoader(data_root='/home/l/liny/ruofan/PhishIntention/src/phishpedia/expand_targetlist',
                            data_list='/home/l/liny/ruofan/PhishIntention/src/phishpedia/train_targets.txt',
                            label_dict='/home/l/liny/ruofan/PhishIntention/src/phishpedia/target_dict.json',
                            transform=transform)
        else:
            dataset = GetLoader(data_root='/home/l/liny/ruofan/PhishIntention/src/phishpedia/expand_targetlist',
                            data_list='/home/l/liny/ruofan/PhishIntention/src/phishpedia/test_targets.txt',
                            label_dict='/home/l/liny/ruofan/PhishIntention/src/phishpedia/target_dict.json',
                            transform=transform)

    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        dataset.classes = dataset.dataset.classes
        dataset.targets = dataset.dataset.targets
    return dataset