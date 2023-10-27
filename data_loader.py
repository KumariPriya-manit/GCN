import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt


def get_train_valid_loader(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    """Train and validation data loaders.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        random_seed: fix seed for reproducibility.
        valid_size: percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle: whether to shuffle the train/validation indices.
        show_sample: plot 9x9 sample grid of the dataset.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((512,512)),transforms.ToTensor(), normalize])

    # load dataset
    # dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)
    
    train_dataset=ImageFolder(root=data_dir+"/train", transform=trans)
    valid_dataset=ImageFolder(root=data_dir+"/validation",transform=trans)

    # dataset = ImageFolder(root=data_dir, transform=trans)
    
    num_train = len(train_dataset)
    indices1 = list(range(num_train))

    num_valid=len(valid_dataset)
    indices2=list(range(num_valid))

    

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices1)
        np.random.shuffle(indices2)

    train_idx = indices1
    valid_idx = indices2

    # print(train_idx,valid_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    data_iter = iter(train_loader)

    # Fetch the first batch of data
    images, labels = next(data_iter)

    # Select one image from the batch (e.g., the first image)
    image = images[0].squeeze().numpy()
    label = labels[0]

    
    # Display the image and its label
    # plt.imshow(image, cmap='gray')
    # plt.title(f'Label: {label}')
    # plt.show()
    # print(train_loader.classes)
    num_classes = len(train_dataset.classes)
    print("Number of classes in training:", num_classes)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader,num_classes)


def get_test_loader(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test datalaoder.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((512,512)),transforms.ToTensor(), normalize])

    
    test_dataset=ImageFoler(root=data_dir+"/test",transform=trans)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
