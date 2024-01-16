
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


# Model Hyperparameters
dataset_path = "datasets"
cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


# make dataset

mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])

train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

torch.save(train_dataset, "datasets/train_dataset.pt")
torch.save(test_dataset, "datasets/test_dataset.pt")