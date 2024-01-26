import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def get_dataloader(batch_size, dataset):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = dataset("./data", train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = dataset(root='./data', train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def imshow(img, text=None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if text is not None:
        plt.title(text)
    plt.show()