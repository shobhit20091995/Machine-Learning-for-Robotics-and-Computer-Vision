import torch
import mlrcv.network
from torch.utils.data.dataloader import DataLoader
import torchvision

import matplotlib.pyplot as plt
from mlrcv.core import *
from typing import Optional

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Trainer:
    def __init__(self, epochs: int, lr: float, net: mlrcv.network.Net, train_loader: DataLoader, val_loader: DataLoader):
        """
        This function initializes the Trainer class and defines optimizers and train/val data:

        Args:
            - epochs (int): number of training epochs
            - lr (float): learning rate used for training
            - net (Net): network architecture model
            - train_loader (DataLoader): training dataloader
            - val_loader (DataLoader): validation dataloader

        Returns:
        """


        self.lr = lr
        self.epochs = epochs
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_loss = []
        self.val_loss = []
        self.log_acc = []
        self.val_acc = []

    def plot_loss_acc(self, epoch: int):
        """
        This function plot the logged losses:

        Args:
            - epoch (int): current epoch

        Returns:
        """
        fig, axs = plt.subplots(2, 2 ,constrained_layout = True)

        x_axis = list(range(len(self.log_loss)))

        axs[0,0].set_title(f'Training loss at epoch {epoch}')
        axs[0,0].set(xlabel='log_steps', ylabel='loss')
        axs[0,0].set_xticks(x_axis)
        axs[0,0].plot(x_axis, self.log_loss, color='g', marker='*')

        
        axs[0,1].set_title(f'Training acc at epoch {epoch}')
        axs[0,1].set(xlabel='log_steps', ylabel='acc')
        axs[0,1].set_xticks(x_axis)
        axs[0,1].set_yticks(list(range(0,100,10)))
        axs[0,1].plot(x_axis, self.log_acc, color='r', marker='o')

        x_axis = list(range(epoch))

        axs[1,0].set_title(f'Validation loss at epoch {epoch}')
        axs[1,0].set(xlabel='epochs', ylabel='loss')
        axs[1,0].set_xticks(x_axis)
        axs[1,0].plot(x_axis, self.val_loss, color='g', marker='*')

        
        axs[1,1].set_title(f'Validation acc at epoch {epoch}')
        axs[1,1].set(xlabel='epochs', ylabel='acc')
        axs[1,1].set_xticks(x_axis)
        axs[1,1].set_yticks(list(range(0,100,5)))
        axs[1,1].plot(x_axis, self.val_acc, color='r', marker='o')

        plt.show()

    def train(self):
        """
        This function train the network over the train dataloader over the number of epochs

        Args:

        Returns:
        """
        dataiter = iter(self.train_loader)
        #images, labels = dataiter.next()
        images, labels = next(dataiter)

        # show images
        imshow(torchvision.utils.make_grid(images))

        self.net.train()
        for epoch in range(self.epochs):
            step_loss = 0.
            step_acc = 0.
            total = 0
            correct = 0
            for i, data in enumerate(self.train_loader,0):
                inputs, labels = data
                
                self.optimizer.zero_grad()
                outputs = self.net(inputs)

                # loss
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                step_loss += loss.item()

                # acc
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                step_acc += (100 * correct / total)

                if (i + 1) % 50 == 0:
                    print(f'[epoch: {epoch}, it: {i+1}] Loss: {step_loss / 50.}')
                    self.log_loss.append(step_loss / 50.)
                    self.log_acc.append(step_acc / 50.)
                    step_loss = 0.
                    step_acc = 0.
                    total = 0
                    correct = 0

            self.net.eval()
            val_loss, val_acc = self.validation()
            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            self.plot_loss_acc(epoch+1)
            self.net.train()

    def validation(self, show_img: Optional[bool] = False) -> (float, float):
        """
        This function calculates the network accuracy over the validation dataloader:

        Args:
            - show_img (bool): flag to show or not the image related to the predictions

        Returns:
            - loss (float): loss calculated over the validation set
            - acc (float): model accuracy on the validation set
        """
        self.net.eval()
        dataiter = iter(self.val_loader)
        #images, labels = dataiter.next()
        images, labels = next(dataiter)

        outputs = self.net(images)

        _, predicted = torch.max(outputs, 1)

        if show_img:
            for j in range(5):
                gt = labels[j].item()
                pred = predicted[j].item()

                imshow(torchvision.utils.make_grid(images[j]), f'GroundTruth: {classes[gt]} - Predicted: {classes[pred]}')

        correct = 0
        total = 0
        loss = 0.
        step = 0
        for data in self.val_loader:
            images, labels = data
            outputs = self.net(images)
            loss += self.criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            step += 1

        return loss / step, (100 * correct / total)

    def save_model(self):
        """
        This function saves the network weights:

        Args:

        Returns:
        """
        torch.save(self.net.state_dict(), 'model_weights.pt')

        