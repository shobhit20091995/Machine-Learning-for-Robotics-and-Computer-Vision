import pytorch_lightning as pl
import torch
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import get_cmap
import numpy as np
from PIL import ImageDraw

def freeze_net_params(model):
    for param in model.parameters():
        param.requires_grad = False

    return model

class ClassificationTrainer(pl.LightningModule):
    def __init__(self, model, criterion, train_loader, val_loader, class_to_lbl, lr, epochs, log_name):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.class_to_lbl = class_to_lbl
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.writer = SummaryWriter(f'train_logs/{log_name}')
        self.train_step = 0
        self.val_step = 0

    def forward(self, x):
        pred = self.model(x)
        
        return pred
    
    def logging(self, train, report):
        if (not (self.train_step % 50) and train) or not train:
            # loss
            summary_scalar = 'train/loss' if train else 'val/loss'
            self.writer.add_scalar(summary_scalar, report['loss'], self.train_step if train else self.val_step)

            # acc
            summary_scalar = 'train/acc' if train else 'val/acc'
            self.writer.add_scalar(summary_scalar, report['acc'], self.train_step if train else self.val_step)

            summary_img = 'train/zimg' if train else 'val/zimg'
            
            for _ in range(2):
                i = np.random.randint(report['img'].shape[0])
                img = (report['img'][i] + torch.abs(report['img'][i].min()))
                img = img / (img.max() - img.min())
                img_pil = torchvision.transforms.functional.to_pil_image(img)
                img_txt = ImageDraw.Draw(img_pil)
                gt_lbl = self.class_to_lbl[report['gt'][i].item()]
                pred_lbl = self.class_to_lbl[report['pred'][i].item()]
                img_txt.text((5,5),f"GT: {gt_lbl}  PRED: {pred_lbl}", (0,255,0))
                img = torch.from_numpy(np.asarray(img_pil).astype(np.float32).copy())
                img = img.permute(2,0,1) / 255.
                self.writer.add_image(summary_img, img.clip(0,1), self.train_step if train else self.val_step)

                if train:
                    self.train_step += 1
                else:
                    self.val_step += 1

        if train:
            self.train_step += 1
        else:
            self.val_step += 1

    def report_dict(self, loss, acc, img_pred):
        return {
            'loss': loss,
            'acc': acc,
            'img': img_pred[0],
            'gt': img_pred[1],
            'pred': img_pred[2]
        }

    def training_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']
        #x, y = x.cuda(), y.cuda()

        pred = self.forward(x)
        loss = self.criterion(pred, y)

        pred_labels = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_labels == y) / y.shape[0]

        self.logging(True, self.report_dict(loss, acc, (x,y,pred_labels)))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']
        #x, y = x.cuda(), y.cuda()
       
        pred = self.forward(x)
        loss = self.criterion(pred, y)

        pred_labels = torch.argmax(pred, dim=1)
        acc = torch.sum(pred_labels == y) / y.shape[0]

        self.logging(False, self.report_dict(loss, acc, (x,y,pred_labels)))

        return loss

    def validation_epoch_end(self, outputs):
        if self.current_epoch % 150 or self.current_epoch == self.epochs - 1:
            torch.save(self.model.state_dict(), f'checkpoint/classification_weights_epoch{self.current_epoch+1}.pt')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optimizer

        return optimizer

    #@pl.data_loader
    def train_dataloader(self):
        return self.train_loader

    #@pl.data_loader
    def val_dataloader(self):
        return self.val_loader

    #@pl.data_loader
    def test_dataloader(self):
        pass
