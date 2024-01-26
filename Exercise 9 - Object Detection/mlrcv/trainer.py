import pytorch_lightning as pl
import torch
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import get_cmap
import numpy as np
from PIL import ImageDraw

class CenterNetTrainer(pl.LightningModule):
    def __init__(self, model, criterion, loss_weights, train_loader, val_loader, lr, epochs):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.writer = SummaryWriter(f'center_logs/centernet_{len(os.listdir("center_logs"))}')
        self.train_step = 0
        self.val_step = 0

    def forward(self, x):
        pred_htm, pred_szm = self.model(x)
        pred_htm = torch.sigmoid(pred_htm).clamp(1e-7, 1.-1e-7)

        return pred_htm, pred_szm

    def draw_bb(self, img, centers, sizes, gt):
        # szm is permuted, img not
        scale_x = img.shape[1] / sizes.shape[0]
        scale_y = img.shape[2] / sizes.shape[1]

        img_pil = torchvision.transforms.functional.to_pil_image(img)
        img_bb = ImageDraw.Draw(img_pil)

        center = centers[0]

        size = sizes[center[0], center[1]]
        pt_1 = (scale_y * (center[1] - int(size[1] / 2)), scale_x * (center[0] - int(size[0] / 2)))
        pt_2 = (scale_y * (center[1] + int(size[1] / 2)), scale_x * (center[0] + int(size[0] / 2)))

        img_bb.rectangle((pt_1, pt_2), outline=(0,255,0) if gt else (255,0,0), width=3)
        bb_type = 'GT' if gt else 'pred'
        img_bb.text((pt_1),f"{bb_type} h:{size[0]} w:{size[1]}", (0,255,0) if gt else (255,0,0))

        img = torch.from_numpy(np.asarray(img_pil).astype(np.float32).copy())

        return img.permute(2,0,1) / 255.

    def compute_miou(self, htm_gt, htm_pred, szm_gt, szm_pred):
        htm_gt = htm_gt.permute(0,2,3,1)
        htm_pred = htm_pred.permute(0,2,3,1)
        szm_gt = szm_gt.permute(0,2,3,1)
        szm_pred = szm_pred.permute(0,2,3,1)

        miou = 0.

        for batch in range(htm_gt.shape[0]):
            gt_center = torch.nonzero(htm_gt[batch][...,0] == htm_gt[batch][...,0].max())[0]
            gt_size = szm_gt[batch, gt_center[0], gt_center[1]]
            gt_bb = {
                    'miny': gt_center[0] - gt_size[0] / 2,
                    'maxy': gt_center[0] + gt_size[0] / 2,
                    'minx': gt_center[1] - gt_size[1] / 2,
                    'maxx': gt_center[1] + gt_size[1] / 2,
                }

            pred_center = torch.nonzero(htm_pred[batch][...,0] == htm_pred[batch][...,0].max())[0]
            pred_size = szm_pred[batch, pred_center[0], pred_center[1]]
            pred_bb = {
                    'miny': pred_center[0] - pred_size[0] / 2,
                    'maxy': pred_center[0] + pred_size[0] / 2,
                    'minx': pred_center[1] - pred_size[1] / 2,
                    'maxx': pred_center[1] + pred_size[1] / 2,
                }
            
            inter_minx = max(gt_bb['minx'], pred_bb['minx'])
            inter_miny = max(gt_bb['miny'], pred_bb['miny'])
            inter_maxx = min(gt_bb['maxx'], pred_bb['maxx'])
            inter_maxy = min(gt_bb['maxy'], pred_bb['maxy'])

            inter_area = max(0, inter_maxx - inter_minx + 1) * max(0, inter_maxy - inter_miny + 1)

            gt_area = (gt_bb['maxx'] - gt_bb['minx'] + 1) * (gt_bb['maxy'] - gt_bb['miny'] + 1)
            pred_area = (pred_bb['maxx'] - pred_bb['minx'] + 1) * (pred_bb['maxy'] - pred_bb['miny'] + 1)

            miou += inter_area / float(gt_area + pred_area - inter_area)

        return miou / htm_gt.shape[0]


    def logging(self, train, report):
        if (not (self.train_step % 50) and train) or not train:
            # loss
            summary_scalar = 'train/loss' if train else 'val/loss'
            self.writer.add_scalar(summary_scalar, report['loss'], self.train_step if train else self.val_step)

            # heatmap loss
            summary_scalar = 'train/loss_heatmap' if train else 'val/loss_heatmap'
            self.writer.add_scalar(summary_scalar, report['loss_htm'], self.train_step if train else self.val_step)

            # object size loss
            summary_scalar = 'train/loss_size' if train else 'val/loss_size'
            self.writer.add_scalar(summary_scalar, report['loss_szm'], self.train_step if train else self.val_step)

            # mIoU
            summary_scalar = 'train/miou' if train else 'val/miou'
            miou = self.compute_miou(report['htm_gt'], report['htm_pred'], report['szm_gt'], report['szm_pred'])
            self.writer.add_scalar(summary_scalar, miou, self.train_step if train else self.val_step)

        if (not (self.train_step % 500) and train) or not train:
            # draw raw img
            summary_img = 'train/zimg' if train else 'val/zimg'
            img = (report['img'][0] + torch.abs(report['img'][0].min()))
            img = img /(img.max() - img.min())
            self.writer.add_image(summary_img, img.clip(0,1), self.train_step if train else self.val_step)

            # draw img bb gt
            summary_img = 'train/zimg_bb' if train else 'val/zimg_bb'
            img = (report['img'][0] + torch.abs(report['img'][0].min()))
            img = img /(img.max() - img.min())

            # gt centers and sizes
            szm = report['szm_gt'][0]
            szm = szm.permute(1,2,0)
            htm = report['htm_gt'][0]
            htm = htm.permute(1,2,0)
            centers = torch.nonzero(htm[...,0] == htm[...,0].max())
            img = self.draw_bb(img, centers, szm, True)

            # pred centers and sizes
            szm = report['szm_pred'][0]
            szm = szm.permute(1,2,0)
            htm = report['htm_pred'][0]
            htm = htm.permute(1,2,0)
            centers = torch.nonzero(htm[...,0] == htm[...,0].max())
            img = self.draw_bb(img, centers, szm, False)

            self.writer.add_image(summary_img, img.clip(0,1), self.train_step if train else self.val_step)

            # draw htm_gt
            summary_gt = 'train/htm_gt' if train else 'val/htm_gt'
            cm = get_cmap('viridis')
            htm = torchvision.transforms.functional.to_pil_image(report['htm_gt'][0])
            htm = np.asarray(cm(htm))[...,:3].astype(np.float32)
            htm = torch.from_numpy(htm)
            htm = htm.permute(2,0,1)
            self.writer.add_image(summary_gt, htm.clip(0,1), self.train_step if train else self.val_step)

            # draw szm_gt
            summary_gt = 'train/szm_gt' if train else 'val/szm_gt'
            szm = report['szm_gt'][0]
            pad = torch.zeros((1, szm.shape[1], szm.shape[2])).cuda()
            szm = torch.cat((szm, pad), dim=0)
            self.writer.add_image(summary_gt, szm.clip(0,1), self.train_step if train else self.val_step)

            # draw htm_pred
            summary_pred = 'train/htm_pred' if train else 'val/htm_pred'
            cm = get_cmap('viridis')
            htm = torchvision.transforms.functional.to_pil_image(report['htm_pred'][0])
            htm = np.asarray(cm(htm))[...,:3].astype(np.float32)
            htm = torch.from_numpy(htm)
            htm = htm.permute(2,0,1)
            self.writer.add_image(summary_pred, htm.clip(0,1), self.train_step if train else self.val_step)

            # draw szm_pred
            summary_pred = 'train/szm_pred' if train else 'val/szm_pred'
            szm = report['szm_pred'][0]
            pad = torch.zeros((1, szm.shape[1], szm.shape[2])).cuda()
            szm = torch.cat((szm, pad), dim=0)
            self.writer.add_image(summary_pred, szm.clip(0,1), self.train_step if train else self.val_step)

        if train:
            self.train_step += 1
        else:
            self.val_step += 1

    def report_dict(self, loss, loss_htm, loss_szm, img, htm, szm, pred_htm, pred_szm):
        return {
            'loss': loss,
            'loss_htm': loss_htm,
            'loss_szm': loss_szm,
            'img': img,
            'htm_gt': htm,
            'szm_gt': szm,
            'htm_pred': pred_htm,
            'szm_pred': pred_szm,
        }

    def training_step(self, batch, batch_idx):
        img, htm, szm, szm_mask = batch
        img, htm, szm = img.cuda(), htm.cuda(), szm.cuda()

        pred_htm, pred_szm = self.forward(img)
        loss, ht_loss, sz_loss = self.criterion(pred_htm, pred_szm, htm, szm, szm_mask, self.loss_weights)

        self.logging(True, self.report_dict(loss, ht_loss, sz_loss, img, htm, szm, pred_htm, pred_szm))

        return loss

    def validation_step(self, batch, batch_idx):
        img, htm, szm, szm_mask = batch
        img, htm, szm = img.cuda(), htm.cuda(), szm.cuda()

        pred_htm, pred_szm = self.forward(img)
        loss, ht_loss, sz_loss = self.criterion(pred_htm, pred_szm, htm, szm, szm_mask, self.loss_weights)

        self.logging(False, self.report_dict(loss, ht_loss, sz_loss, img, htm, szm, pred_htm, pred_szm))

        return loss

    def validation_epoch_end(self, outputs):
        if self.current_epoch % 150 or self.current_epoch == self.epochs - 1:
            torch.save(self.model.state_dict(), f'checkpoint/centernet_weights_epoch{self.current_epoch+1}.pt')

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
