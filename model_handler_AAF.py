import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import utils
from torchmetrics import MetricCollection, Dice, ConfusionMatrix
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from einops import rearrange

from pytorch_lightning import seed_everything

torch.cuda.empty_cache()
seed_everything(42, workers=True)
AVAIL_GPUS = min(1, torch.cuda.device_count())

import json
with open('/home/veronica/Scrivania/RSIm/Fusion/Method_1/config.json', 'r') as f:
  config = json.load(f)

class ModelHandler(pl.LightningModule):
  
    """
  
    """

    def __init__(self,
                 model, 
                 criterion, 
                 params: dict,
                 scheduler: bool = True):
        
        super(ModelHandler, self).__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        self.criterion = criterion
        self.config = params
        self.scheduler = scheduler
        self.num_classes = self.config["num_classes"]
        self.lamda = self.config["lamda"]

        metrics = MetricCollection({
                            "iou_macro": MulticlassJaccardIndex(self.num_classes, ignore_index=0, average="macro"),
                            "iou_micro": MulticlassJaccardIndex(self.num_classes, ignore_index=0, average="micro"),
                            "accuracy_macro": MulticlassAccuracy(self.num_classes, ignore_index=0, average="macro"),
                            "accuracy_micro": MulticlassAccuracy(self.num_classes, ignore_index=0, average="micro"),
                            "precision_macro": MulticlassPrecision(self.num_classes, ignore_index=0, average="macro", top_k=1),
                            "precision_micro": MulticlassPrecision(self.num_classes, ignore_index=0, average="micro", top_k=1),
                            "recall_macro": MulticlassRecall(self.num_classes, ignore_index=0, average="macro", top_k=1),
                            "recall_micro": MulticlassRecall(self.num_classes, ignore_index=0, average="micro", top_k=1),
                            "f1_macro": MulticlassF1Score(self.num_classes, ignore_index=0, average="macro"),
                            "f1_micro": MulticlassF1Score(self.num_classes, ignore_index=0, average="micro"),
                            "dice_macro": Dice(task="multiclass", num_classes=self.num_classes, ignore_index=0, average="macro"),
                            "dice_micro": Dice(task="multiclass", num_classes=self.num_classes, ignore_index=0, average="micro")}) # prefix: str (metterlo per il path dove salvare le cose)
        
        self.train_metrics = metrics.clone(prefix = 'train/')
        self.valid_metrics = metrics.clone(prefix = 'val/')
        self.test_metrics = metrics.clone(prefix = 'test/')

        self.confmat = ConfusionMatrix(task='multiclass', ignore_index=0, num_classes=self.num_classes, normalize='true')

    def forward(self, x): 
         return self.model(x)

    def training_step(self, batch, batch_idx):

        (samples, targets, _) = batch

        #######################################################################################################################

        outputs, masks = self(samples)
        
        loss = 0

        # prima sommatoria
        for i in range(len(outputs)-1):
        # for output in outputs:
            loss += self.criterion(outputs[i], targets)
    
        # seconda sommatoria
        if self.lamda > 0:
            L1_loss = 0
            for mask in masks:
                L1_loss += sum([torch.abs(m).sum().cuda() for m in mask])
            loss += self.lamda * L1_loss

        #######################################################################################################################

        if batch_idx % 500 == 0:
            rgb, hs, dem = samples
            out_max = torch.argmax(outputs[2], dim=1)
            self.show_images(out_max, rgb, hs, dem, targets, step="im_train")

        self.train_metrics.update(outputs[2], targets)
        self.log("losses/train_loss", loss, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config["batch_size"])

        return loss
    

    def on_train_epoch_end(self):

        output = self.train_metrics.compute()
        self.log_dict(output, logger=True, batch_size=self.config["batch_size"])
        self.train_metrics.reset()

    
    def validation_step(self, batch, batch_idx):

        (samples, targets, _) = batch
        
        ##################################################################################

        outputs, masks = self(samples)
        
        loss = 0

        # prima sommatoria
        for i in range(len(outputs)-1):
        # for output in outputs:
            loss += self.criterion(outputs[i], targets)
    
        # seconda sommatoria
        if self.lamda > 0:
            L1_loss = 0
            for mask in masks:
                L1_loss += sum([torch.abs(m).sum().cuda() for m in mask])
            loss += self.lamda * L1_loss

        ###################################################################################
        
        self.valid_metrics.update(outputs[2], targets)
        self.log("losses/val_loss", loss, prog_bar=True, logger=True, on_epoch=True, batch_size=self.config["batch_size"])

        if batch_idx % 500 == 0:
            rgb, hs, dem = samples
            out_max = torch.argmax(outputs[2], dim=1)
            self.show_images(out_max, rgb, hs, dem, targets, step="im_val")

        self.confmat.update(outputs[2], targets)

    def on_validation_epoch_end(self):

        output = self.valid_metrics.compute()
        self.log_dict(output, logger=True, batch_size=self.config["batch_size"])
        self.valid_metrics.reset()

        confusion_mat =  self.confmat.compute()
        cm_out = self.save_confmat(confusion_mat, step='val')
        self.logger.experiment.add_figure("confusion matrix/val", cm_out)


    def test_step(self, batch, batch_idx):

        (samples, targets, im_name) = batch

        outputs, masks = self(samples) 
        self.test_metrics.update(outputs[2], targets)

        # show images
        if batch_idx %2 == 1:
            rgb, hs, dem = samples
            out_max = torch.argmax(outputs[2], dim=1)
            self.show_images(out_max, rgb, hs, dem, targets, step="im_test")

        self.confmat.update(outputs[2], targets)
    
    def on_test_epoch_end(self):
        
        output = self.test_metrics.compute()
        self.log_dict(output, logger=True)
        self.test_metrics.reset()

        confusion_mat =  self.confmat.compute()
        cm_out = self.save_confmat(confusion_mat, step='test')
        self.logger.experiment.add_figure("confusion matrix/test", cm_out)
    

    def configure_optimizers(self):

        params = [
            {'params': self.model.parameters()}
        ]
        optimizer = torch.optim.Adam(params,
                                     weight_decay = self.config["weight_decay"],
                                     lr = self.config["learning_rate"])
        if self.scheduler is not None:
            scheduler = {"scheduler": ReduceLROnPlateau(optimizer, patience=10), "monitor": "losses/val_loss"}
        return ([optimizer], [scheduler])


    def color_coding(self, semlay, n_classes, cmap='Paired'):
        
        cmap = plt.get_cmap('Paired') # get colormap
        im_out = cmap((semlay.cpu().float()/n_classes).numpy()) # apply colormap
        im_out = rearrange(torch.from_numpy(im_out), 'b h w c -> b c h w') # fix dims
        
        return im_out

    def show_images(self, out, rgb, hs, dem, gt, step:str): 

        # move to cpu and detach
        out = out.cpu().detach()
        rgb = rgb.cpu().detach()
        hs = hs.cpu().detach()
        dem = dem.cpu().detach()
        gt = gt.cpu().detach()
    
        # create masked outputs
        out_masked = torch.zeros_like(out)

        # for each image, remove content in background
        # for i in range(out.shape[0]):
        cur_masked = out.clone()
        cur_gt = gt
        cur_masked[cur_gt==0] = 0
        out_masked = cur_masked

        # color coding
        out = self.color_coding(out, self.config['num_classes'])
        out_masked = self.color_coding(out_masked, self.config['num_classes'])
        gt = self.color_coding(gt, self.config['num_classes'])
        # normalize other images
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        dem = (dem - dem.min()) / (dem.max() - dem.min())
        # sanitize images with nan
        out = torch.nan_to_num(out, nan=0.0)
        out_masked = torch.nan_to_num(out_masked, nan=0.0)
        # clamp
        out_masked = torch.clamp(out_masked, min=0, max=1)
        out = torch.clamp(out, min=0, max=1)
        rgb = torch.clamp(rgb, min=0, max=1)
        hs = torch.clamp(hs, min=0, max=1)
        dem = torch.clamp(dem, min=0, max=1)
        # display semantic layouts
        self.logger.experiment.add_image(step+'/out_masked', utils.make_grid(out_masked), self.global_step)
        self.logger.experiment.add_image(step+'/out', utils.make_grid(out), self.global_step)
        self.logger.experiment.add_image(step+'/gt', utils.make_grid(gt), self.global_step)
        # display input images
        self.logger.experiment.add_image(step+'/rgb', utils.make_grid(rgb), self.global_step)
        self.logger.experiment.add_image(step+'/hs', utils.make_grid(hs.mean(1).unsqueeze(1)), self.global_step)
        self.logger.experiment.add_image(step+'/dem', utils.make_grid(dem), self.global_step)
        
    def save_confmat(self, cm, step:str='train'):

        res_mat = ''
        for row_el in range(cm.shape[0]):
            for col_el in range(cm.shape[1]):
                res_mat += str(cm[row_el, col_el].item()) +  " "
            res_mat += "\n"

        # self._save_list_results("./confusion_matrix.txt", (res_mat,)) # list is to not create a different save_list function

        labels = ['Background', 'Building', 'Roads', 'Residential', 'Industrial', 'Forest', 'Farmland', 'Water']
        df_cm = pd.DataFrame(cm.cpu().detach().numpy(), index = range(8), columns=range(8))
        plt.figure(figsize = (10,7))
        fig = sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='.2f', xticklabels=labels, yticklabels=labels).get_figure()
        plt.tight_layout()
        plt.title('Heatmap token Fusion Patch Embedding (Exp. 1)', fontsize = 20, fontweight="bold")
        plt.savefig('./confusion_matrix'+step+'.png', dpi=200)
        plt.close(fig)
        return fig
    
    def confusion_matrix(x, y, n, ignore_label=None, mask=None):
        if mask is None:
            mask = np.ones_like(x) == 1
        k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool))
        return np.bincount(n * x[k].astype(int) + y[k], minlength=n ** 2).reshape(n, n)


    def getScores(conf_matrix):
        if conf_matrix.sum() == 0:
            return 0, 0, 0
        with np.errstate(divide='ignore',invalid='ignore'):
            overall = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
            perclass = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
            IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) \
                - np.diag(conf_matrix)).astype(np.float)
        return overall * 100., np.nanmean(perclass) * 100., np.nanmean(IU) * 100.