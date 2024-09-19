from functools import reduce
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderLoss(nn.Module):
    def __init__(self, args):
        super(AutoencoderLoss, self).__init__()
        self.loss_dict = {}

    def forward(self, output_dict, label_dict):
        """
        Compute loss for autoencoder a41
        Parameters
        ----------
        output_dict : dict
           The dictionary that contains the output.

        target_dict : dict
           The dictionary that contains the target.

        Returns
        -------
        total_loss : torch.Tensor
            Total loss.

        """
        fused_feature_org = output_dict["fused_feature_org"]
        fused_feature_rec = output_dict["fused_feature_rec"]
        loss = nn.MSELoss()
        loss = loss(fused_feature_org, fused_feature_rec)
        self.loss_dict.update({'loss': loss,
                              })

        return loss

    def logging(self, epoch, batch_id, batch_len, writer = None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        loss = self.loss_dict['loss']
        # fusion_sum = self.loss_dict['fusion_sum']


        print("[epoch %d][%d/%d], || Loss: %.4f " % (
                  epoch, batch_id + 1, batch_len,
                  loss))
                  
        if not writer is None:
            writer.add_scalar('loss', loss,
                            epoch * batch_len + batch_id)
    