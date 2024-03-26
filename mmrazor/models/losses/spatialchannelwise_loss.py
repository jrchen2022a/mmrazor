import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class SpatialChannelWiseDivergence(nn.Module):
    """


    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, tau=1.0, loss_weight=1.0):
        super(SpatialChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape

        channel_mask_T = torch.max(preds_T.view(N, C, -1), dim=2).values
        channel_mask_S = torch.max(preds_S.view(N, C, -1), dim=2).values
        spatial_mask_T = torch.max(preds_T, dim=1).values
        spatial_mask_S = torch.max(preds_S, dim=1).values
        softmax_channel_mask_T = F.softmax(channel_mask_T / self.tau, dim=1)
        softmax_spatial_mask_T = F.softmax(spatial_mask_T.view(-1, W * H) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss_c = torch.sum(softmax_channel_mask_T * logsoftmax(channel_mask_T / self.tau)
                           - softmax_channel_mask_T * logsoftmax(channel_mask_S / self.tau)) * (self.tau ** 2)
        loss_s = torch.sum(softmax_spatial_mask_T * logsoftmax(spatial_mask_T.view(-1, W * H) / self.tau)
                           - softmax_spatial_mask_T * logsoftmax(
                            spatial_mask_S.view(-1, W * H) / self.tau)) * (self.tau ** 2)
        loss = self.loss_weight * (loss_c + loss_s) / N

        return loss
