import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class PointNetLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0, reg_weight=0, size_average=True):
        super(PointNetLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reg_weight = reg_weight
        self.size_average = size_average

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, A):

        # get batch size
        bs = predictions.size(0)

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions, targets)

        # reformat predictions and targets
        predictions = predictions.transpose(1, 2) # (b, c, n) -> (b, n, c)
        predictions = predictions.contiguous() \
                                 .view(-1, predictions.size(2)) # (b, n, c) -> (b*n, c)

        # get log softmax of predictions
        log_pn = F.log_softmax(predictions)

        # get predictions at true class indexes
        log_pn = log_pn.gather(1, targets.view(-1,1)).view(-1)

        # get predicted probabilities
        pn = Variable(log_pn.data.exp())

        # get regularization term
        if reg > 0:
            I = torch.eye(64).unsqueeze(0).repeat(A.shape[0], 1, 1) # .to(device)
            reg = torch.linalg.norm(I - torch.bmm(A, A.transpose(2, 1)))
            reg = self.reg_weight*reg/bs
        else:
            reg = 0

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: return loss.mean() + reg
        else: return loss.sum() + reg