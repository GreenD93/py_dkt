import torch
import torch.nn as nn

from constants import *


class lossFunc(nn.Module):
    def __init__(self):
        super(lossFunc, self).__init__()

        self.loss_fn = nn.BCELoss()

    def forward(self, pred, batch):

        acc = 0

        # qt
        delta = batch[:,:,:QUESTION_NUM] + batch[:,:,QUESTION_NUM:]

        # qt+1 & transpose for matrix multiplication
        delta = delta[:,1:,:].permute(0,2,1)

        # yt
        y = pred[:, :MAX_SEQ - 1,:]

        # pred at+1
        temp = torch.matmul(y, delta) # batch, MAX_SEQ, MAX_SEQ-1(prob)

        # get excercise prob from diagonal matrix
        prob = torch.diagonal(temp, dim1=1, dim2=2) # batch, MAX_SEQ-1(prob)

        # at ex) *[1, 0] *[correct, false]
        a = ((batch[:,:,:QUESTION_NUM] - batch[:,:,QUESTION_NUM:]).sum(2) + 1) // 2
        # at+1 ex) *[1, 0] *[correct, false]
        a = a[:, 1:]

        loss = self.loss_fn(prob, a)


        predict = torch.where(prob > 0.5, 1, 0)

        acc = torch.logical_and(predict, a)
        acc = acc.sum() / torch.numel(acc)

        return loss, acc.item()
