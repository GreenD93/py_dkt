import torch
from torch.utils.data.dataset import Dataset

import numpy as np

from constants import *

class DKTDataSet(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)
        return torch.FloatTensor(onehot.tolist())

    def onehot(self, questions, answers):
        # 정오답
        result = np.zeros(shape=[MAX_SEQ, 2 * QUESTION_NUM])

        for i in range(MAX_SEQ):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + QUESTION_NUM] = 1
        return result
