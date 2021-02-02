import numpy as np
import tqdm
import itertools

from dataset import DKTDataSet
from torch.utils.data import DataLoader

from constants import *

class DatasetHandler():

    def __init__(self):
        pass

    def get_data_generator(self, data_path):
        ques, ans = self._read_data(data_path)
        dataset = DKTDataSet(ques, ans)
        data_generator = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        return data_generator

    def _read_data(self, data_path):

        qus_list = np.array([])
        ans_list = np.array([])

        with open(data_path, 'r') as train:
            for num_seq, ques, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 3), desc='loading data:    ', mininterval=2):

                num_seq = int(num_seq.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)

                mod = 0 if num_seq % MAX_SEQ == 0 else (MAX_SEQ - num_seq % MAX_SEQ)

                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)

                qus_list = np.append(qus_list, ques)
                ans_list = np.append(ans_list, ans)

        qus_list = qus_list.astype(np.int).reshape([-1, MAX_SEQ])
        ans_list = ans_list.astype(np.int).reshape([-1, MAX_SEQ])

        return qus_list, ans_list
