import torch
import torch.optim as optim

from models.DKT import DKT
from loss import lossFunc

import tqdm
import time
import os

from constants import *


class ModelHandler():

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.model = None
        self.loss_fn = None
        self.optimizer = None

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if not os.path.isdir('outputs'):
            os.mkdir('outputs')

        print('-' * 30)
        print('-' * 10, self.device, '-' * 10)
        print('-' * 30)

        pass

    def load_model(self, model_path=None):
        if model_path == None:
            self.model = DKT(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
        else:
            self.model = DKT(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim)
            self.model.load_state_dict(torch.load(model_path))
        pass

    def compile_model(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.loss_fn = lossFunc().to(self.device)
        self.model.to(self.device)
        
        print('compile model', '-' * 20)
        print(self.optimizer)
        print('')
        print(self.loss_fn)
        print('')
        print(self.model)
        print('-' * 30)

    def save_model(self, epoch, val_loss, val_acc):

        save_path = f'outputs/{epoch}_{val_loss:.4f}_{val_acc:.4f}.pth'
        torch.save(self.model.state_dict(), save_path)

        pass

    def train(self, train_generator, val_generator, n_epoch):

        min_loss = 10e+8

        for epoch in tqdm.tqdm(range(n_epoch), desc='Training:', mininterval=2):

            # training step
            running_loss, running_acc = self._optimize(train_generator, epoch, train=True)
            print(f"epoch : {epoch}/{n_epoch}  running_acc : {running_acc:.4f}, running_loss : {running_loss.item():.4f}")

            # validation step
            if epoch % 5 == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_loss, val_acc = self._optimize(val_generator, epoch, train=False)
                    print(f"epoch : {epoch}/{n_epoch}  val_acc : {val_acc:.4f}, val_loss : {val_loss.item():.4f}")

                if val_loss < min_loss:
                    min_loss = val_loss
                    self.save_model(epoch, min_loss, val_acc)
        pass

    def _optimize(self, data_generator, epoch, train=True):

        start = time.time()

        if train:

            running_loss = 0
            running_acc = 0

            self.model.train()

            for num, batch in enumerate(data_generator):

                batch = batch.to(self.device)

                # wipe any existing gradients from previous iterations
                self.optimizer.zero_grad()

                pred = self.model(batch)
                loss, acc = self.loss_fn(pred, batch)

                # this step computes all gradients with "autograd"
                # i.e. automatic differentiation
                loss.backward()

                # this actually changes the parameters
                self.optimizer.step()

                # if the current loss is better than any ones we've seen
                # before, save the parameters.

                running_loss += loss
                running_acc += acc

                end = time.time()

                if (num + 1) % 16 == 0:
                    print(
                        f"[{epoch} epoch {num + 1}/{len(data_generator)} iter] batch_running_acc : {acc:.4f}, batch_running_loss : {loss.item():.4f} time : {end - start:.2f} sec",
                        end='\r', flush=True)

            running_loss = running_loss / len(data_generator)
            running_acc = running_acc / len(data_generator)

            return running_loss, running_acc

        else:

            val_loss = 0
            val_acc = 0

            self.model.eval()

            for num, batch in enumerate(data_generator):

                batch = batch.to(self.device)

                with torch.no_grad():

                    pred = self.model(batch)
                    loss, acc = self.loss_fn(pred, batch)
                    val_loss += loss
                    val_acc += acc

                    end = time.time()

                if num % 16 == 1:
                    print(
                        f"[{epoch + 1} epoch {num + 1}/{len(data_generator)} iter] batch_val_acc : {acc:.4f}, batch_val_loss : {loss.item():.4f} time : {end - start:.2f} sec",
                        end='\r', flush=True)

            val_loss = val_loss / len(data_generator)
            val_acc = val_acc / len(data_generator)

            return val_loss, val_acc

    def evaluate(self, test_generator):
        test_loss, test_acc = self._optimize(test_generator, 0, train=False)
        print('-'*50)
        print(f" test_acc : {test_acc:.4f}, test_loss : {test_loss.item():.4f}")
        pass

    def predict(self, x):

        def _cal_prob(x):

            # qt
            delta = x[:,:,:QUESTION_NUM] + x[:,:,QUESTION_NUM:]

            # qt+1
            delta = delta[:,1:,:].permute(0,2,1)

            # yt
            pred = self.model(x)
            y = pred[:, :MAX_SEQ - 1,:]

            # pred at+1
            temp = torch.matmul(y, delta) # 1, MAX_SEQ, MAX_SEQ-1(prob)

            # get excercise prob from diagonal matrix
            prob = torch.diagonal(temp, dim1=1, dim2=2) # 1, MAX_SEQ-1(prob)\

            return prob.squeeze(0)

        def _get_q_sequence(q_seq_one_hot):

            q_sequence = []
            one_hot_excercise_tags = q_seq_one_hot[:, :, :QUESTION_NUM] + q_seq_one_hot[:, :, QUESTION_NUM:]
            one_hot_excercise_tags = one_hot_excercise_tags.squeeze(0)

            for one_hot_excercise_tag in one_hot_excercise_tags:
                try:
                    excercise_tag = torch.nonzero(one_hot_excercise_tag).item()
                except:
                    excercise_tag = -1

                q_sequence.append(excercise_tag)

            return torch.Tensor(q_sequence)

        def _get_a_sequence(q_seq_one_hot):
            q_seq_one_hot = q_seq_one_hot.squeeze(0)
            a_sequence = ((q_seq_one_hot[:, :QUESTION_NUM] - q_seq_one_hot[:, QUESTION_NUM:]).sum(1) + 1) // 2
            return a_sequence


        if len(x.size()) == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        prob = _cal_prob(x)

        q_sequence = _get_q_sequence(x)
        a_sequence = _get_a_sequence(x)

        if -1 in q_sequence:
            last_excercise_tag = torch.nonzero(q_sequence == -1)[0][0].item() - 1
        else:
            last_excercise_tag = len(q_sequence) - 1

        print('-' * 50)
        print(f'sol excercise tags: \n {q_sequence[:last_excercise_tag-1]}')
        print(f'result excercise tags: \n {a_sequence[:last_excercise_tag-1]}')
        print('-' * 50)

        print(f'predict excercise tag {q_sequence[last_excercise_tag]}')
        print(f'ground truth : {a_sequence[last_excercise_tag]}')
        print(f'this student has a {prob[last_excercise_tag-1]*100:.2f}% chance of solving this problem')
