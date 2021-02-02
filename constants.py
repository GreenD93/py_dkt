# data param
train_data_path = 'data/kddcup2010_train.txt'
test_data_path = 'data/kddcup2010_test.txt'

MAX_SEQ = 50
QUESTION_NUM = 661

# model param
BATCH_SIZE = 128
input_dim = QUESTION_NUM * 2
hidden_dim = 200
num_layers = 1
output_dim = QUESTION_NUM
n_epoch = 50
LR = 0.001
