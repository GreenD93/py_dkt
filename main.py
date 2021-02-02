from data_handle import DatasetHandler
from model_handle import ModelHandler

from constants import *


def main():
    data_handler = DatasetHandler()
    train_generator = data_handler.get_data_generator(train_data_path)
    val_generator = data_handler.get_data_generator(test_data_path)

    model_handler = ModelHandler(input_dim, hidden_dim, num_layers, output_dim)
    model_handler.load_model()
    model_handler.compile_model()
    model_handler.train(train_generator, val_generator, n_epoch)

if __name__ == '__main__':
    main()
