import argparse
from functions import *

def train():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Train')

    # Add arguments to the parser
    parser.add_argument('--data_directory', default="flowers/")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--arch', default="vgg16", type=str)
    parser.add_argument('--hidden_units', type=int, default=100)
    parser.add_argument('--save_dir', default="./checkpoint.pth")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gpu', default=False, action='store_true')
    pa_obj = parser.parse_args()

    data_directory = pa_obj.data_directory
    epochs = pa_obj.epochs
    path = pa_obj.save_dir
    architecture = pa_obj.arch
    learning_rate = pa_obj.learning_rate
    hidden_units = pa_obj.hidden_units
    dropout = pa_obj.dropout
    device = "gpu" if pa_obj.gpu else "cpu"
    print_every = 10

    print("Loading Data")
    train_loader, validation_loader, test_loader, train_dataset = load_data(data_directory)
    print("Creating Model")
    model, criterion, optimizer = create_model(architecture, dropout, hidden_units, learning_rate, device)
    print("Training Model")
    train_network(train_loader, validation_loader, model, criterion, optimizer, epochs, print_every, device)
    print("Testing Model")
    test_accuracy(model, test_loader, device)
    print("Saving Checkpoint")
    save_checkpoint(model, train_dataset.class_to_idx, path, architecture, hidden_units, dropout, learning_rate)
    print('Done')

if __name__ == '__main__':
    train()