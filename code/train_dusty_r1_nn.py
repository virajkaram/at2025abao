import torch
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from train_dusty_phoenix_nn import get_dusty_r1_value, get_dusty_parameter_values, normalize_pars
import argparse
import os


def get_r1_values_from_dir(dusty_model_dir):
    dusty_modelfiles = np.sort(glob(f'{dusty_model_dir}/sed*'))
    r1_values = []
    for dusty_modelfile in dusty_modelfiles:
        r1_values.append(get_dusty_r1_value(dusty_modelfile))
    return np.array(r1_values)


# make pytorch Dataset
class DustyDataset(torch.utils.data.Dataset):
    def __init__(self, dusty_pars, dusty_r1):
        self.dusty_pars = dusty_pars
        self.dusty_r1 = dusty_r1

    def __len__(self):
        return len(self.dusty_pars)

    def __getitem__(self, idx):
        return self.dusty_pars[idx].astype(np.float32), self.dusty_r1[idx].astype(np.float32)


class Dusty_r1_Net(torch.nn.Module):
    def __init__(self):
        super(Dusty_r1_Net, self).__init__()
        self.fc1 = torch.nn.Linear(3, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Trainer:
    def __init__(self, net, optimizer, loss_func):
        self.net = net
        self.optimizer = optimizer
        self.loss_func = loss_func

    def train(self, dataloader, epochs=100):
        trainloss = []
        valloss = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            val_loss = self.validate(test_dataloader)
            print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss}')
            trainloss.append(running_loss/len(dataloader))
            valloss.append(val_loss)
        return trainloss, valloss

    def validate(self, dataloader):
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                outputs = self.net(inputs)
                loss = self.loss_func(outputs, labels.view(-1, 1))
                val_loss += loss.item()
        return val_loss / len(dataloader)

if __name__ == '__main__':
    # Train the neural network
    parser = argparse.ArgumentParser(description='Dusty R1 Training')
    parser.add_argument('dusty_model_grid_dir', type=str,
                        help='Directory with dusty model grid')
    parser.add_argument('--output_prefix', type=str,
                        help='output_prefix', default='dusty_nn_r1_model')
    parser.add_argument('--savedir', type=str,
                        default='models/dusty/dusty_ml_interpolator/', )
    args = parser.parse_args()

    dusty_model_dir = args.dusty_model_grid_dir

    dusty_pars = get_dusty_parameter_values(dusty_model_dir)
    norm_pars = normalize_pars(dusty_pars, dusty_pars.min(axis=0),
                               dusty_pars.max(axis=0))
    dusty_datar1_values = get_r1_values_from_dir(dusty_model_dir)
    r1_values = get_r1_values_from_dir(dusty_model_dir)

    dusty_dataset = DustyDataset(norm_pars, np.log10(r1_values))
    # train test split 80-20
    train_size = int(0.8 * len(dusty_dataset))
    test_size = len(dusty_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dusty_dataset,
                                                                [train_size, test_size])
    # train dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                                   shuffle=True)
    # test dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                                  shuffle=True)

    net = Dusty_r1_Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    trainer = Trainer(net, optimizer, loss_func)
    trainloss, valloss = trainer.train(train_dataloader, epochs=100)
    # plot train and validation loss
    plt.plot(trainloss, label='train loss')
    plt.plot(valloss, label='validation loss')

    plt.yscale('log')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(args.savedir, f'{args.output_prefix}_training_loss.pdf'),
                             bbox_inches='tight')

    plt.figure()
    # Plot some predictions and actual values
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            if i == 0:
                plt.scatter(labels, outputs)
                plt.xlabel('actual')
                plt.ylabel('predicted')
                plt.title('Dusty r1 predictions')
                break

        plt.plot(labels, labels, color='red', linestyle='--')

    plt.savefig(os.path.join(args.savedir, f'{args.output_prefix}_predictions.pdf'), bbox_inches='tight')

    torch.save(net.state_dict(), os.path.join(args.savedir, f'{args.output_prefix}_predictor.pth'))