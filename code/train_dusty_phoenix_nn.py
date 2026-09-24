import torch
from glob import glob
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse


file_dir = os.path.dirname(os.path.realpath(__file__))


def modify_fluxes_for_training(fluxes):
    log_fluxes = np.log10(fluxes)
    log_fluxes[log_fluxes < -5] = -5  # set a floor to avoid -inf
    # return log_fluxes + 5
    return fluxes


def get_flux_from_nn_prediction(predictions):
    # return 10 ** (predictions - 5)
    return predictions


def get_flux_from_nn_prediction_log_offset(predictions):
    return 10 ** (predictions - 5)


def get_wavmask_for_training(wavs):
    return (wavs > 0.1) & (wavs < 50)


def get_dusty_pars_from_name(dusty_filename):
    Tstar, Tdust, tau_d = dusty_filename.split('/')[-1].split('_')[1:4]
    return float(Tstar), float(Tdust), np.log10(float(tau_d))


# Function to normalize user-defined parameters
def normalize_pars(pars, min_vals, max_vals):
    return (pars - min_vals)/(max_vals - min_vals)


# make pytorch Dataset
class DustyDataset(torch.utils.data.Dataset):
    def __init__(self, dusty_pars, dusty_seds):
        self.dusty_pars = dusty_pars
        self.dusty_seds = dusty_seds

    def __len__(self):
        return len(self.dusty_pars)

    def __getitem__(self, idx):
        return self.dusty_pars[idx].astype(np.float32), self.dusty_seds[idx].astype(np.float32)


# make a neural network to predict the SEDs from the parameters
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = torch.nn.Linear(3, 32)
#         self.fc2 = torch.nn.Linear(32, 64)
#         self.fc3 = torch.nn.Linear(64, 128)
#         self.fc4 = torch.nn.Linear(128, 251)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x


class Net_spherex(torch.nn.Module):
    def __init__(self):
        super(Net_spherex, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 128)
        self.fc5 = torch.nn.Linear(128, 123)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Net_phoenix(torch.nn.Module):
    def __init__(self):
        super(Net_phoenix, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 128)
        self.fc5 = torch.nn.Linear(128, 155)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 128)
        self.fc5 = torch.nn.Linear(128, 62)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class Net_5params(torch.nn.Module):
    def __init__(self):
        super(Net_5params, self).__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, 64)
        self.fc4 = torch.nn.Linear(64, 128)
        self.fc5 = torch.nn.Linear(128, 62)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


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

    def validate(self, dataloader):
        running_loss = 0.0
        for X, Y in dataloader:
            prediction = self.net(X)
            loss = self.loss_func(prediction, Y)
            running_loss += loss.item()
        return running_loss/len(dataloader)

    def train(self, dataloader, test_dataloader, epochs=100):
        trainloss, valloss = [], []
        for epoch in range(epochs):
            running_loss = 0.0
            for X, Y in dataloader:
                self.optimizer.zero_grad()
                prediction = self.net(X)
                loss = self.loss_func(prediction, Y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            # calculate validation loss
            val_loss = self.validate(test_dataloader)
            print(f'Epoch {epoch}, Loss: {running_loss/len(dataloader)}, Val Loss: {val_loss}')
            trainloss.append(running_loss/len(dataloader))
            valloss.append(val_loss)
        return trainloss, valloss


def get_dusty_r1_value(dusty_filename):
    with open(dusty_filename, 'r') as f:
        dat = f.readline()
    return float(dat.split('#')[1].strip('\n'))


def get_dusty_pars_seds(dusty_model_grid_dir):
    dusty_models = np.sort(glob(f'{dusty_model_grid_dir}/sed*'))


    dusty_pars = []
    for dusty_model in dusty_models:
        Tstar, Tdust, tau = get_dusty_pars_from_name(dusty_model)
        dusty_pars.append([Tstar, Tdust, tau])

    dusty_pars = np.array(dusty_pars)

    dusty_training_fluxes = []
    for dusty_model in dusty_models:
        sed = ascii.read(dusty_model)
        wavmask = get_wavmask_for_training(sed['lam'])
        dusty_training_fluxes.append(modify_fluxes_for_training(sed[wavmask]['flux']))

    dusty_training_fluxes = np.array(dusty_training_fluxes)
    return dusty_pars, dusty_training_fluxes


def get_dusty_parameter_values(dusty_model_grid_dir):
    dusty_models = np.sort(glob(f'{dusty_model_grid_dir}/sed*'))
    dusty_pars = []
    for dusty_model in dusty_models:
        Tstar, Tdust, tau = get_dusty_pars_from_name(dusty_model)
        dusty_pars.append([Tstar, Tdust, tau])

    return np.array(dusty_pars)


def train_dusty_model(dusty_pars, dusty_seds):
    dusty_norm_pars = (dusty_pars - dusty_pars.min(axis=0)) / (
                dusty_pars.max(axis=0) - dusty_pars.min(axis=0))

    dusty_dataset = DustyDataset(dusty_norm_pars, dusty_seds)
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
    net = Net_phoenix() #Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss()

    trainer = Trainer(net, optimizer, loss_func)
    trainloss, valloss = trainer.train(train_dataloader, test_dataloader, epochs=500)
    return net, trainloss, valloss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dusty_model_grid_dir', type=str,
                        help='Directory with dusty model grid')
    parser.add_argument('--output_prefix', type=str,
                        help='output_prefix', default='dusty_nn_model')
    parser.add_argument('--savedir', type=str, default='models/dusty/dusty_ml_interpolator/',)
    args = parser.parse_args()
    dusty_pars, dusty_training_fluxes = get_dusty_pars_seds(args.dusty_model_grid_dir)
    print(f"Loaded {len(dusty_pars)} dusty models")
    print("Min/max of parameters:")
    print(f"Tstar: {dusty_pars[:,0].min()}/{dusty_pars[:,0].max()}")
    print(f"Tdust: {dusty_pars[:,1].min()}/{dusty_pars[:,1].max()}")
    print(f"tau: {dusty_pars[:,2].min()}/{dusty_pars[:,2].max()}")
    print(f"Training the neural network on {len(dusty_pars)} models")
    net, trainloss, valloss = train_dusty_model(dusty_pars, dusty_training_fluxes)
    print("Training complete")

    # Save the model with timestamp
    torch.save(net.state_dict(), f'{args.savedir}/'
                                 f'{args.output_prefix}_predictor.pth')


    plt.figure()
    plt.plot(trainloss, label='Training Loss')
    plt.plot(valloss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig(f'{args.savedir}/{args.output_prefix}_training_loss.pdf', bbox_inches='tight')

    # Make a plot with subplots showing 10 random example predictions and the true SEDs
    fig, axs = plt.subplots(5, 2, figsize=(10, 20))
    for ax in axs:
        for a in ax:
            i = np.random.randint(0, len(dusty_pars))
            normalized_pars = normalize_pars(dusty_pars[i], dusty_pars.min(axis=0),
                                             dusty_pars.max(axis=0))
            prediction = net(torch.tensor(normalized_pars).float())
            a.plot(prediction.detach().numpy(), label='Predicted SED')
            a.plot(dusty_training_fluxes[i], ls='--', label='True SED')
            a.legend()

    plt.savefig(f'{args.savedir}/{args.output_prefix}_predictions.pdf', bbox_inches='tight')

