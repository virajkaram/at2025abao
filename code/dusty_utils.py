import numpy as np
import torch
from astropy.io import ascii

def normalize_pars(pars, min_vals, max_vals):
    norm_pars = (pars - min_vals) / (max_vals - min_vals)
    return norm_pars

def calc_fnu(lam, lam_flam):
    c = 3e10
    nus = c / (lam * 1e-4)
    fnus = lam_flam * lam * 1e-4 / c
    return fnus


def calculate_scale(observed_wavs, observed_fluxes, observed_fluxerrs, model_wavs, model_fluxes):
    model_fluxes_interp = np.interp(x=observed_wavs, xp=model_wavs, fp=model_fluxes)
    # Calculate the weighted scale
    scale = np.sum(observed_fluxes * model_fluxes_interp / observed_fluxerrs ** 2) / np.sum(model_fluxes_interp ** 2 / observed_fluxerrs ** 2)
    return scale


def load_dusty_model(modelname):
    model = ascii.read(modelname)
    model['fnu'] = calc_fnu(lam=model['lam'], lam_flam=model['flux'])
    return model


def predict_dusty_model(net, dusty_pars):
    # Normalize the parameters first
    norm_pars = normalize_pars(pars=dusty_pars, min_vals=np.array([2100, 500, 0.01]), max_vals=np.array([4500, 2000, 10]))
    return net(torch.tensor(norm_pars).float()).detach().numpy()


def predict_dusty_log_r1(r1_net, dusty_pars):
    norm_pars = normalize_pars(pars=dusty_pars, min_vals=np.array([2100, 500, 0.01]), max_vals=np.array([4500, 2000, 10]))
    return r1_net(torch.tensor(norm_pars).float()).detach().numpy()