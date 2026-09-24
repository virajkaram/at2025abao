# Dusty functions similar to those in LRNe_JWST project
import numpy as np
from train_dusty_phoenix_nn import normalize_pars, \
    get_flux_from_nn_prediction, get_flux_from_nn_prediction_log_offset
import torch
from astropy.io import ascii
from pydusty.utils import get_extinction_corrected_fluxes, apply_extinction_to_fluxes
import emcee
import corner
import matplotlib.pyplot as plt


def get_lums_r1s(blobs, dist_mpc):
    int_fluxes = blobs[:,1]
    int_lums = int_fluxes * 4*np.pi*(dist_mpc*3.086e24)**2
    log_r1s = blobs[:,2]
    log_scaled_r1s = log_r1s + 0.5*np.log10(int_lums/3.83e37)
    return int_lums, log_scaled_r1s


def calc_fnu(lam, lam_flam):
    c = 3e10
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



def predict_dusty_model_log_offset(net, dusty_pars, par_min_vals, par_max_vals):
    # Normalize the parameters first
    norm_pars = normalize_pars(pars=dusty_pars, min_vals=par_min_vals,
                               max_vals=par_max_vals)
    dusty_prediction = net(torch.tensor(norm_pars).float()).detach().numpy()
    return get_flux_from_nn_prediction_log_offset(dusty_prediction)


def predict_dusty_model(net, dusty_pars, par_min_vals, par_max_vals):
    # Normalize the parameters first
    norm_pars = normalize_pars(pars=dusty_pars, min_vals=par_min_vals,
                               max_vals=par_max_vals)
    dusty_prediction = net(torch.tensor(norm_pars).float()).detach().numpy()
    return get_flux_from_nn_prediction(dusty_prediction)


def predict_dusty_log_r1(r1_net, dusty_pars, par_min_vals, par_max_vals):
    norm_pars = normalize_pars(pars=dusty_pars, min_vals=par_min_vals,
                               max_vals=par_max_vals)
    return r1_net(torch.tensor(norm_pars).float()).detach().numpy()


def calc_chisq(wavs, fnus, fnuerrs, model_wavs, scaled_model_fnus, error_underestimation_scaling=0):
    scaled_inter_model_fnus = np.interp(x=wavs, xp=model_wavs, fp=scaled_model_fnus)
    chisq = np.sum((fnus - scaled_inter_model_fnus)**2 / (fnuerrs**2 + error_underestimation_scaling*fnus**2))
    return chisq


def calc_chisq_piecewise(wavs, fnus, fnuerrs, model_wavs, scaled_model_fnus, error_underestimation_scaling=0):
    scaled_inter_model_fnus = np.interp(x=wavs, xp=model_wavs, fp=scaled_model_fnus)
    chisq_array = (fnus - scaled_inter_model_fnus)**2 / (fnuerrs**2 + error_underestimation_scaling*fnus**2)
    return chisq_array

def unpack_parameters(theta, vary_ext, vary_error_underestimation, log_tau):
    if not vary_ext and not vary_error_underestimation:
        Tstar, Tdust, tau = theta
        ebv = 0
        err_underestimation = 0

    elif vary_ext and vary_error_underestimation:
        Tstar, Tdust, tau, ebv, err_underestimation = theta

    elif vary_ext:
        Tstar, Tdust, tau, ebv = theta
        err_underestimation = 0

    else:
        Tstar, Tdust, tau, err_underestimation = theta
        ebv = 0
    if not log_tau:
        tau = 10**tau

    dusty_pars = np.array([Tstar, Tdust, tau])
    return dusty_pars, ebv, err_underestimation


def unpack_parameters_tau_Y(theta, vary_ext, vary_error_underestimation, log_tau):
    if not vary_ext and not vary_error_underestimation:
        Tstar, Tdust, tau, Y = theta
        ebv = 0
        err_underestimation = 0

    elif vary_ext and vary_error_underestimation:
        Tstar, Tdust, tau, Y, ebv, err_underestimation = theta

    elif vary_ext:
        Tstar, Tdust, tau, Y, ebv = theta
        err_underestimation = 0

    else:
        Tstar, Tdust, tau, Y, err_underestimation = theta
        ebv = 0
    if not log_tau:
        tau = 10 ** tau

    dusty_pars = np.array([Tstar, Tdust, tau, Y])
    return dusty_pars, ebv, err_underestimation


def uniform_prior(theta, vary_ext=False, vary_error_underestimation=False):
    dusty_pars, ebv, err_underestimation = unpack_parameters(theta, vary_ext, vary_error_underestimation, log_tau=False)
    Tstar, Tdust, tau = dusty_pars
    if 2100 <= Tstar <= 5500 and 500 <= Tdust <= 2000 and 1 <= tau <= 10 and 0 <= ebv <=1 and 0 <= err_underestimation <= 0.1:
        return 0
    return -np.inf


def calc_integrated_flux(wavs_um, fnu_ujy):
    c = 3e10
    nus = c/(wavs_um*1e-4)
    fnu_ujy[fnu_ujy<0] = 0
    try:
        return np.trapezoid(x=nus, y=fnu_ujy*1e-29)*-1.0
    except AttributeError:
        return np.trapz(y=fnu_ujy*1e-29, x=nus)*-1.0


def Mdust(tau_Vs, Rin_Rsun, Y=2.0):
    return 2.4e-11 * (Rin_Rsun/50)**2 * (tau_Vs/0.7) * (Y/5)


def log_likelihood(theta, obs_wavs, obs_fnus, obs_fnuerrs, dusty_wavs,
                   dusty_sed_net, dusty_r1_net, par_min_vals, par_max_vals, vary_ext=False,
                   vary_error_underestimation=False, prior_func=uniform_prior,
                   is_dusty_model_log_tau=False, predict_model=predict_dusty_model,
                   unpack_params_func=unpack_parameters,
                   calc_scale_func=calculate_scale,
                   ):
    dusty_pars, ebv, err_underestimation = unpack_params_func(theta, vary_ext, vary_error_underestimation, is_dusty_model_log_tau)
    prior = prior_func(theta, vary_ext, vary_error_underestimation)
    if not np.isfinite(prior):
        return -np.inf, np.nan, np.nan, np.nan
    ecor_obs_fnu, ecor_obs_fnuerrs = get_extinction_corrected_fluxes(mlam=obs_wavs, mlumobs=obs_fnus, merrobs=obs_fnuerrs, ebv=ebv)
    dusty_model_flux = predict_model(dusty_sed_net, dusty_pars,
                                           par_min_vals=par_min_vals, par_max_vals=par_max_vals)
    dusty_model_fnu = calc_fnu(lam=dusty_wavs, lam_flam=dusty_model_flux)
    scale = calc_scale_func(observed_wavs=obs_wavs, observed_fluxes=ecor_obs_fnu, observed_fluxerrs=ecor_obs_fnuerrs,
                            model_wavs=dusty_wavs, model_fluxes=dusty_model_fnu)
    chisq = calc_chisq(obs_wavs, ecor_obs_fnu, ecor_obs_fnuerrs, dusty_wavs, dusty_model_fnu*scale, error_underestimation_scaling=err_underestimation)

    int_flux = calc_integrated_flux(dusty_wavs, dusty_model_fnu*scale)

    log_r1 = predict_dusty_log_r1(dusty_r1_net, dusty_pars,
                                  par_min_vals=par_min_vals, par_max_vals=par_max_vals)[0]

    return -0.5 * chisq, scale, int_flux, log_r1


def run_mcmc(obs_wavs, obs_fnus, obs_fnuerrs, dusty_wavs, dusty_sed_net, dusty_r1_net,
             dist_mpc, par_min_vals, par_max_vals,
             vary_ext=False, use_error_underestimation=False, plot_label='sed',
             nsteps=1000, plot = True,
             prior_func=uniform_prior,
             nwalkers=32, ndim=3, initial=np.array([3700, 900, 0.5]),
             inital_std=np.array([100, 100, 0.1]),
              labels=['Tstar','Tdust','log_tau'],
             is_dusty_model_log_tau=False,
             predict_model=predict_dusty_model,
             unpack_params_func=unpack_parameters,
             calc_scale_func=calculate_scale,
             ):
    if vary_ext:
        ndim += 1
        initial = np.append(initial, 0.5)
        inital_std = np.append(inital_std, 0.01)
        labels.append('ebv')
    if use_error_underestimation:
        ndim += 1
        initial = np.append(initial, 0.01)
        inital_std = np.append(inital_std, 0.01)
        labels.append('ferr')
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim,
                                    log_prob_fn=log_likelihood, blobs_dtype=float,
                                    args=[obs_wavs, obs_fnus, obs_fnuerrs, dusty_wavs,
                                          dusty_sed_net, dusty_r1_net, par_min_vals, par_max_vals,
                                          vary_ext, use_error_underestimation, prior_func, is_dusty_model_log_tau,
                                          predict_model, unpack_params_func, calc_scale_func,
                                          ])

    inital_pos = initial + inital_std * np.random.randn(nwalkers, ndim)
    sampler.run_mcmc(inital_pos, nsteps, progress=True)

    if plot:
        flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
        blobs = sampler.get_blobs(discard=100, thin=15, flat=True)
        int_lums, log_scaled_r1s = get_lums_r1s(blobs, dist_mpc)
        # add lums and r1s to corner plot
        flat_samples_extend = np.hstack((flat_samples, int_lums[:, None]/3.83e33, log_scaled_r1s[:, None]))
        labels = labels + ['Lint (Lsun)', 'log_scaled_r1']
        fig = corner.corner(flat_samples_extend, labels=labels, show_titles=True)

        plt.figure()
        randinds = np.random.choice(np.arange(len(flat_samples)), 100)
        for ind in randinds:
            dusty_pars, ebv, _ = unpack_params_func(flat_samples[ind], vary_ext, use_error_underestimation,
                                                          log_tau=is_dusty_model_log_tau)
            dusty_model_flux = predict_model(dusty_sed_net, dusty_pars, par_min_vals=par_min_vals,
                                                   par_max_vals=par_max_vals)
            dusty_model_fnu = calc_fnu(lam=dusty_wavs, lam_flam=dusty_model_flux)
            dusty_model_ecor_fnu, _ = apply_extinction_to_fluxes(mlam=dusty_wavs,
                                                                 mlum=dusty_model_fnu,
                                                                 merr=np.zeros_like(dusty_model_fnu),
                                                                 ebv=ebv)
            scale = calc_scale_func(observed_wavs=obs_wavs, observed_fluxes=obs_fnus,
                                    observed_fluxerrs=obs_fnuerrs, model_wavs=dusty_wavs,
                                    model_fluxes=dusty_model_ecor_fnu)
            plt.plot(dusty_wavs, dusty_model_ecor_fnu*scale,
                     alpha=0.1, color='orange')
        plt.errorbar(y=obs_fnus, yerr=obs_fnuerrs, x=obs_wavs, fmt='o', color='black')


        plt.xlim(0.4, 50.0)
        plt.ylim(np.min(obs_fnus/3), np.max(obs_fnus*3))
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'$\lambda$ [$\mu$m]')
        plt.ylabel(r'Flux density [$\mu$Jy]')
    return sampler


