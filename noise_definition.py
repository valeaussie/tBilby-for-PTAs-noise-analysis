#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import json
import numpy as np
import subprocess
import itertools

from enterprise import constants as const
from enterprise.pulsar import Pulsar
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import parameter
from enterprise.signals import selections
from enterprise.signals import gp_priors
from enterprise.signals import utils
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases
from enterprise_extensions.blocks import common_red_noise_block
from enterprise_extensions import hypermodel
from enterprise_extensions.chromatic.solar_wind import solar_wind, createfourierdesignmatrix_solar_dm

from utils import *

parfile = datadir + '/' + psrname + '.par'
print(parfile)
timfile = datadir + '/' + psrname + '.tim'

psr = Pulsar(parfile, timfile, ephem=ephem)
p = psr

components_dict = {key: [] for key in ['red', 'dm', 'band', 'chrom', 'hf']}

"""
Define timing model
"""
tm = gp_signals.MarginalizingTimingModel(use_svd=True)

"""
Define efac model
"""
selection = selections.Selection(selections.by_backend)
efac = parameter.Uniform(0.1, 5.0)
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)

"""
Define ecorr model
"""
log10_ecorr_prior = parameter.Uniform(-10,-4)
ec = white_signals.EcorrKernelNoise(log10_ecorr=log10_ecorr_prior, selection=selection)

"""
Define equad model
"""
log10_equad_prior = parameter.Uniform(-10,-4)
eq = white_signals.TNEquadNoise(log10_tnequad=log10_equad_prior, selection=selection)

"""
Define red noise model
"""
priors = {}
rn_model, rn_lgA_prior, rn_gam_prior, priors = get_informed_rednoise_priors(psr, 'red_noise', {}, {}, priors, use_basic_priors=True, log10_A_min_basic=-18)

Tspan = psr.toas.max() - psr.toas.min()  # seconds
max_cadence = 240.0  # days
red_components = int(Tspan / (max_cadence*86400))
components_dict['red'].append(red_components)
print("Using {} red noise components".format(red_components))
rn = gp_signals.FourierBasisGP(rn_model, components=red_components, name='red_noise')

"""
Define DM noise model (dm)
"""
priors = {}
dm_model, dm_lgA_prior, dm_gam_prior, priors = get_informed_rednoise_priors(psr, 'dm_gp', {}, {}, priors, use_basic_priors=True, log10_A_min_basic=-18)#, noisedict_p0015, noisedict_p9985, priors)

Tspan = psr.toas.max() - psr.toas.min()  # seconds
max_cadence = 60  # days
dm_components = int(Tspan / (max_cadence*86400))
components_dict['dm'].append(dm_components)
print("Using {} DM components".format(dm_components))
dm_basis = gp_bases.createfourierdesignmatrix_dm(nmodes=dm_components)
dm = gp_signals.BasisGP(dm_model, dm_basis, name='dm_gp')

"""
Define chromatic noise model (chrom)
"""
chrom_model, chrom_lgA_prior, chrom_gam_prior, priors = get_informed_rednoise_priors(psr, 'chrom_gp', {}, {}, priors, use_basic_priors=True, log10_A_min_basic=-18)#, noisedict_p0015, noisedict_p9985, priors)

idx = 4  # Define freq^-idx scaling (chromatic index)
max_cadence = 240  # days
chrom_components = int(Tspan / (max_cadence*86400))
components_dict['chrom'].append(chrom_components)
print("Using {} Chrom components".format(chrom_components))
chrom_basis = gp_bases.createfourierdesignmatrix_chromatic(nmodes=chrom_components,
                                                               idx=idx)
chrom = gp_signals.BasisGP(chrom_model, chrom_basis, name='chrom_gp')

"""
Define solar wind (sw)
"""
def get_informed_nearth_priors(psr, priors, n_earth_only=False, n_earth_fixed=False):

    if psr.name[1:5] in ['0030', '0125', '0437', '0613', '1022', '1024', '1545', '1600', '1643', '1713', '1730', '1744', '1824', '1832', '1909', '2145', '2241'] and not n_earth_fixed:
        key_ = psr.name + '_n_earth'
        print('getting prior for {}'.format(key_))
        n_earth_min = 0.0
        n_earth_max = 20.0
        n_earth = parameter.Uniform(n_earth_min, n_earth_max)
        priors[key_ + "_min"] = n_earth_min
        priors[key_ + "_max"] = n_earth_max
    else:
        print('nearth constant')
        n_earth = parameter.Constant(4)

    deter_sw = solar_wind(n_earth=n_earth)
    mean_sw = deterministic_signals.Deterministic(deter_sw)#, name='n_earth')
    sw = mean_sw

    #vary gp_sw for pulsars with constrained gp_sw parameters
    if psr.name[1:5] in ['0437', '0711', '0900', '1024', '1643', '1713', '1730', '1744', '1909', '2145'] and not n_earth_only:
        key_ = psr.name + '_gp_sw_log10_A'
        print('getting prior for {}'.format(key_))
        log10_A_min = -10
        log10_A_max = -3
        gamma_min = -4
        gamma_max = 4
        log10_A_sw = parameter.Uniform(log10_A_min, log10_A_max)
        gamma_sw = parameter.Uniform(gamma_min, gamma_max)
        print(f"""{psr.name}_gp_sw_noise prior:
        log10_A in [{log10_A_min}, {log10_A_max}]
        gamma in [{gamma_min}, {gamma_max}]
        """)
        key_ = psr.name + '_gp_sw_log10_A'
       	priors[key_ + "_min"] = log10_A_min
       	priors[key_ + "_max"] = log10_A_max
       	key_ = psr.name + '_gp_sw_gamma'
       	priors[key_ + "_min"] = gamma_min
       	priors[key_ + "_max"] = gamma_max        

        Tspan = psr.toas.max() - psr.toas.min()
        max_cadence = 60
        sw_components = int(Tspan / (max_cadence*86400))
        sw_prior = utils.powerlaw(log10_A=log10_A_sw, gamma=gamma_sw)
        sw_basis = createfourierdesignmatrix_solar_dm(nmodes=sw_components, Tspan=Tspan)
        sw += gp_signals.BasisGP(sw_prior, sw_basis, name='gp_sw')

    return sw

sw = get_informed_nearth_priors(psr, priors)




