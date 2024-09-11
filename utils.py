import os
import sys
import glob
import json
import numpy as np
import subprocess
import bilby
import corner
import itertools
from pyhelpers.store import load_json\

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
import enterprise.signals.parameter as parameter
from enterprise.signals import gp_priors


# Read in pulsasr name, ephemeris, data and output directory
input_data = load_json('config.json')
psrname = input_data['psrname']
ephem = input_data['ephem']
datadir = input_data['datadir']
outdir = input_data['outdir']
label = input_data['label']

if not os.path.exists(outdir):
    os.makedirs(outdir)

class noise_models:
    def __init__(self, tm, ef):
        self.noise_model_dict = {}
        self.i = 0
        self.model_holder = {}
        self.tm = tm
        self.ef = ef
        self.model_def = {}
    def is_func_in_list(self, func):
        for val in self.noise_model_dict.values():
            if func is val:
                return True
        return False
    def print_noise_models(self):
        print(self.model_def)
    def add_noise_model(self, model, user_string):
        if not self.is_func_in_list(model) and model is not self.ef:
            self.noise_model_dict['n{}'.format(self.i)] = model
            self.model_def['n{}'.format(self.i)] = user_string
            self.i += 1
        else:
            print("Model already in list or model is ef")
    def generate_signal(self):
        s = self.tm + self.ef
        for key in self.noise_model_dict.keys():
            s += self.noise_model_dict[key]
        return s
    def generate_model(self, data):
        n = len(self.noise_model_dict)
        combinations = list(itertools.product([0, 1], repeat=n))   
        for comb in combinations:
            key = '-'.join(map(str,comb))
            non_zero_indices_list = [index for index, value in enumerate(comb) if value != 0]
            model = self.tm + self.ef
            for model_i in non_zero_indices_list:
                model += self.noise_model_dict['n'+str(model_i)]
            self.model_holder[key] = signal_base.PTA(model(data))
        return self.model_holder
    def parameter_mapper(self, key):
        a = [param.name for param in list(self.model_holder[key].params)]
        return a
    def generate_key(self, dictionary):
        key = ''
        for num in range(len(self.model_def)):
            if dictionary ['n{}'.format(num)] == 1:
                key += '-1'
            else:
                key += '-0'
        return key[1:] #remove first '-'

def get_informed_rednoise_priors(psr, noisename, noisedict_3sig_min, noisedict_3sig_max, priors, return_priorvals = False, use_basic_priors = False, log10_A_min_basic=-20, log10_A_min_informed=-18, log10_A_min_bound=-2):
    key_ = psr.name + '_' + noisename + '_log10_A'
    print('getting prior for {}'.format(key_))
    if not use_basic_priors:
        try:
            log10_A_min = np.max([log10_A_min_informed, noisedict_3sig_min[key_] + log10_A_min_bound ])
            log10_A_max = np.min([-11, noisedict_3sig_max[key_] + 1 ])
            print(f'found log10_A_min = {log10_A_min}, log10_A_max = {log10_A_max}')
            key_ = psr.name + '_' + noisename + '_gamma'
            gamma_min = np.max([0, noisedict_3sig_min[key_] - 0.5])
            gamma_max = np.min([7, noisedict_3sig_max[key_] + 0.5])
            print(f'found gamma_min = {gamma_min}, gamma_max = {gamma_max}')
            
        except KeyError as e:
            print('KeyError:', e)
            log10_A_min = -18
            log10_A_max = -11
            gamma_min = 0
            gamma_max = 7
    else:
        log10_A_min = log10_A_min_basic
        log10_A_max = -11
        gamma_min = 0
        gamma_max = 7
    log10_A_prior = parameter.Uniform(log10_A_min, log10_A_max)
    gamma_prior = parameter.Uniform(gamma_min, gamma_max)
    print(f"""{psr.name}_{noisename}_noise prior:
    log10_A in [{log10_A_min}, {log10_A_max}]
    gamma in [{gamma_min}, {gamma_max}]
    """)
    # # powerlaw
    rednoise_model = gp_priors.powerlaw(log10_A=log10_A_prior,
                                        gamma=gamma_prior)
    key_ = psr.name + '_' + noisename + '_log10_A'
    priors[key_ + "_min"] = log10_A_min
    priors[key_ + "_max"] = log10_A_max
    key_ = psr.name + '_' + noisename + '_gamma'
    priors[key_ + "_min"] = gamma_min
    priors[key_ + "_max"] = gamma_max
    if return_priorvals:
        return rednoise_model, log10_A_prior, gamma_prior, log10_A_min, log10_A_max, gamma_min, gamma_max, priors
    
    return rednoise_model, log10_A_prior, gamma_prior, priors

#convert binary list to decimal number
def binary_to_decimal(binary_string):
    binary_list = [int(item) for item in binary_string.split('-')]

    decimal_number = 0
    length = len(binary_list)
    
    for i, bit in enumerate(binary_list):
        decimal_number += bit * (2 ** (length - i - 1))
    
    return decimal_number

class analysis:
    def __init__(self, result_file_name, model_list_object):
        self.result_file_name = result_file_name
        self.model_list_object = model_list_object
        self.result = bilby.result.read_in_result(filename = self.result_file_name)
    def run_analysis(self):
        groups = self.result.posterior.groupby(list(self.model_list_object.model_def.keys()))
        model_freq = {}
        for group in groups:
            n = len(group[1])
            key = self.model_list_object.generate_key(group[1].iloc[0].to_dict())
            param = self.model_list_object.parameter_mapper(key)
            print(group[1].columns)
            print(param)
            samples = group[1][param].values
            model_freq[binary_to_decimal(key)] = n
            fig = corner.corner(samples, labels=param, bins=50, quantiles=[0.025, 0.5, 0.975],
                    show_titles=True, title_kwargs={"fontsize": 12})
            plt.savefig('{}.png'.format(binary_to_decimal(key)))
        plt.bar(model_freq.keys(), model_freq.values())
        plt.savefig('model_freq.png')
    