import bilby
import numpy as np
import matplotlib.pyplot  as plt
import corner
import itertools



file_path = '/fred/oz002/vdimarco/tbiby_PTA_noise/jobs/out_WN'
result = bilby.result.read_in_result(filename = file_path + '/' + 'WN_run_result.json')
with open(file_path + 'model_def.json', 'r') as json_file:
    model_def = json.load(json_file)

print(model_def)
n0 = result.posterior["n0"]

n = count_n_keys(result)
l = generate_all_models(n)
d_list = []
for i in l:
    d = binary_to_decimal(i)
    d_list.append(d)



print(n)
print(l)
print(d_list)
print()

ppp

parameter_list = result.posterior.iloc[0]
print(parameter_list)
labels = list(parameter_list.index)[:-4]
#labels = ['J1909-3744_KAT_MKBF_efac', 'J1909-3744_chrom_gp_gamma', 'J1909-3744_chrom_gp_log10_A', 'J1909-3744_dm_gp_gamma', 'J1909-3744_dm_gp_log10_A']
samples = result.posterior[labels].values
fig = corner.corner(samples, truths=[0.95, 0, np.log10(1.1e-7)], truth_color="red", labels=labels,bins=50, quantiles=[0.025, 0.5, 0.975],
                    show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('WN_bilby.png')

