import uproot
import yaml
import numpy as np
import awkward as ak
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, help='Path to config file', required=True)
args = parser.parse_args()

config_path = args.config
with open(config_path, 'r') as fp:
    config = yaml.safe_load(fp)


filepaths = config['path_train']
if not isinstance(filepaths, list):
    if '[' in filepaths and ']' in filepaths:
        filepaths = eval(filepaths)
    else:
        filepaths = [filepaths]

if len(filepaths) < 10:
    print("Filepaths: ")
    for fp in filepaths:
        print('\t', fp)
else:
    print("Number of filepaths: ", len(filepaths))



n_events = 0
for fp in tqdm(filepaths, desc='Computing number of events'):
    tree = uproot.open(fp)['EventTree']
    n_events += tree.num_entries
print("Number of events: ", n_events)


data_dict = {}
# vars = [
#     'track_pt', 'track_d0', 'track_z0', 'track_chi2', 'track_ndf', 'track_radiusofinnermosthit', 'track_tanlambda', 'track_omega',
#     'topo_e', 'topo_rho', 'topo_sigma_eta', 'topo_sigma_phi', 'topo_sigma_rho', # 'topo_x', 'topo_y', 'topo_z',
#     'particle_pt', 'particle_e']
vars = ['topo_sigma_eta', 'topo_sigma_rho']


for key in tree.keys():
    data_dict[key] = []
for fp in tqdm(filepaths, desc='Loading data'):
    tree = uproot.open(fp)['EventTree']
    for key in tree.keys():
        data_dict[key].append(tree[key].array(library='ak'))
for key in tree.keys():
    data_dict[key] = ak.concatenate(data_dict[key], axis=0)



# pts come from tracks and truth particles
all_pts = ak.concatenate([data_dict['track_pt'], data_dict['particle_pt']], axis=0)
all_pts_pow_xm = np.power(all_pts, 0.5)

# es come from clusters (no cells)
all_energies = data_dict['topo_e']
all_energies_pow_xm = np.power(all_energies, 0.5)

rhos = data_dict['topo_rho']

all_sigma_etas = data_dict['topo_sigma_eta']
all_sigma_etas_pow_xm = np.power(all_sigma_etas, 0.5)

all_sigma_phis = data_dict['topo_sigma_phi']
all_sigma_phis_pow_xm = np.power(all_sigma_phis, 0.5)

all_sigma_rhos = data_dict['topo_sigma_rho']
all_sigma_rhos_pow_xm = np.power(all_sigma_rhos, 0.5)

track_d0 = data_dict['track_d0']
track_z0 = data_dict['track_z0']

chi2 = data_dict['track_chi2']
chi2_pow_xm = np.power(chi2, 0.5)

track_ndf = data_dict['track_ndf']

radiusofinnermosthit = data_dict['track_radiusofinnermosthit']
rad_pow_xm = np.power(radiusofinnermosthit, 0.5)

track_tanlambda = data_dict['track_tanlambda']
track_omega = data_dict['track_omega']


# print the scaling dict
def custom_print(var, x, trans, scale_mode, m=None):
    print(f'"{var}": {{')

    if trans == "null":
        print(f'    "transformation": {trans},')
    else:
        print(f'    "transformation": "{trans}",')
        
    if m is not None:
        print(f'    "m": {m},')

    print(f'    "scale_mode": "{scale_mode}",')    
    print(f'    "mean": {np.nanmean(x):.3f}, "std": {np.nanstd(x):.3f},')
    print(f'    "min": {np.nanmin(x):.3f}, "max": {np.nanmax(x):.3f}, "range": [-1,1]')
    print('},')



custom_print('rho', rhos, trans='null', scale_mode='min_max')

custom_print('pt', all_pts_pow_xm, trans='pow(x,m)', scale_mode='min_max', m=0.5)
custom_print('e', all_energies_pow_xm, trans='pow(x,m)', scale_mode='min_max', m=0.5)


custom_print('sigma_eta', all_sigma_etas_pow_xm, trans='pow(x,m)', scale_mode='min_max', m=0.5)
custom_print('sigma_phi', all_sigma_phis_pow_xm, trans='pow(x,m)', scale_mode='min_max', m=0.5)
custom_print('sigma_rho', all_sigma_rhos_pow_xm, trans='pow(x,m)', scale_mode='min_max', m=0.5)

custom_print('d0', track_d0, trans='null', scale_mode='standard')
custom_print('z0', track_d0, trans='null', scale_mode='standard')

custom_print('chi2', chi2_pow_xm, trans='pow(x,m)', scale_mode='min_max')
custom_print('ndf', track_ndf, trans='null', scale_mode='min_max')
custom_print('radiusofinnermosthit', rad_pow_xm, trans='pow(x,m)', scale_mode='min_max')
custom_print('tanlambda', track_tanlambda, trans='null', scale_mode='min_max')
custom_print('omega', track_omega, trans='null', scale_mode='standard')


# debug
print('debug')
custom_print('sigma_eta', all_sigma_etas, trans='null', scale_mode='min_max')
custom_print('sigma_rho', all_sigma_rhos, trans='null', scale_mode='min_max')


# count nans in sigma_eta and sigma_rho
print('Number of nans in sigma_eta: ', np.sum(np.isnan(all_sigma_etas)))
print('Number of nans in sigma_rho: ', np.sum(np.isnan(all_sigma_rhos)))
print('Number of nans in sigma_eta_pow_xm: ', np.sum(np.isnan(all_sigma_etas_pow_xm)))
print('Number of nans in sigma_rho_pow_xm: ', np.sum(np.isnan(all_sigma_rhos_pow_xm)))