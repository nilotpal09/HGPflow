import uproot
import numpy as np
from tqdm import tqdm
import gc
from ..utility.helper_dicts import pdgid_class_dict, class_mass_dict


def load_pred_hgpflow(pred_path, threshold=0.5):
    tree = uproot.open(pred_path)['event_tree']

    vars_to_load = [
        'pred_ind', 'proxy_pt', 'proxy_eta', 'proxy_phi',
        'hgpflow_pt', 'hgpflow_eta', 'hgpflow_phi', 'hgpflow_class']

    mask = np.array([x > threshold for x in tree['pred_ind'].array(library='np')], dtype=object)

    hgpflow_dict = {}
    for var in tqdm(vars_to_load, desc="Loading HGPFlow predictions...", total=len(vars_to_load)):
        hgpflow_dict[var] = np.array([
            x[m] for x, m in zip(tree[var].array(library='np'), mask)], dtype=object)
    hgpflow_dict['event_number'] = tree['event_number'].array(library='np').astype(int) # - 643_00 # HACK

    # compute mass and energy
    for k in ['mass', 'e', 'charge']:
        hgpflow_dict[f'hgpflow_{k}'] = np.empty_like(hgpflow_dict['hgpflow_pt'], dtype=object)

    for i, cls in tqdm(enumerate(hgpflow_dict['hgpflow_class']), desc="Computing HGPFlow mass...", total=len(hgpflow_dict['hgpflow_class'])):
        hgpflow_dict['hgpflow_mass'][i] = np.array([class_mass_dict[x] for x in cls])
        p = hgpflow_dict['hgpflow_pt'][i] * np.cosh(hgpflow_dict['hgpflow_eta'][i])
        hgpflow_dict['hgpflow_e'][i] = np.sqrt(p**2 + hgpflow_dict['hgpflow_mass'][i]**2)

        hgpflow_dict['hgpflow_charge'][i] = np.full_like(hgpflow_dict['hgpflow_pt'][i], 0)
        hgpflow_dict['hgpflow_charge'][i][cls <= 2] = 1

    return hgpflow_dict



def load_pred_mlpf(pred_path, num_ev_in_one_file, truth_event_number_offset):
    class_remap = {
        1.:0, # ch_had
        2.:3, # neut had
        3.:4, # photon
        4.:1, # electron
        5.:2  # muon
    }

    tree = uproot.open(pred_path)['parts']
    vars_to_load = ['pred_pt', 'pred_phi', 'pred_eta', 'pred_cl', 'pred_e']

    mlpf_class  = tree['pred_cl'].array(library='np')
    mlpf_mask_cl = np.array([x!=0 for x in mlpf_class], dtype=object)

    mlpf_dict = {}
    for var in tqdm(vars_to_load, desc="Loading MLPF predictions...", total=len(vars_to_load)):
        mlpf_dict[var] = np.array([
            x[m] for x, m in zip(tree[var].array(library='np'), mlpf_mask_cl)], dtype=object)

    # class remapping
    mlpf_dict['pred_cl'] = np.array([
        np.array([class_remap[x] for x in cls]) for cls in mlpf_dict['pred_cl']], dtype=object)

    # charge (0 for netural particles, 1 for charged particles)
    mlpf_dict['pred_charge'] = np.empty_like(mlpf_dict['pred_pt'])
    for i, cls in tqdm(enumerate(mlpf_dict['pred_cl']), desc="Computing MLPF charge...", total=len(mlpf_dict['pred_cl'])):
        mlpf_dict['pred_charge'][i] = np.full_like(cls, 0)
        mlpf_dict['pred_charge'][i][cls <= 2] = 1

    # compute event number
    mlpf_dict['event_id'] = tree['event_id'].array(library='np')
    # mlpf_dict['file_id'] = tree['file_id'].array(library='np')
    # mlpf_dict['file_id'] = mlpf_dict['file_id'] - mlpf_dict['file_id'].min()
    # mlpf_dict['event_number'] = mlpf_dict['event_id'] + mlpf_dict['file_id'] * num_ev_in_one_file + truth_event_number_offset

    mlpf_dict['event_number'] = mlpf_dict['event_id'] + truth_event_number_offset

    # compute mass
    mlpf_dict['pred_mass'] = np.empty_like(mlpf_dict['pred_pt'])
    for i, cls in tqdm(enumerate(mlpf_dict['pred_cl']), desc="Computing MLPF mass...", total=len(mlpf_dict['pred_cl'])):
        mlpf_dict['pred_mass'][i] = np.array([class_mass_dict[x] for x in cls])

    return mlpf_dict


def load_truth_cocoa(truth_path, topo=False):
    scale_E_pT=1e-3
    print("\033[96m" + f"E, pT will be scaled by {scale_E_pT}" + "\033[0m")

    tree = uproot.open(truth_path)['Out_Tree']
    n_events = tree.num_entries

    truth_dict = {}
    vars_to_load = [
        'particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdgid',
        'pflow_e', 'pflow_eta', 'pflow_phi', 'pflow_charge', 'pflow_px', 'pflow_py']

    for var in tqdm(vars_to_load, desc="Reading truth tree...", total=len(vars_to_load)):
        truth_dict[var] = tree[var].array(library='np')

    # pflow pt
    truth_dict['pflow_pt'] = np.array([
        np.sqrt(x**2 + y**2) for x, y in zip(truth_dict['pflow_px'], truth_dict['pflow_py'])
    ], dtype=object)

    # MeV to GeV
    truth_dict['particle_pt'] = truth_dict['particle_pt'] * scale_E_pT
    truth_dict['particle_e'] = truth_dict['particle_e'] * scale_E_pT
    truth_dict['pflow_e'] = truth_dict['pflow_e'] * scale_E_pT
    truth_dict['pflow_pt'] = truth_dict['pflow_pt'] * scale_E_pT

    # particle class and charge
    truth_dict['particle_class'] = np.empty_like(truth_dict['particle_pdgid'])
    truth_dict['particle_charge'] = np.empty_like(truth_dict['particle_pdgid'])
    for i, pdgid in tqdm(enumerate(truth_dict['particle_pdgid']), desc="Computing particle class...", total=n_events):
        truth_dict['particle_class'][i] = np.array([pdgid_class_dict[x] for x in pdgid])
        truth_dict['particle_charge'][i] = np.array([1 if x <= 2 else 0 for x in truth_dict['particle_class'][i]])

    # delete unnecessary variables
    vars_to_delete = ['particle_pdgid', 'pflow_px', 'pflow_py']
    for var in vars_to_delete:
        del truth_dict[var]
    gc.collect()

    # fiducial cuts
    for i in tqdm(range(n_events), desc="Applying fiducial cuts...", total=n_events):
        
        # on particles (pt > 1 GeV, |eta| < 3)
        mask = (truth_dict['particle_pt'][i] >= 1) * (abs(truth_dict['particle_eta'][i]) < 3)
        for var in ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_class']:
            truth_dict[var][i] = truth_dict[var][i][mask]

        # on ppflow (|eta| < 3)
        mask = abs(truth_dict['pflow_eta'][i]) < 3
        for var in ['pflow_pt', 'pflow_eta', 'pflow_phi', 'pflow_e', 'pflow_charge']:
            truth_dict[var][i] = truth_dict[var][i][mask]

    # reading topo
    if topo:
        truth_dict['topo_e'] = tree['topo_e'].array(library='np') * scale_E_pT
        truth_dict['topo_eta'] = tree['topo_bary_eta'].array(library='np')
        truth_dict['topo_phi'] = tree['topo_bary_phi'].array(library='np')

        truth_dict['topo_pt'] = np.empty_like(truth_dict['topo_e'])
        for i in tqdm(range(n_events), desc="Computing topo pt...", total=n_events):
            truth_dict['topo_pt'][i] = truth_dict['topo_e'][i] / np.cosh(truth_dict['topo_eta'][i])

    # compute mass (truth)
    truth_dict['particle_mass'] = []
    for pt, eta, e in zip(truth_dict['particle_pt'], truth_dict['particle_eta'], truth_dict['particle_e']):
        p = pt * np.cosh(eta)
        truth_dict['particle_mass'].append(np.sqrt(e**2 - p**2))
    truth_dict['particle_mass'] = np.array(truth_dict['particle_mass'], dtype=object)

    # compute mass (pflow)
    truth_dict['pflow_mass'] = []
    for pt, eta, e in zip(truth_dict['pflow_pt'], truth_dict['pflow_eta'], truth_dict['pflow_e']):
        p = pt * np.cosh(eta)
        truth_dict['pflow_mass'].append(np.sqrt(e**2 - p**2))
    truth_dict['pflow_mass'] = np.array(truth_dict['pflow_mass'], dtype=object)

    if 'event_number' in tree.keys():
        truth_dict['event_number'] = tree['event_number'].array(library='np')
    else:
        truth_dict['event_number'] = np.arange(len(truth_dict['particle_pt']))

    return truth_dict


def load_truth_clic(truth_path, event_number_offset=0):
    scale_E_pT=1
    # pt_min_gev = 0.1
    print("\033[96m" + f"E, pT will be scaled by {scale_E_pT}" + "\033[0m")

    tree = uproot.open(truth_path)['events']
    n_events = tree.num_entries

    truth_dict = {}
    vars_to_load = [
        'particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdg', 'particle_gen_status',
        'pandora_e', 'pandora_eta', 'pandora_phi', 'pandora_pt', 'pandora_pdg']
    
    for var in tqdm(vars_to_load, desc="Reading truth tree...", total=len(vars_to_load)):
        truth_dict[var] = tree[var].array(library='np')

    # MeV to GeV scaling not needed (already in GeV)

    # particle class and charge
    truth_dict['particle_class'] = np.empty_like(truth_dict['particle_pdg'])
    truth_dict['particle_charge'] = np.empty_like(truth_dict['particle_pdg'])
    for i, pdgid in tqdm(enumerate(truth_dict['particle_pdg']), desc="Computing particle class...", total=n_events):
        truth_dict['particle_class'][i] = np.array([pdgid_class_dict.get(x, 4) for x in pdgid])
        truth_dict['particle_charge'][i] = np.array([1 if x <= 2 else 0 for x in truth_dict['particle_class'][i]])

    # pandora class and charge
    truth_dict['pandora_class'] = np.empty_like(truth_dict['pandora_pdg'])
    truth_dict['pandora_charge'] = np.empty_like(truth_dict['pandora_pdg'])
    for i, pdgid in tqdm(enumerate(truth_dict['pandora_pdg']), desc="Computing pandora class...", total=n_events):
        truth_dict['pandora_class'][i] = np.array([pdgid_class_dict.get(x, 4) for x in pdgid])
        truth_dict['pandora_charge'][i] = np.array([1 if x <= 2 else 0 for x in truth_dict['pandora_class'][i]])

    # delete unnecessary variables
    vars_to_delete = ['particle_pdg', 'pandora_pdg']
    for var in vars_to_delete:
        del truth_dict[var]
    gc.collect()

    # fiducial cuts
    for i in tqdm(range(n_events), desc="Applying fiducial cuts...", total=n_events):
            
        # on particles (gen_status=1)
        mask = (truth_dict['particle_gen_status'][i] == 1) # * (truth_dict['particle_pt'][i] >= pt_min_gev)
        for var in ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_class', 'particle_gen_status']:
            truth_dict[var][i] = truth_dict[var][i][mask]

    if 'event_number' in tree.keys():
        truth_dict['event_number'] = tree['event_number'].array(library='np').astype(int)
    else:
        truth_dict['event_number'] = np.arange(len(truth_dict['particle_pt'])) + event_number_offset

    return truth_dict



def load_hgpflow_target(target_path, drop_res=True):
    tree = uproot.open(target_path)['EventTree']
    vars_to_load = ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_pdgid', 'eventNumber']
    remap = {'eventNumber': 'event_number'}

    hgpflow_target_dict_tmp = {}
    for var in tqdm(vars_to_load, desc="Loading HGPflow target (segmented)...", total=len(vars_to_load)):
        new_var = remap.get(var, var)
        hgpflow_target_dict_tmp[new_var] = tree[var].array(library='np')

    # filter out the residual particles
    if drop_res: # will be dafault, here it is just for debugging
        mask = np.array([
            np.array([pdgid_class_dict[xx] for xx in x]) <= 4 \
            for x in hgpflow_target_dict_tmp['particle_pdgid']], dtype=object)
        for key, val in hgpflow_target_dict_tmp.items():
            if key == 'event_number':
                continue
            hgpflow_target_dict_tmp[key] = np.array([
                x[m] for x, m in zip(val, mask)], dtype=object)

    unique_sorted_ev_num = np.sort(np.unique(hgpflow_target_dict_tmp['event_number']))
    hgpflow_target_dict = {}
    for key, val in tqdm(hgpflow_target_dict_tmp.items(), desc="Merging HGPflow target...", total=len(hgpflow_target_dict_tmp)):
        hgpflow_target_dict[key] = []
        for ev_num in unique_sorted_ev_num:
            mask = hgpflow_target_dict_tmp['event_number'] == ev_num
            hgpflow_target_dict[key].append(np.hstack(val[mask]))
        hgpflow_target_dict[key] = np.array(hgpflow_target_dict[key], dtype=object)
    hgpflow_target_dict['event_number'] = unique_sorted_ev_num

    # compute mass
    hgpflow_target_dict['particle_mass'] = np.empty_like(hgpflow_target_dict['particle_pt'])
    for i, pdgid in tqdm(enumerate(hgpflow_target_dict['particle_pdgid']), desc="Computing HGPFlow target mass...", total=len(hgpflow_target_dict['particle_pdgid'])):
        hgpflow_target_dict['particle_mass'][i] = np.array([class_mass_dict[pdgid_class_dict[x]] for x in pdgid])

    return hgpflow_target_dict