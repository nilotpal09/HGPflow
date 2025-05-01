from splitters.clic_event_splitter import CLICEventSplitter
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument('--input_filepath', '-i', type=str, required=True)
argparser.add_argument('--tree_name', '-t', type=str, required=False, default="Out_Tree")
argparser.add_argument('--entry_start', '-estart', type=int, required=False, default=0)
argparser.add_argument('--entry_stop', '-estop', type=int, required=False, default=None)

argparser.add_argument('--output_filepath', '-o', type=str, required=True)
argparser.add_argument('--is_inference', '-is_inf', action='store_true')
argparser.add_argument('--bandwidth', '-bw', type=float, required=False, default=0.4)
argparser.add_argument('--eventNumber_offset', '-eo', type=int, required=False, default=0)
argparser.add_argument('--pt_min', '-pt', type=float, required=False, default=0.1)
argparser.add_argument('--eta_max', '-eta', type=float, required=False, default=4.0)
argparser.add_argument('--num_nodes_min', '-nm', type=int, required=False, default=1)
argparser.add_argument('--skip_cells', '-sc', action='store_true')

argparser.add_argument('--cluster_jet_idx_filepath', '-cji', type=str, required=False)

args = argparser.parse_args()


input_filepath = args.input_filepath
tree_name = args.tree_name
entry_start = args.entry_start
entry_stop = args.entry_stop

output_filepath = args.output_filepath
is_inference = args.is_inference

bandwidth = args.bandwidth
# n_events = args.num_events
eventNumber_offset = args.eventNumber_offset
pt_min_gev = args.pt_min
eta_max = args.eta_max
num_nodes_min = args.num_nodes_min

skip_cells = args.skip_cells

cluster_jet_idx_filepath = args.cluster_jet_idx_filepath

input_branch_mapping_dict = {
    'particle_pt'  : 'particle_pt',
    'particle_eta' : 'particle_eta',
    'particle_phi' : 'particle_phi',
    'particle_e'   : 'particle_e',
    'particle_pdg' : 'particle_pdgid',
    'particle_gen_status': 'particle_gen_status',
    'particle_sim_status': 'particle_sim_status',
    'particle_dep_e': 'particle_dep_energy',
    'particle_track_idx': 'particle_track_idx',
    'particle_interacted': 'particle_interacted',
    'particle_idx': 'particle_idx',
    'particle_parent_idx': 'particle_parent_idx',

    # 'track_elemtype': 'track_elemtype',
    'track_pt'    : 'track_pt',
    'track_eta'   : 'track_eta',
    'track_phi'   : 'track_phi',
    'track_p'     : 'track_p',
    'track_d0'    : 'track_d0',
    'track_z0'    : 'track_z0',
    'track_chi2'  : 'track_chi2',
    'track_ndf'   : 'track_ndf',
    'track_dedx'  : 'track_dedx',
    'track_dedx_error': 'track_dedx_error',
    'track_radiusofinnermosthit' : 'track_radiusofinnermosthit',
    'track_tanlambda' : 'track_tanlambda',
    'track_omega'     : 'track_omega',
    'track_time'      : 'track_time',

    'track_x_int'   : 'track_x_int',
    'track_y_int'   : 'track_y_int',
    'track_z_int'   : 'track_z_int',
    'track_eta_int' : 'track_eta_int',
    'track_phi_int' : 'track_phi_int',

    'track_particle_idx': 'track_particle_idx',

    'topo_x'         : 'topo_x',
    'topo_y'         : 'topo_y',
    'topo_z'         : 'topo_z',

    'topo_eta'       : 'topo_eta',
    'topo_phi'       : 'topo_phi',
    'topo_rho'       : 'topo_rho',
    'topo_e'         : 'topo_e',
    # 'topo_et'        : 'topo_et',
    'topo_num_cells' : 'topo_num_cells',
    'topo_sigma_eta' : 'topo_sigma_eta',
    'topo_sigma_phi' : 'topo_sigma_phi',
    'topo_sigma_rho' : 'topo_sigma_rho',
    'topo_energy_ecal' : 'topo_energy_ecal',
    'topo_energy_hcal' : 'topo_energy_hcal',
    'topo_energy_other': 'topo_energy_other',

    'particle_topo_start': 'particle_topo_start',
    'particle_topo_end'  : 'particle_topo_end',
    'particle_topo_wt'   : 'particle_topo_e',
}

input_cell_branch_mapping_dict = {
    'cell_eta'    : 'cell_eta',
    'cell_phi'    : 'cell_phi',
    'cell_rho'    : 'cell_rho',
    'cell_e'      : 'cell_e',
    'cell_x'      : 'cell_x',
    'cell_y'      : 'cell_y',
    'cell_z'      : 'cell_z',
    'cell_subdet' : 'cell_subdet',

    'cell_to_topo_idx': 'cell_topo_idx',
}


output_branch_dict = {
    'topo_x'        : 'vec(float)',
    'topo_y'        : 'vec(float)',
    'topo_z'        : 'vec(float)',

    'topo_eta'      : 'vec(float)',
    'topo_phi'      : 'vec(float)',
    'topo_rho'      : 'vec(float)',
    'topo_e'        : 'vec(float)',
    # 'topo_et'       : 'vec(float)',
    'topo_num_cells': 'vec(int)',
    'topo_sigma_eta': 'vec(float)',
    'topo_sigma_phi': 'vec(float)',
    'topo_sigma_rho': 'vec(float)',
    'topo_energy_ecal' : 'vec(float)',
    'topo_energy_hcal' : 'vec(float)',
    'topo_energy_other': 'vec(float)',

    'topo_ms_cluster_idx': 'vec(int)',
    'topo_particle_idxs'    : 'vec(vec(int))',
    'topo_particle_energies': 'vec(vec(float))',

    'topo2particle_topo_idx': 'vec(int)',
    'topo2particle_particle_idx': 'vec(int)',
    'topo2particle_energy': 'vec(float)',

    # 'track_elemtype' : 'vec(int)',
    'track_pt'       : 'vec(float)',
    'track_eta'      : 'vec(float)',
    'track_phi'      : 'vec(float)',
    'track_p'        : 'vec(float)',
    'track_d0'       : 'vec(float)',
    'track_z0'       : 'vec(float)',
    'track_chi2'     : 'vec(float)',
    'track_ndf'      : 'vec(float)',
    'track_dedx'     : 'vec(float)',
    'track_dedx_error': 'vec(float)',
    'track_radiusofinnermosthit' : 'vec(float)',
    'track_tanlambda' : 'vec(float)',
    'track_omega'     : 'vec(float)',
    'track_time'      : 'vec(float)',

    'track_x_int'    : 'vec(float)',
    'track_y_int'    : 'vec(float)',
    'track_z_int'    : 'vec(float)',
    'track_eta_int'  : 'vec(float)',
    'track_phi_int'  : 'vec(float)',

    'track_ms_cluster_idx': 'vec(int)',
    'track_particle_idx': 'vec(int)',

    'particle_pt'   : 'vec(float)',
    'particle_eta'  : 'vec(float)',
    'particle_phi'  : 'vec(float)',
    'particle_e'    : 'vec(float)',
    'particle_pdgid': 'vec(int)',
    'particle_gen_status': 'vec(int)',
    'particle_sim_status': 'vec(int)',
    'particle_dep_e': 'vec(float)',
    'particle_old_idx': 'vec(int)',
    'particle_ms_cluster_idx': 'vec(int)',
}

output_cell_branch_dict = {
   'cell_eta'        : 'vec(float)',
    'cell_phi'        : 'vec(float)',
    'cell_rho'        : 'vec(float)',
    'cell_e'          : 'vec(float)',
    'cell_x'          : 'vec(float)',
    'cell_y'          : 'vec(float)',
    'cell_z'          : 'vec(float)',
    'cell_calo_region': 'vec(int)',

    'cell_topo_idx'   : 'vec(int)',
}

if not skip_cells:
    input_branch_mapping_dict.update(input_cell_branch_mapping_dict)
    output_branch_dict.update(output_cell_branch_dict)



base_splitter_init_config = {
    'input_filepath': input_filepath,
    'tree_name': tree_name,
    'entry_start': entry_start,
    'entry_stop': entry_stop,
    'input_br_map_dict': input_branch_mapping_dict,

    'output_filepath': output_filepath,
    'output_br_dict': output_branch_dict,
    'bandwidth': bandwidth,
    'is_inference': is_inference,
    'topo_part_frac_e_th': 0.0,
    'eventNumber_offset': eventNumber_offset,

    'chunk_size': 300,
    'n_proc': 1,
}


ms_event_splitter = CLICEventSplitter(
    base_splitter_init_config, 
    pt_min_gev=pt_min_gev, eta_max=eta_max, num_nodes_min=num_nodes_min, skip_cells=skip_cells,
    cluster_jet_idx_filepath=cluster_jet_idx_filepath)

ms_event_splitter.split_events()
