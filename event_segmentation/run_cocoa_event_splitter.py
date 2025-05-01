from splitters.cocoa_event_splitter import COCOAEventSplitter
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
argparser.add_argument('--num_nodes_min', '-nm', type=int, required=False, default=1)
argparser.add_argument('--mod_e_th', type=float, required=False, default=0.0)
argparser.add_argument('--skip_cells', '-sc', action='store_true')
args = argparser.parse_args()


input_filepath = args.input_filepath
tree_name = args.tree_name
entry_start = args.entry_start
entry_stop = args.entry_stop

output_filepath = args.output_filepath
is_inference = args.is_inference

bandwidth = args.bandwidth
eventNumber_offset = args.eventNumber_offset
num_nodes_min = args.num_nodes_min
mod_e_th = args.mod_e_th

skip_cells = args.skip_cells



input_branch_mapping_dict = {
    'particle_pt'   : 'particle_pt',
    'particle_eta'  : 'particle_eta',
    'particle_phi'  : 'particle_phi',
    'particle_e'    : 'particle_e',
    'particle_pdgid': 'particle_pdgid',
    'particle_track_idx': 'particle_track_idx',

    'track_theta'     : 'track_theta',
    'track_phi'       : 'track_phi',
    'track_qoverp'    : 'track_qoverp',
    'track_parent_idx': 'track_particle_idx',

    'track_d0' : 'track_d0',
    'track_z0' : 'track_z0',
    'track_reconstructed' : 'track_reconstructed',
    'track_in_acceptance' : 'track_in_acceptance',

    'track_x_layer_0' : 'track_x_layer_0',
    'track_x_layer_1' : 'track_x_layer_1',
    'track_x_layer_2' : 'track_x_layer_2',
    'track_x_layer_3' : 'track_x_layer_3',
    'track_x_layer_4' : 'track_x_layer_4',
    'track_x_layer_5' : 'track_x_layer_5',

    'track_y_layer_0' : 'track_y_layer_0',
    'track_y_layer_1' : 'track_y_layer_1',
    'track_y_layer_2' : 'track_y_layer_2',
    'track_y_layer_3' : 'track_y_layer_3',
    'track_y_layer_4' : 'track_y_layer_4',
    'track_y_layer_5' : 'track_y_layer_5',

    'track_z_layer_0' : 'track_z_layer_0',
    'track_z_layer_1' : 'track_z_layer_1',
    'track_z_layer_2' : 'track_z_layer_2',
    'track_z_layer_3' : 'track_z_layer_3',
    'track_z_layer_4' : 'track_z_layer_4',
    'track_z_layer_5' : 'track_z_layer_5',

    'topo_bary_eta'      : 'topo_eta',
    'topo_bary_phi'      : 'topo_phi',
    'topo_e'             : 'topo_e',
    'topo_bary_rho'      : 'topo_rho',
    'topo_bary_sigma_eta': 'topo_sigma_eta',
    'topo_bary_sigma_phi': 'topo_sigma_phi',

    'cell_e'            : 'cell_e', # needed for topo ecal/hcal energy
    'cell_layer'        : 'cell_calo_region', # needed for topo ecal/hcal energy
    'cell_parent_list'  : 'cell_particle_idxs', # needed for incidence computation
    'cell_parent_energy': 'cell_particle_energies', # needed for incidence computation
    'cell_topo_idx'     : 'cell_topo_idx', # needed for incidence computation + topo ecal/hcal energy
}    

input_cell_branch_mapping_dict = {
    'cell_x'            : 'cell_x',
    'cell_y'            : 'cell_y',
    'cell_z'            : 'cell_z',
    'cell_eta'          : 'cell_eta',
    'cell_phi'          : 'cell_phi',

    'cell_to_cell_edge_start' : 'cell_to_cell_edge_start',
    'cell_to_cell_edge_end'   : 'cell_to_cell_edge_end',
    'track_to_cell_edge_start': 'track_to_cell_edge_start',
    'track_to_cell_edge_end'  : 'track_to_cell_edge_end',
}


output_branch_dict = {
    'topo_eta'      : 'vec(float)',
    'topo_phi'      : 'vec(float)',
    'topo_e'        : 'vec(float)',
    'topo_rho'      : 'vec(float)',
    'topo_sigma_eta': 'vec(float)',
    'topo_sigma_phi': 'vec(float)',
    'topo_ms_cluster_idx': 'vec(int)',
    'topo_particle_idxs'    : 'vec(vec(int))',
    'topo_particle_energies': 'vec(vec(float))',
    'topo_ecal_e'   : 'vec(float)',
    'topo_hcal_e'   : 'vec(float)',

    'topo2particle_topo_idx': 'vec(int)',
    'topo2particle_particle_idx': 'vec(int)',
    'topo2particle_energy': 'vec(float)',

    'track_eta'     : 'vec(float)',
    'track_phi'     : 'vec(float)',
    'track_pt'      : 'vec(float)',
    'track_ms_cluster_idx': 'vec(int)',
    'track_particle_idx': 'vec(int)',

    'track_z0' : 'vec(float)',
    'track_d0' : 'vec(float)',

    'track_x_layer_0' : 'vec(float)',
    'track_x_layer_1' : 'vec(float)',
    'track_x_layer_2' : 'vec(float)',
    'track_x_layer_3' : 'vec(float)',
    'track_x_layer_4' : 'vec(float)',
    'track_x_layer_5' : 'vec(float)',

    'track_y_layer_0' : 'vec(float)',
    'track_y_layer_1' : 'vec(float)',
    'track_y_layer_2' : 'vec(float)',
    'track_y_layer_3' : 'vec(float)',
    'track_y_layer_4' : 'vec(float)',
    'track_y_layer_5' : 'vec(float)',

    'track_z_layer_0' : 'vec(float)',
    'track_z_layer_1' : 'vec(float)',
    'track_z_layer_2' : 'vec(float)',
    'track_z_layer_3' : 'vec(float)',
    'track_z_layer_4' : 'vec(float)',
    'track_z_layer_5' : 'vec(float)',

    'track_eta_layer_0' : 'vec(float)',
    'track_eta_layer_1' : 'vec(float)',
    'track_eta_layer_2' : 'vec(float)',
    'track_eta_layer_3' : 'vec(float)',
    'track_eta_layer_4' : 'vec(float)',
    'track_eta_layer_5' : 'vec(float)',

    'track_phi_layer_0' : 'vec(float)',
    'track_phi_layer_1' : 'vec(float)',
    'track_phi_layer_2' : 'vec(float)',
    'track_phi_layer_3' : 'vec(float)',
    'track_phi_layer_4' : 'vec(float)',
    'track_phi_layer_5' : 'vec(float)',

    'track_eta_int': 'vec(float)',
    'track_phi_int': 'vec(float)',

    'particle_pt'   : 'vec(float)',
    'particle_eta'  : 'vec(float)',
    'particle_phi'  : 'vec(float)',
    'particle_e'    : 'vec(float)',
    'particle_pdgid': 'vec(int)',
    'particle_old_idx': 'vec(int)',
    'particle_ms_cluster_idx': 'vec(int)',

}

output_cell_branch_dict = {
    'cell_eta'        : 'vec(float)',
    'cell_phi'        : 'vec(float)',
    'cell_e'          : 'vec(float)',
    'cell_calo_region': 'vec(int)',
    'cell_topo_idx'   : 'vec(int)',
    'cell_x'          : 'vec(float)',
    'cell_y'          : 'vec(float)',
    'cell_z'          : 'vec(float)',

    'cell_to_cell_src' : 'vec(int)',
    'cell_to_cell_dst' : 'vec(int)',
    'track_to_cell_src': 'vec(int)',
    'track_to_cell_dst': 'vec(int)'
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

ms_event_splitter = COCOAEventSplitter(
    base_splitter_init_config, num_nodes_min=num_nodes_min, skip_cells=skip_cells, mod_e_th=mod_e_th)

ms_event_splitter.split_events()
