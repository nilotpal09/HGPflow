import numpy as np
from tqdm import tqdm

from splitters.ms_event_splitter import deltaR, MSEventSplitter
import awkward as ak

class COCOAEventSplitter(MSEventSplitter):

    def __init__(self, base_splitter_init_config, num_nodes_min, skip_cells, mod_e_th):
        super().__init__(**base_splitter_init_config)
        self.num_nodes_min = num_nodes_min
        self.skip_cells = skip_cells
        self.mod_e_th = mod_e_th

    def branch_data_update(self):
        # track
        track_sin_theta = np.array([np.sin(x) for x in self.branch_data['track_theta']], dtype=object)
        self.branch_data['track_pt'] = np.array([np.abs(1/x) * y for x, y in zip(self.branch_data['track_qoverp'], track_sin_theta)], dtype=object)
        self.branch_data['track_eta'] = np.array([-np.log(np.tan(x/2)) for x in self.branch_data['track_theta']], dtype=object)

        # extrapolation from x,y,z to eta,phi
        # doing the empty stuff, otherwise setting array.dtype=obj will force the element to be object (not float) if the array is kinda homogeneous
        for layer in range(6):
            rho = np.empty_like(self.branch_data[f'track_z_layer_{layer}'])
            rho[:] = [np.sqrt(x**2 + y**2) for x, y in zip(self.branch_data[f'track_x_layer_{layer}'], self.branch_data[f'track_y_layer_{layer}'])]

            r = np.empty_like(self.branch_data[f'track_z_layer_{layer}'])
            r[:] = [np.sqrt(rho_**2 + z_**2) for rho_, z_ in zip(rho, self.branch_data[f'track_z_layer_{layer}'])]

            theta = np.empty_like(self.branch_data[f'track_z_layer_{layer}'])
            theta[:] = [np.arccos(z_/r_) for z_, r_ in zip(self.branch_data[f'track_z_layer_{layer}'], r)]

            self.branch_data[f'track_eta_layer_{layer}'] = np.empty_like(self.branch_data[f'track_z_layer_{layer}'])
            self.branch_data[f'track_eta_layer_{layer}'][:] = [-np.log(np.tan(theta_/2)) for theta_ in theta]
            
            self.branch_data[f'track_phi_layer_{layer}'] = np.empty_like(self.branch_data[f'track_z_layer_{layer}'])
            self.branch_data[f'track_phi_layer_{layer}'][:] = [np.arctan2(y_,x_) for x_, y_ in zip(self.branch_data[f'track_x_layer_{layer}'], self.branch_data[f'track_y_layer_{layer}'])]

        self.branch_data['track_eta_int'] = self.branch_data['track_eta_layer_0']
        self.branch_data['track_phi_int'] = self.branch_data['track_phi_layer_0']

        # cell_topo_idx starts from 1. make it start from 0
        self.branch_data['cell_topo_idx'] = self.branch_data['cell_topo_idx'] - 1

        # topo particle energy relation
        self.branch_data['topo_particle_idxs'], self.branch_data['topo_particle_energies'] = \
            self.invert_topo_part(
                self.branch_data['cell_particle_idxs'], self.branch_data['cell_particle_energies'], self.branch_data['cell_topo_idx'])

        # total energy deposited by the particles 
        self.branch_data['particle_dep_energy'] = []
        for i in range(self.n_events):
            tmp = np.zeros(len(self.branch_data['particle_pt'][i]))
            flatten_idx = np.hstack(self.branch_data['topo_particle_idxs'][i]).astype(int) if len(self.branch_data['topo_particle_idxs'][i]) != 0 else np.array([], dtype=int)
            flatten_energies = np.hstack(self.branch_data['topo_particle_energies'][i]) if len(self.branch_data['topo_particle_energies'][i]) != 0 else np.array([], dtype=float)
            np.add.at(tmp, flatten_idx, flatten_energies)
            self.branch_data['particle_dep_energy'].append(tmp)
        self.branch_data['particle_dep_energy'] = np.array(self.branch_data['particle_dep_energy'], dtype=object)

        # remap -1 to -9999 for particle_track_idx (our convention). 
        #   It appears track_particle_idx is never -1 (i.e. no fake tracks), so skipping it for now.
        # self.branch_data['track_particle_idx'] = np.array([np.where(idxs<0, -9999, idxs) for idxs in self.branch_data['track_particle_idx']], dtype=object)
        self.branch_data['particle_track_idx'] = np.array([np.where(idxs<0, -9999, idxs) for idxs in self.branch_data['particle_track_idx']], dtype=object)

        # MeV to GeV
        for k in ['particle_pt', 'particle_e', 'track_pt', 'topo_e', 'cell_e', 'particle_dep_energy']:
            self.branch_data[k] = self.branch_data[k] / 1000.
        self.branch_data['topo_particle_energies'] = [x/1000. for x in self.branch_data['topo_particle_energies']]

        self.particle_dep_e_total = self.branch_data['particle_dep_energy'].copy()

        # track phi needs to be in the range [-pi, pi] # should be fixed now
        self.branch_data['track_phi'] = (self.branch_data['track_phi'] + np.pi) % (2*np.pi) - np.pi

        # annoying must dos :(
        if track_sin_theta[0].dtype == object:
            self.branch_data['track_pt'] = self.branch_data['track_pt'].astype(np.float64)
            self.branch_data['track_eta'] = self.branch_data['track_eta'].astype(np.float64)

            self.branch_data['particle_dep_energy'] = self.branch_data['particle_dep_energy'].astype(np.float64)
            self.branch_data['particle_track_idx'] = self.branch_data['particle_track_idx'].astype(np.float64)


    def invert_topo_part(self, cell_parent_list, cell_parent_energy, cell_topo_idx):

        topo_particle_idx = []; topo_particle_energy = []
        for event in tqdm(range(len(cell_parent_list)), desc='Inverting topo-particle relation'):
            tmp_topo_particle_idx = []; tmp_topo_particle_energy = []
            unique_topo_idx = np.unique(cell_topo_idx[event])
            for topo_idx in unique_topo_idx:
                cell_idx_in_topo = np.where(cell_topo_idx[event] == topo_idx)[0]
                # slow
                # p_indices = np.hstack([ak.to_numpy(x).astype(int) for x in cell_parent_list[event][cell_idx_in_topo]])
                # p_energies = np.hstack([ak.to_numpy(x).astype(float) for x in cell_parent_energy[event][cell_idx_in_topo]])

                # also slow (faster with another uproot version. no need for ak->np, root_env)
                p_indices = ak.to_numpy(ak.concatenate([x for x in cell_parent_list[event][cell_idx_in_topo]])).astype(int)
                p_energies = ak.to_numpy(ak.concatenate([x for x in cell_parent_energy[event][cell_idx_in_topo]]))

                if len(p_indices) == 0:
                    tmp_topo_particle_idx.append(np.array([]))
                    tmp_topo_particle_energy.append(np.array([]))
                    continue

                max_index = np.max(p_indices)
                summed_energies = np.zeros(max_index + 1)
                np.add.at(summed_energies, p_indices, p_energies)

                part_indices = np.nonzero(summed_energies)[0]
                part_energies = summed_energies[part_indices]

                tmp_topo_particle_idx.append(part_indices)
                tmp_topo_particle_energy.append(part_energies)
            
            topo_particle_idx.append(np.array(tmp_topo_particle_idx, dtype=object))
            topo_particle_energy.append(np.array(tmp_topo_particle_energy, dtype=object))

        return topo_particle_idx, topo_particle_energy



    def get_track_topo_part_mask_ev(self, ev_i):

        # part mask ev
        nonzero_dep_e_nu_mask = (self.branch_data['particle_dep_energy'][ev_i] > 0) + (self.branch_data['particle_track_idx'][ev_i] >= 0) #!= -9999)
        mask_part = nonzero_dep_e_nu_mask

        if self.is_inference:
            mask_topo = np.ones_like(self.branch_data['topo_eta'][ev_i], dtype=bool)
            mask_track = np.ones_like(self.branch_data['track_eta'][ev_i], dtype=bool)

            return mask_track, mask_topo, mask_part


        TRACK_PART_PT_RES_TH = 2
        TRACK_PART_DR_TH = 0.15

        # topo mask ev
        # eta_mask = np.abs(self.branch_data['topo_eta'][ev_i]) < self.eta_max
        # e_mask   = self.branch_data['topo_e'][ev_i] > self.pt_min_gev
        # mask_topo = eta_mask * e_mask
        mask_topo = np.ones_like(self.branch_data['topo_eta'][ev_i], dtype=bool)

        # track mask ev
        # no direct cut on track, cuts are applied based on associated particles

        # mask on associated particles for training        
        tpi = self.branch_data['track_particle_idx'][ev_i]
        assoc_part_pt = self.branch_data['particle_pt'][ev_i][tpi]
        assoc_part_eta = self.branch_data['particle_eta'][ev_i][tpi]
        assoc_part_phi = self.branch_data['particle_phi'][ev_i][tpi]

        tr_pt = self.branch_data['track_pt'][ev_i]
        tr_eta = self.branch_data['track_eta'][ev_i]; tr_phi = self.branch_data['track_phi'][ev_i]

        resolution_mask = np.abs((tr_pt - assoc_part_pt) / (assoc_part_pt + 1e-8)) < TRACK_PART_PT_RES_TH
        dr_with_assoc_part_mask = deltaR(tr_eta, tr_phi, assoc_part_eta, assoc_part_phi) < TRACK_PART_DR_TH

        mask_track = resolution_mask * dr_with_assoc_part_mask

        return mask_track, mask_topo, mask_part



    def decorate_mscluster(self, track_mask, topo_mask, part_mask_ev, ev_i, ms_i):

        particle_dep_e_frac_msi_num = np.zeros_like(self.branch_data['particle_pt'][ev_i], dtype=float)
        particle_mask_msi = np.zeros_like(part_mask_ev, dtype=bool)

        track_indices_to_keep = np.where(track_mask)[0]
        topo_indices_to_keep = np.where(topo_mask)[0]
        particle_indices_to_keep = [] # we can drop neutral particles based on certain conditions, but charged particles always kept


        # topos
        _topo_vars = ['topo_eta', 'topo_phi', 'topo_e', 'topo_rho', 'topo_sigma_eta', 'topo_sigma_phi']
        topo_write_dict = {}
        for var in _topo_vars:
            topo_write_dict[var] = self.branch_data[var][ev_i][topo_mask]
        topo_write_dict['topo_ms_cluster_idx'] = np.full_like(topo_write_dict['topo_eta'], int(ms_i), dtype=int)
        
        # topo ecal and hcal energy
        cell_ecal_mask = self.branch_data['cell_calo_region'][ev_i] <= 2
        cell_hcal_mask = self.branch_data['cell_calo_region'][ev_i] > 2
        topo_write_dict['topo_ecal_e'] = np.zeros(len(topo_indices_to_keep), dtype=float)
        topo_write_dict['topo_hcal_e'] = np.zeros(len(topo_indices_to_keep), dtype=float)
        for _i, topo_i in enumerate(topo_indices_to_keep):
            this_topo_cell_mask = self.branch_data['cell_topo_idx'][ev_i] == topo_i
            topo_write_dict['topo_ecal_e'][_i] = self.branch_data['cell_e'][ev_i][this_topo_cell_mask * cell_ecal_mask].sum()
            topo_write_dict['topo_hcal_e'][_i] = self.branch_data['cell_e'][ev_i][this_topo_cell_mask * cell_hcal_mask].sum()


        # cells
        if not self.skip_cells:
            _cell_vars = ['cell_eta', 'cell_phi', 'cell_e', 'cell_x', 'cell_y', 'cell_z']

            # if no cell (coz no topo)
            cell_write_dict = {}
            for var in _cell_vars:
                cell_write_dict[var] = np.array([])
            cell_write_dict['cell_calo_region'] = np.array([], dtype=int)
            cell_write_dict['cell_topo_idx'] = np.array([], dtype=int)

            new_topo_idx = 0
            cell_indices_to_keep = []

            for topo_i in topo_indices_to_keep:
                cell_mask_for_this_topo = self.branch_data['cell_topo_idx'][ev_i] == topo_i
                cell_write_dict['cell_topo_idx'] = np.hstack([
                    cell_write_dict['cell_topo_idx'], np.full(cell_mask_for_this_topo.sum(), new_topo_idx, dtype=int)])
                cell_indices_to_keep.append(np.where(cell_mask_for_this_topo)[0])
                new_topo_idx += 1

            if len(cell_indices_to_keep) != 0:
                cell_indices_to_keep = np.hstack(cell_indices_to_keep)
                for var in _cell_vars:
                    cell_write_dict[var] = self.branch_data[var][ev_i][cell_indices_to_keep]
                cell_write_dict['cell_calo_region'] = self.branch_data['cell_calo_region'][ev_i][cell_indices_to_keep].astype(int)
            else:
                cell_indices_to_keep = np.array([], dtype=int)

        
        # particle_dep_e_frac_msi_num
        for topo_i in topo_indices_to_keep:
            topo_p_idxs = np.array(self.branch_data['topo_particle_idxs'][ev_i][topo_i], dtype='int32')
            topo_p_es = np.array(self.branch_data['topo_particle_energies'][ev_i][topo_i], dtype='float32')
            topo_p_e_frac = topo_p_es / topo_p_es.sum()

            topo_p_e_frac_mask = topo_p_e_frac > self.topo_part_e_th
            topo_p_idxs = topo_p_idxs[topo_p_e_frac_mask]
            if len(topo_p_idxs) != 0:
                particle_mask_msi[topo_p_idxs] = True # modify particle_mask_msi
                particle_dep_e_frac_msi_num[topo_p_idxs] += topo_p_es[topo_p_e_frac_mask]

        # particle_dep_e_frac_msi (fraction of the energy deposited by the particle in all the topos of this ms cluster)
        particle_dep_e_frac_msi = particle_dep_e_frac_msi_num / (self.branch_data['particle_dep_energy'][ev_i] + 1e-8)
        particle_dep_e_frac_msi = np.clip(particle_dep_e_frac_msi, 0, 1)


        # tracks
        _track_vars = ['track_eta', 'track_phi', 'track_pt', 'track_d0', 'track_z0']
        track_write_dict = {}
        for var in _track_vars:
            track_write_dict[var] = self.branch_data[var][ev_i][track_mask]
        for lay_i in range(6):
            for var in ['x', 'y', 'z', 'eta', 'phi']:
                track_write_dict[f'track_{var}_layer_{lay_i}'] = self.branch_data[f'track_{var}_layer_{lay_i}'][ev_i][track_mask]
        track_write_dict['track_eta_int'] = self.branch_data['track_eta_layer_0'][ev_i][track_mask]
        track_write_dict['track_phi_int'] = self.branch_data['track_phi_layer_0'][ev_i][track_mask]        
        track_write_dict['track_ms_cluster_idx'] = np.full_like(track_write_dict['track_eta'], int(ms_i), dtype=int)

        for track_i in track_indices_to_keep:
            if self.branch_data['track_particle_idx'][ev_i][track_i] < 0:
                continue
            particle_mask_msi[self.branch_data['track_particle_idx'][ev_i][track_i]] = True

        
        # if no topo or track, then just return
        if len(topo_indices_to_keep) + len(track_indices_to_keep) < self.num_nodes_min:
            return
            

        if not self.skip_cells:

            # edges (cell to cell)
            src_mask = np.isin(self.branch_data['cell_to_cell_edge_start'][ev_i], cell_indices_to_keep)
            dst_mask = np.isin(self.branch_data['cell_to_cell_edge_end'][ev_i], cell_indices_to_keep)
            edge_mask = src_mask * dst_mask

            filtered_src = self.branch_data['cell_to_cell_edge_start'][ev_i][edge_mask]
            filtered_dst = self.branch_data['cell_to_cell_edge_end'][ev_i][edge_mask]
            cell_to_cell_write_dict = {
                'src': np.where(cell_indices_to_keep[None, :] == filtered_src[:, None])[1],
                'dst': np.where(cell_indices_to_keep[None, :] == filtered_dst[:, None])[1]
            }

            # edges (track to cell)
            src_mask = np.isin(self.branch_data['track_to_cell_edge_start'][ev_i], track_indices_to_keep)
            dst_mask = np.isin(self.branch_data['track_to_cell_edge_end'][ev_i], cell_indices_to_keep)
            edge_mask = src_mask * dst_mask

            filtered_src = self.branch_data['track_to_cell_edge_start'][ev_i][edge_mask]
            filtered_dst = self.branch_data['track_to_cell_edge_end'][ev_i][edge_mask]
            track_to_cell_write_dict = {
                'src': np.where(track_indices_to_keep[None, :] == filtered_src[:, None])[1],
                'dst': np.where(cell_indices_to_keep[None, :] == filtered_dst[:, None])[1]
            }


        # particles (empty if no particles)
        particle_write_dict = {}
        for var in ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_dep_e', 'particle_pdgid',
                'particle_ms_cluster_idx', 'particle_old_idx', 'particle_track_idx']:
            particle_write_dict[var] = np.array([])

        # update links (topo-particle) (empty if no particles)
        topo2particle_write_dict = {}
        for var in ['topo2particle_topo_idx', 'topo2particle_particle_idx', 'topo2particle_energy']:
            topo2particle_write_dict[var] = []


        # particle_mask_msi was created based on tracks and topos kept in this MS cluster
        # now we set them to False if the modified energy(or pT? idk) is below certain threshold
        mod_e = np.zeros_like(self.branch_data['particle_e'][ev_i]) + 1e7 # charged particles are always kept
        trackless_part_mask = self.branch_data['particle_track_idx'][ev_i] < 0 # trackless particles
        mod_e[trackless_part_mask] = self.branch_data['particle_e'][ev_i][trackless_part_mask] * particle_dep_e_frac_msi[trackless_part_mask]
        particle_mask_msi[mod_e < self.mod_e_th] = False


        # particles
        particle_mask_msi = particle_mask_msi * part_mask_ev
        particle_indices_to_keep = np.where(particle_mask_msi)[0]

        if len(particle_indices_to_keep) != 0:
            particle_write_dict['particle_eta'] = self.branch_data['particle_eta'][ev_i][particle_mask_msi]
            particle_write_dict['particle_phi'] = self.branch_data['particle_phi'][ev_i][particle_mask_msi]
            particle_write_dict['particle_track_idx'] = self.branch_data['particle_track_idx'][ev_i][particle_mask_msi]

            particle_write_dict['particle_pt'] = self.branch_data['particle_pt'][ev_i][particle_mask_msi]
            particle_write_dict['particle_e'] = self.branch_data['particle_e'][ev_i][particle_mask_msi]
            particle_write_dict['particle_dep_e'] = self.branch_data['particle_dep_energy'][ev_i][particle_mask_msi]

            has_track_ev = self.branch_data['particle_track_idx'][ev_i] >= 0
            track_kept_ev_msi = np.isin(self.branch_data['particle_track_idx'][ev_i], track_indices_to_keep)

            # track not kept in this MS cluster
            trackless_mask_msi = (~(track_kept_ev_msi * has_track_ev)) * particle_mask_msi
            trackless_msi_e = self.branch_data['particle_e'][ev_i][trackless_mask_msi]
            trackless_msi_eta = self.branch_data['particle_eta'][ev_i][trackless_mask_msi]
            trackless_msi_p = self.branch_data['particle_pt'][ev_i][trackless_mask_msi] * np.cosh(trackless_msi_eta)
            trackless_msi_m = np.sqrt(np.clip(trackless_msi_e**2 - trackless_msi_p**2, 0, None))

            # technically tmp_frac is only modifying the kinetic energy of the particle (i.e. the part which the detector can sense),
            # but we apply it to the total particle energy for energy conservation when recombining the MS clusters
            tmp_frac = particle_dep_e_frac_msi[trackless_mask_msi]
            mod_e = trackless_msi_e * tmp_frac
            mod_pt = np.sqrt(np.clip(mod_e**2 - trackless_msi_m**2, 0, None)) / np.clip(np.cosh(trackless_msi_eta), 1e-6, None)

            particle_write_dict['particle_e'][trackless_mask_msi[particle_mask_msi]] = mod_e
            particle_write_dict['particle_pt'][trackless_mask_msi[particle_mask_msi]] = mod_pt
            particle_write_dict['particle_dep_e'][trackless_mask_msi[particle_mask_msi]] = self.branch_data['particle_dep_energy'][ev_i][trackless_mask_msi] * tmp_frac



            particle_write_dict['particle_pdgid'] = self.branch_data['particle_pdgid'][ev_i][particle_mask_msi]
            track_in_other_ms_cluster_mask = has_track_ev & ~track_kept_ev_msi
            particle_write_dict['particle_pdgid'][track_in_other_ms_cluster_mask[particle_mask_msi]] = -999

            particle_write_dict['particle_ms_cluster_idx'] = np.full(len(particle_indices_to_keep), int(ms_i), dtype=int)
            particle_write_dict['particle_old_idx'] = np.array(particle_indices_to_keep, dtype=int)

            particle_remap_dict = {old: new for new, old in enumerate(particle_indices_to_keep)}


            # update links (topo-particle)
            for topo_new_i, topo_i in enumerate(topo_indices_to_keep):
                topo_p_idxs     = np.array(self.branch_data['topo_particle_idxs'][ev_i][topo_i], dtype='int32')
                topo_p_energies = np.array(self.branch_data['topo_particle_energies'][ev_i][topo_i], dtype='float32')
                
                topo_p_idxs_mask = np.isin(topo_p_idxs, particle_indices_to_keep)
                topo_p_idxs = topo_p_idxs[topo_p_idxs_mask]

                topo_p_idxs = np.array([particle_remap_dict[x] for x in topo_p_idxs], dtype='int32')
                topo_p_energies = topo_p_energies[topo_p_idxs_mask].astype('float32')

                for pi, pe in zip(topo_p_idxs, topo_p_energies):
                    topo2particle_write_dict['topo2particle_topo_idx'].append(int(topo_new_i))
                    topo2particle_write_dict['topo2particle_particle_idx'].append(int(pi))
                    topo2particle_write_dict['topo2particle_energy'].append(pe)

        for var in ['topo2particle_topo_idx', 'topo2particle_particle_idx', 'topo2particle_energy']:
            topo2particle_write_dict[var] = np.array(topo2particle_write_dict[var])


        # update links (track-particle)
        track_write_dict['track_particle_idx'] = []
        for track_i in track_indices_to_keep:
            if self.branch_data['track_particle_idx'][ev_i][track_i] not in particle_indices_to_keep:
                print('\033[91m' + f'ev_i: {ev_i}, ms_i: {ms_i} Keeping track with no associated particle (this is expected if in inference mode)' + '\033[0m')
                track_write_dict['track_particle_idx'].append(-9999)
                continue
            track_write_dict['track_particle_idx'].append(particle_remap_dict[self.branch_data['track_particle_idx'][ev_i][track_i]])
        track_write_dict['track_particle_idx'] = np.array(track_write_dict['track_particle_idx'], dtype=int)


        # event number
        eventNumber = int(ev_i) + self.eventNumber_offset
        
        _dict = {
            'topo': {k[5:]: v for k, v in topo_write_dict.items()}, 
            'track': {k[6:]: v for k, v in track_write_dict.items()},
            'particle': {k[9:]: v for k, v in particle_write_dict.items()},
            'topo2particle': {k[14:]: v for k, v in topo2particle_write_dict.items()},
            'eventNumber': eventNumber
        }
        if not self.skip_cells:
            _dict['cell'] = {k[5:]: v for k, v in cell_write_dict.items()}
            _dict['cell_to_cell'] = cell_to_cell_write_dict
            _dict['track_to_cell'] = track_to_cell_write_dict

        self.tree_writer_obj.fill_one_event(_dict)