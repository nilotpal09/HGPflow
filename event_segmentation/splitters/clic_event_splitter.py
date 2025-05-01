import numpy as np
from tqdm import tqdm
import uproot

from splitters.ms_event_splitter import deltaR, MSEventSplitter

class CLICEventSplitter(MSEventSplitter):

    def __init__(self, base_splitter_init_config, pt_min_gev, eta_max, num_nodes_min, skip_cells, cluster_jet_idx_filepath=None):
        super().__init__(**base_splitter_init_config)
        self.pt_min_gev = pt_min_gev
        self.eta_max = eta_max
        self.num_nodes_min = num_nodes_min
        self.skip_cells = skip_cells
        print("skip_cells:", self.skip_cells)

        self.corrupt_topo = False
        if cluster_jet_idx_filepath is not None:
            self.corrupt_topo = True

            cl_tree = uproot.open(cluster_jet_idx_filepath)['cluster_data']
            cl_jet_idx = cl_tree['jet_idx'].array(library='np')
            cl_energy = cl_tree['energy'].array(library='np')
            cl_eta = cl_tree['eta'].array(library='np')
            cl_phi = cl_tree['phi'].array(library='np')

            self.corrupt_topo_e = []; self.corrupt_topo_eta = []; self.corrupt_topo_phi = []
            for ev_i, cl_jet_idx_ev in enumerate(cl_jet_idx):
                subleading_mask = cl_jet_idx_ev == 1
                self.corrupt_topo_e.append(cl_energy[ev_i][subleading_mask])
                self.corrupt_topo_eta.append(cl_eta[ev_i][subleading_mask])
                self.corrupt_topo_phi.append(cl_phi[ev_i][subleading_mask])

    def branch_data_update(self):

        # fiducial cuts to require particle interaction and reduce double counting
        self.fiducial_mask, self.new_particle_topo_start = self.get_fiducial_mask()

        self.branch_data['topo_particle_idxs'], self.branch_data['topo_particle_energies'] = \
            self.invert_topo_part(
                self.new_particle_topo_start,
                self.branch_data['particle_topo_end'],
                self.branch_data['particle_topo_e']
            )
        
        # chi2 is always positive, but we sometimes have negative values (no clue why)
        for i in range(len(self.branch_data['track_chi2'])):
            self.branch_data['track_chi2'][i] = np.clip(self.branch_data['track_chi2'][i], 1e-8, None)

    def invert_topo_part(self, particle_topo_start, particle_topo_end, particle_topo_e):

        n_events = len(particle_topo_start)
        topo_particle_idx = []; topo_particle_energy = []
        for ev_i in tqdm(range(n_events), desc='Inverting topo-particle'):
            ev_topo_particle_idx = []; ev_topo_particle_energy = []
            for topo_i in range(len(self.branch_data['topo_eta'][ev_i])):
                mask = particle_topo_end[ev_i] == topo_i
                ev_topo_particle_idx.append(particle_topo_start[ev_i][mask])
                ev_topo_particle_energy.append(particle_topo_e[ev_i][mask])

            topo_particle_idx.append(np.array(ev_topo_particle_idx, dtype=object))
            topo_particle_energy.append(np.array(ev_topo_particle_energy, dtype=object))

        return topo_particle_idx, topo_particle_energy


    def get_fiducial_mask(self):

        mask_keep, new_particle_topo_start = [], []

        n_events = len(self.branch_data['particle_idx'])
        for ev_i in tqdm(range(n_events), desc='Getting fiducial mask'):

            idx        = self.branch_data['particle_idx'][ev_i]
            parent_idx = self.branch_data['particle_parent_idx'][ev_i]
            interacted = self.branch_data['particle_interacted'][ev_i] == 1
            is_child   = self.branch_data['particle_parent_idx'][ev_i] >= 0
            has_track  = self.branch_data['particle_track_idx'][ev_i]  >= 0

            parent_has_track = np.in1d(parent_idx, idx[has_track])

            # only children without track are excluded (calo deposits will be reassigned to parent)
            mask_keep_ev = interacted & np.logical_or.reduce([
                ~is_child,
                is_child  & ~parent_has_track,
                is_child  &  parent_has_track & has_track
            ])

            # compute list of children indices which get absorbed and corresponding parent indices
            mask_absorb_ev = interacted & is_child & parent_has_track & ~has_track
            absorbed_c2p_parent_idx_ev = np.where(mask_absorb_ev, parent_idx, idx)
            old_particle_topo_start_ev = self.branch_data['particle_topo_start'][ev_i]
            new_particle_topo_start_ev = absorbed_c2p_parent_idx_ev[old_particle_topo_start_ev]

            mask_keep.append(mask_keep_ev)
            new_particle_topo_start.append(new_particle_topo_start_ev)

        mask_keep = np.array(mask_keep, dtype=object)
        new_particle_topo_start = np.array(new_particle_topo_start, dtype=object)

        return mask_keep, new_particle_topo_start


    def get_track_topo_part_mask_ev(self, ev_i):

        # part mask ev
        pt_mask = self.branch_data['particle_pt'][ev_i] > self.pt_min_gev
        eta_mask = np.abs(self.branch_data['particle_eta'][ev_i]) < self.eta_max
        fid_mask = self.fiducial_mask[ev_i]
        nonzero_dep_e_nu_mask = (self.branch_data['particle_dep_energy'][ev_i] > 0) + (self.branch_data['particle_track_idx'][ev_i] != -9999)
        mask_part = pt_mask * eta_mask * fid_mask * nonzero_dep_e_nu_mask

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
        assoc_part_pt = np.zeros_like(self.branch_data['track_particle_idx'][ev_i], dtype=float) - 9999
        assoc_part_eta = np.zeros_like(self.branch_data['track_particle_idx'][ev_i], dtype=float) - 9999
        assoc_part_phi = np.zeros_like(self.branch_data['track_particle_idx'][ev_i], dtype=float) - 9999

        assoc_part_exist_mask = self.branch_data['track_particle_idx'][ev_i] >= 0 # != -9999
        tpi_filtered = self.branch_data['track_particle_idx'][ev_i][assoc_part_exist_mask]
        
        assoc_part_pt[assoc_part_exist_mask] = self.branch_data['particle_pt'][ev_i][tpi_filtered]
        assoc_part_eta[assoc_part_exist_mask] = self.branch_data['particle_eta'][ev_i][tpi_filtered]
        assoc_part_phi[assoc_part_exist_mask] = self.branch_data['particle_phi'][ev_i][tpi_filtered]

        tr_pt = self.branch_data['track_pt'][ev_i]
        tr_eta = self.branch_data['track_eta'][ev_i]; tr_phi = self.branch_data['track_phi'][ev_i]

        resolution_mask = np.abs((tr_pt - assoc_part_pt) / (assoc_part_pt + 1e-8)) < TRACK_PART_PT_RES_TH
        dr_with_assoc_part_mask = deltaR(tr_eta, tr_phi, assoc_part_eta, assoc_part_phi) < TRACK_PART_DR_TH

        assoc_part_survived_mask = np.zeros_like(self.branch_data['track_particle_idx'][ev_i], dtype=bool)
        assoc_part_survived_mask[assoc_part_exist_mask] = mask_part[tpi_filtered]

        mask_track = assoc_part_survived_mask * resolution_mask * dr_with_assoc_part_mask

        return mask_track, mask_topo, mask_part



    def decorate_mscluster(self, track_mask, topo_mask, part_mask_ev, ev_i, ms_i):
        
        particle_dep_e_frac_msi_num = np.zeros_like(self.branch_data['particle_pt'][ev_i], dtype=float)
        particle_mask_msi = np.zeros_like(part_mask_ev, dtype=bool)

        track_indices_to_keep = np.where(track_mask)[0]
        topo_indices_to_keep = np.where(topo_mask)[0]
        particle_indices_to_keep = [] # we can drop neutral particles based on certain conditions, but charged particles always kept


        # topos
        _topo_vars = ['topo_x', 'topo_y', 'topo_z', 'topo_eta', 'topo_phi', 'topo_rho', 'topo_e',
            'topo_num_cells', 'topo_sigma_eta', 'topo_sigma_phi', 'topo_sigma_rho',
            'topo_energy_ecal', 'topo_energy_hcal', 'topo_energy_other']
        
        topo_write_dict = {}
        for var in _topo_vars:
            topo_write_dict[var] = self.branch_data[var][ev_i][topo_mask]
        topo_write_dict['topo_ms_cluster_idx'] = np.full(len(topo_indices_to_keep), int(ms_i), dtype=int)
        topo_write_dict['topo_num_cells'] = topo_write_dict['topo_num_cells'].astype(np.float32) # easier this way

        # corrupt topo_e for locality test
        if self.corrupt_topo:
            zipped_obj = zip(self.corrupt_topo_e[ev_i], self.corrupt_topo_eta[ev_i], self.corrupt_topo_phi[ev_i])
            for e_target, eta_target, phi_target in zipped_obj:
                match_mask = (
                    np.isclose(topo_write_dict['topo_e'], e_target, atol=1e-5) &
                    np.isclose(topo_write_dict['topo_eta'], eta_target, atol=1e-5) &
                    np.isclose(topo_write_dict['topo_phi'], phi_target, atol=1e-5)
                )
                # Reduce topo_e by 80% for matching entries
                topo_write_dict['topo_e'][match_mask] *= 0.2


        # cells
        if not self.skip_cells:
            _cell_vars = ['cell_eta', 'cell_phi', 'cell_rho', 'cell_e', 'cell_x', 'cell_y', 'cell_z']
            
            # if no cell (coz no topo)
            cell_write_dict = {}
            for var in _cell_vars:
                cell_write_dict[var] = np.array([])
            cell_write_dict['cell_calo_region'] = np.array([], dtype=int)
            cell_write_dict['cell_topo_idx'] = np.array([], dtype=int)

            new_topo_idx = 0
            for topo_i in topo_indices_to_keep:
                cell_mask_for_this_topo = self.branch_data['cell_topo_idx'][ev_i] == topo_i
                for var in _cell_vars:
                    cell_write_dict[var] = self.branch_data[var][ev_i][cell_mask_for_this_topo]
                cell_write_dict['cell_calo_region'] = self.branch_data['cell_subdet'][ev_i][cell_mask_for_this_topo]
                cell_write_dict['cell_topo_idx'] = np.full(len(cell_write_dict['cell_eta']), new_topo_idx, dtype=int)
                new_topo_idx += 1


        # particle_dep_e_frac_msi_num
        for topo_i in topo_indices_to_keep:
            topo_p_idxs = np.array(self.branch_data['topo_particle_idxs'][ev_i][topo_i], dtype='int32')
            topo_p_es = np.array(self.branch_data['topo_particle_energies'][ev_i][topo_i], dtype='float32')
            topo_p_e_frac = topo_p_es / topo_p_es.sum()

            topo_p_e_frac_mask = topo_p_e_frac > self.topo_part_e_th
            topo_p_idxs = topo_p_idxs[topo_p_e_frac_mask]
            if len(topo_p_idxs) != 0:
                particle_mask_msi[topo_p_idxs] = True
                particle_dep_e_frac_msi_num[topo_p_idxs] += topo_p_es[topo_p_e_frac_mask]

        # particle_dep_e_frac_msi (fraction of the energy deposited by the particle in all the topos of this ms cluster)
        particle_dep_e_frac_msi = particle_dep_e_frac_msi_num / (self.branch_data['particle_dep_energy'][ev_i] + 1e-8)
        particle_dep_e_frac_msi = np.clip(particle_dep_e_frac_msi, 0, 1)


        # tracks
        _track_vars = ['track_pt', 'track_eta', 'track_phi', 'track_p', 'track_d0', 'track_z0', 'track_chi2', 'track_ndf',
            'track_dedx', 'track_dedx_error', 'track_radiusofinnermosthit', 'track_tanlambda', 'track_omega',
            'track_time', 'track_x_int', 'track_y_int', 'track_z_int', 'track_eta_int', 'track_phi_int']
        
        track_write_dict = {}
        for var in _track_vars:
            track_write_dict[var] = self.branch_data[var][ev_i][track_mask]
        track_write_dict['track_ms_cluster_idx'] = np.full(len(track_indices_to_keep), int(ms_i), dtype=int)
        track_write_dict['track_ndf'] = track_write_dict['track_ndf'].astype(np.float32) # easier this way

        for track_i in track_indices_to_keep:
            if self.branch_data['track_particle_idx'][ev_i][track_i] == -9999: # can hit this if we are in inference mode
                continue
            particle_mask_msi[self.branch_data['track_particle_idx'][ev_i][track_i]] = True


        # no topo or track, then just return
        if len(topo_indices_to_keep) + len(track_indices_to_keep) < self.num_nodes_min:
            return


        # particles (empty if no particles)
        particle_write_dict = {}
        for var in ['particle_pt', 'particle_eta', 'particle_phi', 'particle_e', 'particle_dep_e', 'particle_pdgid',
                'particle_ms_cluster_idx', 'particle_old_idx', 'particle_track_idx']:
            particle_write_dict[var] = np.array([])

        # update links (topo-particle) (empty if no particles)
        topo2particle_write_dict = {}
        for var in ['topo2particle_topo_idx', 'topo2particle_particle_idx', 'topo2particle_energy']:
            topo2particle_write_dict[var] = []


        # particles
        particle_mask_msi = particle_mask_msi * part_mask_ev
        particle_indices_to_keep = np.where(particle_mask_msi)[0]

        if len(particle_indices_to_keep) != 0:
            particle_write_dict['particle_eta'] = self.branch_data['particle_eta'][ev_i][particle_mask_msi]
            particle_write_dict['particle_phi'] = self.branch_data['particle_phi'][ev_i][particle_mask_msi]

            particle_write_dict['particle_pt'] = self.branch_data['particle_pt'][ev_i][particle_mask_msi]
            particle_write_dict['particle_e'] = self.branch_data['particle_e'][ev_i][particle_mask_msi]
            particle_write_dict['particle_dep_e'] = self.branch_data['particle_dep_energy'][ev_i][particle_mask_msi]
            particle_write_dict['particle_track_idx'] = self.branch_data['particle_track_idx'][ev_i][particle_mask_msi]

            has_track_ev = self.branch_data['particle_track_idx'][ev_i] != -9999 # >= 0
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

        self.tree_writer_obj.fill_one_event(_dict)
