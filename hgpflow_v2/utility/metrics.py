import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import os


def l_split_ind(l, n):
    r = l%n
    return np.cumsum([0] + [l//n+1]*r + [l//n]*(n-r))

@ray.remote
def lsa(arr, indicator, s, e):
    return [linear_sum_assignment(p[ind]) for p,ind in zip(arr[s:e], indicator[s:e])]

def ray_lsa(arr, indicator, n):
    l = arr.shape[0]
    n = min(n, l)
    ind = l_split_ind(l, n)
    arr_id = ray.put(arr)
    indicator_id = ray.put(indicator)
    res = [lsa.remote(arr_id, indicator_id, ind[i], ind[i+1]) for i in range(n)]
    res_ = []
    for r in res:
        res_.extend(ray.get(r))
    return res_



class Metrics:
    def __init__(self, config, n_ray=0):
        self.n_ray = n_ray

        self.device = config['device']
        self.EPS = 1e-8
        self.IND_LOSS_WT = config['ind_loss_wt']
        self.INC_LOSS_WT = config['inc_loss_wt']
        self.TRACK_FIX_WT = config['track_fix_wt']

        fnp = np.load('incidence_dist_0.01_0.99_20bins.npz')
        xs = 0.5 * (fnp['arr_0'][1:] + fnp['arr_0'][:-1])
        self.INC_CONTWT_X = torch.FloatTensor(xs[:-1])
        self.INC_CONTWT_Y = torch.FloatTensor(120_000 * 1/fnp['arr_1'])
        self.INC_CONTWT_Y[0] = 1.0

        if self.device == 'cuda':
            self.INC_CONTWT_X = self.INC_CONTWT_X.cuda()
            self.INC_CONTWT_Y = self.INC_CONTWT_Y.cuda()

        if 'loss_discount_factor' in config:
            self.discount_factor = eval(config['loss_discount_factor'])
            self.discount_factor = self.discount_factor.to(self.device)

        self.do_consistent_match = config.get('do_consistent_match', False)


    def get_inc_and_ind_pdist(self, input, target, node_is_track):
        '''
        Args:
            input : (input_inc  (b, ne, nv), input_ind  (b, ne), input_is_charged  (b, ne))
            target: (target_inc (b, ne, nv), target_ind (b, ne), target_is_charged (b, ne))
        '''

        input_inc, input_ind_logit, input_is_charged = input
        target_inc, target_ind, target_is_charged = target

        # reshaping
        input_inc_reshaped = input_inc.unsqueeze(1).expand(-1, target_inc.size(1), -1, -1)
        target_inc_reshaped = target_inc.unsqueeze(2).expand(-1, -1, input_inc.size(1), -1)

        input_ind_logit_reshaped = input_ind_logit.unsqueeze(1).expand(-1, target_ind.size(1), -1)
        target_ind_reshaped = target_ind.unsqueeze(2).expand(-1, -1, input_ind_logit.size(1))

        # getting the charged_particles - tracks block from the incidence matrix
        input_inc_ch_track  = input_inc *  input_is_charged.unsqueeze(-1).float()  * node_is_track.unsqueeze(1).float()
        target_inc_ch_track = target_inc * target_is_charged.unsqueeze(-1).float() * node_is_track.unsqueeze(1).float()

        input_inc_ch_track_reshaped = input_inc_ch_track.unsqueeze(1).expand(-1, target_inc_ch_track.size(1), -1, -1)
        target_inc_ch_track_reshaped = target_inc_ch_track.unsqueeze(2).expand(-1, -1, input_inc_ch_track.size(1), -1)

        # charged particles already have theur assignments based on tracks
        # same value -> 0, different value -> 1 (TRACK_FIX_WT)
        track_fix_wt = torch.abs(input_inc_ch_track_reshaped - target_inc_ch_track_reshaped)
        track_fix_wt = self.TRACK_FIX_WT * track_fix_wt

        # continuos weight for inc
        if self.INC_CONTWT_X is not None:
            cont_wt = self.INC_CONTWT_Y[torch.searchsorted(self.INC_CONTWT_X, target_inc)]
            cont_wt = cont_wt.unsqueeze(2).expand(-1, -1, input_inc.size(1), -1)

        # incidence loss (kld = -qlog(p))
        pdist_inc = - target_inc_reshaped * torch.log(input_inc_reshaped + self.EPS)

        # for logging
        pdist_inc_unscaled = pdist_inc.clone().detach()
        pdist_inc_unscaled = (pdist_inc_unscaled + track_fix_wt).mean(3)

        pdist_inc = pdist_inc * cont_wt
        pdist_inc = pdist_inc + track_fix_wt
        pdist_inc = self.INC_LOSS_WT * pdist_inc.mean(3)

        # indicator loss (bce with logits)
        pdist_ind = F.binary_cross_entropy_with_logits(
            input_ind_logit_reshaped, target_ind_reshaped, reduction='none')

        # for logging        
        pdist_ind_unscaled = pdist_ind.clone().detach()

        pdist_ind = self.IND_LOSS_WT * pdist_ind

        return pdist_inc, pdist_ind, pdist_inc_unscaled, pdist_ind_unscaled


    def get_lsa_indices(self, pdist, target_ind):
        pdist_ = pdist.detach().cpu().numpy()

        b, n_part, _ = pdist.size()
        indices_torch = torch.arange(n_part).unsqueeze(0).unsqueeze(0).expand(b, 2, -1).clone() # step1
        _arange = torch.arange(n_part)

        if self.n_ray > 0:
            raise NotImplementedError
            ray_return = ray_lsa(pdist_, target_ind.bool().cpu().numpy(), self.n_ray)
            for i, (row_ind, col_ind) in enumerate(ray_return):
                pass                
            
        else:
            for i, (p, ind) in enumerate(zip(pdist_, target_ind.cpu().numpy())):
                ind_mask = ind == 1
                row_ind, col_ind = linear_sum_assignment(p[ind_mask])
                col_ind = torch.from_numpy(col_ind)

                indices_torch[i, 1, ind_mask] = col_ind # step2
                
                unmatched_mask = torch.full((n_part,), True)
                unmatched_mask[col_ind] = False
                indices_torch[i, 1, ~ind_mask] = _arange[unmatched_mask] # step3

        indices_torch = indices_torch.to(device=pdist.device) 

        return indices_torch


    def process_pdists(self, pdist_inc, pdist_ind, target_ind, pdist_inc_unscaled, pdist_ind_unscaled):

        pdist = pdist_inc + pdist_ind
        indices_torch = self.get_lsa_indices(pdist, target_ind)
        indices_to_return = indices_torch[:, 1, :]

        indices_torch = indices_torch.shape[2] * indices_torch[:, 0] + indices_torch[:, 1]
        total_loss = torch.gather(pdist.flatten(1,2), 1, indices_torch).mean()

        # for book-keeping
        inc_loss = torch.gather(pdist_inc.flatten(1,2), 1, indices_torch).mean()
        ind_loss = torch.gather(pdist_ind.flatten(1,2), 1, indices_torch).mean()

        inc_loss_unscaled = torch.gather(pdist_inc_unscaled.flatten(1,2), 1, indices_torch).mean()
        ind_loss_unscaled = torch.gather(pdist_ind_unscaled.flatten(1,2), 1, indices_torch).mean()

        loss_componenets = {
            'inc_loss': inc_loss.item(),
            'ind_loss': ind_loss.item(),
            'inc_loss_unscaled': inc_loss_unscaled.item(),
            'ind_loss_unscaled': ind_loss_unscaled.item()
        }

        return total_loss, loss_componenets, indices_to_return


    def LAP_loss_single(self, input, target, node_is_track):
        '''
        Args:
            input : (input_inc  (b, ne, nv), input_ind_logit  (b, ne), input_is_charged  (b, ne))
            target: (target_inc (b, ne, nv), target_ind_logit (b, ne), target_is_charged (b, ne))
        '''

        pdist_inc, pdist_ind, pdist_inc_unscaled, pdist_ind_unscaled = \
            self.get_inc_and_ind_pdist(input, target, node_is_track)

        return self.process_pdists(pdist_inc, pdist_ind, target[1], pdist_inc_unscaled, pdist_ind_unscaled)


    def LAP_loss_multi(self, input_list, target, node_is_track):
        '''
        Args:
            input_list : [(t, (input_inc  (b, ne, nv), input_ind_logit  (b, ne), input_is_charged  (b, ne))) t_bptt times]
            target:           (target_inc (b, ne, nv), target_ind_logit (b, ne), target_is_charged (b, ne))
        '''

        if self.do_consistent_match:
            return self.LAP_loss_multi_consistent_match(input_list, target, node_is_track)

        else:
            total_loss = 0; 
            loss_componenets = {
                'inc_loss': 0,
                'ind_loss': 0
            }
            for t, input in input_list:
                loss, loss_components, _ = self.LAP_loss_single(input, target, node_is_track)
                total_loss += loss * self.discount_factor[t]
                for k in loss_components.keys():
                    loss_componenets[k] += loss_components[k] * self.discount_factor[t]
            
            total_loss = total_loss / sum(self.discount_factor)
            for k in loss_componenets.keys():
                loss_componenets[k] = loss_componenets[k] / sum(self.discount_factor)

            return total_loss, loss_componenets, None


    def LAP_loss_multi_consistent_match(self, input_list, target, node_is_track):
        '''
        Args:
            input_list : [(t, (input_inc  (b, ne, nv), input_ind_logit  (b, ne), input_is_charged  (b, ne))) t_bptt times]
            target:           (target_inc (b, ne, nv), target_ind_logit (b, ne), target_is_charged (b, ne))
        '''

        pdist_inc, pdist_ind = 0, 0; discount_total = 0
        pdist_inc_unscaled, pdist_ind_unscaled = 0, 0
        for t, input in input_list:
            pdist_inc_t, pdist_ind_t, pdist_inc_unscaled_t, pdist_ind_unscaled_t = \
                self.get_inc_and_ind_pdist(input, target, node_is_track)
            pdist_inc = pdist_inc + pdist_inc_t * self.discount_factor[t]
            pdist_ind = pdist_ind + pdist_ind_t * self.discount_factor[t]
            pdist_inc_unscaled += pdist_inc_unscaled_t * self.discount_factor[t]
            pdist_ind_unscaled += pdist_ind_unscaled_t * self.discount_factor[t]
            discount_total += self.discount_factor[t]
        
        pdist_inc = pdist_inc / discount_total
        pdist_ind = pdist_ind / discount_total
        pdist_inc_unscaled = pdist_inc_unscaled / discount_total
        pdist_ind_unscaled = pdist_ind_unscaled / discount_total

        return self.process_pdists(pdist_inc, pdist_ind, target[1], pdist_inc_unscaled, pdist_ind_unscaled)