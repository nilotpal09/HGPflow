import torch
import torch.nn.functional as F
from torch import nn
from ..helpers.dense import Dense
from ..helpers.diffusion_transformer import DiTEncoder




class IterativeRefiner(nn.Module):
    def __init__(self, config, max_edges):
        super().__init__()
        self.config = config
        self.n_edges = max_edges

        self.T = self.config['T_TOTAL']
        self.d_in = self.config['d_in']
        self.d_hid = self.config['d_hid']

        self.t_backprops_last = [False] * (self.T - 1) + [True]

        self.proj_inputs = nn.Linear(self.d_in, self.d_hid)
        self.refiner = HypergraphRefiner(self.d_hid, config)

        self.edges_mu = nn.Parameter(torch.randn(1, 1, self.d_hid))
        self.edges_logsigma = nn.Parameter(torch.zeros(1, 1, self.d_hid))
        nn.init.xavier_uniform_(self.edges_logsigma)

    def get_initial(self, inputs, track_mask, n_edges=None):
        b, n_v, _, device = *inputs.shape, inputs.device
        n_e = n_edges if n_edges is not None else self.n_edges

        mu = self.edges_mu.expand(b, n_e, -1)
        sigma = self.edges_logsigma.exp().expand(b, n_e, -1)
        e_t = mu + sigma * torch.randn(mu.shape, device = device)

        v_t = self.proj_inputs(inputs)
        i_t = torch.zeros(b, n_e, n_v, device=device) + 1.0/self.n_edges

        track_eye, ch_mask_from_tracks = self.refiner.get_track_eye(track_mask, i_t.shape)
        i_t = self.refiner.set_track(i_t, track_mask, track_eye)
        return e_t, v_t, i_t, track_eye, ch_mask_from_tracks

    def refine(self, inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks, t_backprops):
        inputs = self.proj_inputs(inputs)
        pred_bp = []

        for t, do_bp in enumerate(t_backprops):
            if not do_bp:
                with torch.no_grad():
                    _, e_t, v_t, i_t = self.refiner(inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks)
            else:
                p, e_t, v_t, i_t = self.refiner(inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks)
                pred_bp.append((t, p))

        return pred_bp, (e_t, v_t, i_t)

    def forward(self, inputs, track_mask):
        e_t, v_t, i_t, track_eye, ch_mask_from_tracks = self.get_initial(inputs, track_mask)
        return self.refine(
            inputs, e_t, v_t, i_t, track_mask, track_eye, ch_mask_from_tracks, self.t_backprops_last)




class HypergraphRefiner(nn.Module):
    def __init__(self, dim, config):
        super().__init__()

        self.config = config

        self.mlp_n = DeepSet(3*dim, config['deepset_n']['hidden_layers'])
        self.transformer_e = DiTEncoder(**config['transformer_e'])

        self.norm_pre_n  = nn.LayerNorm(3*dim)
        self.norm_pre_e  = nn.LayerNorm(2*dim)
        self.norm_n = nn.LayerNorm(dim)
        self.norm_e = nn.LayerNorm(dim)

        self.mlp_incidence = nn.Sequential(
            OutCatLinear(dim, dim, 1, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1)
        )
        self.edge_indicator = Dense(**config['edge_indicator'])


    def get_track_eye(self, track_mask, i_t_shape):
        b_size, e_size, n_size = i_t_shape

        # identity tensor with shape (b, n, n)
        track_eye = torch.eye(n_size, device=track_mask.device).unsqueeze(0).repeat(b_size, 1, 1)

        # zero pad track_eye to shape (b, e, n)
        track_eye = torch.nn.functional.pad(track_eye, (0, 0, 0, e_size - n_size))
        track_eye = track_eye * track_mask.unsqueeze(1)

        ch_mask_from_tracks = track_eye.sum(dim=2, keepdim=True).bool()

        return track_eye, ch_mask_from_tracks


    def set_track(self, i_t, track_mask, track_eye):
        i_t = i_t * (~track_mask).unsqueeze(1) # setting to zero
        i_t = i_t + track_eye # adding track identity matrix

        return i_t # , track_eye


    def forward(self, inputs, e_t, n_t, i_t, track_mask, track_eye, ch_mask_from_tracks):
        b_size, e_size, n_size = i_t.shape

        i_t = self.mlp_incidence((e_t, n_t, i_t)).squeeze(3) # B, n_hyperedge, n_node

        # softmax i_t such that for each node, the sum of incidence is 1
        i_t = nn.Softmax(dim=1)(i_t) # B, n_hyperedge, n_node

        # hard coding the track part
        i_t = self.set_track(i_t, track_mask, track_eye)

        # update the indicator
        i_t_sum = i_t.sum(dim=2, keepdim=True) 
        e_ind_logit = self.edge_indicator(torch.cat([e_t, i_t_sum], dim=2))

        # set the indicator equal to one for the track part
        e_ind_logit = e_ind_logit * (~ch_mask_from_tracks) + ch_mask_from_tracks * 1e6

        e_ind = F.sigmoid(e_ind_logit)

        # update the incidence according to the indicator
        im_t = i_t * e_ind

        updates_e = torch.einsum("ben,bnd->bed", im_t, n_t)
        e_t = self.transformer_e(
            q=torch.cat([e_t, updates_e], dim=-1),
            context=inputs.mean(dim=1)    
        )

        updates_n = torch.einsum("ben,bed->bnd", im_t, e_t)
        n_t = self.norm_n(n_t + self.mlp_n(self.norm_pre_n(torch.cat([inputs, n_t, updates_n], dim=-1))))

        pred_is_charged = track_eye.sum(dim=2).bool()
        pred = (i_t, e_ind_logit.squeeze(-1), pred_is_charged)

        return pred, e_t, n_t, i_t


class OutCatLinear(nn.Module):
    def __init__(self, d_e, d_n, d_i, d_out):
        super().__init__()
        self.proj_e = nn.Linear(d_e, d_out)
        self.proj_n = nn.Linear(d_n, d_out)
        self.proj_i = nn.Linear(d_i, d_out)

    def forward(self, inputs):
        e_t, n_t, i_t = inputs
        o0 = self.proj_n(n_t).unsqueeze(1)
        o1 = self.proj_e(e_t).unsqueeze(2)
        o2 = self.proj_i(i_t.unsqueeze(3))
        return o0 + o1 + o2


class DeepSet(nn.Module):
    def __init__(self, d_in, d_hids):
        super().__init__()
        layers = []
        layers.append(DeepSetLayer(d_in, d_hids[0]))
        for i in range(1, len(d_hids)):
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(d_hids[i-1]))
            layers.append(DeepSetLayer(d_hids[i-1], d_hids[i]))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class DeepSetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        x0 = self.layer1(x)
        x1 = self.layer2(x - x.mean(dim=1, keepdim=True))
        x = x0 + x1
        return x