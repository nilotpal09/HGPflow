import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import warnings


def plot_inc_ed(refiner_dict):
    n_nodes = refiner_dict['incidence_truth'].shape[1]
    x = np.arange(refiner_dict['indicator_truth'].shape[0])
    bins = np.arange(refiner_dict['indicator_truth'].shape[0]+1)
    ncol = 3; nrow = int(np.ceil((n_nodes + 2) / ncol))
    fig = plt.figure(figsize=(12, nrow*2), dpi=100) #, tight_layout=True) ##### WAS 6 BEFORE
    for i in range(n_nodes):
        ax = fig.add_subplot(nrow, ncol, i+1)
        ax.hist(x, weights=refiner_dict['incidence_truth'][:,i], bins=bins, histtype='bar', label='truth', color='cornflowerblue', alpha=0.3)
        ax.hist(x, weights=refiner_dict['incidence_pred'][:,i], bins=bins, histtype='step', label='pred', color='red')
        if refiner_dict['node_is_track'][i]:
            node_type = 'track'
            node_pt_raw_or_e_raw = refiner_dict['node_pt_raw'][i]
        else:
            node_type = 'topo'
            node_pt_raw_or_e_raw = refiner_dict['node_e_raw'][i]
        ax.set_title(f'Node {i} ({node_type}) {node_pt_raw_or_e_raw:.1f} GeV')
        ax.set_ylim(0, 1.05)

    ax = fig.add_subplot(nrow, ncol, n_nodes+1)
    ax.hist(x, weights=refiner_dict['indicator_truth'], bins=bins, histtype='bar', label='truth', color='cornflowerblue', alpha=0.3)
    ax.hist(x, weights=refiner_dict['indicator_pred'], bins=bins, histtype='step', label='pred', color='red')
    ax.set_title(f'Indicator')

    # make a new subplot and add the legend from the previous plot
    ax = fig.add_subplot(nrow, ncol, n_nodes+2)
    ax.axis('off')
    ax.legend(*ax.get_legend_handles_labels(), loc='center', ncol=2)

    return fig


def plot_hg_summary(hg_summaries):
    '''
        hg_summaries: list of dictionaries
    '''

    n_inc_edges = len(hg_summaries[0]['inc_diff_means'])
    inc_diff_sums = np.zeros(n_inc_edges); inc_var_sums = np.zeros(n_inc_edges)
    inc_count = np.zeros(n_inc_edges)

    ind_thresholds = hg_summaries[0]['ind_thresholds']
    mr_num_sum = np.zeros(len(ind_thresholds)); mr_den_sum = np.zeros(len(ind_thresholds))
    fr_num_sum = np.zeros(len(ind_thresholds)); fr_den_sum = np.zeros(len(ind_thresholds))

    for hist_inc in hg_summaries:

        inc_diff_sums += np.nan_to_num(hist_inc['inc_diff_means'])
        inc_var_sums += np.nan_to_num(hist_inc['inc_diff_variances'])
        inc_count += np.array(hist_inc['inc_count'])

        mr_num_sum += hist_inc['mr_num']
        mr_den_sum += hist_inc['mr_den']
        fr_num_sum += hist_inc['fr_num']
        fr_den_sum += hist_inc['fr_den']

    # ignore the division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        inc_diff_means = inc_diff_sums / inc_count
        inc_diff_vars = inc_var_sums / inc_count

        mr = mr_num_sum / mr_den_sum
        fr = fr_num_sum / fr_den_sum

    fig = plt.figure(figsize=(12, 5), dpi=100) # , tight_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.2, width_ratios=[1.5, 1])

    ax = fig.add_subplot(gs[0, 0])
    x_edges = np.linspace(0, 1, n_inc_edges+1)
    x_vals = (x_edges[1:] + x_edges[:-1]) / 2
    x_width = x_edges[1] - x_edges[0]
    ax.hlines(0, 0, 1, color='black', linestyle='--', alpha=0.5)
    ax.errorbar(x_vals, inc_diff_means, xerr=x_width/2, yerr=np.sqrt(inc_diff_vars), fmt='o', c='r')
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlabel('incidence vals')
    ax.set_ylabel('pred - truth')
    ax.set_title('indicator_truth == True; is_track == False')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ind_thresholds, mr, label='miss rate')
    ax.plot(ind_thresholds, fr, label='fake rate')
    ax.set_xlabel('indicator threshold')
    ax.legend()

    log_dict = {
        'inc_diff_mean' : np.sum(inc_diff_sums) / np.sum(inc_count),
        'inc_diff_std' : np.sqrt(np.sum(inc_var_sums) / np.sum(inc_count)),
    }
    return fig, log_dict


def plot_inc_dep_e(inc_dep_es):
    '''
        inc_dep_es: list of dictionaries
    '''

    cmap = plt.cm.get_cmap('viridis')
    cmap.set_under('white')

    truth_raw_dep_e_list = []; pred_raw_dep_e_list = []
    truth_ind_list = []; pred_ind_list = []
    truth_is_charged_list = []; pred_is_charged_list = [] # they are bools

    for inc_dep_e in inc_dep_es:
        truth_raw_dep_e_list.append(inc_dep_e['truth_raw_dep_e'].flatten())
        pred_raw_dep_e_list.append(inc_dep_e['pred_raw_dep_e'].flatten())
        truth_ind_list.append(inc_dep_e['truth_ind'].flatten())
        pred_ind_list.append(inc_dep_e['pred_ind'].flatten())
        truth_is_charged_list.append(inc_dep_e['truth_is_charged'].flatten())
        pred_is_charged_list.append(inc_dep_e['pred_is_charged'].flatten())

    truth_raw_dep_e = np.hstack(truth_raw_dep_e_list)
    pred_raw_dep_e = np.hstack(pred_raw_dep_e_list)

    assert np.isfinite(truth_raw_dep_e).all() and np.isfinite(pred_raw_dep_e).all(), \
        'truth_raw_dep_e and pred_raw_dep_e should be finite'


    truth_ind = np.hstack(truth_ind_list)
    pred_ind = np.hstack(pred_ind_list)
    truth_is_charged = np.hstack(truth_is_charged_list)
    pred_is_charged = np.hstack(pred_is_charged_list)

    assert (truth_is_charged == pred_is_charged).all()

    fig = plt.figure(figsize=(12, 6), dpi=100) # , tight_layout=True)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.4)

    for i, mask in enumerate([truth_is_charged, ~truth_is_charged]):
        ax = fig.add_subplot(gs[i, 0])

        if mask.sum() == 0: continue
        del_dep_e = pred_raw_dep_e[mask] - truth_raw_dep_e[mask]
        bins_x = np.linspace(truth_raw_dep_e[mask].min(), truth_raw_dep_e[mask].max(), 50)
        bins_y = np.linspace(del_dep_e.min(), del_dep_e.max(), 50)
        _, _, _, im = ax.hist2d(truth_raw_dep_e[mask], del_dep_e, bins=[bins_x, bins_y], cmap=cmap, norm=LogNorm())
        fig.colorbar(im, ax=ax)

        ax.set_xlabel('truth raw dep e')
        ax.set_ylabel('(pred - truth) raw dep e')
        ax.set_title('charged' if i == 0 else 'neutral')

    tr_ind_mask = truth_ind > 0.5
    pr_ind_mask = pred_ind > 0.5

    for i, tr_mask in enumerate([tr_ind_mask, ~tr_ind_mask]):
        if tr_mask.sum() == 0: continue
        for j, pr_mask in enumerate([pr_ind_mask, ~pr_ind_mask]):
            ax = fig.add_subplot(gs[i, j+1])
            ax.set_xlabel('truth raw dep e')
            ax.set_ylabel('(pred - truth) raw dep e')
            title = 'tr_ind > 0.5' if i == 0 else 'tr_ind < 0.5'
            title += '; pr_ind > 0.5' if j == 0 else '; pr_ind < 0.5'
            ax.set_title(title)

            mask = tr_mask & pr_mask & (~truth_is_charged) # truth_is_charged is bool and is the same for both
            if mask.sum() == 0: continue

            del_dep_e = pred_raw_dep_e[mask] - truth_raw_dep_e[mask]
            x = truth_raw_dep_e[mask]

            bins_x = np.linspace(x.min() - 1e-6, x.max() + 1e-6, 50)
            bins_y = np.linspace(del_dep_e.min() - 1e-6, del_dep_e.max() + 1e-6, 50)
            h, _, _, im = ax.hist2d(x, del_dep_e, bins=[bins_x, bins_y], cmap=cmap, norm=LogNorm())

            if h.min() != h.max():
                fig.colorbar(im, ax=ax)

    log_dict = {}

    del_dep_e = pred_raw_dep_e - truth_raw_dep_e
    log_dict['del_dep_e_mean'] = del_dep_e.mean()
    log_dict['del_dep_e_std'] = del_dep_e.std()
    log_dict['del_dep_e_iqr'] = np.percentile(del_dep_e, 75) - np.percentile(del_dep_e, 25)

    del_dep_e_charged = del_dep_e[truth_is_charged]
    if truth_is_charged.sum() > 0:
        log_dict['del_dep_e_charged_mean'] = del_dep_e_charged.mean()
        log_dict['del_dep_e_charged_std'] = del_dep_e_charged.std()
        log_dict['del_dep_e_charged_iqr'] = np.percentile(del_dep_e_charged, 75) - np.percentile(del_dep_e_charged, 25)

    del_dep_e_neutral = del_dep_e[~truth_is_charged]
    if (~truth_is_charged).sum() > 0:
        log_dict['del_dep_e_neutral_mean'] = del_dep_e_neutral.mean()
        log_dict['del_dep_e_neutral_std'] = del_dep_e_neutral.std()
        log_dict['del_dep_e_neutral_iqr'] = np.percentile(del_dep_e_neutral, 75) - np.percentile(del_dep_e_neutral, 25)

    del_dep_e_neutral_trind = del_dep_e[~truth_is_charged & tr_ind_mask]
    if (~truth_is_charged & tr_ind_mask).sum() > 0:
        log_dict['del_dep_e_neutral_trind_mean'] = del_dep_e_neutral_trind.mean()
        log_dict['del_dep_e_neutral_trind_std'] = del_dep_e_neutral_trind.std()
        log_dict['del_dep_e_neutral_trind_iqr'] = np.percentile(del_dep_e_neutral_trind, 75) - np.percentile(del_dep_e_neutral_trind, 25)

    del_dep_e_neutral_trind_prind = del_dep_e[~truth_is_charged & tr_ind_mask & pr_ind_mask]
    if (~truth_is_charged & tr_ind_mask & pr_ind_mask).sum() > 0:
        log_dict['del_dep_e_neutral_trind_prind_mean'] = del_dep_e_neutral_trind_prind.mean()
        log_dict['del_dep_e_neutral_trind_prind_std'] = del_dep_e_neutral_trind_prind.std()
        log_dict['del_dep_e_neutral_trind_prind_iqr'] = np.percentile(del_dep_e_neutral_trind_prind, 75) - np.percentile(del_dep_e_neutral_trind_prind, 25)

    return fig, log_dict



def plot_hyperedge(hyperedge_dicts, apply_truth_ind_mask=False):

    cmap = plt.cm.get_cmap('winter_r')
    cmap.set_under('white')

    flat_dict = {}
    for k in hyperedge_dicts[0].keys():
        flat_dict[k] = np.hstack([he_dict[k] for he_dict in hyperedge_dicts])

    if apply_truth_ind_mask:
        truth_ind_mask = flat_dict['truth_ind'] > 0.5
        for k, v in flat_dict.items():
            flat_dict[k] = v[truth_ind_mask]

    # classification
    fig_class = plt.figure(figsize=(25, 8), dpi=100)
    gs = fig_class.add_gridspec(2, 3, wspace=0.15, hspace=0.6)

    class_labels = ['ch had', 'e', 'mu', 'neut had', 'photon', 'residual']
    n_class = len(class_labels)
    pt_or_ke_thresholds = [[0, 500],[0, 1], [1, 3], [3, 10], [10, 25], [25, 500]]
    for mm_i, min_max in enumerate(pt_or_ke_thresholds):
        ax = fig_class.add_subplot(gs[mm_i])
        mask_ch = (flat_dict['truth_pt_raw'] > min_max[0]) & (flat_dict['truth_pt_raw'] < min_max[1])
        mask_neut = (flat_dict['truth_ke_raw'] > min_max[0]) & (flat_dict['truth_ke_raw'] < min_max[1])
        mask = mask_ch * (flat_dict['truth_class'] <= 2) + mask_neut * (flat_dict['truth_class'] > 2)
        pred_class = flat_dict['pred_class'][mask]
        truth_class = flat_dict['truth_class'][mask]
        cm = confusion_matrix(pred_class, truth_class, labels=np.arange(n_class), normalize=None)
        df_cm = pd.DataFrame(cm, 
            index = [class_labels[i] for i in range(n_class)], 
            columns = [class_labels[i] for i in range(n_class)]
        )
        sn.heatmap(df_cm, annot=True, ax=ax, cmap=cmap, vmin=1e-8)
        ax.set_xlabel('truth class')
        ax.set_ylabel('pred class')
        ax.set_title(f'{min_max[0]} GeV < truth pt(ch) or ke(neut) < {min_max[1]} GeV')

    if 'truth_eta_raw' not in flat_dict:
        return fig_class, None


    # regerssion
    vars = ['pt_or_ke_raw', 'eta_raw', 'phi']
    cl_names = ['ch_had', 'e', 'mu', 'neut_had', 'photon']

    fig_regression = plt.figure(figsize=(len(vars)*8, len(cl_names)*2.5), dpi=150)
    gs = fig_regression.add_gridspec(len(cl_names), len(vars), wspace=0.15, hspace=0.6)

    for cl_i, cl_name in enumerate(cl_names):
        mask = flat_dict['truth_class'] == cl_i
        if mask.sum() == 0:
            continue

        for v_i, var in enumerate(vars):
            sub_gs = gs[cl_i, v_i].subgridspec(1, 2, wspace=0.1, width_ratios=[1.5, 1])

            if var == 'pt_or_ke_raw':
                if cl_name in ['ch_had', 'e', 'mu']: var = 'pt_raw'
                else: var = 'ke_raw'

            proxy_res = flat_dict[f'truth_{var}'][mask] - flat_dict[f'proxy_{var}'][mask]
            pred_res  = flat_dict[f'truth_{var}'][mask] - flat_dict[f'pred_{var}'][mask]
    
            # residual as a function of proxy_x
            min_proxy = flat_dict[f'proxy_{var}'][mask].min()
            max_proxy = flat_dict[f'proxy_{var}'][mask].max()
            bins = np.linspace(min_proxy-1e-8, max_proxy+1e-8, 20)
            
            x_vals = (bins[1:] + bins[:-1])/2
            bin_width = bins[1] - bins[0]

            y_vals_proxy = []; y_errs_proxy = []; y_vals_pred = []; y_errs_pred = []
            for i in range(len(bins)-1):
                mask_2 = (flat_dict[f'proxy_{var}'][mask] >= bins[i]) & (flat_dict[f'proxy_{var}'][mask] < bins[i+1])

                # ignore warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y_vals_proxy.append(proxy_res[mask_2].mean())
                        y_errs_proxy.append(proxy_res[mask_2].std())
                        y_vals_pred.append(pred_res[mask_2].mean())
                        y_errs_pred.append(pred_res[mask_2].std())

            # this may happen at low stat
            if np.isnan(y_vals_proxy).all():
                continue

            ax = fig_regression.add_subplot(sub_gs[0])
            ax.errorbar(x_vals, y_vals_proxy, xerr=bin_width/2, yerr=y_errs_proxy, fmt="none", label='proxy', color='blue')    
            ax.errorbar(x_vals, y_vals_pred, xerr=bin_width/2, yerr=y_errs_pred, fmt="none", label='pred', color='red')
            ax.set_xlabel(f'proxy')
            ax.set_ylabel(f'truth - X')
            ax.set_title(f'{cl_name} ({var})')
            ax.hlines(0, min_proxy, max_proxy, color='black', linestyle='--', lw=0.5)
            ax.grid(zorder=3)
            ax.legend()

        
            # scatters        
            ax = fig_regression.add_subplot(sub_gs[1])

            var = var.replace('_raw', '') if 'raw' in var else var
            target = flat_dict[f'truth_{var}'][mask] - flat_dict[f'proxy_{var}'][mask]
            pred = flat_dict[f'pred_{var}'][mask] - flat_dict[f'proxy_{var}'][mask]
            comb = np.concatenate([target, pred])

            min_ = np.percentile(comb, 3); max_ = np.percentile(comb, 97)
            bins = np.linspace(min_, max_, 50)
            ax.hist2d(target, pred, bins=bins, cmap=cmap, norm=LogNorm())

            ax.plot([min_, max_], [min_, max_], 'r--',lw=0.5)
            ax.set_aspect('equal')
            ax.set_xlabel(f'truth - proxy'); ax.set_ylabel(f'pred - proxy')
            ax.set_title(f'NN space')


    return fig_class, fig_regression
