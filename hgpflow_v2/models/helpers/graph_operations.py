import torch

def custom_update_all(edge_fn, node_fn,
        **kwargs):
    '''
        kwargs with prefix 'efn_' will be passed to edge_fn
        kwargs with prefix 'nfn_' will be passed to node_fn
    '''

    # edge_fn kwargs
    edge_fn_kwargs = {}; node_fn_kwargs = {}
    for k, v in kwargs.items():
        if k.startswith('efn_'):
            edge_fn_kwargs[k[4:]] = v
        elif k.startswith('nfn_'):
            node_fn_kwargs[k[4:]] = v
        else:
            raise ValueError('invalid kwarg prefix')

    # B, N, M, D
    edge_msg = edge_fn(**edge_fn_kwargs)

    # B, N, D
    node_fn_kwargs['edge_msg'] = edge_msg
    dst_feat = node_fn(**node_fn_kwargs)

    return dst_feat





#---------------------------------------
# edge_fns: B, N, D -> B, N, M, D
#---------------------------------------

def custom_copy_u(src_feat, num_nodes_dst,
    src_mask=None, dst_mask=None, edge_mask=None):
    '''
        Function to copy the src features to the edges

        src_feat      : B, N, D
        num_nodes_dst : M
        src_mask      : B, N
        dst_mask      : B, M
        edge_mask     : B, N, M
    '''
    M = num_nodes_dst

    # B, N, D -> B, N, M, D
    edge_msg = src_feat.unsqueeze(2).repeat(1, 1, M, 1)

    # B, N, M, 1
    mask = torch.ones_like(edge_msg, dtype=torch.bool)
    if src_mask is not None:
        # B, N -> B, N, 1, 1
        mask = mask * src_mask.unsqueeze(-1).unsqueeze(-1) 
    if dst_mask is not None:
        # B, 1 -> B, 1, M, 1
        mask = mask * dst_mask.unsqueeze(1).unsqueeze(-1)
    if edge_mask is not None:
        # B, N, M -> B, N, M, 1
        mask = mask * edge_mask.unsqueeze(-1)

    edge_msg = edge_msg * mask
    return edge_msg


def custom_u_mul_e(src_feat, num_nodes_dst, edge_feat,
    src_mask=None, dst_mask=None, edge_mask=None):
    '''
        Function to copy the src features to the edges

        src_feat      : B, N, D
        num_nodes_dst : M
        edge_feat     : B, N, M, D
        src_mask      : B, N
        dst_mask      : B, M
        edge_mask     : B, N, M
    '''
    M = num_nodes_dst

    # B, N, D -> B, N, M, D
    edge_msg = src_feat.unsqueeze(2).repeat(1, 1, M, 1)
    edge_msg = edge_msg * edge_feat

    # B, N, M, 1
    mask = torch.ones_like(edge_msg, dtype=torch.bool)
    if src_mask is not None:
        # B, N -> B, N, 1, 1
        mask = mask * src_mask.unsqueeze(-1).unsqueeze(-1) 
    if dst_mask is not None:
        # B, 1 -> B, 1, M, 1
        mask = mask * dst_mask.unsqueeze(1).unsqueeze(-1)
    if edge_mask is not None:
        # B, N, M -> B, N, M, 1
        mask = mask * edge_mask.unsqueeze(-1)

    edge_msg = edge_msg * mask
    return edge_msg






#---------------------------------------
# node_fns: B, N, D -> B, N, M, D
#---------------------------------------

def custom_sum_mailbox(edge_msg, edge_mask=None):
    '''
        Function to sum the edge messages

        edge_msg: B, N, M, D
    '''

    if edge_mask is None:
        edge_mask = torch.ones(edge_msg.shape[:-1], dtype=torch.float32, device=edge_msg.device)

    edge_msg = edge_msg * edge_mask.unsqueeze(-1)

    # B, N, M, D -> B, M, D
    edge_msg = edge_msg.sum(dim=1)

    return edge_msg



def custom_mean_mailbox(edge_msg, edge_mask=None):
    '''
        Function to mean the edge messages

        edge_msg: B, N, M, D
        edge_mask: B, N, M
    '''

    if edge_mask is None:
        edge_mask = torch.ones(edge_msg.shape[:-1], dtype=torch.float32, device=edge_msg.device)

    edge_msg = edge_msg * edge_mask.unsqueeze(-1)

    # B, N, M, D -> B, M, D
    edge_msg = edge_msg.sum(dim=1)
    edge_msg = edge_msg / (edge_mask.sum(dim=1).unsqueeze(-1) + 1e-8)

    return edge_msg

