import torch
import torch.nn.functional as F

from .utils import shuffle_joint, tanh_clip

"""
Score functions
-Simplified log_softmax scores
"""

def abstract_scores_action(model_t,model_t_p_1,s_t,s_t_p_1,s_t_p_k,a_t,encoder_shape,score_fn,device):
    """
    Evaluate psi(s_t), psi(s_t+1), psi(s_t+k) and psi(a_t), then compute NCE scores
    """
    if a_t is None:
        t = model_t(s_t)
    else:
        t = model_t(s_t,a_t)
    t_p_1 = model_t_p_1(s_t_p_1)
    if s_t_p_k is not None:
        t_p_k = model_t_p_1(s_t_p_k)
    else:
        t_p_k = None
    nce, reg = score_fn(t,t_p_1,t_p_k,encoder_shape,device)
    return nce, reg

def nce_scores_log_softmax(reference_samples,positive_samples,negative_samples,encoder_shape,device):
    """
    Compute NCE + and - scores with log_softmax.
    Re-use the expanded (n_locs x n_batch x n_batch) version
    """
    nce_scores, lgt_reg = nce_scores_log_softmax_expanded(reference_samples,positive_samples,negative_samples,encoder_shape,device)
    nce_scores = nce_scores.sum(2)

    return nce_scores, lgt_reg

def nce_scores_log_softmax_expanded(reference_samples,positive_samples,negative_samples,encoder_shape,device):
    """
    Compute NCE + and - scores with log_softmax but return a vector of n_locs instead of a mean
    """
    if len(reference_samples.shape) == 2:
        reference_samples = reference_samples.unsqueeze(-1).unsqueeze(-1).repeat(1,1,encoder_shape[2],encoder_shape[3])
    if len(positive_samples.shape) == 2:
        positive_samples = positive_samples.unsqueeze(-1).unsqueeze(-1).repeat(1,1,encoder_shape[2],encoder_shape[3])
    if negative_samples is not None and len(negative_samples.shape) == 2:
        negative_samples = negative_samples.unsqueeze(-1).unsqueeze(-1).repeat(1,1,encoder_shape[2],encoder_shape[3])

    reference_samples = reference_samples.view(encoder_shape[:2]+[-1]).permute(2,0,1) # before permute: n_batch x n_rkhs x n_locs , desired: n_loc x n_batch x n_rkhs
    positive_samples = positive_samples.view(encoder_shape[:2]+[-1]).permute(2,1,0) # before permute: n_batch x n_rkhs x n_locs, desired: n_locs x n_rkhs x n_batch

    lgt_reg = 0
    positive_pairs = torch.matmul(reference_samples, positive_samples) / encoder_shape[1]**0.5 # n_locs x n_batch x n_batch
    lgt_reg += (positive_pairs**2.).mean()
    positive_pairs = tanh_clip(positive_pairs)
    score_shape = positive_pairs.shape

    scores = F.log_softmax(positive_pairs, 2) # (n_locs, n_batch, n_batch)

    mask = torch.eye(score_shape[-1]).unsqueeze(0).repeat(score_shape[0],1,1).to(device) # n_locs x n_batch x n_batch
    nce_scores = ( scores * mask ) # n_locs x n_batch x n_batch


    return nce_scores, lgt_reg


"""
Loss functions
-InfoNCE (s->s')
-InfoNCE + action (sa->s')
"""

def InfoNCE_action_loss(model, s_t, a_t, r_t, s_t_p_1,args,s_t_p_k=None,target=None):
    device = args['device']
    score_fn = globals()[args['score_fn']]

    local_encoder = model.local_encoder # Local encoder
    global_encoder = model.global_encoder # Global encoder
    action_encoder = model.action_encoder

    # Extract features phi(s)
    s_t_local = local_encoder(s_t.float())
    
    s_t_p_1_local = model.local_encoder(s_t_p_1.float())
    if args['lambda_GL'] != 0 or args['lambda_GG'] != 0:
        s_t_global = global_encoder(s_t_local)
    if args['lambda_LG'] != 0 or args['lambda_GG'] != 0:
        s_t_p_1_global = model.global_encoder(s_t_p_1_local)
    if args['score_fn'] not in ('nce_scores_log_softmax','nce_scores_log_softmax_expanded'):
        s_t_p_k_local = model.local_encoder(shuffle_joint(s_t_p_1.float()))
        s_t_p_k_global = model.global_encoder(s_t_p_k_local)
    else:
        s_t_p_k_local = None
        s_t_p_k_global = None

    a_t_global = action_encoder(a_t)
    a_t_local = a_t_global
    a_t_local = a_t_local.unsqueeze(-1).unsqueeze(-1).repeat(1,1,s_t_local.shape[2],s_t_local.shape[3]) # repeat HxW times to be compatible with local concats

    encoder_shape = list(model.psi_local_LL_t_p_1(s_t_p_1_local).shape)

    # Local -> Local
    if args['lambda_LL'] != 0:
        psi_local_LL_t = model.psi_local_LL_t
        psi_local_LL_t_p_1 = model.psi_local_LL_t_p_1
        nce_L_L, reg_L_L = abstract_scores_action(psi_local_LL_t,psi_local_LL_t_p_1,
                            s_t_local,
                            s_t_p_1_local,
                            s_t_p_k_local,
                            a_t_local,
                            encoder_shape,score_fn,device)
    else:
        nce_L_L = torch.zeros(1).to(device)
        reg_L_L = torch.zeros(1).to(device)

    # Local -> Global
    if args['lambda_LG'] != 0:
        psi_local_LG = model.psi_local_LG
        psi_global_LG = model.psi_global_LG
            
        nce_L_G, reg_L_G = abstract_scores_action(psi_local_LG,psi_global_LG,
                            s_t_local,
                            s_t_p_1_global,
                            s_t_p_k_global,
                            a_t_local,
                            encoder_shape,score_fn,device)
    else:
        nce_L_G = torch.zeros(1).to(device)
        reg_L_G = torch.zeros(1).to(device)

    # Global -> Local
    if args['lambda_GL'] != 0:
        psi_local_GL = model.psi_local_GL
        psi_global_GL = model.psi_global_GL
        
        nce_G_L, reg_G_L = abstract_scores_action(psi_global_GL,psi_local_GL,
                            s_t_global,
                            s_t_p_1_local,
                            s_t_p_k_local,
                            a_t_global,
                            encoder_shape,score_fn,device)
    else:
        nce_G_L = torch.zeros(1).to(device)
        reg_G_L = torch.zeros(1).to(device)

    # Global -> Global
    if args['lambda_GG'] != 0:
        psi_global_GG_t = model.psi_global_GG_t
        psi_global_GG_t_p_1 = model.psi_global_GG_t_p_1
        
        nce_G_G, reg_G_G = abstract_scores_action(psi_global_GG_t,psi_global_GG_t_p_1,
                            s_t_global,
                            s_t_p_1_global,
                            s_t_p_k_global,
                            a_t_global,
                            encoder_shape,score_fn,device)
    else:
        nce_G_G = torch.zeros(1).to(device)
        reg_G_G = torch.zeros(1).to(device)

    return {'nce_L_L':nce_L_L,
            'nce_L_G':nce_L_G,
            'nce_G_L':nce_G_L,
            'nce_G_G':nce_G_G,
            'reg_L_L':reg_L_L,
            'reg_L_G':reg_L_G,
            'reg_G_L':reg_G_L,
            'reg_G_G':reg_G_G
    }


def InfoNCE_no_action_loss(model,s_t, a_t, r_t, s_t_p_1,args,s_t_p_k=None,target=None):
    device = args['device']
    score_fn = globals()[args['score_fn']]

    local_encoder = model.local_encoder # Local encoder
    global_encoder = model.global_encoder # Global encoder

    # Extract features phi(s)
    s_t_local = local_encoder(s_t.float())
    s_t_global = global_encoder(s_t_local)
    s_t_p_1_local = local_encoder(s_t_p_1.float())
    s_t_p_1_global = global_encoder(s_t_p_1_local)
    if args['score_fn'] not in ('nce_scores_log_softmax','nce_scores_log_softmax_expanded'):
        s_t_p_k_local = local_encoder(shuffle_joint(s_t_p_1.float()))
        s_t_p_k_global = global_encoder(s_t_p_k_local)
    else:
        s_t_p_k_local = None
        s_t_p_k_global = None

    encoder_shape = list(model.psi_local_LL(s_t_p_1_local).shape)

    # Local -> Local
    if args['lambda_LL'] != 0:
        psi_local_LL_t = model.psi_local_LL
        psi_local_LL_t_p_1 = model.psi_local_LL
        nce_L_L, reg_L_L = abstract_scores_action(psi_local_LL_t,psi_local_LL_t_p_1,
                            s_t_local,
                            s_t_p_1_local,
                            s_t_p_k_local,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_L_L = torch.zeros(1).to(device)
        reg_L_L = torch.zeros(1).to(device)

    # Local -> Global
    if args['lambda_LG'] != 0:
        psi_local_LG = model.psi_local_LG
        psi_global_LG = model.psi_global_LG
        nce_L_G, reg_L_G = abstract_scores_action(psi_local_LG,psi_global_LG,
                            s_t_local,
                            s_t_p_1_global,
                            s_t_p_k_global,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_L_G = torch.zeros(1).to(device)
        reg_L_G = torch.zeros(1).to(device)

    # Global -> Local
    if args['lambda_GL'] != 0:
        psi_local_GL = model.psi_local_GL
        psi_global_GL = model.psi_global_GL
        nce_G_L, reg_G_L = abstract_scores_action(psi_global_GL,psi_local_GL,
                            s_t_global,
                            s_t_p_1_local,
                            s_t_p_k_local,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_G_L = torch.zeros(1).to(device)
        reg_G_L = torch.zeros(1).to(device)

    # Global -> Global
    if args['lambda_GG'] != 0:
        psi_global_GG_t = model.psi_global_GG
        psi_global_GG_t_p_1 = model.psi_global_GG
        nce_G_G, reg_G_G = abstract_scores_action(psi_global_GG_t,psi_global_GG_t_p_1,
                            s_t_global,
                            s_t_p_1_global,
                            s_t_p_k_global,
                            None,
                            encoder_shape,score_fn,device)
    else:
        nce_G_G = torch.zeros(1).to(device)
        reg_G_G = torch.zeros(1).to(device)

    return {'nce_L_L':nce_L_L,
            'nce_L_G':nce_L_G,
            'nce_G_L':nce_G_L,
            'nce_G_G':nce_G_G,
            'reg_L_L':reg_L_L,
            'reg_L_G':reg_L_G,
            'reg_G_L':reg_G_L,
            'reg_G_G':reg_G_G
    }
