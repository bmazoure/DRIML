import torch
import numpy as np
import random

def init(module, weight_init, bias_init):
    """
    Apply the initialization function to both weights and bias of input module
    """
    weight_init(module.weight.data)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def make_one_hot(labels, C=2):
    """
    Scatters integer labels into a one-hot encoding
    """
    one_hot = torch.FloatTensor(size=(labels.size(0),C)).zero_()
    if torch.cuda.is_available():
        one_hot = one_hot.cuda()
    target = one_hot.scatter_(1, labels.unsqueeze(-1).long(), 1).float()
    return target

def compute_network_output_size(h,w,kernels_h,kernels_w,strides_h,strides_w):
    """
    Automatically compute the output size of standard convolutional layers
    """
    for (k_h,k_w,s_h,s_w) in zip(kernels_h,kernels_w,strides_h,strides_w):
        h = (h-k_h) / s_h + 1
        w = (w-k_w) / s_w + 1
    return int(h) * int(w)

def select_architecture(args,class_list):
    """
    Method to select the relevant network based on the loss
    """
    loss_fn = args.nce_loss
    if 'action' in loss_fn:
        return class_list['infoNCE_Mnih_84x84_action']
    else:
        return class_list['infoNCE_Mnih_84x84']

class Arguments(object):
    """
    Dummy class to mimic argparse
    """
    def __init__(self,args):
        for k,v in args.items():
            setattr(self, k, v)

def make_procgen_action_matrix():
    """
    Procgen has hidden actions (i.e. 1 to 15 keys), and visible ones (e.g. in running games, the special action key maps to a no-op).
    This matrix maps every hidden action in every Procgen game to the actual action which is visible to the agent.
    """
    tmp_map_procgen = {'bigfish':[[a,4] for a in range(9,15)], #9,10,11,12,13,14 -> no-op (4)
                                'bossfight':[[a,4] for a in range(10,15)], # 10,11,12,13,14 -> no-op since only 1 special move
                                'caveflyer':[[a,4] for a in range(10,15)],
                                'chaser':[[a,4] for a in range(9,15)],
                                'climber':[[0,1],[3,4],[6,7]] + [[a,4] for a in range(9,15)],  # clip vel_y to >= 0 and no special move
                                'coinrun':[[a,4] for a in range(9,15)],
                                'dodgeball':[[a,4] for a in range(10,15)],
                                'fruitbot':[[0,1],[3,4],[6,7],[2,1],[5,4],[8,7]]+[[a,4] for a in range(10,15)],
                                'heist':[[a,4] for a in range(9,15)],
                                'jumper':[[0,1],[3,4],[6,7]]+[[a,4] for a in range(9,15)],
                                'leaper':[[a,4] for a in range(9,15)],
                                'maze':[[0,1],[6,7],[2,1],[8,7]] + [[a,4] for a in range(9,15)],
                                'miner':[[0,1],[6,7],[2,1],[8,7]] + [[a,4] for a in range(9,15)],
                                'ninja':[[0,1],[3,4],[6,7],[13,4],[14,4]],
                                'plunder':[[0,1],[3,4],[6,7],[2,1],[5,4],[8,7]]+[[a,4] for a in range(10,15)],
                                'starpilot':[]
                                }
    PROCGEN_ACTION_MAT = {}
    for env_name, vec in tmp_map_procgen.items():
        mat = np.eye(15)
        for (x,y) in vec:
            mat[x] = 0.
            mat[x,y] = 1.
        PROCGEN_ACTION_MAT[env_name] = mat
    return tmp_map_procgen, PROCGEN_ACTION_MAT

def shuffle_joint(x):
    """
    Shuffles input x on first (batch) dimension
    Args:
        x (torch.Tensor): (n,**) A >1D tensor, gets shuffled along batch dimension
    """
    n = len(x)
    idx = np.array(range(n))
    np.random.shuffle(idx)
    return x[idx]

def tanh_clip(x, clip_val=20.):
    '''
    soft clip values to the range [-clip_val, +clip_val]
    Trick from AM-DIM
    '''
    if clip_val is not None:
        x_clip = clip_val * torch.tanh((1. / clip_val) * x)
    else:
        x_clip = x
    return x_clip

def set_seed(seed,cuda):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_config(algo):
    if algo == 'c51':
        config = dict(
            agent=dict(
                eps_init=0.1,
                eps_final=0.01
                ),
            algo=dict(
                discount=0.99,
                batch_size=256, # -> 128
                delta_clip=1.,
                learning_rate=2.5e-4, # -> 6.5e-4 (rainbow), 2.5e-4 (C51)
                target_update_interval=int(312),
                clip_grad_norm=10., # -> 10
                min_steps_learn=int(1000), # -> 5e4
                target_update_tau=0.95, # tau * new + (1-tau) * old

                double_dqn=True,
                prioritized_replay=True,
                n_step_return=3, # -> 1

                replay_size=int(1e6),
                replay_ratio=8, # -> 8,

                pri_alpha=0.5,
                pri_beta_init=0.4,
                pri_beta_final=1.,
                pri_beta_steps=int(50e6),

                eps_steps=int(1e5)
            ),
            env=dict(
                game=None,
                episodic_lives=False,
                clip_reward=False,
                horizon=int(27e3),
                max_start_noops=0,
                repeat_action_probability=0.,
                frame_skip=1,
                num_img_obs=4
            ),
            eval_env=dict(
                game=None,
                episodic_lives=False,
                horizon=int(27e3),
                clip_reward=False,
                max_start_noops=0,
                repeat_action_probability=0.,
                frame_skip=1,
                num_img_obs=4
            ),
            model=dict(dueling=False),
            optim=dict(),
            runner=dict(
                n_steps=5e7,
                log_interval_steps=5e5,
            ),
            sampler=dict(
                batch_T=7,
                batch_B=32, # this should be batch_size // replay_ratio
                max_decorrelation_steps=1000,
                eval_n_envs=4,
                eval_max_steps=int(125e3),
                eval_max_trajectories=100,
            ),
        )
    return config