import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.models.mlp import MlpModel

from .utils import make_one_hot, init, compute_network_output_size

"""
Rlpyt models
"""

class Action_net(nn.Module):
    def __init__(self,n_actions):
        """
        The very simple action encoder
        """
        super(Action_net,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_actions, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.layers(x)

class DistributionalHeadModel(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, n_atoms,fc_1=None):
        """
        Distributional C51 head copied from rlpyt
        """
        super().__init__()
        if fc_1 is None:
            self.fc_1 = nn.Linear(input_size, layer_sizes)
        else:
            self.fc_1 = fc_1
        self.fc_2 = nn.Linear(layer_sizes, output_size * n_atoms)
        self.mlp = nn.Sequential(*[self.fc_1,nn.ReLU(),self.fc_2])
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.mlp(input).view(-1, self._output_size, self._n_atoms)

class DistributionalDuelingHeadModel(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            n_atoms,
            grad_scale=2 ** (-1 / 2),
            fc_1_V = None
            ):
        """
        Dueling distributional C51 head copied from rlpyt
        """
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.advantage_hidden = MlpModel(input_size, hidden_sizes)
        self.advantage_out = torch.nn.Linear(hidden_sizes[-1],
            output_size * n_atoms, bias=False)
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms))
        if fc_1_V is None:
            self.value = MlpModel(input_size, hidden_sizes, output_size=n_atoms)
        else:
            self.value = nn.Sequential(*[fc_1_V,nn.ReLU(),nn.Linear(hidden_sizes[0],n_atoms)])
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

"""
DRIML encoders based off the DQN (Mnih) architecture
"""

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, input_batch):
        features = self.global_encoder(self.local_encoder(input_batch))
        return features

class infoNCE_Mnih_84x84_action(Encoder):
    def __init__(self, obs_space, num_outputs, options):
        """
        Model with action encoder
        """

        super().__init__()
        (w, h, in_channels) = obs_space.shape

        action_dim = 64
        rkhs_dim = 512
        init_fn = lambda m: init(m,
                               lambda x:nn.init.orthogonal_(x,gain=nn.init.calculate_gain('relu')),
                               #lambda x:torch.nn.init.kaiming_uniform_(x,a=0,mode='fan_in',nonlinearity='relu'),
                               lambda x: nn.init.constant_(x, 0))

        self.out_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64
        self.fc_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64

        self.conv1 = init_fn( nn.Conv2d(in_channels, 32, [8,8], 4) )
        self.conv2 = init_fn( nn.Conv2d(32, 64, [4,4], 2) )
        self.conv3 = init_fn( nn.Conv2d(64, 64, [3,3], 1) )
        self.flatten = Flatten()
        self.fc_1 = init_fn( nn.Linear(self.fc_channels, 512) )

        self.psi_local_LL_t = ResBlock_conv(64+action_dim,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_LL_t_p_1 = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_LG = ResBlock_conv(64+action_dim,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_GL = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)

        self.psi_global_LG = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GL = ResBlock_fc(512+action_dim,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GG_t = ResBlock_fc(512+action_dim,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GG_t_p_1 = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)

        self.layers = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten,self.fc_1,nn.ReLU()])
        self.convs  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

        self.global_encoder = nn.Sequential(*[self.flatten,self.fc_1])
        self.local_encoder  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])
        self.action_encoder = OneHot_fc(num_outputs,action_dim,action_dim,init_fn)

class infoNCE_Mnih_84x84(Encoder):
    def __init__(self, obs_space, num_outputs, options):
        """
        Model without action encoder
        """

        super().__init__()
        (w, h, in_channels) = obs_space.shape

        rkhs_dim = 128
        init_fn = lambda m: init(m,
                               lambda x:torch.nn.init.kaiming_uniform_(x,a=0,mode='fan_in',nonlinearity='relu'),
                               lambda x: nn.init.constant_(x, 0))

        self.out_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64
        self.fc_channels = compute_network_output_size(h,w,[8,4,3],[8,4,3],[4,2,1],[4,2,1]) * 64

        self.conv1 = init_fn( nn.Conv2d(in_channels, 32, [8,8], 4) )
        self.conv2 = init_fn( nn.Conv2d(32, 64, [4,4], 2) )
        self.conv3 = init_fn( nn.Conv2d(64, 64, [3,3], 1) )
        self.flatten = Flatten()
        self.fc_1 = init_fn( nn.Linear(self.fc_channels, 512) )

        self.psi_local_LL = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_LG = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)
        self.psi_local_GL = ResBlock_conv(64,rkhs_dim,rkhs_dim,init_fn)

        self.psi_global_LG = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GL = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)
        self.psi_global_GG = ResBlock_fc(512,rkhs_dim,rkhs_dim,init_fn)

        self.layers = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU(),self.flatten,self.fc_1,nn.ReLU()])
        self.convs  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

        self.global_encoder = nn.Sequential(*[self.flatten,self.fc_1])
        self.local_encoder  = nn.Sequential(*[self.conv1,nn.ReLU(),self.conv2,nn.ReLU(),self.conv3,nn.ReLU()])

"""
Helper layer (flatten, ResBlocks, 1-hot)
"""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ResBlock_conv(nn.Module):
    """
    Simple 1 hidden layer resblock
    """

    def __init__(self, in_features, hidden_features, out_features, init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0))):
        super(ResBlock_conv, self).__init__()

        self.psi_1 = init_fn( nn.Conv2d(in_features,hidden_features, [1,1], 1, bias=True) )
        self.psi_2 = init_fn( nn.Conv2d(hidden_features,out_features, [1,1], 1, bias=True) )

        self.W = init_fn( nn.Conv2d(in_features,hidden_features, [1,1], 1, bias=True) )

    def forward(self,x,action=None):
        if action is not None:
            x = torch.cat([x,action],dim=1)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x

class ResBlock_fc(nn.Module):
    """
    Simple 1 hidden layer resblock (for fully-connected inputs)
    """

    def __init__(self, in_features, hidden_features, out_features,init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0))):
        super(ResBlock_fc, self).__init__()

        self.psi_1 =  nn.Linear(in_features, hidden_features, bias=True) 
        self.psi_2 =  nn.Linear(hidden_features, out_features, bias=True) 

        self.W = init_fn( nn.Linear(in_features,out_features,bias=False) )

    def forward(self,x,action=None):
        if action is not None:
            x = torch.cat([x,action],dim=1)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x

class OneHot_fc(nn.Module):
    """
    Simple 1 hidden layer resblock for a one-hot vector
    """

    def __init__(self, in_features, hidden_features, out_features,init_fn=lambda m: init(m, nn.init.orthogonal_,lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))):
        super().__init__()

        self.in_features = in_features
        self.psi_1 = init_fn( nn.Linear(in_features, hidden_features) )
        self.psi_2 = init_fn( nn.Linear(hidden_features, out_features) )
        self.W = init_fn( nn.Linear(in_features,out_features,bias=True) )

    def forward(self,x):
        x = make_one_hot(x,self.in_features)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x