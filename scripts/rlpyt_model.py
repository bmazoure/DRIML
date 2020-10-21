import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from .models import *
from .utils import Arguments, make_procgen_action_matrix, select_architecture


class AtariCatDqnModel_nce(nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms=51,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            frame_stack=4,
            nce_loss='InfoNCE_action_loss',
            algo='c51',
            env_name=None
            ):
        super().__init__()
        self.dueling = dueling
        c, h, w = image_shape
        self.args = Arguments({'frame_stack':int(frame_stack==1),'nce_loss':nce_loss,'algo':algo})
        dummy_state = np.zeros((h,w,c))
        
        network = select_architecture(self.args,globals())
        self.model = network(dummy_state,output_size,{})

        self.conv = self.model.convs
        conv_out_size = self.model.out_channels

        # Pick the right head if dueling or not (only ever worked with non-dueling)
        if dueling:
            self.head = DistributionalDuelingHeadModel(conv_out_size, fc_sizes,
                output_size=output_size, n_atoms=n_atoms, fc_1_V=self.model.fc_1)
        else:
            self.head = DistributionalHeadModel(conv_out_size, fc_sizes,
                output_size=output_size, n_atoms=n_atoms,fc_1=self.model.fc_1 )   
        self.model.fc_1.requires_grad = True

        # Make the procgen action mapping matrix
        tmp_map_procgen, PROCGEN_ACTION_MAT = make_procgen_action_matrix()

        if env_name.split('-')[1] in tmp_map_procgen.keys(): # procgen n_step_nce = -1
            mat = torch.LongTensor(PROCGEN_ACTION_MAT[env_name.split('-')[1]])
            N_hidden = 15
            N_visible = (mat.sum(0)>0).sum()
            # There is 15 actions in total, but some actions map to other, making this set reduced
            self.action_net = Action_net( 2 *  N_visible * N_hidden )
        self.A_hat_visible = nn.Parameter(torch.zeros(15,15))
        

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        
        p = self.head(conv_out.view(T * B, -1))
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p
