import math
import os
from collections import Counter, defaultdict, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.algos.utils import valid_from_done
from rlpyt.replays.non_sequence.frame import UniformReplayFrameBuffer
from rlpyt.replays.sequence.frame import (
    AsyncPrioritizedSequenceReplayFrameBuffer,
    AsyncUniformSequenceReplayFrameBuffer,
    PrioritizedSequenceReplayFrameBuffer, UniformSequenceReplayFrameBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from torchvision import transforms

from .losses import *
from .utils import make_procgen_action_matrix

tmp_map_procgen, PROCGEN_ACTION_MAT = make_procgen_action_matrix()

EPS = 1e-6  # (NaN-guard)

OptInfoNCE = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr","lossNCE"]+['action%d'%i for i in range(15)])

# OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr", "priority"])
SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn", SamplesToBuffer._fields + ("prev_rnn_state",))
PrioritiesSamplesToBuffer = namedarraytuple("PrioritiesSamplesToBuffer",["priorities", "samples"])



class CategoricalDQN_nce(DQN):

    def __init__(self, V_min=-10, V_max=10,args=None, **kwargs):
        super().__init__(**kwargs)
        # self.agent = a
        self.V_min = V_min
        self.V_max = V_max
        self.args = args
        if "eps" not in self.optim_kwargs:  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.batch_size
        self.warmup_T = 0

        device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.reset_nce_accumulators(device)

        self.A_mat_itr = 0
        self.itr = 0

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        if self.args['n_step_nce'] > 1 or self.args['n_step_nce'] < 0:
            ReplayCls = UniformSequenceReplayFrameBuffer
            replay_kwargs['rnn_state_interval'] = 0
            replay_kwargs['batch_T'] = batch_spec.T + self.warmup_T
        elif self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = (AsyncPrioritizedReplayFrameBuffer if async_ else
                PrioritizedReplayFrameBuffer)
        else:
            ReplayCls = (AsyncUniformReplayFrameBuffer if async_ else
                UniformReplayFrameBuffer)
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optim_initialize(self, rank=0):
        """Called by async runner."""
        self.rank = rank
        self.optimizer = self.OptimCls(self.agent.parameters(),
                lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)
        # the q network, which learns pairwise action affinities
        self.action_net_optimizer = self.OptimCls(self.agent.model.action_net.parameters(),
                lr=self.learning_rate, **self.optim_kwargs)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.agent.give_V_min_max(self.V_min, self.V_max)

    def async_initialize(self, *args, **kwargs):
        buffer = super().async_initialize(*args, **kwargs)
        self.agent.give_V_min_max(self.V_min, self.V_max)
        return buffer

    def reset_nce_accumulators(self,device):
        """
        Batch accumulators for NCE loss, pre-allocated on the GPU
        """
        self.device = device
        self.actions = torch.zeros(size=(self.args['nce_batch_size'],),dtype=torch.int64,device=self.device)
        self.returns = torch.zeros(size=(self.args['nce_batch_size'],),dtype=torch.float32,device=self.device)
        self.nonterminals = torch.zeros(size=(self.args['nce_batch_size'],),dtype=torch.float32,device=self.device)
        self.states = torch.zeros(size=(self.args['nce_batch_size'],self.args['frame_stack'],104, 80),dtype=torch.float32,device=self.device) # 104 x 80 for ProcGen
        self.next_states = torch.zeros(size=(self.args['nce_batch_size'],self.args['frame_stack'],104, 80),dtype=torch.float32,device=self.device)
        if self.prioritized_replay:
            self.weights = torch.zeros(size=(self.args['nce_batch_size'],),dtype=torch.float32,device=self.device)
        self.nce_counter = 0
        
    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        
        self.A_mat_itr += (itr-self.itr)
        self.itr = itr
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
            
        opt_info = OptInfoNCE(*([] for _ in range(len(OptInfoNCE._fields))))
        if itr < self.min_itr_learn:
            self.agent.A_hat_visible = None
            return opt_info
        for _ in range(self.updates_per_optimize):
            action_counts = np.zeros(shape=15)
            average_repeating_action_len = 0.
            average_N = 0.
            is_procgen = ('procgen' in self.args['env_name'])

            if self.args['n_step_nce'] == -1:
                """
                DRIML-ada
                """
                t = 0
                OneStepSamples = namedtuple("OneStepSamples", ["agent_inputs", "action", "return_","target_inputs","done","done_n","is_weights"])
                ObservedTensor = namedtuple("ObservedTensor",["observation","prev_action"])
                samples = self.replay_buffer.sample_batch(self.batch_size)

                actions = samples.all_action
                

                def loss_fn(p,q):
                    log_2 = math.log(2.)
                    # return -F.log_softmax(p).mean()
                    return (log_2 - F.softplus(- p)).mean() - (F.softplus(-q) + q - log_2).mean()
                
                device = self.agent.device
                B = len(actions[0])
                T = len(actions)
                Y = torch.ones(size=(B,))
                C = torch.zeros(size=(B,))
                
                if is_procgen:
                    action_type = 'hybrid'
                    mat = torch.LongTensor(PROCGEN_ACTION_MAT[self.args['env_name'].split('-')[1]])
                    mapped_actions = (F.one_hot(actions) @ mat).max(2)[1]
                    N_hidden = 15
                    N_visible = (mat.sum(0)>0).sum()
                else:
                    action_type = 'hidden'
                    N_hidden = 18 # ALE
                    N_visible = 18

                if action_type == 'hybrid':
                    N_actions = N_visible * N_hidden
                
                if not hasattr(self, 'A_mat'):
                    self.A_mat = np.random.uniform(size=(N_actions,N_actions))

                for tt in range(1,T):
                    """
                    |Hidden| = 15
                    |Visible| <= 15
                    0 <= |Mixed| <= |Hidden|*|Visible|, hidden_0_visible_0,...,hidden_0_visible_14,hidden_1_visible_0,etc

                    A) Estimate K for each entry
                    """
                    if action_type == 'hybrid':
                        aa = torch.stack([actions[tt-1],mapped_actions[tt-1]],1)
                        bb = torch.stack([actions[tt],mapped_actions[tt]],1)

                        a_t = (N_visible-1)*aa[:,0]+aa[:,1]
                        a_tp1 = (N_visible-1)*bb[:,0]+bb[:,1]

                    P = torch.FloatTensor( np.array([self.A_mat[a_t[i],a_tp1[i] ] for i in range(B)])  )
                    S = torch.bernoulli(P)
                    Y = Y * S
                    C = C + Y

                """
                B) Update A_mat
                """
                
                true_inp = torch.cat([F.one_hot(torch.LongTensor(actions[:-1].view(-1)),N_actions),F.one_hot(torch.LongTensor(actions[1:].view(-1)),N_actions)],1).float().to(device)
                p = self.agent.model.action_net( true_inp )
                
                idx = torch.randperm(B*(T-1))
                perm_inp = torch.cat([F.one_hot(torch.LongTensor(actions[:-1].view(-1)),N_actions),F.one_hot(torch.LongTensor(actions[1:].view(-1)),N_actions)[idx]],1).float().to(device)
                q = self.agent.model.action_net( perm_inp )
                
                loss = -loss_fn(p,q)
                self.action_net_optimizer.zero_grad()
                loss.backward()
                self.action_net_optimizer.step()

                """
                C) Do smoothing on A_mat to prevent fast changes
                """
                
                A_hat = np.zeros(shape=(N_actions,N_actions))
                v1 = F.one_hot(torch.arange(N_actions).repeat_interleave(N_actions),N_actions)
                v2 = F.one_hot(torch.arange(N_actions).repeat(N_actions),N_actions)
                v = torch.cat([v1,v2],dim=1).float().to(device)
                A_hat = torch.softmax(self.agent.model.action_net(v).detach().cpu().view(N_actions,N_actions),dim=1).numpy()

                
                self.A_mat = (0.9) * self.A_mat + (0.1) * A_hat


                optimal_Ns = np.maximum(1,C.detach().cpu().numpy())

                adx = torch.arange(0, len(actions[0])).long()

                average_N = optimal_Ns.mean()

                agent_inputs = ObservedTensor(observation=samples.all_observation[t],prev_action=samples.all_action[t])
                action = samples.all_action[t+1]
                return_ = samples.return_[t]
                target_inputs_rl = ObservedTensor(observation=samples.all_observation[t+1],prev_action=samples.all_action[t+1])
                target_inputs_nce = ObservedTensor(observation=samples.all_observation.transpose(1,0)[adx[None,:],optimal_Ns][0],prev_action=samples.all_action.transpose(1,0)[adx[None,:],optimal_Ns][0])
                done = samples.done[t]
                done_n = samples.done[t]
                is_weights = None

                samples_from_replay_nce = OneStepSamples(agent_inputs=agent_inputs,
                                                         action=action,
                                                         return_=return_,
                                                         target_inputs=target_inputs_nce,
                                                         done=done,
                                                         done_n=done_n,
                                                         is_weights=is_weights)
                samples_from_replay_rl = OneStepSamples(agent_inputs=agent_inputs,
                                                         action=action,
                                                         return_=return_,
                                                         target_inputs=target_inputs_rl,
                                                         done=done,
                                                         done_n=done_n,
                                                         is_weights=is_weights)
                
                a_diff = samples.all_action.transpose(0,1)[:,:-1] - samples.all_action.transpose(0,1)[:,1:]
                idx = torch.where(a_diff==0)[0]
                average_repeating_action_len = np.mean(list(Counter(idx.cpu().numpy()).values()))

                for a in range(N_hidden):
                    action_counts[a] = torch.mean((samples.all_action.transpose(0,1)==a).sum(1).float())
            elif self.args['n_step_nce'] == -2:
                """
                DRIML-randk
                """
                t = 0
                OneStepSamples = namedtuple("OneStepSamples", ["agent_inputs", "action", "return_","target_inputs","done","done_n","is_weights"])
                ObservedTensor = namedtuple("ObservedTensor",["observation","prev_action"])
                samples = self.replay_buffer.sample_batch(self.batch_size)

                actions = samples.all_action

                optimal_Ns = np.random.randint(1,3,size=(actions.shape[1]))  

                adx = torch.arange(0, len(actions[0])).long()

                agent_inputs = ObservedTensor(observation=samples.all_observation[t],prev_action=samples.all_action[t])
                action = samples.all_action[t+1]
                return_ = samples.return_[t]
                target_inputs_rl = ObservedTensor(observation=samples.all_observation[t+1],prev_action=samples.all_action[t+1])
                target_inputs_nce = ObservedTensor(observation=samples.all_observation.transpose(1,0)[adx[None,:],optimal_Ns][0],prev_action=samples.all_action.transpose(1,0)[adx[None,:],optimal_Ns][0])
                done = samples.done[t]
                done_n = samples.done[t]
                is_weights = None

                samples_from_replay_nce = OneStepSamples(agent_inputs=agent_inputs,
                                                         action=action,
                                                         return_=return_,
                                                         target_inputs=target_inputs_nce,
                                                         done=done,
                                                         done_n=done_n,
                                                         is_weights=is_weights)
                samples_from_replay_rl = OneStepSamples(agent_inputs=agent_inputs,
                                                         action=action,
                                                         return_=return_,
                                                         target_inputs=target_inputs_rl,
                                                         done=done,
                                                         done_n=done_n,
                                                         is_weights=is_weights)
            elif self.args['n_step_nce'] > 1:
                """
                DRIML-fix
                """
                t = 0
                t_p_k = t+self.args['n_step_nce']
                OneStepSamples = namedtuple("OneStepSamples", ["agent_inputs", "action", "return_","target_inputs","done","done_n","is_weights"])
                ObservedTensor = namedtuple("ObservedTensor",["observation","prev_action"])
                samples = self.replay_buffer.sample_batch(self.batch_size)

                agent_inputs = ObservedTensor(observation=samples.all_observation[t],prev_action=samples.all_action[t])
                action = samples.all_action[t+1]
                return_ = samples.return_[t]
                target_inputs_rl = ObservedTensor(observation=samples.all_observation[t+1],prev_action=samples.all_action[t+1])
                target_inputs_nce = ObservedTensor(observation=samples.all_observation[t_p_k],prev_action=samples.all_action[t_p_k])
                done = samples.done[t]
                done_n = samples.done[t]
                is_weights = None


                samples_from_replay_nce = OneStepSamples(agent_inputs=agent_inputs,
                                                         action=action,
                                                         return_=return_,
                                                         target_inputs=target_inputs_nce,
                                                         done=done,
                                                         done_n=done_n,
                                                         is_weights=is_weights)
                samples_from_replay_rl = OneStepSamples(agent_inputs=agent_inputs,
                                                         action=action,
                                                         return_=return_,
                                                         target_inputs=target_inputs_rl,
                                                         done=done,
                                                         done_n=done_n,
                                                         is_weights=is_weights)
                
                a_diff = samples.all_action.transpose(0,1)[:,:-1] - samples.all_action.transpose(0,1)[:,1:]
                idx = torch.where(a_diff==0)[0]
                average_repeating_action_len = np.mean(list(Counter(idx.cpu().numpy()).values()))

                for a in range(15):
                    action_counts[a] = torch.mean((samples.all_action.transpose(0,1)==a).sum(1).float())
            else:
                """
                C51
                """
                samples_from_replay_rl = samples_from_replay_nce = self.replay_buffer.sample_batch(self.batch_size)
            
            self.optimizer.zero_grad()
            loss, td_abs_errors, _, _, loss_nce_raw = self.loss(samples_from_replay_rl, itr,samples_from_replay_nce)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            if self.prioritized_replay:# and itr >= self.args['NCE_pretrain_steps']:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm)
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.

            opt_info.lossNCE.append(loss_nce_raw.item())

            opt_info.action0.append(action_counts[0])
            opt_info.action1.append(action_counts[1])
            opt_info.action2.append(action_counts[2])
            opt_info.action3.append(action_counts[3])
            opt_info.action4.append(action_counts[4])
            opt_info.action5.append(action_counts[5])
            opt_info.action6.append(action_counts[6])
            opt_info.action7.append(action_counts[7])
            opt_info.action8.append(action_counts[8])
            opt_info.action9.append(action_counts[9])
            opt_info.action10.append(action_counts[10])
            opt_info.action11.append(action_counts[11])
            opt_info.action12.append(action_counts[12])
            opt_info.action13.append(action_counts[13])
            opt_info.action14.append(action_counts[14])

            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()
        self.update_itr_hyperparams(itr)
        return opt_info

    def loss(self, samples, itr,samples_nce):
        """Samples have leading batch dimension [B,..] (but not time)."""
        self.args['device'] = self.agent.device
        
        """
        Get rlpyt batch inputs and write them to GPU tensors
        """
        rl_agent_inputs = AgentInputs(observation=samples.agent_inputs.observation,prev_action=samples.agent_inputs.prev_action,prev_reward=None)
        rl_action = samples.action
        rl_return_ = samples.return_
        rl_target_inputs = AgentInputs(observation=samples.target_inputs.observation,prev_action=samples.target_inputs.prev_action,prev_reward=None)
        rl_done = samples.done
        rl_done_n = samples.done_n

        self.states[(self.nce_counter*self.args['batch_size']):(self.nce_counter+1)*self.args['batch_size']] = samples_nce.agent_inputs.observation.type(torch.float32).to(self.args['device']) /255.
        self.actions[(self.nce_counter*self.args['batch_size']):(self.nce_counter+1)*self.args['batch_size']] = samples_nce.action.type(torch.int64).to(self.args['device'])
        self.returns[(self.nce_counter*self.args['batch_size']):(self.nce_counter+1)*self.args['batch_size']] = samples_nce.return_.type(torch.float32).to(self.args['device'])
        self.next_states[(self.nce_counter*self.args['batch_size']):(self.nce_counter+1)*self.args['batch_size']] = samples_nce.target_inputs.observation.type(torch.float32).to(self.args['device']) /255.
        self.nonterminals[(self.nce_counter*self.args['batch_size']):(self.nce_counter+1)*self.args['batch_size']] = samples_nce.done
        if self.prioritized_replay:
            rl_is_weights = samples.is_weights
            self.weights[(self.nce_counter*self.args['batch_size']):(self.nce_counter+1)*self.args['batch_size']] = samples_nce.is_weights

        self.nce_counter += 1

        """
        C51 code from rlpyt (unchanged)
        """
        
        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        # Makde 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        next_z = z * (self.discount ** self.n_step_return)  # [P']
        next_z = torch.ger(1 - rl_done_n.float(), next_z)  # [B,P']
        ret = rl_return_.unsqueeze(1)  # [B,1]
        next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)  # [B,P']

        z_bc = z.view(1, -1, 1)  # [1,P,1]
        next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.
        # projection_coeffs is a 3-D tensor: [B,P,P']
        # dim-0: independent data entries
        # dim-1: base_z atoms (remains after projection)
        # dim-2: next_z atoms (summed in projection)
        
        with torch.no_grad():
            target_ps = self.agent.target(*rl_target_inputs)  # [B,A,P']
            if self.double_dqn:
                next_ps = self.agent(*rl_target_inputs)  # [B,A,P']
                next_qs = torch.tensordot(next_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(next_qs, dim=-1)  # [B]
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(target_qs, dim=-1)  # [B]
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
            target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B,P]
        ps = self.agent(*rl_agent_inputs)  # [B,A,P]
        p = select_at_indexes(rl_action, ps)  # [B,P]
        p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * torch.log(p), dim=1)  # Cross-entropy.

        if self.prioritized_replay:
            losses *= rl_is_weights

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - torch.log(p.detach())), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        if not self.mid_batch_reset:
            valid = valid_from_done(rl_done)
            loss = valid_mean(losses, valid)
            KL_div *= valid
        else:
            loss = torch.mean(losses)
        # else:
        #     KL_div = torch.tensor([0.]).cpu()
        #     loss = torch.tensor([0.]).to(self.args['device'])

        """
        NCE loss
        """
        loss_device = loss.get_device()
        if self.args['lambda_LL'] != 0 or self.args['lambda_LG'] != 0 or self.args['lambda_GL'] != 0 or self.args['lambda_GG'] != 0:
            """
            Compute this only if one of the 4 lambdas != 0
            """
            if self.args['nce_batch_size'] // self.args['batch_size'] <= self.nce_counter:
                target = None
                # Select the proper NCE loss passed as argument
                dict_nce = globals()[self.args['nce_loss']](self.agent.model.model,self.states,self.actions,self.returns,self.next_states,self.args,target=target)
                
                nce_scores = self.args['lambda_LL'] * dict_nce['nce_L_L'] + self.args['lambda_LG'] * dict_nce['nce_L_G'] + self.args['lambda_GL'] * dict_nce['nce_G_L'] + self.args['lambda_GG'] * dict_nce['nce_G_G']
                device_ = nce_scores.device
                nce_scores_raw = (dict_nce['nce_L_L'] if self.args['lambda_LL'] > 0 else torch.tensor(0.).to(device_)).mean()
                nce_scores_raw += (dict_nce['nce_L_G'] if self.args['lambda_LG'] > 0 else torch.tensor(0.).to(device_)).mean()
                nce_scores_raw += (dict_nce['nce_G_L'] if self.args['lambda_GL'] > 0 else torch.tensor(0.).to(device_)).mean()
                nce_scores_raw += (dict_nce['nce_G_G'] if self.args['lambda_GG'] > 0 else torch.tensor(0.).to(device_)).mean()
                if self.prioritized_replay:
                    nce_device = nce_scores.get_device()
                    if nce_device < 0:
                        nce_scores *= samples.is_weights
                    else:
                        nce_scores *= samples.is_weights.to(nce_device)
                info_nce_loss_weighted = (-nce_scores).mean() # decay by epsilon
                nce_scores_raw = (-nce_scores_raw).mean()

                if loss_device < 0:
                    info_nce_loss_weighted = info_nce_loss_weighted.to('cpu')
                    nce_scores_raw = nce_scores_raw.to('cpu')

                # self.reset_nce_accumulators(self.agent.device)
                self.nce_counter = 0
            else:
                if loss_device > 0:
                    info_nce_loss_weighted = torch.tensor(0.).to(loss_device)
                    nce_scores_raw = torch.tensor(0.).to(loss_device)
                else:
                    info_nce_loss_weighted = torch.tensor(0.).cpu()
                    nce_scores_raw = torch.tensor(0.).cpu()
        else:
            if self.args['nce_batch_size'] // self.args['batch_size'] <= self.nce_counter:
                # self.reset_nce_accumulators(self.agent.device)
                self.nce_counter = 0
            if loss_device > 0:
                info_nce_loss_weighted = torch.tensor(0.).to(loss_device)
                nce_scores_raw = torch.tensor(0.).to(loss_device)
            else:
                info_nce_loss_weighted = torch.tensor(0.).cpu()
                nce_scores_raw = torch.tensor(0.).cpu()

        return loss + (self.args['nce_batch_size'] // self.batch_size) *  info_nce_loss_weighted, KL_div, loss, info_nce_loss_weighted, nce_scores_raw

