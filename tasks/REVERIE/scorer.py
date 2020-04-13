import torch
import torch.nn as nn
from utilsFast import try_cuda
from speaker import batch_observations_and_actions
import numpy as np

class Scorer:
    def __init__(self):
        self.scorer = scorer
        self.text_encoder = encoder
        self.traj_encoder = None
        self.sm = try_cuda(nn.Softmax(dim=1))
        self.gamma = 0.0 # how much follower_logit to consider

    def init_traj(self, batch_size):
        return self.encoder.init_state(batch_size)

    def encode_traj(self, cand_obs, cand_acs):
        assert len(cand_obs) == len(cand_acs)
        batch_size = len(cand_obs)
        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
            path_lengths, _, _ = \
            batch_observations_and_actions(
                cand_obs, cand_acs, None) # no permutation
        ctx,_,_ = self.encoder(batched_action_embeddings, batched_image_features)
        path_lengths = np.array(path_lengths) - 1
        h_t = ctx[np.arange(batch_size),path_lengths]
        return h_t

    def score(self, instr_enc, traj_enc):
        return self.scorer(instr_enc, traj_enc.unsqueeze(1)).squeeze(1)

    def prepare_proposals(self,batch_h,batch_c,batch_obs,batch_acs):
        ''' for each action proposal, prepare its h,c
        input: existing traj h,c; observation; actions
        output: proposed (h,c) * [batch_size, max_proposal_size]
        '''
        batch_size, ac_size, _ = batch_acs.size()
        hidden_size = self.encoder.hidden_size
        proposal_h = try_cuda(torch.zeros(batch_size,ac_size,hidden_size))
        proposal_c = try_cuda(torch.zeros(batch_size,ac_size,hidden_size))
        for i in range(batch_size):
            h = batch_h[i].expand(ac_size,-1)
            c = batch_c[i].expand(ac_size,-1)
            obs = batch_obs[i].expand(ac_size,-1,-1)
            proposal_h[i], proposal_c[i] = self.encoder._forward_one_step(h,c,batch_acs[i],obs)
        return proposal_h.detach(), proposal_c.detach()

    def combine_logit(self, scorer_logit, follower_logit):
        #import pdb;pdb.set_trace()
        if self.gamma == 0.0:
            return scorer_logit
        if self.gamma == 1.0:
            return follower_logit
        g, h = self.gamma, 1-self.gamma
        prob = h * self.sm(scorer_logit) + g * self.sm(follower_logit)
        return try_cuda(torch.log(prob))

    def modules(self):
        return [self.encoder, self.scorer]

    def modules_path(self, path):
        return (path + '_scorer_enc',
                path + '_scorer_' + self.scorer.__class__.__name__)

    def save(self, path):
        ''' Snapshot models '''
        enc_path, s_path = self.modules_path(path)
        torch.save(self.encoder.state_dict(), enc_path)
        torch.save(self.scorer.state_dict(), s_path)

    def load(self, path):
        ''' Snapshot models '''
        enc_path, s_path = self.modules_path(path)
        self.encoder.load_state_dict(torch.load(enc_path))
        self.scorer.load_state_dict(torch.load(s_path))

    def load_traj_encoder(self, base_path, **kwargs):
        encoder_path = base_path + '_enc'
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))

