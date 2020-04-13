''' Agents: stop/random/shortest/seq2seq  '''

import json
import sys
import numpy as np
import networkx as nx
import random
from collections import namedtuple, defaultdict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utilsFast import vocab_pad_idx, vocab_eos_idx, flatten, structured_map, try_cuda
from utilsFast import PriorityQueue
from running_mean_std import RunningMean

InferenceState = namedtuple("InferenceState", "prev_inference_state, world_state, observation, flat_index, last_action, last_action_embedding, action_count, score, h_t, c_t, last_alpha")
SearchState = namedtuple("SearchState", "flogit,flogp, world_state, observation, action, action_embedding, action_count, h_t,c_t,father") # flat_index,
CandidateState = namedtuple("CandidateState", "flogit,flogp,world_states,actions,pm,speaker,scorer") # flat_index,
Cons = namedtuple("Cons", "first, rest")

def cons_to_list(cons):
    l = []
    while True:
        l.append(cons.first)
        cons = cons.rest
        if cons is None:
            break
    return l

def backchain_inference_states(last_inference_state):
    states = []
    observations = []
    actions = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        states.append(inf_state.world_state)
        observations.append(inf_state.observation)
        actions.append(inf_state.last_action)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(states)), list(reversed(observations)), list(reversed(actions))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:] # exclude start action

def least_common_viewpoint_path(inf_state_a, inf_state_b):
    # return inference states traversing from A to X, then from Y to B,
    # where X and Y are the least common ancestors of A and B respectively that share a viewpointId
    path_to_b_by_viewpoint =  {
    }
    b = inf_state_b
    b_stack = Cons(b, None)
    while b is not None:
        path_to_b_by_viewpoint[b.world_state.viewpointId] = b_stack
        b = b.prev_inference_state
        b_stack = Cons(b, b_stack)
    a = inf_state_a
    path_from_a = [a]
    while a is not None:
        vp = a.world_state.viewpointId
        if vp in path_to_b_by_viewpoint:
            path_to_b = cons_to_list(path_to_b_by_viewpoint[vp])
            assert path_from_a[-1].world_state.viewpointId == path_to_b[0].world_state.viewpointId
            return path_from_a + path_to_b[1:]
        a = a.prev_inference_state
        path_from_a.append(a)
    raise AssertionError("no common ancestor found")

def batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False,
                                    sort=False, tok=None, addEos=True):
    # encoded_instructions: list of lists of token indices (should not be padded, or contain BOS or EOS tokens)
    # make sure pad does not start any sentence
    num_instructions = len(encoded_instructions)
    seq_tensor = np.full((num_instructions, max_length), vocab_pad_idx)
    seq_lengths = []
    inst_mask = []

    for i, inst in enumerate(encoded_instructions):
        if len(inst) > 0:
            assert inst[-1] != vocab_eos_idx
        if reverse:
            inst = inst[::-1]
        if addEos:
            inst = np.concatenate((inst, [vocab_eos_idx]))
        inst = inst[:max_length]
        if tok:
            inst_mask.append(tok.filter_verb(inst,sel_verb=False)[1])
        seq_tensor[i,:len(inst)] = inst
        seq_lengths.append(len(inst))

    seq_tensor = torch.from_numpy(seq_tensor)
    if sort:
        seq_lengths, perm_idx = torch.from_numpy(np.array(seq_lengths)).sort(0, True)
        seq_lengths = list(seq_lengths)
        seq_tensor = seq_tensor[perm_idx]
    else:
        perm_idx = np.arange((num_instructions))

    mask = (seq_tensor == vocab_pad_idx)[:, :max(seq_lengths)]

    if tok:
        for i,idx in enumerate(perm_idx):
            mask[i][inst_mask[idx]] = 1

    ret_tp = try_cuda(Variable(seq_tensor, requires_grad=False).long()), \
             try_cuda(mask), \
             seq_lengths
    if sort:
        ret_tp = ret_tp + (list(perm_idx),)
    return ret_tp

def final_text_enc(encoded_instructions, max_length, encoder):
    seq, seq_mask, seq_lengths = batch_instructions_from_encoded(encoded_instructions, max_length, reverse=False)
    ctx,h_t,c_t = encoder(seq, seq_lengths)
    return h_t, c_t

def stretch_tensor(alphas, lens, target_len):
    ''' Given a batch of sequences of various lengths, stretch to a target_len
        and normalize the sum to 1
    '''
    batch_size, _ = alphas.shape
    r = torch.zeros(batch_size, target_len)
    al = alphas.unsqueeze(1)
    for idx,_len in enumerate(lens):
        r[idx] = F.interpolate(al[idx:idx+1,:,:_len],size=(target_len),mode='linear',align_corners=False)
        r[idx] /= r[idx].sum()
    return r

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

    def write_test_results(self):
        results = []
        for key, item in self.results.items():
            results.append({
                'instr_id': item['instr_id'],
                'trajectory': item['trajectory'],
            })
        with open(self.results_path, 'w') as f:
            json.dump(results, f)

    def write_results(self, results=None, results_path=None):
        if results is None:
            results = {}
            for key, item in self.results.items():
                results[key] = {
                    'instr_id': item['instr_id'],
                    'trajectory': item['trajectory'],
                }
        if results_path is None:
            results_path = self.results_path
        with open(results_path, 'w') as f:
            json.dump(results, f)

    def rollout(self):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def test(self):
        self.phase = 'test'
        self.env.reset_epoch()
        self.losses = []
        self.results = {}
        self.clean_results = {}

        # We rely on env showing the entire batch before repeating anything
        #print 'Testing %s' % self.__class__.__name__
        looped = False
        rollout_scores = []
        beam_10_scores = []
        with torch.no_grad():
            while True:
                rollout_results = self.rollout()
                for result in rollout_results:
                    if result['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[result['instr_id']] = result

                if looped:
                    break
        # if self.feedback == 'argmax':
        #     print("avg rollout score: ", np.mean(rollout_scores))
        #     print("avg beam 10 score: ", np.mean(beam_10_scores))
        return self.results

def path_element_from_observation(ob):
    return (ob['viewpoint'], ob['heading'], ob['elevation'])

def realistic_jumping(graph, start_step, dest_obs):
    if start_step == path_element_from_observation(dest_obs):
        return []
    s = start_step[0]
    t = dest_obs['viewpoint']
    path = nx.shortest_path(graph,s,t)
    traj = [(vp,0,0) for vp in path[:-1]]
    traj.append(path_element_from_observation(dest_obs))
    return traj

class StopAgent(BaseAgent):
    ''' An agent that doesn't move! '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob) ]
        } for ob in obs]
        return traj


class RandomAgent(BaseAgent):
    ''' An agent that picks a random direction then tries to go straight for
        five viewpoint steps and then stops. '''

    def rollout(self):
        world_states = self.env.reset()
        obs = self.env.observe(world_states)
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)]
        } for ob in obs]
        ended = [False] * len(obs)

        self.steps = [0] * len(obs)
        for t in range(9):
            actions = []
            for i, ob in enumerate(obs):
                if self.steps[i] >= 9:
                    actions.append(0)  # do nothing, i.e. end
                    ended[i] = True
                elif self.steps[i] == 0:
                    # a = np.random.randint(len(ob['adj_loc_list']) - 1) + 1# ori
                    a = np.random.randint(len(ob['adj_loc_list']) - 1)
                    actions.append(a)  # choose a random adjacent loc
                    self.steps[i] += 1
                    if a ==0:
                        ended[i] = True
                else:
                    assert len(ob['adj_loc_list']) > 1
                    actions.append(1)  # go forward
                    self.steps[i] += 1
            world_states = self.env.step(world_states, actions, obs)
            obs = self.env.observe(world_states)
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
        return traj

class ShortestAgent(BaseAgent):
    ''' An agent that always takes the shortest path to goal. '''

    def rollout(self):
        world_states = self.env.reset()
        #obs = self.env.observe(world_states)
        all_obs, all_actions = self.env.shortest_paths_to_goals(world_states, 20)
        return [
            {
                'instr_id': obs[0]['instr_id'],
                # end state will appear twice because stop action is a no-op, so exclude it
                'trajectory': [path_element_from_observation(ob) for ob in obs[:-1]]
            }
            for obs in all_obs
        ]

class Seq2SeqAgent(BaseAgent):
    ''' An agent based on an LSTM seq2seq model with attention. '''

    # For now, the agent can't pick which forward move to make - just the one in the middle
    # env_actions = FOLLOWER_ENV_ACTIONS
    # start_index = START_ACTION_INDEX
    # ignore_index = IGNORE_ACTION_INDEX
    # forward_index = FORWARD_ACTION_INDEX
    # end_index = END_ACTION_INDEX

    def __init__(self, env, results_path, encoder, decoder, episode_len=10, beam_size=1, reverse_instruction=True, max_instruction_length=80, attn_only_verb=False):
        super(self.__class__, self).__init__(env, results_path)

        self.encoder = encoder
        self.decoder = decoder
        self.episode_len = episode_len
        self.losses = []
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.pm_criterion = nn.MSELoss()
        self.beam_size = beam_size
        self.reverse_instruction = reverse_instruction
        self.max_instruction_length = max_instruction_length
        self.attn_only_verb = attn_only_verb
        self.scorer = None
        self.goal_button = None
        self.gb = None
        self.speaker = None
        self.soft_align = False
        self.want_loss = False

    def _feature_variables(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        feature_lists = list(zip(*[ob['feature'] for ob in (flatten(obs) if beamed else obs)]))
        assert len(feature_lists) == len(self.env.image_features_list)
        batched = []
        for featurizer, feature_list in zip(self.env.image_features_list, feature_lists):
            batched.append(featurizer.batch_features(feature_list))
        return batched

    def getObjLanFeat(self,obs):
        objFeats = []
        for i, ob in enumerate(obs):
            lanInput = np.zeros((len(ob['adj_loc_list']), 25))
            for j, candi_view in enumerate(ob['adj_loc_list']):
                # print('i=%d,j=%d'%(i,j))
                if j==0:
                    if self.useStopFeat:
                        featKey = ob['instr_id'] + '_%s_%s_%02d' % (ob['scan'], ob['viewpoint'],
                                                                         ob['viewIndex'])
                    else:
                        continue
                else:
                    featKey = ob['instr_id'] + '_%s_%s_%02d' % (ob['scan'], ob['viewpoint'],
                                                                     candi_view['absViewIndex'])
                if featKey in self.ObjEachViewLanFeat:
                    lanInput[j, :] = self.ObjEachViewLanFeat[featKey]
                else:
                    if os.path.exists(self.ObjEachViewLanFeatPath + featKey + '.json'):
                        # print(self.ObjEachViewLanFeatPath + featKey + '.json')
                        with open(self.ObjEachViewLanFeatPath + featKey + '.json', 'r') as f1:
                            data = np.array(json.load(f1))
                            lanInput[j, :] = data
                            self.ObjEachViewLanFeat[featKey] = data
                    else:
                        if j==0:
                            _, data = self.pointer.groundingForView(ob['scan'], ob['viewpoint'],
                                                       ob['Label'], ob['viewIndex'], 'one')
                        else:
                            _, data = self.pointer.groundingForView(ob['scan'], ob['viewpoint'],
                                                       ob['Label'], candi_view['absViewIndex'],'one')
                        if not data is None:
                            self.ObjEachViewLanFeat[featKey] = data
                            lanInput[j, :] = data
                            with open(self.ObjEachViewLanFeatPath + featKey + '.json', 'w') as f1:
                                json.dump(data.tolist(), f1, indent=2)
            # go through lstm
            if self.useStopFeat:
                seq, seq_mask, seq_lengths, perm_indices = \
                    batch_instructions_from_encoded(lanInput, 25, reverse=False, sort=True)

                addSelf = self.objLabelEncoder(seq, seq_lengths)  # check requires_grad
            else:
                seq, seq_mask, seq_lengths, perm_indices = \
                    batch_instructions_from_encoded(lanInput[1:,:], 25, reverse=False, sort=True)

                objctx = self.objLabelEncoder(seq, seq_lengths)  # check requires_grad
                addSelf = torch.cat((torch.zeros(1,512).cuda(), objctx), 0)
            objFeats.append(addSelf)
        return objFeats

    def getObjVisFeat(self,obs):
        objFeats = []
        visDim = self.objVisFeatDim
        for i, ob in enumerate(obs):
            temp = np.zeros((len(ob['adj_loc_list']),visDim),dtype=np.float32)
            for j, candi_view in enumerate(ob['adj_loc_list']):
                if j==0:
                    if self.useStopFeat:
                        featKey = ob['instr_id'] + '_%s_%s_%02d' % (ob['scan'], ob['viewpoint'],
                                                                         ob['viewIndex'])
                    else:
                        continue
                else:
                    featKey = ob['instr_id'] + '_%s_%s_%02d' % (ob['scan'], ob['viewpoint'],
                                                                     candi_view['absViewIndex'])

                if featKey in self.ObjEachViewVisFeat:
                    temp[j, :] = self.ObjEachViewVisFeat[featKey]
                else:
                    try:
                        with open(self.ObjEachViewVisFeatPath + featKey + '.json', 'r') as f1:
                            data = np.array(json.load(f1))
                            temp[j, :] = data
                            self.ObjEachViewVisFeat[featKey] = data
                    except:
                        if os.path.exists(self.ObjEachViewVisFeatPath + featKey + '.json'):
                            print('regenerate '+ self.ObjEachViewVisFeatPath + featKey + '.json')
                            os.remove(self.ObjEachViewVisFeatPath + featKey + '.json')

                        if j==0:
                            data, _ = self.pointer.groundingForView(ob['scan'], ob['viewpoint'],
                                                  ob['Label'], ob['viewIndex'], 'one')
                        else:
                            data, _ = self.pointer.groundingForView(ob['scan'], ob['viewpoint'],
                                                  ob['Label'], candi_view['absViewIndex'],'one')

                        if not data is None:
                            self.ObjEachViewVisFeat[featKey] = data
                            temp[j, :] = data
                            with open(self.ObjEachViewVisFeatPath + featKey + '.json', 'w') as f1:
                                json.dump(data.tolist(), f1, indent=2)

            objFeats.append(torch.from_numpy(temp).cuda())
        return objFeats

    def getObjFeat(self, obs):
        if self.pointer.useObjLabelOrVis == 'label':
            objFeats = self.getObjLanFeat(obs)

        elif self.pointer.useObjLabelOrVis == 'vis':
            objFeats = self.getObjVisFeat(obs)

        elif self.pointer.useObjLabelOrVis == 'both':
            lanFeats = self.getObjLanFeat(obs)
            visFeats = self.getObjVisFeat(obs)

            objFeats = []
            for i in range(len(obs)):
                temp = torch.cat((visFeats[i],lanFeats[i]),1) # check cat cols
                objFeats.append(temp)

        return objFeats

    def _action_variable(self, obs):
        # get the maximum number of actions of all sample in this batch
        max_num_a = -1
        for i, ob in enumerate(obs):
            max_num_a = max(max_num_a, len(ob['adj_loc_list']))

        is_valid = np.zeros((len(obs), max_num_a), np.float32)
        # add objFeat
        if not self.pointer is None:
            objFeats = self.getObjFeat(obs)

        action_embedding_dim = obs[0]['action_embedding'].shape[-1]

        if not self.pointer is None:
            if self.useObjLabelOrVis == 'label':
                objDim = self.objLanFeatDim
            elif self.useObjLabelOrVis == 'vis':
                objDim = self.objVisFeatDim
            elif self.useObjLabelOrVis == 'both':
                objDim = self.objVisFeatDim + self.objLanFeatDim

        action_embeddings = torch.zeros((len(obs), max_num_a, action_embedding_dim+objDim),\
                                        dtype=torch.float32).cuda()

        if not self.pointer is None:
            for i, ob in enumerate(obs):
                adj_loc_list = ob['adj_loc_list']
                num_a = len(adj_loc_list)
                is_valid[i, 0:num_a] = 1.
                action_embeddings[i, :num_a, :] = torch.cat((torch.from_numpy(ob['action_embedding']).cuda(),
                                                              objFeats[i]),dim=1)
        else:
            for i, ob in enumerate(obs):
                adj_loc_list = ob['adj_loc_list']
                num_a = len(adj_loc_list)
                is_valid[i, 0:num_a] = 1.

                action_embeddings[i, :num_a, :] = torch.from_numpy(ob['action_embedding']).cuda() #

        return (
            action_embeddings,
            Variable(torch.from_numpy(is_valid), requires_grad=False).cuda(),
            is_valid)

    def _teacher_action(self, obs, ended):
        ''' Extract teacher actions into variable. '''
        a = torch.LongTensor(len(obs))
        for i,ob in enumerate(obs):
            # Supervised teacher only moves one axis at a time
            a[i] = ob['teacher'] if not ended[i] else -1
        return try_cuda(Variable(a, requires_grad=False))

    def _progress_target(self, obs, ended, monitor_score):
        t = [None] * len(obs)
        num_elem = 0
        for i,ob in enumerate(obs):
            num_elem += int(not ended[i])
            t[i] = ob['progress'][0] if not ended[i] else monitor_score[i].item()
            #t[i] = ob['progress'][0] if not ended[i] else 1.0
        return try_cuda(torch.tensor(t, requires_grad=False)), num_elem

    def _progress_soft_align(self, alpha_t, seq_lengths):
        if not hasattr(self,'dummy'):
            self.dummy = torch.arange(80).float().cuda()
        score = torch.matmul(alpha_t,self.dummy[:alpha_t.size()[-1]])
        score /= torch.tensor(seq_lengths).float().cuda()
        return score

    def _deviation_target(self, obs, ended, computed_score):
        t = [ob['deviation'] if not ended[i] else computed_score[i].item() for i,ob in enumerate(obs)]
        return try_cuda(torch.tensor(t, requires_grad=False).float())

    def _proc_batch(self, obs, beamed=False):
        encoded_instructions = [ob['instr_encoding'] for ob in (flatten(obs) if beamed else obs)]
        tok = self.env.tokenizer if self.attn_only_verb else None
        return batch_instructions_from_encoded(encoded_instructions, self.max_instruction_length, reverse=self.reverse_instruction, tok=tok)

    def rollout(self):
        if hasattr(self,'search'):
            self.records = defaultdict(list)
            return self._rollout_with_search()
        if self.beam_size == 1:
            return self._rollout_with_loss()
        else:
            assert self.beam_size >= 1
            beams, _, _ = self.beam_search(self.beam_size)
            return [beam[0] for beam in beams]

    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions):
        batch_size = len(path_obs)
        assert len(path_actions) == batch_size
        assert len(encoded_instructions) == batch_size
        for path_o, path_a in zip(path_obs, path_actions):
            assert len(path_o) == len(path_a) + 1

        seq, seq_mask, seq_lengths, perm_indices = \
            batch_instructions_from_encoded(
                encoded_instructions, self.max_instruction_length,
                reverse=self.reverse_instruction, sort=True)
        loss = 0

        ctx,h_t,c_t = self.encoder(seq, seq_lengths)
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size)
        sequence_scores = try_cuda(torch.zeros(batch_size))

        traj = [{
            'instr_id': path_o[0]['instr_id'],
            'trajectory': [path_element_from_observation(path_o[0])],
            'actions': [],
            'scores': [],
            'observations': [path_o[0]],
            'instr_encoding': path_o[0]['instr_encoding']
        } for path_o in path_obs]

        obs = None
        for t in range(self.episode_len):
            next_obs = []
            next_target_list = []
            for perm_index, src_index in enumerate(perm_indices):
                path_o = path_obs[src_index]
                path_a = path_actions[src_index]
                if t < len(path_a):
                    next_target_list.append(path_a[t])
                    next_obs.append(path_o[t])
                else:
                    next_target_list.append(-1)
                    next_obs.append(obs[perm_index])

            obs = next_obs

            target = try_cuda(Variable(torch.LongTensor(next_target_list), requires_grad=False))

            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')

            # Supervised training
            loss += self.criterion(logit, target)

            # Determine next model inputs
            a_t = torch.clamp(target, min=0)  # teacher forcing
            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()

            action_scores = -F.cross_entropy(logit, target, ignore_index=-1, reduction='none').data
            sequence_scores += action_scores

            # Save trajectory output
            for perm_index, src_index in enumerate(perm_indices):
                ob = obs[perm_index]
                if not ended[perm_index]:
                    traj[src_index]['trajectory'].append(path_element_from_observation(ob))
                    traj[src_index]['score'] = float(sequence_scores[perm_index])
                    traj[src_index]['scores'].append(action_scores[perm_index])
                    traj[src_index]['actions'].append(a_t.data[perm_index])
                    # traj[src_index]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].item()
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        return traj, loss

    def _rollout_with_loss(self):
        initial_world_states = self.env.reset(sort=True)
        initial_obs = self.env.observe(initial_world_states)
        initial_obs = np.array(initial_obs)
        batch_size = len(initial_obs)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        # TODO consider not feeding this into the decoder, and just using attention

        self._init_loss()
        last_dev = try_cuda(torch.zeros(batch_size).float())
        last_logit = try_cuda(torch.zeros(batch_size).float())
        ce_criterion = self.criterion
        pm_criterion = self.pm_criterion
        total_num_elem = 0

        feedback = self.feedback

        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        # Record starting point
        traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
            'actions': [],
            'scores': [],
            'observations': [ob],
            'instr_encoding': ob['instr_encoding']
        } for ob in initial_obs]

        obs = initial_obs
        world_states = initial_world_states

        # Initial action
        u_t_prev = self.decoder.u_begin.expand(batch_size, -1)  # init action
        ended = np.array([False] * batch_size) # Indices match permuation of the model, not env

        # Do a sequence rollout and calculate the loss
        env_action = [None] * batch_size
        sequence_scores = try_cuda(torch.zeros(batch_size))

        if self.scorer:
            traj_h, traj_c = self.scorer.init_traj(batch_size)


        for t in range(self.episode_len):
            f_t_list = self._feature_variables(obs) # Image features from obs
            all_u_t, is_valid, _ = self._action_variable(obs) # add obj feature

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'

            # follower logit
            prev_h_t = h_t
            h_t, c_t, t_ground, v_ground, alpha_t, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx, seq_mask)

            if self.phase == 'train':
                if self.soft_align:
                    progress_score = self._progress_soft_align(alpha_t, seq_lengths)
                    target_score, num_elem = self._progress_target(obs, ended, progress_score)
                    self.pm_loss += pm_criterion(progress_score, target_score)
                    if torch.isnan(self.pm_loss):
                        import pdb;pdb.set_trace()
                if self.prog_monitor:
                    monitor_score = self.prog_monitor(prev_h_t,c_t,v_ground,alpha_t)
                    target_score, num_elem = self._progress_target(obs, ended, monitor_score)
                    self.pm_loss += pm_criterion(monitor_score, target_score)

                if self.dev_monitor:
                    dv_criterion = nn.MSELoss()
                    dev_score = self.dev_monitor(last_dev,prev_h_t,h_t,c_t,v_ground,t_ground, alpha_v, alpha_t, last_logit)
                    target_score = self._deviation_target(obs, ended, dev_score)
                    self.dv_loss += dv_criterion(dev_score, target_score)
                    last_dev = dev_score

            # scorer logit
            if self.scorer:
                # encode traj
                proposal_h, proposal_c = self.scorer.prepare_proposals(
                        traj_h, traj_c, f_t_list[0], all_u_t)

                # feed to scorer
                scorer_logit = self.scorer.scorer(t_ground, proposal_h)

                # combine logit
                logit = self.scorer.combine_logit(scorer_logit, logit)

            # Mask outputs of invalid actions
            _logit = logit.detach()
            _logit[is_valid == 0] = -float('inf')

            # Supervised training
            target = self._teacher_action(obs, ended)
            self.ce_loss += ce_criterion(logit, target)
            total_num_elem += np.size(ended) - np.count_nonzero(ended)

            # Determine next model inputs
            if feedback == 'teacher':
                # turn -1 (ignore) to 0 (stop) so that the action is executable
                a_t = torch.clamp(target, min=0)
            elif feedback == 'sample':
                m = D.Categorical(logits=_logit)
                a_t = m.sample()
            elif feedback == 'argmax':
                _,a_t = _logit.max(1)        # student forcing - argmax
                a_t = a_t.detach()
            elif feedback == 'recover':
                m = D.Categorical(logits=_logit)
                a_t = m.sample()
                for i,ob in enumerate(obs):
                    if ob['deviation'] > 0:
                        a_t[i] = -1
            else:
                if 'sample' in feedback:
                    m = D.Categorical(logits=_logit)
                    a_t = m.sample()
                elif 'argmax' in feedback:
                    _,a_t = _logit.max(1)
                else:
                    import pdb;pdb.set_trace()
                deviation = int(''.join([n for n in feedback if n.isdigit()]))
                for i,ob in enumerate(obs):
                    if ob['deviation'] >= deviation:
                        a_t[i] = target[i]
                a_t = torch.clamp(a_t, min=0)

            # check actions
            for i in range(batch_size):
                if a_t[i].item() >= len(obs[i]['adj_loc_list']):
                    while True:
                        nm = D.Categorical(logits=_logit[i,:])
                        a_t[i] = nm.sample()
                        if a_t[i].item()<len(obs[i]['adj_loc_list']):
                            break

            # update the previous action
            u_t_prev = all_u_t[np.arange(batch_size), a_t, :].detach()
            last_logit = _logit[np.arange(batch_size), a_t]

            # update the traj
            if self.scorer:
                traj_h = proposal_h[np.arange(batch_size), a_t, :]
                traj_c = proposal_c[np.arange(batch_size), a_t, :]

            action_scores = -F.cross_entropy(_logit, a_t, ignore_index=-1, reduce=False).data
            sequence_scores += action_scores

            # Make environment action
            for i in range(batch_size):
                action_idx = a_t[i].item()
                env_action[i] = action_idx

            # update
            world_states = self.env.step(world_states, env_action, obs)
            obs = self.env.observe(world_states)
            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.item(), sequence_scores[0]))

            # Save trajectory output
            for i,ob in enumerate(obs):
                if not ended[i]:
                    traj[i]['trajectory'].append(path_element_from_observation(ob))
                    traj[i]['score'] = sequence_scores[i]
                    traj[i]['scores'].append(action_scores[i])
                    traj[i]['actions'].append(a_t.data[i])
                    traj[i]['observations'].append(ob)

            # Update ended list
            for i in range(batch_size):
                action_idx = a_t[i].item()
                if action_idx == 0:
                    ended[i] = True

            # Early exit if all ended
            if ended.all():
                break

        # per step loss
        if self.phase == 'train':
            if self.prog_monitor or self.soft_align:
                self.loss = 0.5*self.ce_loss + 0.5*self.pm_loss
                self.pm_losses.append(self.pm_loss.item())
            else:
                self.loss = self.ce_loss
            if self.dev_monitor:
                self.loss += 0.1 * self.dv_loss
                self.dv_losses.append(self.dv_loss.item())
            self.losses.append(self.loss.item())
            self.ce_losses.append(self.ce_loss.item())
        return traj

    def _search_collect(self, batch_queue, wss, current_idx, ended):
        cand_wss = []
        cand_acs = []
        for idx,_q in enumerate(batch_queue):
            _wss = [wss[idx]]
            _acs = [0]
            _step = current_idx[idx]
            while not ended[idx] and _step > 0:
                _wss.append(_q.queue[_step].world_state)
                _acs.append(_q.queue[_step].action)
                _step = _q.queue[_step].father
            cand_wss.append(list(reversed(_wss)))
            cand_acs.append(list(reversed(_acs)))
        return cand_wss, cand_acs

    def _wss_to_obs(self, cand_wss, instr_ids):
        cand_obs = []
        for _wss,_instr_id in zip(cand_wss, instr_ids):
            ac_len = len(_wss)
            cand_obs.append(self.env.observe(_wss, instr_id=_instr_id))# how to observe when there are many ws
        return cand_obs

    def _rollout_with_search(self):
        if self.env.notTest:
            self._init_loss()
            ce_criterion = self.criterion
            pm_criterion = self.pm_criterion
            bt_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        world_states = self.env.reset(sort=True)
        obs = self.env.observe(world_states)# get panoramic feature and adjacent vps and corresponding action
        batch_size = len(obs)

        seq, seq_mask, seq_lengths = self._proc_batch(obs)
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        traj = [{
            'instr_id': ob['instr_id'],
            'instr_encoding': ob['instr_encoding'],
            'trajectory': [path_element_from_observation(ob)],
        } for ob in obs]

        clean_traj = [{
            'instr_id': ob['instr_id'],
            'trajectory': [path_element_from_observation(ob)],
        } for ob in obs]

        batch_queue = [PriorityQueue() for _ in range(batch_size)]
        ending_queue = [PriorityQueue() for _ in range(batch_size)]

        visit_graphs = [nx.Graph() for _ in range(batch_size)]
        for ob, g in zip(obs, visit_graphs): g.add_node(ob['viewpoint'])

        ended = np.array([False] * batch_size)

        for i, (ws, o) in enumerate(zip(world_states, obs)):
            batch_queue[i].push(
                SearchState(
                    flogit=RunningMean(),
                    flogp=RunningMean(),
                    world_state=ws,
                    observation=o,
                    action=0,
                    action_embedding=self.decoder.u_begin.view(-1).detach(),
                    action_count=0,
                    h_t=h_t[i].detach(),c_t=c_t[i].detach(),
                    father=-1),
                0)

        for t in range(self.episode_len):

            current_idx, priority, current_batch = \
                    zip(*[_q.pop() for _q in batch_queue])
            (last_logit,last_logp,last_world_states,last_obs,acs,acs_embedding,
                    ac_counts,prev_h_t,prev_c_t,prev_father) = zip(*current_batch)

            if t > 0:
                for i,ob in enumerate(last_obs):
                    if not ended[i]:
                        last_vp = traj[i]['trajectory'][-1]
                        traj[i]['trajectory'] += realistic_jumping(
                            visit_graphs[i], last_vp, ob)#visited vp will not be added to traj

                world_states = self.env.step(last_world_states,acs,last_obs)#make action to new vp
                obs = self.env.observe(world_states)#the 1st action in ob's adj_list is stop
                for i in range(batch_size):
                    if (not ended[i] and
                        not visit_graphs[i].has_edge(last_obs[i]['viewpoint'], obs[i]['viewpoint'])):
                        traj[i]['trajectory'].append(path_element_from_observation(obs[i]))
                        visit_graphs[i].add_edge(last_obs[i]['viewpoint'],
                                             obs[i]['viewpoint'])
                for idx, ac in enumerate(acs):
                    if ac == 0:
                        ended[idx] = True
                        batch_queue[idx].lock()

            if ended.all(): break

            u_t_prev = torch.stack(acs_embedding, dim=0)
            prev_h_t = torch.stack(prev_h_t,dim=0)
            prev_c_t = torch.stack(prev_c_t,dim=0)

            f_t_list = self._feature_variables(obs)
            all_u_t, is_valid, _ = self._action_variable(obs) # t=0, how

            # 1. local scorer
            h_t, c_t, t_ground, v_ground, alpha_t, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], prev_h_t, prev_c_t, ctx, seq_mask)

            # 2. prog monitor
            progress_score = [0] * batch_size
            if self.soft_align:
                progress_score = self._progress_soft_align(alpha_t, seq_lengths).tolist()
            if self.prog_monitor:
                progress_score = self.prog_monitor(prev_h_t,c_t,v_ground,alpha_t).tolist()

            # Mask outputs of invalid actions
            _logit = logit.detach()
            _logit[is_valid == 0] = -float('inf')

            # Expand nodes
            is_valid_cpu = is_valid.cpu()
            ac_lens = np.argmax(is_valid_cpu == 0, axis=1).detach()
            ac_lens[ac_lens == 0] = is_valid_cpu.shape[1]

            h_t_data, c_t_data = h_t.detach(),c_t.detach()
            u_t_data = all_u_t.detach()
            log_prob = F.log_softmax(_logit, dim=1).detach()

            # 4. prepare ending evaluation
            cand_instr= [_traj['instr_encoding'] for _traj in traj]
            cand_wss, cand_acs = self._search_collect(batch_queue, world_states, current_idx, ended)
            instr_ids = [_traj['instr_id'] for _traj in traj]
            cand_obs = self._wss_to_obs(cand_wss, instr_ids)

            speaker_scores = [0] * batch_size
            if self.speaker is not None:
                speaker_scored_cand, _ = \
                    self.speaker._score_obs_actions_and_instructions(
                        cand_obs,cand_acs,cand_instr,feedback='teacher')
                speaker_scores = [_s['score'] for _s in speaker_scored_cand]

            goal_scores = [0] * batch_size

            if self.gb:
                text_enc = [ob['instr_encoding'][-40:] for ob in obs]
                text_len = [len(_enc) for _enc in text_enc]
                text_tensor = np.zeros((batch_size, max(text_len)))
                for _idx, _enc in enumerate(text_enc):
                    text_tensor[_idx][:len(_enc)] = _enc
                text_tensor = torch.from_numpy(text_tensor).long().cuda()
                stop_button = self.gb(text_tensor, text_len, f_t_list[0])

            for idx in range(batch_size):
                if ended[idx]: continue

                _len = ac_lens[idx]
                new_logit = last_logit[idx].fork(_logit[idx][:_len].cpu().tolist())
                new_logp = last_logp[idx].fork(log_prob[idx][:_len].cpu().tolist())

                # entropy
                entropy = torch.sum(-log_prob[idx][:_len] * torch.exp(log_prob[idx][:_len]))

                # record
                if self.env.notTest:
                    _dev = obs[idx]['deviation']
                    self.records[_dev].append(
                        (ac_counts[idx],
                         obs[idx]['progress'],
                         _logit[idx][:_len].cpu().tolist(),
                         log_prob[idx][:_len].cpu().tolist(),
                         entropy.item(),
                         speaker_scores[idx]))

                # selectively expand nodes
                K = self.K
                select_k = _len if _len < K else K
                top_ac = list(torch.topk(_logit[idx][:_len],select_k)[1])
                if self.inject_stop and 0 not in top_ac:
                    top_ac.append(0)

                # compute heuristics
                new_heur = new_logit if self.search_logit else new_logp

                if self.search_mean:
                    _new_heur = [_h.mean for _h in new_heur]
                else:
                    _new_heur = [_h.sum for _h in new_heur]

                visitedVps = [ws[1] for ws in cand_wss[idx]]
                for ac_idx, ac in enumerate(top_ac):
                    nextViewpointId = obs[idx]['adj_loc_list'][ac]['nextViewpointId']

                    if not self.revisit and (ac > 0 and nextViewpointId in visitedVps):
                        # avoid re-visiting
                        continue

                    if not self.beam and ac_idx == 0:
                        _new_heur[ac] = float('inf')

                    if ac == 0:
                        if not self.gb or (ac_idx == 0 and stop_button[idx][1] > stop_button[idx][0]):
                            # Don't stop unless the stop button says so
                            ending_heur = _new_heur[ac]
                            new_ending = CandidateState(
                                    flogit=new_logit[ac],
                                    flogp=new_logp[ac],
                                    world_states=cand_wss[idx],
                                    actions=cand_acs[idx],
                                    pm=progress_score[idx],
                                    speaker=speaker_scores[idx],
                                    scorer=_logit[idx][ac],
                                    )

                            ending_queue[idx].push(new_ending, ending_heur)

                    if ac > 0 or self.search_early_stop:

                        new_node = SearchState(
                            flogit=new_logit[ac],
                            flogp=new_logp[ac],
                            world_state=world_states[idx],
                            observation=obs[idx],
                            action=ac,
                            action_embedding=u_t_data[idx,ac],
                            action_count=ac_counts[idx]+1,
                            h_t=h_t_data[idx],c_t=c_t_data[idx],
                            father=current_idx[idx])
                        batch_queue[idx].push(new_node, _new_heur[ac])

                if batch_queue[idx].size() == 0:
                    batch_queue[idx].lock()
                    ended[idx] = True

        # cache the candidates
        if hasattr(self, 'cache_candidates'):
            for idx in range(batch_size):
                instr_id = traj[idx]['instr_id']
                if instr_id not in self.cache_candidates:
                    cand = []
                    for item in ending_queue[idx].queue:
                        cand.append((instr_id, item.world_states, item.actions, item.flogit.sum, item.flogit.mean, item.flogp.sum, item.flogp.mean, item.pm, item.speaker, item.scorer))
                    self.cache_candidates[instr_id] = cand

        # cache the search progress
        if hasattr(self, 'cache_search'):
            for idx in range(batch_size):
                instr_id = traj[idx]['instr_id']
                if instr_id not in self.cache_search:
                    cand = []
                    for item in batch_queue[idx].queue:
                        cand.append((item.world_state, item.action, item.father, item.flogit.sum, item.flogp.sum))
                    self.cache_search[instr_id] = cand

        # actually move the cursor
        for idx in range(batch_size):
            instr_id = traj[idx]['instr_id']
            if ending_queue[idx].size() == 0:
                #print("Warning: some instr does not have ending, ",
                #        "this can be a desired behavior though")
                self.clean_results[instr_id] = {
                        'instr_id': traj[idx]['instr_id'],
                        'trajectory': traj[idx]['trajectory'],
                        }
                continue

            last_vp = traj[idx]['trajectory'][-1]
            if hasattr(self, 'reranker') and ending_queue[idx].size() > 1:
                inputs = []
                inputs_idx = []
                num_candidates = 100
                while num_candidates > 0 and ending_queue[idx].size() > 0:
                    _idx, _pri, item = ending_queue[idx].pop()
                    inputs_idx.append(_idx)
                    inputs.append([len(item.world_states), item.flogit.sum, item.flogit.mean, item.flogp.sum, item.flogp.mean, item.pm, item.speaker] * 4)
                    num_candidates -= 1
                inputs = try_cuda(torch.Tensor(inputs))
                reranker_scores = self.reranker(inputs)
                sel_cand = inputs_idx[torch.argmax(reranker_scores)]
                cur = ending_queue[idx].queue[sel_cand]
            else:
                cur = ending_queue[idx].peak()[-1]
            # keep switching if cur is not the shortest path?

            ob = self.env.observe([cur.world_states[-1]], instr_id=instr_id)
            traj[idx]['trajectory'] += realistic_jumping(
                visit_graphs[idx], last_vp, ob[0])
            ended[idx] = 1

            for _ws in cur.world_states: # we don't collect ws0, this is fine.
                clean_traj[idx]['trajectory'].append((_ws.viewpointId, _ws.heading, _ws.elevation))
                self.clean_results[instr_id] = clean_traj[idx]

        return traj

    def _init_loss(self):
        self.loss = 0
        self.ce_loss = 0
        self.pm_loss = 0
        self.bt_loss = 0
        self.dv_loss = 0

    def beam_search(self, beam_size, load_next_minibatch=True, mask_undo=False):
        assert self.env.beam_size >= beam_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            world_state=ws[0],
                            observation=o[0],
                            flat_index=i,
                            last_action=-1,
                            last_action_embedding=self.decoder.u_begin.view(-1),
                            action_count=0,
                            score=0.0, h_t=None, c_t=None, last_alpha=None)]
            for i, (ws, o) in enumerate(zip(world_states, obs))
        ]

        # Do a sequence rollout and calculate the loss
        for t in range(self.episode_len):
            flat_indices = []
            beam_indices = []
            u_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    u_t_list.append(inf_state.last_action_embedding)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            flat_obs = flatten(obs)
            f_t_list = self._feature_variables(flat_obs) # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t[flat_indices], c_t[flat_indices], ctx[beam_indices], seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #action_scores, action_indices = log_probs.topk(min(beam_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            new_beams = []
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states, beam_obs) in enumerate(zip(beams, world_states, obs)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                assert len(beam_obs) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, ob, action_score_row, action_index_row) in \
                            enumerate(zip(beam, beam_world_states, beam_obs, action_scores[start_index:end_index], action_indices[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_score, action_index in zip(action_score_row, action_index_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state, # will be updated later after successors are pruned
                                               observation=ob, # will be updated later after successors are pruned
                                               flat_index=flat_index,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=float(inf_state.score + action_score), h_t=None, c_t=None,
                                               last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs, beamed=True)
            successor_obs = self.env.observe(successor_world_states, beamed=True)

            all_successors = structured_map(lambda inf_state, world_state, obs: inf_state._replace(world_state=world_state, observation=obs),
                                   all_successors, successor_world_states, successor_obs, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_action == 0 or t == self.episode_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]

            obs = [
                [inf_state.observation for inf_state in beam]
                for beam in beams
            ]

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

        trajs = []

        for this_completed in completed:
            assert this_completed
            this_trajs = []
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        traversed_lists = None # todo
        return trajs, completed, traversed_lists

    def state_factored_search(self, completion_size, successor_size, load_next_minibatch=True, mask_undo=False, first_n_ws_key=4):
        assert self.env.beam_size >= successor_size
        world_states = self.env.reset(sort=True, beamed=True, load_next_minibatch=load_next_minibatch)
        initial_obs = self.env.observe(world_states, beamed=True)
        batch_size = len(world_states)

        # get mask and lengths
        seq, seq_mask, seq_lengths = self._proc_batch(initial_obs, beamed=True)

        # Forward through encoder, giving initial hidden state and memory cell for decoder
        ctx,h_t,c_t = self.encoder(seq, seq_lengths)

        completed = []
        completed_holding = []
        for _ in range(batch_size):
            completed.append({})
            completed_holding.append({})

        state_cache = [
            {ws[0][0:first_n_ws_key]: (InferenceState(prev_inference_state=None,
                                                      world_state=ws[0],
                                                      observation=o[0],
                                                      flat_index=None,
                                                      last_action=-1,
                                                      last_action_embedding=self.decoder.u_begin.view(-1),
                                                      action_count=0,
                                                      score=0.0, h_t=h_t[i], c_t=c_t[i], last_alpha=None), True)}
            for i, (ws, o) in enumerate(zip(world_states, initial_obs))
        ]

        beams = [[inf_state for world_state, (inf_state, expanded) in sorted(instance_cache.items())]
                 for instance_cache in state_cache] # sorting is a noop here since each instance_cache should only contain one


        # traversed_lists = None
        # list of inference states containing states in order of the states being expanded
        last_expanded_list = []
        traversed_lists = []
        for beam in beams:
            assert len(beam) == 1
            first_state = beam[0]
            last_expanded_list.append(first_state)
            traversed_lists.append([first_state])

        def update_traversed_lists(new_visited_inf_states):
            assert len(new_visited_inf_states) == len(last_expanded_list)
            assert len(new_visited_inf_states) == len(traversed_lists)

            for instance_index, instance_states in enumerate(new_visited_inf_states):
                last_expanded = last_expanded_list[instance_index]
                # todo: if this passes, shouldn't need traversed_lists
                assert last_expanded.world_state.viewpointId == traversed_lists[instance_index][-1].world_state.viewpointId
                for inf_state in instance_states:
                    path_from_last_to_next = least_common_viewpoint_path(last_expanded, inf_state)
                    # path_from_last should include last_expanded's world state as the first element, so check and drop that
                    assert path_from_last_to_next[0].world_state.viewpointId == last_expanded.world_state.viewpointId
                    assert path_from_last_to_next[-1].world_state.viewpointId == inf_state.world_state.viewpointId
                    traversed_lists[instance_index].extend(path_from_last_to_next[1:])
                    last_expanded = inf_state
                last_expanded_list[instance_index] = last_expanded

        # Do a sequence rollout and calculate the loss
        while any(len(comp) < completion_size for comp in completed):
            beam_indices = []
            u_t_list = []
            h_t_list = []
            c_t_list = []
            flat_obs = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    u_t_list.append(inf_state.last_action_embedding)
                    h_t_list.append(inf_state.h_t.unsqueeze(0))
                    c_t_list.append(inf_state.c_t.unsqueeze(0))
                    flat_obs.append(inf_state.observation)

            u_t_prev = torch.stack(u_t_list, dim=0)
            assert len(u_t_prev.shape) == 2
            f_t_list = self._feature_variables(flat_obs) # Image features from obs
            all_u_t, is_valid, is_valid_numpy = self._action_variable(flat_obs)
            h_t = torch.cat(h_t_list, dim=0)
            c_t = torch.cat(c_t_list, dim=0)

            assert len(f_t_list) == 1, 'for now, only work with MeanPooled feature'
            h_t, c_t, alpha, logit, alpha_v = self.decoder(
                u_t_prev, all_u_t, f_t_list[0], h_t, c_t, ctx[beam_indices], seq_mask[beam_indices])

            # Mask outputs of invalid actions
            logit[is_valid == 0] = -float('inf')
            # # Mask outputs where agent can't move forward
            # no_forward_mask = [len(ob['navigableLocations']) <= 1 for ob in flat_obs]

            if mask_undo:
                masked_logit = logit.clone()
            else:
                masked_logit = logit

            log_probs = F.log_softmax(logit, dim=1).data

            # force ending if we've reached the max time steps
            # if t == self.episode_len - 1:
            #     action_scores = log_probs[:,self.end_index].unsqueeze(-1)
            #     action_indices = torch.from_numpy(np.full((log_probs.size()[0], 1), self.end_index))
            # else:
            #_, action_indices = masked_logit.data.topk(min(successor_size, logit.size()[1]), dim=1)
            _, action_indices = masked_logit.data.topk(logit.size()[1], dim=1) # todo: fix this
            action_scores = log_probs.gather(1, action_indices)
            assert action_scores.size() == action_indices.size()

            start_index = 0
            assert len(beams) == len(world_states)
            all_successors = []
            for beam_index, (beam, beam_world_states) in enumerate(zip(beams, world_states)):
                successors = []
                end_index = start_index + len(beam)
                assert len(beam_world_states) == len(beam)
                if beam:
                    for inf_index, (inf_state, world_state, action_score_row) in \
                            enumerate(zip(beam, beam_world_states, log_probs[start_index:end_index])):
                        flat_index = start_index + inf_index
                        for action_index, action_score in enumerate(action_score_row):
                            if is_valid_numpy[flat_index, action_index] == 0:
                                continue
                            successors.append(
                                InferenceState(prev_inference_state=inf_state,
                                               world_state=world_state, # will be updated later after successors are pruned
                                               observation=flat_obs[flat_index], # will be updated later after successors are pruned
                                               flat_index=None,
                                               last_action=action_index,
                                               last_action_embedding=all_u_t[flat_index, action_index].detach(),
                                               action_count=inf_state.action_count + 1,
                                               score=inf_state.score + action_score,
                                               h_t=h_t[flat_index], c_t=c_t[flat_index],
                                               last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)
                all_successors.append(successors)

            successor_world_states = [
                [inf_state.world_state for inf_state in successors]
                for successors in all_successors
            ]

            successor_env_actions = [
                [inf_state.last_action for inf_state in successors]
                for successors in all_successors
            ]

            successor_last_obs = [
                [inf_state.observation for inf_state in successors]
                for successors in all_successors
            ]

            successor_world_states = self.env.step(successor_world_states, successor_env_actions, successor_last_obs, beamed=True)

            all_successors = structured_map(lambda inf_state, world_state: inf_state._replace(world_state=world_state),
                                            all_successors, successor_world_states, nested=True)

            # if all_successors[0]:
            #     print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, all_successors[0][0].world_state, all_successors[0][0].last_action, all_successors[0][0].score))

            assert len(all_successors) == len(state_cache)

            new_beams = []

            for beam_index, (successors, instance_cache) in enumerate(zip(all_successors, state_cache)):
                # early stop if we've already built a sizable completion list
                instance_completed = completed[beam_index]
                instance_completed_holding = completed_holding[beam_index]
                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                    continue
                for successor in successors:
                    ws_keys = successor.world_state[0:first_n_ws_key]
                    if successor.last_action == 0 or successor.action_count == self.episode_len:
                        if ws_keys not in instance_completed_holding or instance_completed_holding[ws_keys][0].score < successor.score:
                            instance_completed_holding[ws_keys] = (successor, False)
                    else:
                        if ws_keys not in instance_cache or instance_cache[ws_keys][0].score < successor.score:
                            instance_cache[ws_keys] = (successor, False)

                # third value: did this come from completed_holding?
                uncompleted_to_consider = ((ws_keys, inf_state, False) for (ws_keys, (inf_state, expanded)) in instance_cache.items() if not expanded)
                completed_to_consider = ((ws_keys, inf_state, True) for (ws_keys, (inf_state, expanded)) in instance_completed_holding.items() if not expanded)
                import itertools
                import heapq
                to_consider = itertools.chain(uncompleted_to_consider, completed_to_consider)
                ws_keys_and_inf_states = heapq.nlargest(successor_size, to_consider, key=lambda pair: pair[1].score)

                new_beam = []
                for ws_keys, inf_state, is_completed in ws_keys_and_inf_states:
                    if is_completed:
                        assert instance_completed_holding[ws_keys] == (inf_state, False)
                        instance_completed_holding[ws_keys] = (inf_state, True)
                        if ws_keys not in instance_completed or instance_completed[ws_keys].score < inf_state.score:
                            instance_completed[ws_keys] = inf_state
                    else:
                        instance_cache[ws_keys] = (inf_state, True)
                        new_beam.append(inf_state)

                if len(instance_completed) >= completion_size:
                    new_beams.append([])
                else:
                    new_beams.append(new_beam)

            beams = new_beams

            # Early exit if all ended
            if not any(beam for beam in beams):
                break

            world_states = [
                [inf_state.world_state for inf_state in beam]
                for beam in beams
            ]
            successor_obs = self.env.observe(world_states, beamed=True)
            beams = structured_map(lambda inf_state, obs: inf_state._replace(observation=obs),
                                   beams, successor_obs, nested=True)
            update_traversed_lists(beams)

        completed_list = []
        for this_completed in completed:
            completed_list.append(sorted(this_completed.values(), key=lambda t: t.score, reverse=True)[:completion_size])
        completed_ws = [
            [inf_state.world_state for inf_state in comp_l]
            for comp_l in completed_list
        ]
        completed_obs = self.env.observe(completed_ws, beamed=True)
        completed_list = structured_map(lambda inf_state, obs: inf_state._replace(observation=obs),
                                        completed_list, completed_obs, nested=True)
        # TODO: consider moving observations and this update earlier so that we don't have to traverse as far back
        update_traversed_lists(completed_list)

        # TODO: sanity check the traversed lists here

        trajs = []
        for this_completed in completed_list:
            assert this_completed
            this_trajs = []
            for inf_state in this_completed:
                path_states, path_observations, path_actions, path_scores, path_attentions = backchain_inference_states(inf_state)
                # this will have messed-up headings for (at least some) starting locations because of
                # discretization, so read from the observations instead
                ## path = [(obs.viewpointId, state.heading, state.elevation)
                ##         for state in path_states]
                trajectory = [path_element_from_observation(ob) for ob in path_observations]
                this_trajs.append({
                    'instr_id': path_observations[0]['instr_id'],
                    'instr_encoding': path_observations[0]['instr_encoding'],
                    'trajectory': trajectory,
                    'observations': path_observations,
                    'actions': path_actions,
                    'score': inf_state.score,
                    'scores': path_scores,
                    'attentions': path_attentions
                })
            trajs.append(this_trajs)
        # completed_list: list of lists of final inference states corresponding to the candidates, one list per instance
        # traversed_lists: list of "physical states" that the robot has explored, one per instance
        return trajs, completed_list, traversed_lists

    def set_beam_size(self, beam_size):
        if self.env.beam_size < beam_size:
            self.env.set_beam_size(beam_size)
        self.beam_size = beam_size

    def cache_search_candidates(self):
        for m in self.modules():
            m.eval()
        self.cache_candidates = {}
        self.cache_search = {}
        super(self.__class__, self).test()


    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
            if not self.objLabelEncoder is None:
                self.objLabelEncoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            if not self.objLabelEncoder is None:
                self.objLabelEncoder.eval()

        self.set_beam_size(beam_size)
        return super(self.__class__, self).test()

    def train(self, optimizers, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        self.phase = 'train'
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        if not self.objLabelEncoder is None:
            self.objLabelEncoder.train()

        self.losses = []
        self.ce_losses = []
        self.dv_losses = []
        self.pm_losses = []
        self.bt_losses = []
        it = range(1, n_iters + 1)
        try:
            import tqdm
            it = tqdm.tqdm(it)
        except:
            pass
        self.attn_t = np.zeros((n_iters,self.episode_len,self.env.batch_size,80))
        for self.it_idx in it:
            for opt in optimizers:
                opt.zero_grad()
            self.rollout()
            #self._rollout_with_loss()
            if type(self.loss) is torch.Tensor:
                self.loss.backward()
            for opt in optimizers:
                opt.step()

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_enc", base_path + "_dec"

    def save(self, path):
        ''' Snapshot models '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)
        if self.scorer:
            self.scorer.save(path)
        if self.prog_monitor:
            torch.save(self.prog_monitor.state_dict(), path + "_pm")
        if self.dev_monitor:
            torch.save(self.dev_monitor.state_dict(), path+ "_dv")
        if self.bt_button:
            torch.save(self.bt_button.state_dict(), path+ "_bt")
        if self.objLabelEncoder:
            torch.save(self.objLabelEncoder.state_dict(), path+'_objLabelEncoder')

    def load(self, path, load_scorer=False, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))
        if self.prog_monitor:
            self.prog_monitor.load_state_dict(torch.load(path+"_pm", **kwargs))
        if load_scorer and self.scorer:
            self.scorer.load(path)
        if self.dev_monitor:
            self.dev_monitor.load_state_dict(torch.load(path+"_dv", **kwargs))
        if self.bt_button:
            if os.path.isfile(path + '_bt'):
                self.bt_button.load_state_dict(torch.load(path+"_bt", **kwargs))
        if self.objLabelEncoder:
            self.objLabelEncoder.load_state_dict(torch.load(path+"_objLabelEncoder", **kwargs))

    def modules(self):
        _m = [self.encoder, self.decoder]
        if self.prog_monitor: _m.append(self.prog_monitor)
        if self.dev_monitor: _m.append(self.dev_monitor)
        if self.bt_button: _m.append(self.bt_button)
        if self.scorer: _m += self.scorer.modules()
        return _m

    def modules_paths(self, base_path):
        _mp = list(self._encoder_and_decoder_paths(base_path))
        if self.prog_monitor: _mp.append(base_path + "_pm")
        if self.scorer: _mp += self.scorer.modules_path(base_path)
        if self.bt_button: _mp += _mp.append(base_path + "_bt")
        return _mp

    def get_loss_info(self):
        val_loss = np.average(self.losses)
        ce_loss = np.average(self.ce_losses)
        dv_loss = np.average(self.dv_losses)
        pm_loss = np.average(self.pm_losses)
        loss_str = 'loss {:.3f}|ce {:.3f}|pm {:.3f}|dv {:.3f} '.format(
                val_loss, ce_loss, pm_loss, dv_loss)
        return loss_str, {'loss': val_loss,
                          'ce' : ce_loss,
                          'pm' : pm_loss,
                          'dv' : dv_loss}
