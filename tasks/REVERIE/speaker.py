import json
import sys
import numpy as np
import random
from collections import namedtuple

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D

from utilsFast import vocab_pad_idx, vocab_bos_idx, vocab_eos_idx, flatten, try_cuda
from follower import batch_instructions_from_encoded

InferenceState = namedtuple("InferenceState", "prev_inference_state, flat_index, last_word, word_count, score, last_alpha")

def backchain_inference_states(last_inference_state):
    word_indices = []
    inf_state = last_inference_state
    scores = []
    last_score = None
    attentions = []
    while inf_state is not None:
        word_indices.append(inf_state.last_word)
        attentions.append(inf_state.last_alpha)
        if last_score is not None:
            scores.append(last_score - inf_state.score)
        last_score = inf_state.score
        inf_state = inf_state.prev_inference_state
    scores.append(last_score)
    return list(reversed(word_indices))[1:], list(reversed(scores))[1:], list(reversed(attentions))[1:] # exclude BOS

def batch_observations_and_actions(path_obs, path_actions, encoded_instructions):
    seq_lengths = np.array([len(a) for a in path_actions])
    max_path_length = seq_lengths.max()

    # DO NOT permute the sequence, since here we are doing manual LSTM unrolling in encoder
    # perm_indices = np.argsort(-seq_lengths)
    perm_indices = np.arange(len(path_obs))
    #path_obs, path_actions, encoded_instructions, seq_lengths = zip(*sorted(zip(path_obs, path_actions, encoded_instructions, seq_lengths), key=lambda p: p[-1], reverse=True))
    # path_obs = [path_obs[i] for i in perm_indices]
    # path_actions = [path_actions[i] for i in perm_indices]
    # if encoded_instructions:
    #     encoded_instructions = [encoded_instructions[i] for i in perm_indices]
    # seq_lengths = [seq_lengths[i] for i in perm_indices]

    batch_size = len(path_obs)
    assert batch_size == len(path_actions)

    mask = np.ones((batch_size, max_path_length), np.uint8)
    action_embedding_dim = path_obs[0][0]['action_embedding'].shape[-1]
    batched_action_embeddings = [
        np.zeros((batch_size, action_embedding_dim), np.float32)
        for _ in range(max_path_length)]
    feature_list = path_obs[0][0]['feature']
    assert len(feature_list) == 1
    image_feature_shape = feature_list[0].shape
    batched_image_features = [
        np.zeros((batch_size,) + image_feature_shape, np.float32)
        for _ in range(max_path_length)]
    for i, (obs, actions) in enumerate(zip(path_obs, path_actions)):
        # don't include the last state, which should result after the stop action
        if len(obs) == len(actions) + 1:
            obs = obs[:-1]
        assert len(obs) == len(actions)
        mask[i, :len(actions)] = 0
        for t, (ob, a) in enumerate(zip(obs, actions)):
            batched_image_features[t][i] = ob['feature'][0]
            batched_action_embeddings[t][i] = ob['action_embedding'][a]
    batched_action_embeddings = [
        try_cuda(Variable(torch.from_numpy(act), requires_grad=False))
        for act in batched_action_embeddings]
    batched_image_features = [
        try_cuda(Variable(torch.from_numpy(feat), requires_grad=False))
        for feat in batched_image_features]
    mask = try_cuda(torch.from_numpy(mask))

    start_obs = [obs[0] for obs in path_obs]

    return start_obs, \
           batched_image_features, \
           batched_action_embeddings, \
           mask, \
           list(seq_lengths), \
           encoded_instructions, \
           list(perm_indices)

class Seq2SeqSpeaker(object):
    feedback_options = ['teacher', 'argmax', 'sample']

    def __init__(self, env, results_path, encoder, decoder, instruction_len, max_episode_len=10):
        self.env = env
        self.results_path = results_path
        random.seed(1)
        self.results = {}
        self.losses = [] # For learning agents

        self.encoder = encoder
        self.decoder = decoder
        self.instruction_len = instruction_len

        self.losses = []
        self.max_episode_len = max_episode_len

    def write_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f)

    # def n_inputs(self):
    #     return self.decoder.vocab_size
    #
    # def n_outputs(self):
    #     return self.decoder.vocab_size-1 # Model doesn't output start

    def _feature_variable(self, obs, beamed=False):
        ''' Extract precomputed features into variable. '''
        features = [ob['feature'] for ob in (flatten(obs) if beamed else obs)]
        assert all(len(f) == 1 for f in features)  #currently only support one image featurizer (without attention)
        features = np.stack(features)
        return try_cuda(Variable(torch.from_numpy(features), requires_grad=False))

    def _score_obs_actions_and_instructions(self, path_obs, path_actions, encoded_instructions, feedback):
        assert len(path_obs) == len(path_actions)
        assert len(path_obs) == len(encoded_instructions)
        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
            path_lengths, encoded_instructions, perm_indices = \
            batch_observations_and_actions(
                path_obs, path_actions, encoded_instructions)

        instr_seq, _, _ = batch_instructions_from_encoded(encoded_instructions, self.instruction_len)

        batch_size = len(start_obs)

        ctx,h_t,c_t = self.encoder(batched_action_embeddings, batched_image_features)

        w_t = try_cuda(Variable(torch.from_numpy(np.full((batch_size,), vocab_bos_idx, dtype='int64')).long(),
                                requires_grad=False))
        ended = np.array([False] * batch_size)

        assert len(perm_indices) == batch_size
        outputs = [None] * batch_size
        for perm_index, src_index in enumerate(perm_indices):
            outputs[src_index] = {
                'instr_id': start_obs[perm_index]['instr_id'],
                'word_indices': [],
                'scores': [],
                #'actions': ' '.join(FOLLOWER_MODEL_ACTIONS[ac] for ac in path_actions[src_index]),
            }
        assert all(outputs)

        # for i in range(batch_size):
        #     assert outputs[i]['instr_id'] != '1008_0', "found example at index {}".format(i)

        # Do a sequence rollout and calculate the loss
        loss = 0
        sequence_scores = try_cuda(torch.zeros(batch_size))
        for t in range(self.instruction_len):
            h_t,c_t,alpha,logit = self.decoder(w_t.view(-1, 1), h_t, c_t, ctx, path_mask)
            # Supervised training

            # BOS are not part of the encoded sequences
            target = instr_seq[:,t].contiguous()

            # Determine next model inputs
            if feedback == 'teacher':
                w_t = target
            elif feedback == 'argmax':
                _,w_t = logit.max(1)        # student forcing - argmax
                w_t = w_t.detach()
            elif feedback == 'sample':
                probs = F.softmax(logit)    # sampling an action from model
                m = D.Categorical(probs)
                w_t = m.sample()
                #w_t = probs.multinomial(1).detach().squeeze(-1)
            else:
                sys.exit('Invalid feedback option')

            log_probs = F.log_softmax(logit, dim=1)
            word_scores = -F.nll_loss(log_probs, w_t, ignore_index=vocab_pad_idx, reduction='none')
            sequence_scores += word_scores.data
            loss += F.nll_loss(log_probs, target, ignore_index=vocab_pad_idx, reduction='elementwise_mean')

            for perm_index, src_index in enumerate(perm_indices):
                word_idx = w_t[perm_index].item()
                if not ended[perm_index]:
                    outputs[src_index]['word_indices'].append(int(word_idx))
                    outputs[src_index]['score'] = float(sequence_scores[perm_index])
                    outputs[src_index]['scores'].append(word_scores[perm_index].data.tolist())
                if word_idx == vocab_eos_idx:
                    ended[perm_index] = True

            # print("t: %s\tstate: %s\taction: %s\tscore: %s" % (t, world_states[0], a_t.data[0], sequence_scores[0]))

            # Early exit if all ended
            if ended.all():
                break

        for item in outputs:
            item['words'] = self.env.tokenizer.decode_sentence(item['word_indices'], break_on_eos=True, join=False)

        return outputs, loss

    def rollout(self, load_next_minibatch=True):
        path_obs, path_actions, encoded_instructions = self.env.gold_obs_actions_and_instructions(self.max_episode_len, load_next_minibatch=load_next_minibatch)
        outputs, loss = self._score_obs_actions_and_instructions(path_obs, path_actions, encoded_instructions, self.feedback)
        self.loss = loss
        self.losses.append(loss.data[0])
        return outputs

    def beam_search(self, beam_size, path_obs, path_actions):

        # TODO: here
        assert len(path_obs) == len(path_actions)

        start_obs, batched_image_features, batched_action_embeddings, path_mask, \
            path_lengths, _, perm_indices = \
            batch_observations_and_actions(path_obs, path_actions, None)
        batch_size = len(start_obs)
        assert len(perm_indices) == batch_size

        ctx,h_t,c_t = self.encoder(batched_action_embeddings, batched_image_features)

        completed = []
        for _ in range(batch_size):
            completed.append([])

        beams = [
            [InferenceState(prev_inference_state=None,
                            flat_index=i,
                            last_word=vocab_bos_idx,
                            word_count=0,
                            score=0.0,
                            last_alpha=None)]
            for i in range(batch_size)
        ]

        for t in range(self.instruction_len):
            flat_indices = []
            beam_indices = []
            w_t_list = []
            for beam_index, beam in enumerate(beams):
                for inf_state in beam:
                    beam_indices.append(beam_index)
                    flat_indices.append(inf_state.flat_index)
                    w_t_list.append(inf_state.last_word)
            w_t = try_cuda(Variable(torch.LongTensor(w_t_list), requires_grad=False))
            if len(w_t.shape) == 1:
                w_t = w_t.unsqueeze(0)

            h_t,c_t,alpha,logit = self.decoder(w_t.view(-1, 1), h_t[flat_indices], c_t[flat_indices], ctx[beam_indices], path_mask[beam_indices])

            log_probs = F.log_softmax(logit, dim=1).data
            _, word_indices = logit.data.topk(min(beam_size, logit.size()[1]), dim=1)
            word_scores = log_probs.gather(1, word_indices)
            assert word_scores.size() == word_indices.size()

            start_index = 0
            new_beams = []
            all_successors = []
            for beam_index, beam in enumerate(beams):
                successors = []
                end_index = start_index + len(beam)
                if beam:
                    for inf_index, (inf_state, word_score_row, word_index_row) in \
                        enumerate(zip(beam, word_scores[start_index:end_index], word_indices[start_index:end_index])):
                        for word_score, word_index in zip(word_score_row, word_index_row):
                            flat_index = start_index + inf_index
                            successors.append(
                                InferenceState(
                                    prev_inference_state=inf_state,
                                    flat_index=flat_index,
                                    last_word=word_index,
                                    word_count=inf_state.word_count + 1,
                                    score=inf_state.score + word_score,
                                    last_alpha=alpha[flat_index].data)
                            )
                start_index = end_index
                successors = sorted(successors, key=lambda t: t.score, reverse=True)[:beam_size]
                all_successors.append(successors)

            for beam_index, successors in enumerate(all_successors):
                new_beam = []
                for successor in successors:
                    if successor.last_word == vocab_eos_idx or t == self.instruction_len - 1:
                        completed[beam_index].append(successor)
                    else:
                        new_beam.append(successor)
                if len(completed[beam_index]) >= beam_size:
                    new_beam = []
                new_beams.append(new_beam)

            beams = new_beams

            if not any(beam for beam in beams):
                break

        outputs = []
        for _ in range(batch_size):
            outputs.append([])

        for perm_index, src_index in enumerate(perm_indices):
            this_outputs = outputs[src_index]
            assert len(this_outputs) == 0

            this_completed = completed[perm_index]
            instr_id = start_obs[perm_index]['instr_id']
            for inf_state in sorted(this_completed, key=lambda t: t.score, reverse=True)[:beam_size]:
                word_indices, scores, attentions = backchain_inference_states(inf_state)
                this_outputs.append({
                    'instr_id': instr_id,
                    'word_indices': word_indices,
                    'score': inf_state.score,
                    'scores': scores,
                    'words': self.env.tokenizer.decode_sentence(word_indices, break_on_eos=True, join=False),
                    'attentions': attentions,
                })
        return outputs

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, beam_size=1):
        ''' Evaluate once on each instruction in the current environment '''
        if not allow_cheat: # permitted for purpose of calculating validation loss only
            assert feedback in ['argmax', 'sample'] # no cheating by using teacher at test time!
        self.feedback = feedback
        if use_dropout:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
        self.beam_size = beam_size
        self.env.reset_epoch()
        self.losses = []
        self.results = {}


        # We rely on env showing the entire batch before repeating anything
        looped = False
        # rollout_scores = []
        # beam_10_scores = []
        while True:
            rollout_results = self.rollout()
            # if self.feedback == 'argmax':
            #     path_obs, path_actions, _ = self.env.gold_obs_actions_and_instructions(self.max_episode_len, load_next_minibatch=False)
            #     beam_results = self.beam_search(1, path_obs, path_actions)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         assert rollout_traj['instr_id'] == beam_trajs[0]['instr_id']
            #         assert rollout_traj['word_indices'] == beam_trajs[0]['word_indices']
            #         assert np.allclose(rollout_traj['score'], beam_trajs[0]['score'])
            #     print("passed check: beam_search with beam_size=1")
            #
            #     self.env.set_beam_size(10)
            #     beam_results = self.beam_search(10, path_obs, path_actions)
            #     assert len(rollout_results) == len(beam_results)
            #     for rollout_traj, beam_trajs in zip(rollout_results, beam_results):
            #         rollout_score = rollout_traj['score']
            #         rollout_scores.append(rollout_score)
            #         beam_score = beam_trajs[0]['score']
            #         beam_10_scores.append(beam_score)
            #         # assert rollout_score <= beam_score
            # # print("passed check: beam_search with beam_size=10")

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

    def train(self, encoder_optimizer, decoder_optimizer, n_iters, feedback='teacher'):
        ''' Train for a given number of iterations '''
        assert feedback in self.feedback_options
        self.feedback = feedback
        self.encoder.train()
        self.decoder.train()
        self.losses = []
        it = range(1, n_iters + 1)
        try:
            import tqdm
            it = tqdm.tqdm(it)
        except:
            pass
        for _ in it:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            self.rollout()
            self.loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

    def _encoder_and_decoder_paths(self, base_path):
        return base_path + "_enc", base_path + "_dec"

    def save(self, path):
        ''' Snapshot models '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def load(self, path, **kwargs):
        ''' Loads parameters (but not training state) '''
        encoder_path, decoder_path = self._encoder_and_decoder_paths(path)
        self.encoder.load_state_dict(torch.load(encoder_path, **kwargs))
        self.decoder.load_state_dict(torch.load(decoder_path, **kwargs))
