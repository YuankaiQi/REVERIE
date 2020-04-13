import torch
from torch import optim

import os
import os.path as osp
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse

file_path = osp.dirname(__file__)
root_path = osp.split(osp.split(file_path)[0])[0]

import utilsFast
from utilsFast import build_vocab, write_vocab, read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda
from utilsFast import module_grad, colorize, filter_param
from eval_release import Evaluation
from env import R2RBatch, ImageFeatures
from modelFast import TransformerEncoder, EncoderLSTM, AttnDecoderLSTM, CogroundDecoderLSTM, \
    ProgressMonitor, DeviationMonitor, EncoderLSTMGlove
from modelFast import SpeakerEncoderLSTM, DotScorer
from modelFast import Pointer,DectPointer
from follower import Seq2SeqAgent
from scorer import Scorer
# import evalFast

from vocab import SUBTRAIN_VOCAB, TRAINVAL_VOCAB, TRAIN_VOCAB

MAX_INPUT_LENGTH = 80 # TODO make this an argument
max_episode_len = 10


hidden_size = 512
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005

log_every = 100
save_every = 100


def get_model_prefix(args, image_feature_list):
    image_feature_name = "+".join(
        [featurizer.get_name() for featurizer in image_feature_list])
    nn = ('{}{}{}{}{}{}'.format(
            ('_ts' if args.transformer else ''),
            ('_sc' if args.scorer else ''),
            ('_mh' if args.num_head > 1 else ''),
            ('_cg' if args.coground else ''),
            ('_pm' if args.prog_monitor else ''),
            ('_sa' if args.soft_align else ''),
            ))
    model_prefix = 'NP{}_{}_{}_{}heads'.format(
        nn, args.feedback_method, image_feature_name, args.num_head)
    if args.use_train_subset:
        model_prefix = 'trainsub_' + model_prefix
    if args.bidirectional:
        model_prefix = model_prefix + "_bidirectional"
    if args.use_pretraining:
        model_prefix = model_prefix.replace(
            'follower', 'follower_with_pretraining', 1)
    return model_prefix


def eval_model(agent, results_path, use_dropout, feedback, allow_cheat=False):
    agent.results_path = results_path
    agent.test(
        use_dropout=use_dropout, feedback=feedback, allow_cheat=allow_cheat)


def train(args, train_env, agent, optimizers, n_iters, log_every=log_every, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''

    if val_envs is None:
        val_envs = {}

    print('Training with %s feedback' % args.feedback_method)

    data_log = defaultdict(list)
    start = time.time()

    split_string = "-".join(train_env.splits)

    def make_path(n_iter):
        return osp.join(
            args.SNAPSHOT_DIR, '%s_%s_iter_%d' % (
                get_model_prefix(args, train_env.image_features_list),
                split_string, n_iter))

    best_metrics = {}
    last_model_saved = {}
    for idx in range(0, n_iters, log_every):
        agent.env = train_env

        interval = min(log_every, n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)
        loss_str = ''

        # Train for log_every interval
        env_name = 'train'
        agent.train(optimizers, interval, feedback=args.feedback_method)
        _loss_str, losses = agent.get_loss_info()
        loss_str += env_name + ' ' + _loss_str
        for k,v in losses.items():
            data_log['%s %s' % (env_name,k)].append(v)

        save_log = []
        # # Run validation
        # for env_name, (val_env, evaluator) in sorted(val_envs.items()):
        #     agent.env = val_env
        #     # Get validation loss under the same conditions as training
        #     agent.test(use_dropout=True, feedback=args.feedback_method,
        #                allow_cheat=True)
        #     _loss_str, losses = agent.get_loss_info()
        #     loss_str += ', ' + env_name + ' ' + _loss_str
        #     for k,v in losses.items():
        #         data_log['%s %s' % (env_name,k)].append(v)
        #
        #     agent.results_path = '%s/%s_%s_iter_%d.json' % (
        #         args.RESULT_DIR, get_model_prefix(
        #             args, train_env.image_features_list),
        #         env_name, iter)
        #
        #     # Get validation distance from goal under evaluation conditions
        #     agent.test(use_dropout=False, feedback='argmax')
        #
        #     print("evaluating on {}".format(env_name))
        #     score_summary, _ = evaluator.score_results(agent.results)
        #
        #     for metric, val in sorted(score_summary.items()):
        #         data_log['%s %s' % (env_name, metric)].append(val)
        #         if metric in ['success_rate']:
        #             loss_str += ', %s: %.3f' % (metric, val)
        #
        #             key = (env_name, metric)
        #             if key not in best_metrics or best_metrics[key] < val:
        #                 best_metrics[key] = val
        #                 if not args.no_save:
        #                     model_path = make_path(iter) + "_%s-%s=%.3f" % (
        #                         env_name, metric, val)
        #                     save_log.append(
        #                         "new best, saved model to %s" % model_path)
        #                     agent.save(model_path)
        #                     agent.write_results()
        #                     if key in last_model_saved:
        #                         for old_model_path in last_model_saved[key]:
        #                             if osp.isfile(old_model_path):
        #                                 os.remove(old_model_path)
        #                     #last_model_saved[key] = [agent.results_path] +\
        #                     last_model_saved[key] = [] +\
        #                         list(agent.modules_paths(model_path))

        print(('%s (%d %d%%) %s' % (
            timeSince(start, float(iter)/n_iters),
            iter, float(iter)/n_iters*100, loss_str)))
        for s in save_log:
            print(colorize(s))

        if not args.no_save:
            if save_every and iter>8000 and iter % save_every == 0:
                agent.save(make_path(iter))

            df = pd.DataFrame(data_log)
            df.set_index('iteration')
            df_path = '%s/%s_%s_log.csv' % (
                args.PLOT_DIR, get_model_prefix(
                    args, train_env.image_features_list), split_string)
            df.to_csv(df_path)

def setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_more_train_env(args, train_vocab_path, train_splits):
    setup(args.seed)
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=train_splits, tokenizer=tok)
    return train_env


def make_scorer(args):

    bidirectional = args.bidirectional
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    if   args.useObjLabelOrVis =='none':
        feature_size, action_embedding_size = 2048+128, 2048+128
    elif args.useObjLabelOrVis == 'vis':
        feature_size, action_embedding_size = 2048 +128+ args.objVisFeatDim, 2048 + 128 + args.objVisFeatDim
    elif args.useObjLabelOrVis =='label':
        feature_size, action_embedding_size = 2048+128+args.objLanFeatDim, 2048 + 128 +args.objLanFeatDim
    elif args.useObjLabelOrVis =='both':
        feature_size  = 2048+128+args.objVisFeatDim +args.objLanFeatDim
        action_embedding_size = 2048 + args.objVisFeatDim + args.objLanFeatDim +128

    traj_encoder = try_cuda(SpeakerEncoderLSTM(action_embedding_size, feature_size,
                                          enc_hidden_size, dropout_ratio, bidirectional=args.bidirectional))
    scorer_module = try_cuda(DotScorer(enc_hidden_size, enc_hidden_size))
    scorer = Scorer(scorer_module, traj_encoder)
    if args.load_scorer is not '':
        scorer.load(args.load_scorer)
        print(colorize('load scorer traj '+ args.load_scorer))
    elif args.load_traj_encoder is not '':
        scorer.load_traj_encoder(args.load_traj_encoder)
        print(colorize('load traj encoder '+ args.load_traj_encoder))
    return scorer


def make_follower(args, vocab):
    enc_hidden_size = hidden_size//2 if args.bidirectional else hidden_size
    glove_path = osp.join(file_path, 'data', 'train_glove.npy')  # not used
    glove = np.load(glove_path) if args.use_glove else None

    if args.useObjLabelOrVis == 'none':
        feature_size, action_embedding_size = 2048+128, 2048 + 128
    elif args.useObjLabelOrVis == 'vis':
        feature_size, action_embedding_size = 2048+128 + args.objVisFeatDim, 2048 + 128 + args.objVisFeatDim
    elif args.useObjLabelOrVis == 'label':
        feature_size, action_embedding_size = 2048+128 + args.objLanFeatDim, 2048 + 128 + args.objLanFeatDim
    elif args.useObjLabelOrVis == 'both':
        feature_size = 2048 +128 + args.objVisFeatDim + args.objLanFeatDim
        action_embedding_size = 2048 + args.objVisFeatDim + args.objLanFeatDim + 128

    Encoder = TransformerEncoder if args.transformer else EncoderLSTM
    Decoder = CogroundDecoderLSTM if args.coground else AttnDecoderLSTM
    word_embedding_size = 256 if args.coground else 300
    encoder = try_cuda(Encoder(
        len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
        dropout_ratio, bidirectional=args.bidirectional, glove=glove))
    decoder = try_cuda(Decoder(
        action_embedding_size, hidden_size, dropout_ratio,
        feature_size=feature_size, num_head=args.num_head))
    prog_monitor = try_cuda(ProgressMonitor(action_embedding_size,
                            hidden_size)) if args.prog_monitor else None
    bt_button = try_cuda(BacktrackButton()) if args.bt_button else None
    dev_monitor = try_cuda(DeviationMonitor(action_embedding_size,
                            hidden_size)) if args.dev_monitor else None

    agent = Seq2SeqAgent(
        None, "", encoder, decoder, max_episode_len,
        max_instruction_length=MAX_INPUT_LENGTH,
        attn_only_verb=args.attn_only_verb)
    agent.prog_monitor = prog_monitor
    agent.dev_monitor = dev_monitor # not used
    agent.bt_button = bt_button # not used
    agent.soft_align = args.soft_align # not used

    if args.useObjLabelOrVis!='none':
        if args.useDect:
            print('Using detectoin-based pointer')
            agent.pointer = DectPointer(args)
        else:
            print('Using gt-based pointer')
            agent.pointer = Pointer(args)
        agent.useObjLabelOrVis = args.useObjLabelOrVis
        agent.objTopK = args.objTopK
        agent.objVisFeatDim = args.objVisFeatDim
        agent.objLanFeatDim = args.objLanFeatDim
        agent.ObjEachViewVisFeatPath = osp.join(root_path,'img_features',args.ObjEachViewVisFeatDir)
        agent.ObjEachViewLanFeatPath = osp.join(root_path,'img_features',args.ObjEachViewLanFeatDir)

        agent.ObjEachViewVisFeat = {}
        agent.ObjEachViewLanFeat = {}
        dict_glove = np.load(args.labelGlovePath) # for object label encoding
        if args.useObjLabelOrVis in ['label', 'both']:
            agent.objLabelEncoder =try_cuda(EncoderLSTMGlove(
                dict_glove.shape[0], 300, int(enc_hidden_size/2), vocab_pad_idx,
                dropout_ratio, bidirectional=True, glove=dict_glove))
        else:
            agent.objLabelEncoder = None
    else:
        agent.pointer = None


    if args.scorer: # not used
        agent.scorer = make_scorer(args)

    if args.load_follower is not '':
        scorer_exists = osp.isfile(args.load_follower + '_scorer_enc')
        agent.load(args.load_follower, load_scorer=(args.load_scorer is '' and scorer_exists))
        print(colorize('load follower '+ args.load_follower))

    return agent

def make_env_and_models(args, train_vocab_path, train_splits, test_splits):
    setup(args.seed)
    image_features_list = ImageFeatures.from_args(args)
    if args.job == None:# create vocab only during training (job == none)
        vocab = build_vocab(train_splits)
        write_vocab(vocab, TRAIN_VOCAB)

    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=train_splits, tokenizer=tok) if len(train_splits) > 0 else None
    test_envs = {
        split: (R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=[split], tokenizer=tok),
                Evaluation(split, args.instrType))
        for split in test_splits}

    agent = make_follower(args, vocab)
    agent.env = train_env

    if args.useObjLabelOrVis in [ 'label','both']:
        if not train_env is None:
            agent.pointer.wtoi = train_env.wtoi
        else:
            agent.pointer.wtoi = test_envs[test_splits[0]][0].wtoi

    return train_env, test_envs, agent


def train_setup(args, train_splits=['train']):

    if args.job == 'train' or args.job is None:
        val_splits = []
    else:
        val_splits = ['val_seen', 'val_unseen']

    if args.use_test_set:
        val_splits = ['test']
    if args.debug:
        log_every = 5
        args.n_iters = 10
        train_splits = val_splits = ['val_seen']

    vocab = TRAIN_VOCAB

    if args.use_train_subset:
        train_splits = ['sub_' + split for split in train_splits]
        val_splits = ['sub_' + split for split in val_splits]
        vocab = SUBTRAIN_VOCAB

    train_env, val_envs, agent = make_env_and_models(
        args, vocab, train_splits, val_splits)

    agent.useStopFeat = args.useStopFeat
    if not train_env is None:
        agent.pointer.word2count = train_env.word2count
    else:
        agent.pointer.word2count = val_envs[val_splits[0]][0].word2count
    if args.use_pretraining:
        pretrain_splits = args.pretrain_splits
        assert len(pretrain_splits) > 0, \
            'must specify at least one pretrain split'
        pretrain_env = make_more_train_env(
            args, vocab, pretrain_splits)

    if args.use_pretraining:
        return agent, train_env, val_envs, pretrain_env
    else:
        return agent, train_env, val_envs


def train_val(args):
    ''' Train on the training set, and validate on seen and unseen splits. '''


    if args.use_pretraining:
        agent, train_env, val_envs, pretrain_env = train_setup(args)
    else:
        agent, train_env, val_envs = train_setup(args)

    m_dict = {
            'follower': [agent.encoder,agent.decoder],
            'pm': [agent.prog_monitor],
            'follower+pm': [agent.encoder, agent.decoder, agent.prog_monitor],
            'all': agent.modules() # this is not really `all` now
        }
    if args.useObjLabelOrVis in ['label', 'both']:
        m_dict['objLabelEncoder']=[agent.objLabelEncoder]

    if agent.scorer:
        m_dict['scorer_all'] = agent.scorer.modules()
        m_dict['scorer_scorer'] = [agent.scorer.scorer]

    optimizers = [optim.Adam(filter_param(m), lr=learning_rate,
        weight_decay=weight_decay) for m in m_dict[args.grad] if len(filter_param(m))]
    optimizers.append(optim.Adam(filter_param(agent.objLabelEncoder), lr=learning_rate,
        weight_decay=weight_decay) )
    if args.use_pretraining:
        train(args, pretrain_env, agent, optimizers,
              args.n_pretrain_iters, val_envs=val_envs)

    val_envs = {}

    train(args, train_env, agent, optimizers,
          args.n_iters, val_envs=val_envs)

def make_arg_parser():
    parser = argparse.ArgumentParser()
    ImageFeatures.add_args(parser)
    parser.add_argument("--load_scorer", type=str, default='')
    parser.add_argument("--load_follower", type=str, default='')
    parser.add_argument("--load_traj_encoder", type=str, default='')
    parser.add_argument("--feedback_method",
            choices=["sample", "teacher", "sample1step","sample2step","sample3step","teacher+sample","recover"], default="sample")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--bidirectional", action='store_true')
    parser.add_argument("--transformer", action='store_true')
    parser.add_argument("--scorer", action='store_true')
    parser.add_argument("--coground", action='store_false')
    parser.add_argument("--prog_monitor", action='store_false')
    parser.add_argument("--dev_monitor", action='store_true')
    parser.add_argument("--bt_button", action='store_true')
    parser.add_argument("--soft_align", action='store_true')
    parser.add_argument("--n_iters", type=int, default=10900)
    parser.add_argument("--num_head", type=int, default=1)
    parser.add_argument("--use_pretraining", action='store_true')
    parser.add_argument("--grad", type=str, default='all')
    parser.add_argument("--pretrain_splits", nargs="+", default=[])
    parser.add_argument("--n_pretrain_iters", type=int, default=50000)
    parser.add_argument("--no_save", action='store_true')
    parser.add_argument("--use_glove", action='store_true')
    parser.add_argument("--WIDTH", type=int, default=640)
    parser.add_argument("--HEIGHT", type=int, default=480)
    parser.add_argument("--VFOV", type=int, default=60)

    parser.add_argument("--useStopFeat", type=int, default=1)
    parser.add_argument("--useObjLabelOrVis", type=str, default='both',
                        help="options: vis, label, both, none")
    parser.add_argument("--objFeatType", type=str, default='fc7',
                        help="options: pool5, fc7")
    parser.add_argument("--objVisFeatDim", type=int, default=2048)
    parser.add_argument("--objLanFeatDim", type=int, default=512)
    parser.add_argument("--objTopK", type=int, default=3)
    parser.add_argument("--useDect", type=bool, default=False)
    parser.add_argument("--instrType", type=str, default='instructions',
                        help="options: instrutions, instructions_l")

    parser.add_argument("--ObjEachViewVisFeatDir", type=str,
                        default='ObjEachViewVisFeatFc7Top3/')
    parser.add_argument("--ObjEachViewLanFeatDir", type=str,
                        default='ObjEachViewLanFeatTop3/')
    parser.add_argument("--labelGlovePath", type=str, # for object label embedding
                        default='tasks/REVERIE/data/reverie4_reverie4.npy')
    parser.add_argument("--grdModelPrefix", type=str,  # for visual grounding
                        default='MAttNet3/output/reverie4_reverie4/mrcn_cmr_with_st')
    parser.add_argument("--matterportDir", type=str,
                        default='/home/qyk/dataset/Matterport/')

    parser.add_argument("--attn_only_verb", action='store_true')
    parser.add_argument("--use_train_subset", action='store_true',
                       help="use a subset of the original train data for validation")
    parser.add_argument("--use_test_set", action='store_true')
    parser.add_argument("--seed", type=int, default=1)

    return parser


if __name__ == "__main__":
    utilsFast.run(make_arg_parser(), train_val)

