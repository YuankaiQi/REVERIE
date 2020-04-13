from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import re
import string
import sys
import os
import os.path as osp
import numpy as np
import torch
from torch.autograd import Variable
import random
from modelFast import Pointer
from trainFast import make_arg_parser

from eval_release import run_eval
# load refer
file_path = os.path.dirname(__file__)
root_path = osp.split(osp.split(file_path)[0])[0]
sys.path.insert(0, osp.join(root_path,'MAttNet3','pyutils','refer'))

from refer import REFER
# simulator path and parameters

def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab

def getWord2count():
    referPath = osp.join(root_path, 'MAttNet3', 'data/')
    refer = REFER(referPath, 'reverie4', 'reverie4')
    sentToTokens = refer.sentToTokens

    # count up the number of words
    word2count = {}
    for sent_id, tokens in sentToTokens.items():
        for wd in tokens:
            word2count[wd] = word2count.get(wd, 0) + 1
    # add category words
    category_names = list(refer.Cats.values()) + ['__background__']
    for cat_name in category_names:
        for wd in cat_name.split():
            if wd not in word2count or word2count[wd] <= 5:
                word2count[wd] = 1e5
    return word2count

def split_sentence(sentence):
    '''
    Break sentence into a list of words and punctuation '''
    toks = []
    for word in [s.strip().lower() for s in re.compile(r'(\W+)').split(sentence.strip()) if len(s.strip()) > 0]:
        # # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
        # if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
        #     toks += list(word)
        if all(c in string.punctuation for c in word):
            pass
        else:
            toks.append(word)
    return toks

def sentToToken(sent, word2count):
    tokens = split_sentence(sent)
    final = []
    for wd in tokens:
        if wd in word2count:
            if word2count[wd] > 5:
                final.append(wd)
            else:
                final.append('<UNK>')
        else:
            final.append('<UNK>')
    return final

def idToScan(split):
    id2scan = {}
    with open('tasks/REVERIE/data/REVERIE_%s.json'%split,'r') as f:
        data = json.load(f)
        for item in data:
            id = item['id']
            id2scan[id] = (item['scan'], item['instructions'])

    return id2scan

if __name__ == '__main__':

    splits = ['val_seen','val_unseen', 'test']#
    evalType = 'whole'
    experimentName = 'releaseCheck'
    resBaseDir = 'tasks/REVERIE/experiments/%s/results/'%experimentName

    vocabPath = osp.join(root_path,'MAttNet3','cache','prepro','reverie4_reverie4','vocab.txt')

    max_length = 40

    args = make_arg_parser().parse_args()
    pointer = Pointer(args)
    word2count = getWord2count()
    vocab = read_vocab(vocabPath)
    wtoi = {w: i for i, w in enumerate(vocab)}

    for split in splits:
        id2scan = idToScan(split)

        navResPath = resBaseDir+'NP_cg_pm_sample_imagenet_mean_pooled_1heads_%s'%split+'.json'

        with open(navResPath, 'r') as f:
            navRes = json.load(f)

        submission,acc = [],0

        for res in navRes:
            pathObjsent = res['instr_id'].split('_')

            if split=='test':
                id = pathObjsent[0]
                sentence = id2scan[id][1][int(pathObjsent[1])]
            else:
                id = pathObjsent[0]+'_'+pathObjsent[1]
                sentence = id2scan[id][1][int(pathObjsent[2])]

            scan = id2scan[id][0]
            vp = res['trajectory'][-1][0]

            newItem = dict(res)

            Label = np.zeros((1, max_length), dtype=np.int32)

            tokens = sentToToken(sentence, word2count)
            for j, w in enumerate(tokens):
                if j < max_length:
                    Label[0, j] = wtoi[w]
            predictions = pointer.groundingForView(scan, vp, Label,\
                            viewId=None, oneOrAll='all', resType='prediction')
            if predictions is None:
                newItem['predObjId'] = -1
            else:
                maxid = np.argmax(predictions['scores'])
                pred_objId = predictions['objIds'][maxid]
                newItem['predObjId'] = int(pred_objId)

            submission.append(newItem)

        grdFile = '%s/%s_submission.json'%(resBaseDir, split)
        with open(grdFile, 'w') as f11:
            json.dump(submission,f11,indent=2)
        if split != 'test':
            run_eval([grdFile], [split], evalType)
