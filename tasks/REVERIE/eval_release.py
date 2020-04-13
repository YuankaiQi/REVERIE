''' Evaluation of agent trajectories '''

import json
import os
import os.path as osp
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
file_path = osp.dirname(__file__)
root_path = osp.split(osp.split(file_path)[0])[0]

from utilsFast import load_datasets, load_nav_graphs


usePano = True  # This means selecting one viewpoint as the next action
                # It affects how agent steps are counted.


class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, split_tag, instrType,mapFile=''):
        self.error_margin = 3.0
        self.splits = split_tag
        bboxDir = osp.join(file_path,'data','BBox')
        self.objProposals, self.obj2viewpoint = self.loadObjProposals(bboxDir)
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.instrType = instrType
        for item in load_datasets([split_tag]):
            self.gt[item['id']] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%s_%d' % (item['id'], i) for i in
                               range(len(item[instrType]))]
        self.scans = set(self.scans)

        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, item_in, evalType):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        path = item_in['trajectory']
        if evalType=='whole':
            predObjId = item_in['predObjId'] # open when release
        if self.splits=='test':
            gt = self.gt[instr_id.split('_')[0]]
        else:
            gt = self.gt[instr_id.split('_')[0]+'_'+instr_id.split('_')[1]]
        objId = str(gt['objId'])
        start = gt['path'][0]
        assert start == path[0][0], print(item_in)
        goal = gt['path'][-1]

        # correct the goal
        scan = gt['scan']
        candidate_vps = []
        for cvp in self.obj2viewpoint[scan + '_' + objId]:
            if self.distances[scan][start].__contains__(cvp):
                candidate_vps.append(cvp)
        # remote grounding success or not #open when release
        if evalType == 'whole':
            if objId==str(predObjId):
                self.scores['rgs'].append(1)
            else:
                self.scores['rgs'].append(0)
        # success or not
        if self.objProposals.__contains__(scan+'_'+path[-1][0]):
            if objId in self.objProposals[scan+'_'+path[-1][0]]['objId']:
                self.scores['visible'].append(1)
            else:
                self.scores['visible'].append(0)
        else:
            self.scores['visible'].append(0)

        # oracle success or not
        oracle_succ = 0
        for passvp in path:
            if self.objProposals.__contains__(scan+'_'+passvp[0]):
                if objId in self.objProposals[scan+'_'+passvp[0]]['objId']:
                    oracle_succ = 1
                    break
        self.scores['oracle_visible'].append(oracle_succ)



        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])

    def loadObjProposals(self, bboxDir):
        objProposals = {}
        obj2viewpoint = {}
        for efile in os.listdir(bboxDir):
            if efile.endswith('.json'):
                with open(osp.join(bboxDir, efile)) as f:
                    scan = efile.split('_')[0]
                    scanvp, _ = efile.split('.')
                    data = json.load(f)
                    for vp, vv in data.items():
                        for objid, objinfo in vv.items():
                            if objinfo['visible_pos']:
                                if obj2viewpoint.__contains__(scan+'_'+objid):
                                    if vp not in obj2viewpoint[scan+'_'+objid]:
                                        obj2viewpoint[scan+'_'+objid].append(vp)
                                else:
                                    obj2viewpoint[scan+'_'+objid] = [vp,]

                                if objProposals.__contains__(scanvp):
                                    for ii, bbox in enumerate(objinfo['bbox2d']):
                                        objProposals[scanvp]['bbox'].append(bbox)
                                        objProposals[scanvp]['visible_pos'].append(objinfo['visible_pos'][ii])
                                        objProposals[scanvp]['objId'].append(objid)

                                else:
                                    objProposals[scanvp] = {'bbox': objinfo['bbox2d'],
                                                            'visible_pos': objinfo['visible_pos']}
                                    objProposals[scanvp]['objId'] = []
                                    for _ in objinfo['visible_pos']:
                                        objProposals[scanvp]['objId'].append(objid)

        return objProposals, obj2viewpoint

    def score(self, output_file, evalType):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids) 
        with open(output_file) as f:
            data = json.load(f)

            for item in data:
                # Check against expected ids
                if item['instr_id'] in instr_ids:
                    instr_ids.remove(item['instr_id'])
                    self._score_item(item['instr_id'], item, evalType)
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        if evalType == 'whole':
            assert len(self.scores['rgs']) == len(self.instr_ids)
            num_rgs = sum(self.scores['rgs'])

        num_successes = sum(self.scores['visible'])
        oracle_successes = sum(self.scores['oracle_visible'])

        spls = []
        for visible, length, sp in zip(self.scores['visible'], self.scores['trajectory_lengths'],
                                   self.scores['shortest_path_lengths']):
            if visible:
                spls.append(sp / max(length, sp))
            else:
                spls.append(0)
        if evalType == 'whole':
            wrgs = []
            for rgs, length, sp in zip(self.scores['rgs'], self.scores['trajectory_lengths'],
                                           self.scores['shortest_path_lengths']):
                if rgs:
                    wrgs.append(sp / max(length, sp))
                else:
                    wrgs.append(0)

        score_summary = {
            'length': np.average(self.scores['trajectory_lengths']),
            'success_rate': float(num_successes)/float(len(self.scores['visible'])),
            'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_visible'])),
            'spl': np.average(spls)
        }
        if evalType == 'whole':
            score_summary['rgs'] = float(num_rgs)/float(len(self.scores['rgs']))
            score_summary['rgspl'] = np.average(wrgs)

        return score_summary

def eval_seq2seq(resfiles, instrType, split_tags, resDir, evalType):
    ''' Eval sequence to sequence models on val splits '''

    for resfile in resfiles:
        for split_tag in split_tags:
            ev = Evaluation(split_tag, instrType)
            score_summary = ev.score(resDir+resfile, evalType)
            print('\n%s %s' % ( split_tag,resfile))
            pp.pprint(score_summary)

def run_eval(resfiles, split_tag, evalType='nav'):
    '''evalType: nav or whole'''
    instrType = 'instructions'
    resDir = ''
    eval_seq2seq(resfiles, instrType, split_tag,resDir, evalType)

