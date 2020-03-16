''' Evaluation of agent trajectories '''

import json
import os
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from utils import load_datasets_REVERIE, load_nav_graphs

class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, split, split_tag, instrType, bboxDir):
        self.error_margin = 3.0
        self.splits = split_tag
        self.objProposals, self.obj2viewpoint = self.loadObjProposals(bboxDir)
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.instrType = instrType
        for item in load_datasets_REVERIE(split):
            self.gt[str(item['path_id'])+'_'+str(item['objId'])] = item
            self.scans.append(item['scan'])
            self.instr_ids += ['%d_%d_%d' % (item['path_id'],item['objId'],i) for i in range(len(item[instrType]))]

        self.scans = set(self.scans)

        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.iteritems(): # compute all shortest paths
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

    def _score_item(self, instr_id, path):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        gt = self.gt[instr_id.split('_')[0]+'_'+instr_id.split('_')[1]]
        objId = instr_id.split('_')[1] # this can be used to measure whether this task is successful
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]

        # correct the goal
        scan = gt['scan']
        candidate_vps = []
        for cvp in self.obj2viewpoint[scan + '_' + objId]:
            if self.distances[scan][start].has_key(cvp):
                candidate_vps.append(cvp)

        # success or not
        if self.objProposals.has_key(scan+'_'+path[-1][0]):
            if objId in self.objProposals[scan+'_'+path[-1][0]]['objId']:
                self.scores['visible'].append(1)
            else:
                self.scores['visible'].append(0)
        else:
            self.scores['visible'].append(0)

        # oracle success or not
        oracle_succ = 0
        for passvp in path:
            if self.objProposals.has_key(scan+'_'+passvp[0]):
                if objId in self.objProposals[scan+'_'+passvp[0]]['objId']:
                    oracle_succ = 1
                    break
        self.scores['oracle_visible'].append(oracle_succ)

        # distance
        final_position = path[-1][0]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal]) #not reported
        if usePano:
            self.scores['trajectory_steps'].append(len(path) - 1)
        else:
            sstep, preStep = -1, ''
            for tstep in path:
                if tstep[0]!=preStep:
                    sstep += 1
                    preStep = tstep[0]
            self.scores['trajectory_steps'].append(sstep)

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
                with open(bboxDir + efile) as f:
                    scan = efile.split('_')[0]
                    scanvp, _ = efile.split('.')
                    data = json.load(f)
                    for vp, vv in data.iteritems():
                        for objid, objinfo in vv.iteritems():
                            if objinfo['visible_pos']:
                                if obj2viewpoint.has_key(scan+'_'+objid):
                                    if vp not in obj2viewpoint[scan+'_'+objid]:
                                        obj2viewpoint[scan+'_'+objid].append(vp)
                                else:
                                    obj2viewpoint[scan+'_'+objid] = [vp,]

                                if objProposals.has_key(scanvp):
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

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids) 
        with open(output_file) as f:
            data = json.load(f)
            if isinstance(data,dict):
                for _, item in data.items():
                    # Check against expected ids
                    if item['instr_id'] in instr_ids:
                        instr_ids.remove(item['instr_id'])
                        self._score_item(item['instr_id'], item['trajectory'])
            else:
                for item in data:
                    # Check against expected ids
                    if item['instr_id'] in instr_ids:
                        instr_ids.remove(item['instr_id'])
                        self._score_item(item['instr_id'], item['trajectory'])
        assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                       % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
        assert len(self.scores['nav_errors']) == len(self.instr_ids)

        num_successes = sum(self.scores['visible'])
        oracle_successes = sum(self.scores['oracle_visible'])

        spls = []
        for visible, length, sp in zip(self.scores['visible'], self.scores['trajectory_lengths'],
                                   self.scores['shortest_path_lengths']):
            if max(length, sp)==0:
                stop=1
            if visible:
                spls.append(sp / max(length, sp))
            else:
                spls.append(0)


        score_summary = {
            'length': np.average(self.scores['trajectory_lengths']),
            'nav_error': np.average(self.scores['nav_errors']),
            'success_rate': float(num_successes)/float(len(self.scores['visible'])),
            'oracle success_rate': float(oracle_successes)/float(len(self.scores['oracle_visible'])),
            'spl': np.average(spls),
            # RGS can be added here,
        }

        return score_summary, self.scores

def eval_seq2seq(resfiles, instrType, split_tags, splits):
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''

    for resfile in resfiles:
        for i, split_tag in enumerate(split_tags):
            ev = Evaluation([splits[i]], split_tag, instrType, bboxDir)
            score_summary, _ = ev.score(resfile % split_tag)
            print '\n%s' % (resfile % split_tag)
            pp.pprint(score_summary)

if __name__ == '__main__':
    instrType = 'instructions'
    usePano = True  # This means selecting one viewpoint as the next action
                    # It affects how agent steps are counted.
    VAL_SEEN_FILE = 'tasks/REVERIE/data/REVERIE_val_seen.json'
    VAL_UNSEEN_FILE = 'tasks/REVERIE/data/REVERIE_val_unseen.json'
    bboxDir = 'tasks/REVERIE/bbox/'
    RESULT_DIR = 'results/'
    dataDir = 'path to /Matterport/v1/scans/'

    resfiles = [
        RESULT_DIR + 'xxxx_%s.json',
    ]

    split_tag = ['val_seen', 'val_unseen']
    splits = [VAL_SEEN_FILE, VAL_UNSEEN_FILE]

    eval_seq2seq(resfiles, instrType, split_tag, splits)
