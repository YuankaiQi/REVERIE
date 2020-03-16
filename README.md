# REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments
Here are the pre-released data for the CVPR 2020 paper [REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments](https://arxiv.org/abs/1904.10151)
<div align="center">
<img src="REVERIE_task.png" width = "300" height = "300" alt="REVERIE task example" align=center />
</div>

## Definition of the REVERIE Task
As shown in the above figure, a robot agent is given a natural language instruction referring to a remote object (here in the red bounding box) in a photo-realistic 3D environment. The agent must navigate to an appropriate location and identify the object from multiple distracting candidates. The blue discs indicate nearby navigable viewpoints provided the simulator.

## Data Organization
Unzip the data.zip and bbox.zip files. Then in the data folder, you get REVERIE_train.json, REVERIE_val_seen.json, and REVERIE_val_unseen.json three files, which provide instructions, paths, and target object of each task. In the bbox folder, you get json files that record objects observed at each viewpoint within 3 meters.

+ **Example of tarin/val_seen/val_unseen.json file**
```
[
  {
    "distance" : 11.65, # distance to the goal viewpoint
    "ix": 208,  # Reserved data, not used
    "scan": "qoiz87JEwZ2", # building ID
    "heading": 4.59, # initial parameters for agent
    "path_id": 1357, # inherited from the R2R dataset
    "objId": 66, # the unique object id in the current building 
    "id": "1357_66" # task id
    "instructions":[ # collected instructions for REVERIE
        "Go to the entryway and clean the coffee table", 
        "Go to the foyer and wipe down the coffee table", 
        "Go to the foyer on level 1 and pull out the coffee table further from the chair"
     ]
    "path": [ # inherited from the R2R dataset
        "bdb1023cb7cc4ebd8245b9291fcbc1a2", 
        "a6ba3f53b7964464b23341896d3c75fa", 
        "c407e34577aa4724b7e5d447a5d859d1", 
        "9f68b19f50d14f5d8371447f73c3a2e3", 
        "150763c717894adc8ccbbbe640fa67ef", 
        "59b190857cfe47f691bf0d866f1e5aeb", 
        "267a7e2459054db7952fc1e3e45e98fa"
      ]
     "instructions_l":[ # inherited from the R2R dataset and provided just for convenience 
        "Walk into the dining room and continue past the table. Turn left when you xxx ", 
       ...
       ]
  },
  ...
]
```
+ **Example of json file in the bbox folder**

    File name format: ScanID_ViewpointID.json, e.g.,VzqfbhrpDEA_57fba128d2f042f7a59793c665a3f587.json
```
{ # note that this is in the type of dict not list
  "57fba128d2f042f7a59793c665a3f587":{ # this key is the id of viewpoint
    "827":{ # this key is the id of object 
      "name": "toilet",
      "visible_pos":[
        6,7,8,9,19,20  # these are view index (0~35) which contain the object. Index is consitent with that in  R2R 
        ],
      "bbox2d":[
        [585,382,55,98], # [x,y,w,h] and corresponds to the views listed in the "visible_pos"
        ...
       ]
    },
    "833": {
       ...
    },
  }
}
```
## Integrating into Your Project
+ If you are new to this task, we recommend to first review the VLN task by [Self-Monitor](https://github.com/chihyaoma/selfmonitoring-agent), which helps you set up the R2R simulator. And then conduct the following instructions.

The easiest way to integrate these object infomation into your project is to preload all the objects bounding box/label/visible_pos with the **loadObjProposals()** function as in the eval.py file. Then you can access visible objects using ScanID_ViewpointID as key. You can use any referring expression methods to get matched objects with an instruction
## Note
+ We modify the method to load dataset, see **load_datasets_REVERIE()** in utils.py
+ The number of instructions may vary across the dataset, we recommend the following way to index an instruction:
```
instrType = "instructions"
self.instr_ids += ['%s_%d' % (item['id'],i) for i in range(len(item[instrType]))]

```
+ To get the Remote Grounding Success Rate in the eval.py file, you need to implement the method to read the predicted object id from your results file. And then compare it against the 'objId' in the \_score_item() function.
## Acknowledgements
We would like to thank Matterport for allowing the Matterport3D dataset to be used by the academic community. This project is supported by the [Australian Centre for Robotic Vision](https://www.roboticvision.org/). We also thank [Philip Roberts](mailto:philip.roberts@adelaide.edu.au), [Zheng Liu](mailto:marco.liu19@imperial.ac.uk), and [Zizheng Pan](mailto:zizheng.pan@student.adelaide.edu.au), and [Sam Bahrami](https://www.roboticvision.org/rv_person/sam-bahrami/) for their great help in building the dataset.
## Reference
The REVERIE task and dataset are descriped in:
```
@inproceedings{reverie,
  title={REVERIE: Remote Embodied Visual Referring Expression in Real Indoor Environments},
  author={Yuankai Qi and Qi Wu and Peter Anderson and Xin Wang and William Yang Wang and Chunhua Shen and Anton van den Hengel},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}

```
