POST_pt = "/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/POST/prior/prior_stage_3.pt"

prior_filename = "/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/POST_smil/smil/smil_pose_prior.pkl"
# smil_pkl = "/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/SMIL_Prior/smil/smil_web.pkl"

import pickle

class Mahalanobis(object):

    def __init__(self, mean, prec, prefix):
        self.mean = mean
        self.prec = prec
        self.prefix = prefix

    def __call__(self, pose):
        if len(pose.shape) == 1:
            return (pose[self.prefix:]-self.mean).reshape(1, -1).dot(self.prec)
        else:
            return (pose[:, self.prefix:]-self.mean).dot(self.prec)

prior = pickle.load(open(prior_filename, 'rb'), encoding='latin1')
print('prior')            
print(prior.mean)

import torch

model = torch.load(POST_pt)

