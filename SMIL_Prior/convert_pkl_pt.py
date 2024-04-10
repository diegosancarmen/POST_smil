prior_filename = "/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/POST_smil/smil/smil_pose_prior.pkl"
smil_pkl = "/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/POST_smil/smil/smil_web.pkl"

from pickle import load
import sys
import os
sys.path.append(os.path.abspath('/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/POST_smil/smil/smil_webuser'))

from serialization import load_model

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

prior = load(open(prior_filename, 'rb'), encoding='latin1')

## Load SMIL model
m = load_model(smil_pkl)

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * .2
m.betas[:] = np.random.rand(m.betas.size) * .03
m.pose[0] = np.pi

## Alternatively assign mean pose from pose prior
prior = load(open(prior_filename, 'rb'), encoding='latin1')
m.pose[3:] = prior.mean # first three pose parameters contain global rotation

