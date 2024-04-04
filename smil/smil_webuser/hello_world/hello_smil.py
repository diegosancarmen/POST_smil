'''

This script is adapted from the SMPL package, copyright 2015 Matthew Loper, Naureen Mahmood and
the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMIL is available here http://s.fhg.de/smil
For comments or questions, please email us at: smil@iosb.fraunhofer.de


Please Note:
============
This is a demo version of the script for driving the SMIL model with python.
We would be happy to receive comments, help and suggestions on improving this code
and in making it available on more platforms.

System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMIL model. The code shows how to:
  - Load the SMIL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smil_webuser/hello_world directory. You can run
the hello world script now by typing the following:
>	python hello_smil.py

'''

import sys
import os
sys.path.append(os.path.abspath('/home/coeguest/hdelacruz/DAIP/Experiments_2024/042024/POST_smil/smil/smil_webuser'))

from serialization import load_model
import numpy as np

## Load SMIL model
## Make sure path is correct
m = load_model( '../../smil_web.pkl' )

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * .2
m.betas[:] = np.random.rand(m.betas.size) * .03

## Write to an .obj file
outmesh_path = './hello_smil.obj'
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print('..Output mesh saved to: ', outmesh_path )
