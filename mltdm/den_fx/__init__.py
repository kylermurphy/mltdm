# -*- coding: utf-8 -*-

# load the fixed altitude model as soon as 
# the module is imported
# this saves time as the model will not
# need to loaded for each instance of 
# den_fx.fx_den

from mltdm.config import config
from skops.io import load as sio_load

c_dat = config().dat

den_fx_mod = sio_load(c_dat['fx_dir'])