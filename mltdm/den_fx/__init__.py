# -*- coding: utf-8 -*-

from os.path import join
from os.path import exists

from .fx_den import *
from .setup import *



# check for required files
from mltdm.config import config
c_dat = config().dat

r_f = []


r_f.append(join(c_dat['data_dir'],c_dat['fx_m']))
r_f.append(join(c_dat['data_dir'],c_dat['fxnoAE_m']))
r_f.append(join(c_dat['data_dir'],'fx_den_feat.hdf'))

m_f = [f for f in r_f if not exists(f)]

if len(m_f) > 0:
    print('Some files required for running the model are missing:')
    for f in m_f:
        print(f'\t{f}')
    print('Run setup() to download files:')
    print('\timport mltdm.den_fx as den_fx')
    print('\tden_fx.setup()')