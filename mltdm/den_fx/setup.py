# -*- coding: utf-8 -*-

import os

from urllib.parse import urljoin

from mltdm.config import config
from mltdm.io import dl_file


def setup():
    
    # get the config data
    c_d = config().dat
    
    #check if models exist
    # if they don't download them
    d = c_d['data_dir']
    z_url = urljoin(c_d['zenodo'],'files/')
    
    
    m1 = os.path.join(d,c_d['fx_m'])
    m2 = os.path.join(d,c_d['fxnoAE_m'])
    
    if not os.path.exists(d):
        os.mkdir(d)

    if not os.path.exists(m1):
        url =  urljoin(z_url,c_d['fx_m'])
        dl_file(url,m1)
        
    if not os.path.exists(m2):
        url =  urljoin(z_url,c_d['fxnoAE_m'])
        dl_file(url,m2)
        
    #check for feature file
    ft_f = 'fx_den_feat.hdf'
    ft_d = os.path.join(d,ft_f)
    
    if not os.path.exists(ft_d):
        url = urljoin(z_url,ft_f)
        dl_file(url,ft_d)

    print('Setup Complete')
        