# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np

import mltdm
from mltdm import io


def create_feat():
    #create feature dataset that we can load
    
    # use the flare.nc file to create an initial 
    # feature dataset
    tol = pd.Timedelta('2.5 minute')
    fism_cols = ['1300_02', '43000_09', '85550_13', '94400_18', 'DateTime']
    omni_cols = ['SYM_H index', 'AE', 'DateTime']
    
    fism, _ = io.fism_flare(rcols=fism_cols)
    
    sdate = fism['DateTime'].min()+pd.offsets.YearBegin(-1)
    edate = fism['DateTime'].max()+pd.offsets.YearBegin(0)
    
    omni = io.omni(sdate=sdate,edate=edate, rcols=omni_cols)
    omni = omni[(omni['DateTime'] >= fism['DateTime'].min()) &
                (omni['DateTime'] <= fism['DateTime'].max())]
    
    
    omni = omni.set_index('DateTime')
    fism = fism.set_index('DateTime')
    
    feat_dat = pd.merge_asof(left=fism,right=omni,
                             right_index=True,left_index=True,
                             direction='nearest',tolerance=tol)
    
    out_f = os.path.join(mltdm.c_dat['data_dir'],'fx_den_feat.hdf')
    
    feat_dat = feat_dat.reset_index()
    feat_dat.to_hdf(out_f,key='fx_den_feat',
                    complevel=2,format='table', data_columns=['DateTime'])
     
    
def append_feat():
    pass


def load_feat(sdate: str=None, edate: str=None):
    
    log_cols = ['1300_02', '43000_09', '85550_13', '94400_18']
    
    feat_f = os.path.join(mltdm.c_dat['data_dir'],'fx_den_feat.hdf')
    
    # if the feature data doesn't exist
    # creat it
    if not os.path.exists(feat_f):
        create_feat()
    
        
    # TODO check if sdate and edate are outside file ranges
    # if so might have to call append to fix the file.
    
    # setup where statements to only read in data that we need    
    if sdate and edate:
        where = f'(DateTime>="{sdate}") & (DateTime<="{edate}")'
    elif sdate:
        # 
        srnd = pd.to_datetime(sdate).round('5min') 
        where = f'DateTime=="{srnd}"'
    else:
        where = None  
    
    feat_dat = pd.read_hdf(feat_f, where=where)
    
    # log the columns for predictions
    for i in log_cols:
        feat_dat[i] = np.log10(feat_dat[i])
        
    return feat_dat

