# -*- coding: utf-8 -*-

import pandas as pd

from mltdm import io


def create_feat():
    
    # use the flare.nc file to create an initial 
    # feature dataset
    tol = pd.Timedelta('2.5 minute')
    fism_cols = ['1300_02', '43000_09', '85550_13', '94400_18', 'DateTime']
    omni_cols = ['SYM_H index', 'AE', 'DateTime']
    
    fism = io.fism_flare(rcols=fism_cols)
    
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
    
    return feat_dat
    
    
    
def append_feat():
    pass


def load_feat():
    pass

