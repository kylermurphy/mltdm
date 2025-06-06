# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np

from urllib.parse import urljoin


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
    
    # the netcdf file has large gaps after 2022-01-01
    # hard code edate so only continous data is loaded
    sdate = fism['DateTime'].min()+pd.offsets.YearBegin(-1)
    edate = pd.to_datetime('2022-01-01')
    
    omni = io.omni(sdate=sdate,edate=edate, rcols=omni_cols)
    omni = omni[(omni['DateTime'] >= fism['DateTime'].min()) &
                (omni['DateTime'] < edate)]
    
    
    omni = omni.set_index('DateTime')
    fism = fism.set_index('DateTime')
    
    feat_dat = pd.merge_asof(left=fism,right=omni,
                             right_index=True,left_index=True,
                             direction='nearest',tolerance=tol)
    
    out_f = os.path.join(mltdm.c_dat['data_dir'],'fx_den_feat.hdf')
    
    feat_dat = feat_dat.reset_index()
    feat_dat.to_hdf(out_f,key='fx_den_feat',
                    complevel=2,format='table', data_columns=['DateTime'])
     
    
def append_feat(edate: str=None):
    
    feat_f = os.path.join(mltdm.c_dat['data_dir'],'fx_den_feat.hdf')
    feat_dat = pd.read_hdf(feat_f)
    
    tol = pd.Timedelta('2.5 minute')
    fism_cols = ['1300_02', '43000_09', '85550_13', '94400_18', 'DateTime']
    omni_cols = ['SYM_H index', 'AE', 'DateTime']
    
    sdate = feat_dat['DateTime'].max()
    sdate = f'{sdate.year:04}-{sdate.month:02}-{sdate.day:02}'
    
    # read in the new data
    fism = io.fism_flare_day(sdate=sdate,edate=edate, rcols=fism_cols)
    omni = io.omni(sdate=sdate,edate=edate, rcols=omni_cols)
    omni = omni[(omni['DateTime'] >= fism['DateTime'].min()) &
                (omni['DateTime'] <= fism['DateTime'].max())]
    
    omni = omni.set_index('DateTime')
    fism = fism.set_index('DateTime')
    
    # merge the omni and FISM data
    feat_app = pd.merge_asof(left=fism,right=omni,
                             right_index=True,left_index=True,
                             direction='nearest',tolerance=tol)
    
    feat_app = feat_app.reset_index()
    
    # append the new data to original data
    feat_new = pd.concat([feat_dat,feat_app],ignore_index=True)
    feat_new = feat_new.drop_duplicates(subset='DateTime')
    feat_new = feat_new.reset_index(drop=True)
    
    out_f = os.path.join(mltdm.c_dat['data_dir'],'fx_den_feat.hdf')
    feat_new.to_hdf(out_f,key='fx_den_feat',
                    complevel=2,format='table', data_columns=['DateTime'])
    
    return feat_new


def load_feat(sdate: str=None, edate: str=None):
    
    log_cols = ['1300_02', '43000_09', '85550_13', '94400_18']
    
    feat_f = os.path.join(mltdm.c_dat['data_dir'],'fx_den_feat.hdf')
    
    # if the feature data doesn't exist
    # download it
    # if it can't be downloaded create it
    if not os.path.exists(feat_f):
        try:
           z_url = urljoin(mltdm.c_dat['zenodo'],'files/') 
           f_url = urljoin(z_url,'fx_den_feat.hdf')
           io.dl_file(f_url,feat_f)
        except:
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
    
    if not feat_dat.empty:
        dt = pd.to_datetime(edate)-feat_dat['DateTime'].max()
        dt = dt.total_seconds()
    else:
        dt = -1
         
    if feat_dat.empty or dt > 86399:
        feat_dat = append_feat(edate=edate)
        gd = (feat_dat['DateTime']>=sdate) & (feat_dat['DateTime']<=edate)
        feat_dat = feat_dat[gd]
    
    # log the columns for predictions
    for i in log_cols:
        feat_dat[i] = np.log10(feat_dat[i])
        
    return feat_dat


