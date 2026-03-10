# -*- coding: utf-8 -*-


import os
import socket

from urllib.parse import urljoin
from dateutil.relativedelta import relativedelta


import pandas as pd
import numpy as np



from EUVpy.tools import processIndices
from EUVpy.tools import spectralAnalysis
from EUVpy.NEUVAC import neuvac


import mltdm
from mltdm import io

# timeout for retrieving data from
# the web
socket.setdefaulttimeout(30)

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
    try:
        fism = io.fism_flare_day(sdate=sdate,edate=edate, rcols=fism_cols)
        omni = io.omni(sdate=sdate,edate=edate, rcols=omni_cols)
    except:
        # no data to append
        return feat_dat
    
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


def load_feat(sdate: str=None, edate: str=None, prelim: bool=False):
    
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
        feat_dat = feat_dat[gd].reset_index(drop=True)

    # load preliminary data
    p_dt = (pd.to_datetime(edate)-feat_dat['DateTime'].max()).total_seconds()
    if prelim and (p_dt > 3600 or feat_dat.empty):
        if feat_dat.empty:
            psdate = sdate
        else:
            psdate = feat_dat['DateTime'].max()
        pedate = edate

        prelim_feat = load_prelim_feat(sdate=psdate, edate=pedate)

        gd_date = prelim_feat['DateTime'] > sdate

        feat_dat = pd.concat([feat_dat,prelim_feat], ignore_index=True).reset_index(drop=True)

    # log the columns for predictions
    for i in log_cols:
        feat_dat[i] = np.log10(feat_dat[i])

    if 'index' in feat_dat.columns:
        feat_dat = feat_dat.drop(columns='index')    
        
    return feat_dat

def load_prelim_feat(sdate: str=None, edate: str=None):

    log_cols = ['1300_02', '43000_09', '85550_13', '94400_18']
    # kyoto base URL for DST data
    kyoto_base = 'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/'
    # load omni data first
    # drop rows where Sym-H is NaN (this is the variable typically used)
    omni_cols = ['SYM_H index', 'AE', 'DateTime']
    
    try:
        omni_data = omni(sdate=sdate, edate=edate)[omni_cols].dropna(subset=['SYM_H index'])

        gd_omni = (omni_data['DateTime'] >= sdate) & (omni_data['DateTime'] <= edate)
        omni_data = omni_data[gd_omni]
    except:
        omni_data = pd.DataFrame({'DateTime':[]})
    
    # check what data we are missing
    # if we are missing more than 1 hour
    # data fill it with data from Kyoto (which is more complete but less accurate)
    ky_df = -1
    if omni_data.empty or (pd.to_datetime(edate)-omni_data['DateTime'].max()).total_seconds() > 3600.:
        ke = pd.to_datetime(edate)
        if omni_data.empty:
            ks = pd.to_datetime(sdate)
        else:
            ks = io.omni_data['DateTime'].max()
        if ks.month == ke.month:
            ks = ks.replace(month=ke.month-1) 
        dt_ran = pd.date_range(start=ks,end=ke, freq='MS')    
        kyoto_urls = [f'{kyoto_base}/{dt.strftime("%Y%m")}/dst{dt.strftime("%y%m")}.for.request' for dt in dt_ran]
        df_l = [io.stream_kyoto_dst(url) for url in kyoto_urls]

        if len(df_l) == 1:
            ky_df = df_l[0]
        else:
            ky_df = pd.concat(df_l, ignore_index=True)

        ky_df['SYM_H index'] = ky_df['DST'].astype(float)
        ky_df['AE'] = np.nan

        gd_val = (ky_df['DST'] != '9999')

        gd_t = (ky_df['DateTime'] > ks) & (ky_df['DateTime'] <= ke)

        ky_df = ky_df[gd_t & gd_val]

        omni_data = pd.concat([omni_data, ky_df[['DateTime','SYM_H index','AE']]], ignore_index=True)

    # load modeled fism data from neuvac
    neu_sdate = pd.to_datetime(sdate)-relativedelta(days=1)
    neu_edate = pd.to_datetime(edate)+relativedelta(days=1)

    try:
        f107times, f107, _, f107b = processIndices.getCLSF107(neu_sdate.strftime("%Y-%m-%d"), 
                                                          neu_edate.strftime("%Y-%m-%d"), 
                                                          truncate=False, rewrite=True)
    except Exception as e:
        f107times, f107, _, f107b = processIndices.getCelestTrackInd(
                                        dateStart=neu_sdate.strftime("%Y-%m-%d"),
                                        dateEnd=neu_edate.strftime("%Y-%m-%d"))
    
    # get neuvac Soloman bands
    neuvacIrrSolomon, _, _, _ = neuvac.neuvacEUV(f107, f107b, bands='SOLOMON')

    fism_neuvac = pd.DataFrame( )
    fism_neuvac['DateTime'] = pd.to_datetime(f107times)
    fism_neuvac['1300_02'] = spectralAnalysis.spectralFlux(neuvacIrrSolomon[:,2],13.)/100.**2
    fism_neuvac['43000_09'] = spectralAnalysis.spectralFlux(neuvacIrrSolomon[:,9],430.)/100.**2
    fism_neuvac['85550_13'] = spectralAnalysis.spectralFlux(neuvacIrrSolomon[:,13],855.5)/100.**2
    fism_neuvac['94400_18'] = spectralAnalysis.spectralFlux(neuvacIrrSolomon[:,18],944.)/100.**2

    # merge the arrays
    feat_df = pd.merge_asof(omni_data.sort_values('DateTime'), 
                            fism_neuvac.sort_values('DateTime'), 
                            on='DateTime', direction='backward', 
                            tolerance=pd.Timedelta('36 hours'))

    return feat_df
