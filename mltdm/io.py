# -*- coding: utf-8 -*-

import os
import requests
import pathlib
import functools
import shutil

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import netCDF4 as nc

from scipy.io import readsav
from urllib.parse import urljoin
from urllib.request import urlretrieve


import mltdm


# TODO I think this should be it's own submodule
# TODO it could get quite big as we use more datasets

def dl_file(url, filename):
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))
    
    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)


def fism_flare(http_path: str=mltdm.c_dat['fism_flare'],
               data_path: str=mltdm.c_dat['data_dir'],
               rcols: list=None):
    
    flare_f = 'flare_bands.nc'
    http_f = os.path.join(http_path, flare_f)
    local_f = os.path.join(data_path, flare_f)
    
    if os.path.exists(local_f):
        dat = nc.Dataset(local_f)
    else:
        with requests.get(http_f,stream=True) as r:
            dat = nc.Dataset('in-mem-file', mode='r', memory=r.content)    
    
    wvlen = np.array(dat.variables['wavelength'][:]) # median wavelengt
    wvbwd = np.array(dat.variables['band_width'][:]) # wavelength bandwidth
    ddoy = np.array(dat.variables['date'][:]) # day of year
    sec = np.array(dat.variables['date_sec'][:]) # second of day
    ssi = np.array(dat.variables['ssi'][:]) # Solar Spectral Irridance
    
    #ssi is a 3D array
    # [doy,sec of day, wavelength]
    # need to unravel it as well 
    # as the time index

    # loop over all days and all seconds in 
    # days and generate a single dimension time index
    t_i = [pd.to_datetime(b,format='%Y%j')+pd.DateOffset(seconds=int(i)) 
           for b in ddoy for i in np.array(sec)]

    # unravel ssi 
    df = pd.DataFrame()
    df['DateTime'] = t_i
    for i in np.arange(0,wvlen.shape[0]):
        df[f'{round(wvlen[i]*1000)}_{i:02}'] = ssi[:,:,i].flatten()

    desc = {'data':'Solar Spectral Irridance Stan Bands, fism2 from LISIRD', 
            'units':'photons/cm^2/s',
            'url':'https://lasp.colorado.edu/lisird/',
            'wave lengths (nm)':wvlen, 'wavelength bandwidth (nm)':wvbwd,
            'columns titles':'wave legnth*1000'}
    
    # TODO check if values are actually in columns
    if rcols:
        df=df[rcols]
    
    return df, desc


def fism_flare_day(http_path: str=mltdm.c_dat['fism_flare'],
                   data_path: str=mltdm.c_dat['data_dir'],
                   sdate: str='2024-01-01', edate: str='2024-01-02',
                   rcols: list=None):
    # todo read in daily files to append to larger data file
    pre = 'FISM_bands_'
    suf = '_v02_01.sav'

    d_ser = pd.date_range(start=sdate, end=edate, freq='1D')
    fn_http = [urljoin(http_path,
                       f'{x.year}/{pre}{x.year}{x.day_of_year:03}{suf}')
               for x in d_ser]
    fn_local = [os.path.join(data_path,
                       f'{x.year}',
                       f'{pre}{x.year}{x.day_of_year:03}{suf}')
                for x in d_ser]
    
    df_r = pd.DataFrame()
    for fn_l, fn_w in zip(fn_local, fn_http):
        df_i = pd.DataFrame()
        
        if os.path.exists(fn_l):
            dat = readsav(fn_l)
        else:
            fn_t = os.path.join(data_path,'temp.sav')
            urlretrieve(fn_w, fn_t)
            dat = readsav(fn_t)
        
        wvlen = dat['wavelength']
        ssi = dat['ssi']
        
        df_i['DateTime'] = pd.to_datetime(dat['jd'], unit='D',origin='julian')
        for i in np.arange(0,wvlen.shape[0]):
            df_i[f'{round(wvlen[i]*1000)}_{i:02}'] = ssi[:,i]
        
        if df_r.empty:
            df_r = df_i.copy()
        else:
            df_r = pd.concat([df_r,df_i],ignore_index=True)
        
    if os.path.exists(os.path.join(data_path,'temp.sav')):
        os.remove(os.path.join(data_path,'temp.sav'))
    
    df_r['DateTime'] = df_r['DateTime'].dt.round('1min')
    # TODO check if values are actually in columns
    if rcols:
        df_r=df_r[rcols]
       
    return df_r
    

def omni(http_path: str=mltdm.c_dat['omni'],
         sdate: str='2003', edate: str='2004',
         rcols: list=None):
    
    sdate = pd.to_datetime(sdate)
    edate = pd.to_datetime(edate)
    #
    d_ser = pd.date_range(start=f'{sdate.year}', 
                          end=f'{edate.year}', freq='12MS')
    if d_ser.empty:
        d_ser = pd.date_range(start=f'{sdate.year}', 
                              end=f'{sdate.year+1}', freq='12MS')
        
    fn = [os.path.join(http_path,f'omni_5min{x.year}.asc') for x in d_ser]
    
    # TODO maybe at this to a _utils.py and import so it's less messy?
    # col names and bad data values
    dcols = {'Year':None, 'DOY':None, 'Hour':None, 'Minute':None,
            'IMF_id':99, 'SW_id':99, 'IMF_pt':999, 'SW_pt':999,
            'Per_int':999, 'Timeshift':999999, 'RMS_Timeshift':999999,
            'RMS_PhaseFrontNormal':99.99, 'Time_btwn_observations':999999,
            'B':9999.99, 'Bx_GSEGSM':9999.99, 'By_GSE':9999.99, 'Bz_GSE':9999.99,
            'By_GSM':9999.99, 'Bz_GSM':9999.99, 'RMS_SD_B':9999.99,
            'RMS_SD_field_vector':9999.99, 'Vsw':99999.9, 'Vx_GSE':99999.9,
            'Vy_GSE':99999.9, 'Vz_GSE':99999.9, 'Prho':999.99,'Tp':9999999.,
            'dynP':99.99, 'Esw':999.99, 'Beta':999.99, 'AlfvenMach':999.9,
            'X(s/c), GSE':9999.99, 'Y(s/c), GSE':9999.99, 'Z(s/c), GSE':9999.99,
            'BSN location, Xgse':9999.99, 'BSN location, Ygse':9999.99, 'BSN location, Zgse':9999.99,
            'AE':99999, 'AL':99999, 'AU':99999, 'SYM_D index':99999, 'SYM_H index':99999, 
            'ASY_D index':99999, 'ASY_H index':99999, 'PC index':999.99, 'Na_Np Ratio':9.999,
            'MagnetosonicMach':99.9, 'Goes Proton flux (>10 MeV)':99999.99,
            'Goes Proton flux (>30 MeV)':99999.99, 'Goes Proton flux (>60 MeV)':99999.99}
    
    # columns which hold dates
    dates = ['Year','DOY','Hour','Minute']
    
    # read in all the data
    if len(fn) > 1:
        om_dat = pd.concat((pd.read_csv(f,sep='\s+', engine='python', 
                                    names=list(dcols.keys()),header=None, 
                                    on_bad_lines='skip') 
                                    for f in fn), 
                           ignore_index=True)
    else:
        om_dat = pd.read_csv(fn[0],sep='\s+', engine='python', 
                                    names=list(dcols.keys()),header=None, 
                                    on_bad_lines='skip') 
    
    # replace missing or bad data with NaN's
    for k, kval in dcols.items():
        if kval is None:
            continue
        om_dat.loc[om_dat[k] == kval, k] = np.nan
    
    # create a datetime index
    dv = om_dat.loc[:,dates].astype('int32')
    dt = [f"{x['Year']:04}{x['DOY']:03}{x['Hour']:02}{x['Minute']:02}" 
          for index, x in dv.iterrows()]
    
    om_dat['DateTime'] = pd.to_datetime(dt, format='%Y%j%H%M')
    
    # TODO check if values are actually in columns
    if rcols:
        om_dat = om_dat[rcols]
    
    return om_dat
