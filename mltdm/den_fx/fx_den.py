import pandas as pd
import numpy as np
import os



# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade

# loading models
from skops.io import load as sio_load


from mltdm.subsol import subsol
from mltdm.config import config
from mltdm.io import dl_file
from mltdm.den_fx.fx_feat import load_feat


# get varaibale from the configuration file
# this is loaded previously but it's loaded
# again here for transparency
c_dat = config().dat

# set up global variables for loading
# models so that models are only loaded
# once 
fx_noAE_mod = None
fx_mod = None

class fx_den():

    def __init__(self, n_lat: int=30, n_mlt: int=24,
                 input_f: str=None, dropAE: bool=True):
        
        if dropAE:
            self.feat_cols = ['1300_02','43000_09','85550_13','94400_18',
                              'SYM_H index']
            
            global fx_noAE_mod
            # only load the model once
            if fx_noAE_mod:
                self.rfmod = fx_noAE_mod
            else:
                fx_noAE_mod = self.load(c_dat['fxnoAE_m'])
                self.rfmod = fx_noAE_mod
            
            self.AE = False
        else:
            self.feat_cols = ['1300_02','43000_09','85550_13','94400_18',
                              'SYM_H index','AE']
            
            global fx_mod
            #only load the model once
            if fx_mod:
                self.rfmod = fx_mod
            else:
                fx_mod = self.load(c_dat['fx_m'])
                self.rfmod = fx_mod
        
            self.AE = True
            
            
        self.input_f = input_f
        
        self.n_lat = n_lat
        self.n_mlt = n_mlt
        self.setup_grid()

    def load(self, mod):
        
        file_path = os.path.join(c_dat['data_dir'],mod)
        
        if not os.path.exists(file_path):
            from urllib.parse import urljoin
            url_path = urljoin(c_dat['zenodo'],f'files/{mod}')
            
            print('Model file does not exist: {file_path}')
            print('Downloading model file from Zenodo to data directory')
            print('Downloading: {url_path}')
            
            dl_file(url_path,file_path)
            
        print(f'Loading {mod}, this will take a few minutes.')    
        return sio_load(os.path.join(c_dat['data_dir'], mod))
            

    def setup_grid(self):

        self.radii = np.linspace(0.999,0.001,self.n_lat) # start at 0.01 to avoid pole singularity
        self.satLat_N = np.arccos(self.radii)*180/np.pi
        self.satLat_S = -self.satLat_N

        self.mlt = np.linspace(0,24,self.n_mlt, endpoint=False) # don't include endpoint to avoid overlap
        self.theta = self.mlt*2*np.pi/24

        # setup two dimensional grid
        self.radMesh, self.thtMesh = np.meshgrid(self.radii, self.theta, indexing='ij')
        self.satlt_mesh_N, self.MLT_mesh = np.meshgrid(self.satLat_N, self.mlt, indexing='ij')
        self.satlt_mesh_S = -self.satlt_mesh_N

        # and flatten for use in appending to feature data
        self.satLat_N = self.satlt_mesh_N.flatten()
        self.satLat_S = self.satlt_mesh_S.flatten()
        self.satMLT = self.MLT_mesh.flatten()
        self.cos_theta = np.cos(self.thtMesh.flatten())
        self.sin_theta = np.sin(self.thtMesh.flatten())

        # create a grid in Geographic Coordinates
        # subsol.py is used to convert the longitude values
        # to a quasi-magnetic local time grid assuming 
        # for GEO map
        self.geoLat = np.linspace(-89.,89.,self.n_lat*2) # is satLat
        self.geoLon = np.linspace(-179.,179., self.n_mlt)
        self.geoLat_msh, self.geoLon_msh = np.meshgrid(self.geoLat, self.geoLon, indexing='ij')
    
    def make_grid(self, event, north: bool):
        if isinstance(event, pd.Series):
            # need to convert to dataFrame for grid generation
            evt = event.to_frame().transpose()
        else:
            evt = event

        grid = pd.DataFrame()
        for _, row in evt.iterrows():
            gs = pd.concat([row.to_frame().transpose()]*self.n_lat*self.n_mlt,ignore_index=True)

            gs['SatLat'] = self.satLat_N if north else self.satLat_S
            gs['cos_SatMagLT'] = self.cos_theta
            gs['sin_SatMagLT'] = self.sin_theta

            if grid.empty:
                grid = gs
            else:
                grid = pd.concat([grid,gs],ignore_index=True)
        
        return grid

    def make_geo_grid(self, event):
        if isinstance(event, pd.Series):
            # need to convert to dataFrame for grid generation
            evt = event.to_frame().transpose()
        else:
            evt = event

        grid = pd.DataFrame()
        for _, row in evt.iterrows():
            # factor of 2 for both hemispheres (latitudes)
            gridGEO = pd.concat([row.to_frame().transpose()]*2*self.n_lat*self.n_mlt, ignore_index=True)

            # GEO MLT grid requires subsolar point longitude
            _, evt_sbslon = subsol(row['DateTime'])
            GEO_MLT = ((self.geoLon_msh.flatten() - evt_sbslon) / 15. + 12)%24
            cos_MLT = np.cos(GEO_MLT*2*np.pi/24.)
            sin_MLT = np.sin(GEO_MLT*2*np.pi/24.)

            gridGEO["SatLat"] = self.geoLat_msh.flatten()
            gridGEO["cos_SatMagLT"] = cos_MLT
            gridGEO["sin_SatMagLT"] = sin_MLT
            gridGEO["sbslon"] = evt_sbslon

            if grid.empty:
                grid = gridGEO
            else:
                grid = pd.concat([grid,gridGEO],ignore_index=True)

        return grid

    def pred_den_mlt(self, sdate: str='2003-01-01 00:00:00', edate: str=None, hemisphere: str='North'):
        
        self.feat = load_feat(sdate=sdate,edate=edate)[self.feat_cols+['DateTime']]
        
        if edate:
            ev_id = (self.feat['DateTime'] >= sdate) & (self.feat['DateTime'] <= edate)
            event = self.feat.loc[ev_id, self.feat_cols+['DateTime']].copy()
        else:
            ev_id = self.feat.set_index('DateTime').index.get_loc(sdate) 
            event = self.feat.loc[ev_id, self.feat_cols+['DateTime']].copy().to_frame().transpose()
        
        north = True if hemisphere.upper() == 'NORTH' else False

        grid = self.make_grid(event,north=north)

        self.north = north
        
        grid['400 km den'] = self.rfmod.predict(grid.drop(columns='DateTime'))*(10**-12)
        
        self.den = grid[['DateTime','400 km den']]

        return grid
    
    def pred_den_geo(self, sdate: str='2003-01-01 00:00:00', edate: str=None):
        
        self.feat = load_feat(sdate=sdate,edate=edate)[self.feat_cols+['DateTime']]
        
        if edate:
            ev_id = (self.feat['DateTime'] >= sdate) & (self.feat['DateTime'] <= edate)
            event = self.feat.loc[ev_id, self.feat_cols+['DateTime']].copy()
        else:
            ev_id = self.feat.set_index('DateTime').index.get_loc(sdate) 
            event = self.feat.loc[ev_id, self.feat_cols+['DateTime']].copy().to_frame().transpose()

        grid = self.make_geo_grid(event)

        grid['400 km den'] = self.rfmod.predict(grid.drop(columns=['DateTime','sbslon']))*(10**-12)

        return grid

    def pred_den_orb(self, spos: np.ndarray, 
                     tol: pd.Timedelta=pd.Timedelta('2.5 minute')):
        
        # spos should have datetime
        # sat lat and sat mlt to derive densities
        # the dataframe is then combined with the feature
        # dataframe to derive densities
        
        # load feature data
        feat = load_feat(sdate=spos['DateTime'].min(), 
                         edate=spos['DateTime'].max())
        
        # add cos and sin to the position array
        spos["cos_SatMagLT"] = np.cos(spos['SatMagLT']*2*np.pi/24.)
        spos["sin_SatMagLT"] = np.sin(spos['SatMagLT']*2*np.pi/24.)
        
        spos = pd.merge_asof(left=spos,right=feat,
                                 left_on='DateTime',right_on='DateTime',
                                 direction='nearest',tolerance=tol)
        
        pred_cols = self.feat_cols + ["SatLat","cos_SatMagLT","sin_SatMagLT"]
        
        spos['400 km den'] = self.rfmod.predict(
            spos[pred_cols])*(10**-12)
        
        return spos
    
    def plot_dpolar(self, den, ax=None, fig=None, date=None, theta_offset: float =-0.5*np.pi, **kwargs):
        # check if an axis and figure are passed
        # if not grab the current ones
        if ax is None:
            # get current axis if none is passed
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        if date:
            date = pd.to_datetime(date)
            den_p = den.loc[den['DateTime'] == date,'400 km den'].to_numpy()
        else:
            den_p = den['400 km den'].to_numpy()[0:self.n_lat*self.n_mlt]
            date = den.loc[0,'DateTime']

        # check if the current axis is polar
        # if not change it to polar by deleting
        # it and adding another
        if ax.name != 'polar':
            rows, cols, start, _ = ax.get_subplotspec().get_geometry()
            ax.remove()
            ax = fig.add_subplot(rows, cols, start+1, projection='polar')
        
        im = ax.pcolormesh(self.thtMesh, self.satlt_mesh_N, 
                           den_p.reshape(self.n_lat,self.n_mlt),
                           **kwargs)

        if self.north:
            ax.set_rlim(bottom=90, top=0)
        else:
            ax.set_rlim(bottom=-90, top=0)
        
        ax.set_theta_offset(theta_offset)
        fig.colorbar(im, ax=ax)

        return im

    def plot_dgeo(self, den, ax=None, fig=None, date=None, **kwargs):
        
        if ax is None:
            # get current axis if none is passed
            ax = plt.gca()
        if fig is None:
            fig = plt.gcf()

        if date:
            date = pd.to_datetime(date)
            den_p = den.loc[den['DateTime'] == date,'400 km den'].to_numpy()
            evt_sbslon = den.loc[den['DateTime'] == date,'sbslon'].mean()
        else:
            den_p = den['400 km den'].to_numpy()[0:2*self.n_lat*self.n_mlt]
            date = den.loc[0,'DateTime']
            evt_sbslon = den.loc[0,'sbslon'] 

        ax.coastlines()
        ax.gridlines(draw_labels=True)
        im = ax.pcolormesh(self.geoLon_msh, self.geoLat_msh, 
                           den_p.reshape(self.n_lat*2, self.n_mlt),
                           **kwargs)
        ax.set_title(f"Density at: {date}", size=14)
        ax.add_feature(Nightshade(date, alpha=0.25))
        ax.axvline(evt_sbslon, ls='--',lw=2.5,color='#3fd7f3',
                   zorder=max([_.zorder for _ in ax.get_children()])+1)
        

        fig.colorbar(im, ax=ax)

        return im

    def plot_dorbit():
        pass























