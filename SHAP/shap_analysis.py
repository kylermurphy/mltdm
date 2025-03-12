"""
General classes for shapley analysis of
Neutral Density Random Forest Model (Halford/Murphy et al.)
as published in Murphy et al. 2025 (https://doi.org/10.1029/2024SW003928)

author : C. Bard
for : AIMFAHR
(also see K. Murphy Python notebook for training model)

requires:
numpy
matplotlib
pandas
fasttreeshap
cartopy

sklearn is required to train random forest (RF) model (see KM's notebook);
following classes assume that RF model has been
saved to pickle with file option 'wb', e.g.

> with open(fname, 'wb') as f:
>     pickle.dump(forest, f)

and that the train-test split of the original satdrag_database*hdf5 files
(as performed in KM's notebook) has been saved to hdf5 in the file
"FI_GEO_RFdat_AIMFAHR_data.h5"

"""
from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import pandas as pd
import fasttreeshap as fts
import time as pytime
import numpy.random as rand
import numpy as np
import numpy.typing as npt
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt

#default seed
gen = rand.default_rng(693993)

fgeo_col = ['1300_02', '43000_09', '85550_13', '94400_18', 'SYM_H index', 'AE', 'SatLat', 'cos_SatMagLT', 'sin_SatMagLT']
_fgeo_col_dict = {'1300_02':0, '43000_09':1, '85550_13':2, '94400_18':3, 'SYM_H index':4, 'AE':5, 'SatLat':6, 'cos_SatMagLT':7, 'sin_SatMagLT':8}
# other options, but RF_model needs to be run for them (to have data and forest)
# si_col = ['F10', 'F81', 'S10', 'S81c', 'M10', 'M81c', 'Y10', 'Y81c', 'SatLat', 'cos_SatMagLT', 'sin_SatMagLT']
# fi_col = ['1300_02', '43000_09', '85550_13', '94400_18', 'SatLat', 'cos_SatMagLT', 'sin_SatMagLT']

def load_tools(fname, n_jobs: int = 4):
    """
    Loads forest from file `fname`
    and instantiates Shap Tree Explainer
    Output: tuple (forest, explainer)

    Assumes forest was saved to pickle

    Parameter 'n_jobs' is used for fasttreeshap (how many cores does your CPU have?)
    """
    rf = pickle.load(open(fname, 'rb'))
    return (rf, fts.TreeExplainer(rf, n_jobs = n_jobs))


def _load_datafile(option: str = "test_d"):
    """
    Loads forest training/test data from CB's checkpointing of
    the train/test split in KM's Notebook

    option : str
        1. "test_d" : Grace B test data
        2. "train_d" : Grace B training data
        3. "oos_d" : Grace A
        4. "oos2_d" : CHAMP

    """

    return pd.read_hdf("FI_GEO_RFdat_AIMFAHR_data.h5", option)


def load_data(storm: bool = None, option: str = "test_d", all_cols: bool = False):
    """
    Loads pandas data array from file in `load_datafile`

    `storm` : bool | None
        1. None : load full data
        2. True : load storm data ("storm" == 1)
        3. False : load nonstorm data ("storm" == -1)

    option : str
        1. "test_d" : Grace B test data
        2. "train_d" : Grace B training data
        3. "oos_d" : Grace A
        4. "oos2_d" : CHAMP

    all_cols : bool
        If True, load all data columns
        If False, load just fgeo_columns used in best RF model
    """

    dat = _load_datafile(option)

    if storm is not None:
        dat = dat[dat['storm'] == ((storm == True) - (storm == False))]

    return dat if all_cols else dat[fgeo_col]

#---------------------------
# following function 'subsol' is from Apexpy: https://github.com/aburrell/apexpy/
# taken from apexpy.helpers.subsol on 2025/02/05
# and used under MIT license (Copyright (c) 2015 Christer van der Meeren)

def subsol(datetime):
    """Finds subsolar geocentric latitude and longitude.

    Parameters
    ----------
    datetime : :class:`datetime.datetime` or :class:`numpy.ndarray[datetime64]`
        Date and time in UTC (naive objects are treated as UTC)

    Returns
    -------
    sbsllat : float
        Latitude of subsolar point
    sbsllon : float
        Longitude of subsolar point

    Notes
    -----
    Based on formulas in Astronomical Almanac for the year 1996, p. C24.
    (U.S. Government Printing Office, 1994). Usable for years 1601-2100,
    inclusive. According to the Almanac, results are good to at least 0.01
    degree latitude and 0.025 degrees longitude between years 1950 and 2050.
    Accuracy for other years has not been tested. Every day is assumed to have
    exactly 86400 seconds; thus leap seconds that sometimes occur on December
    31 are ignored (their effect is below the accuracy threshold of the
    algorithm).

    After Fortran code by A. D. Richmond, NCAR. Translated from IDL
    by K. Laundal.

    """
    # Convert to year, day of year and seconds since midnight
    if isinstance(datetime, dt.datetime):
        year = np.asanyarray([datetime.year])
        doy = np.asanyarray([datetime.timetuple().tm_yday])
        ut = np.asanyarray([datetime.hour * 3600 + datetime.minute * 60
                            + datetime.second + datetime.microsecond / 1.0e6])
    elif isinstance(datetime, np.ndarray):
        # This conversion works for datetime of wrong precision or unit epoch
        times = datetime.astype('datetime64[us]')
        year_floor = times.astype('datetime64[Y]')
        day_floor = times.astype('datetime64[D]')
        year = year_floor.astype(int) + 1970
        doy = (day_floor - year_floor).astype(int) + 1
        ut = (times.astype('datetime64[us]') - day_floor).astype(float)
        ut /= 1e6
    else:
        raise ValueError("input must be datetime.datetime or numpy array")

    if not (np.all(1601 <= year) and np.all(year <= 2100)):
        raise ValueError('Year must be in [1601, 2100]')

    yr = year - 2000

    nleap = np.floor((year - 1601.0) / 4.0).astype(int)
    nleap -= 99
    mask_1900 = year <= 1900
    if np.any(mask_1900):
        ncent = np.floor((year[mask_1900] - 1601.0) / 100.0).astype(int)
        ncent = 3 - ncent
        nleap[mask_1900] = nleap[mask_1900] + ncent

    l0 = -79.549 + (-0.238699 * (yr - 4.0 * nleap) + 3.08514e-2 * nleap)
    g0 = -2.472 + (-0.2558905 * (yr - 4.0 * nleap) - 3.79617e-2 * nleap)

    # Days (including fraction) since 12 UT on January 1 of IYR:
    df = (ut / 86400.0 - 1.5) + doy

    # Mean longitude of Sun:
    lmean = l0 + 0.9856474 * df

    # Mean anomaly in radians:
    grad = np.radians(g0 + 0.9856003 * df)

    # Ecliptic longitude:
    lmrad = np.radians(lmean + 1.915 * np.sin(grad)
                       + 0.020 * np.sin(2.0 * grad))
    sinlm = np.sin(lmrad)

    # Obliquity of ecliptic in radians:
    epsrad = np.radians(23.439 - 4e-7 * (df + 365 * yr + nleap))

    # Right ascension:
    alpha = np.degrees(np.arctan2(np.cos(epsrad) * sinlm, np.cos(lmrad)))

    # Declination, which is also the subsolar latitude:
    sslat = np.degrees(np.arcsin(np.sin(epsrad) * sinlm))

    # Equation of time (degrees):
    etdeg = lmean - alpha
    nrot = np.round(etdeg / 360.0)
    etdeg = etdeg - 360.0 * nrot

    # Subsolar longitude calculation. Earth rotates one degree every 240 s.
    sslon = 180.0 - (ut / 240.0 + etdeg)
    nrot = np.round(sslon / 360.0)
    sslon = sslon - 360.0 * nrot

    # Return a single value from the output if the input was a single value
    if isinstance(datetime, dt.datetime):
        return sslat[0], sslon[0]
    return sslat, sslon

#----------------------------
# resume CB's code
class PolarBear():

    def __init__(self, n_lat: int = 10, n_mlt: int = 20):
        """
        Parameters
        ----------
        n_lat : int
            Number of points for sat Lat between (0,90)
        n_mlt : int
            Number of points for MLT between (0,24)
        """
        self.n_lat = n_lat
        self.n_mlt = n_mlt
        self.setup_grid()

    def setup_grid(self):
        """
        Sets up needed arrays
        """
        self.radii = np.linspace(0.999,0.001,self.n_lat) # start at 0.01 to avoid pole singularity
        self.satLat_N = np.arccos(self.radii)*180/np.pi
        self.satLat_S = -self.satLat_N

        self.theta = np.linspace(0,2.*np.pi,self.n_mlt)

        self.MLT_all = 24.*self.theta/(2*np.pi)

        self.radMesh, self.thtMesh = np.meshgrid(self.radii, self.theta, indexing='ij')
        self.satlt_mesh_N, self.MLT_mesh = np.meshgrid(self.satLat_N, self.MLT_all, indexing='ij')
        self.satlt_mesh_S = -self.satlt_mesh_N

        self.satLat_N = self.satlt_mesh_N.flatten()
        self.satLat_S = self.satlt_mesh_S.flatten()
        self.cos_theta = np.cos(self.thtMesh.flatten())
        self.sin_theta = np.sin(self.thtMesh.flatten())

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

        evt = evt[fgeo_col]

        grid = pd.concat([evt]*self.n_lat*self.n_mlt, ignore_index=True)
        satlt_mesh = self.satlt_mesh_N if north else self.satlt_mesh_S

        grid["SatLat"] = self.satLat_N if north else self.satLat_S
        grid["cos_SatMagLT"] = self.cos_theta
        grid["sin_SatMagLT"] = self.sin_theta

        return grid, satlt_mesh

    def make_geo_grid(self, event):
        evt_date = event['DateTime']
        if isinstance(event, pd.Series):
            # need to convert to dataFrame for grid generation
            evt = event.to_frame().transpose()
        else:
            evt = event

        evt = evt[fgeo_col]

        # factor of 2 for both hemispheres (latitudes)
        gridGEO = pd.concat([evt]*2*self.n_lat*self.n_mlt, ignore_index=True)

        # GEO MLT grid requires subsolar point longitude
        _, evt_sbslon = subsol(evt_date)
        GEO_MLT = ((self.geoLon_msh.flatten() - evt_sbslon) / 15. + 12)%24
        cos_MLT = np.cos(GEO_MLT*2*np.pi/24.)
        sin_MLT = np.sin(GEO_MLT*2*np.pi/24.)

        gridGEO["SatLat"] = self.geoLat_msh.flatten()
        gridGEO["cos_SatMagLT"] = cos_MLT
        gridGEO["sin_SatMagLT"] = sin_MLT

        return gridGEO

    def plot_density(self, ax, rf, event, grid, satlat_mesh, north: bool, vr=[0.5,6.5]):
        """Dial plot of density prediction

        Parameters
        ----------
        fig: plt.Figure
            Figure to plot on
        ax : plt.Axes
            Axes to plot on (must have projection = "polar")
        rf : Random Forest
            Random Forest model (from scikit-learn)
        event : Pandas Series or DataFrame
            single event to pass to RF model
        grid : Pandas DataFrame
            from 'self.make_grid()', Holds grid of SatLat, cosMLT, sinMLT
        satlat_mesh : np.NDArray
            Holds SatLat mesh
        north : bool
            Plot north hemisphere? (or south?)
        prefix : str, optional
            File prefix for saving density figure, by default ""
        vr : list, optional
            Min/Max for plotting, by default [0.5,6.5]

        Returns
        -------
        img :
            Handle to Canvas on ax

        Notes
        -----
        'ax' must be instantiated with keyword arg: projection="polar"
        """
        vmin, vmax = vr

        if isinstance(event, pd.Series):
            # need to convert to dataFrame for prediction
            point_satLat = event["SatLat"]
            point_theta = np.arctan2(event["sin_SatMagLT"], event["cos_SatMagLT"])
            evt = event.to_frame().transpose()
        else:
            evt = event
            point_satLat = event["SatLat"].item()
            point_theta = np.arctan2(event["sin_SatMagLT"].item(), event["cos_SatMagLT"].item())

        denPred = rf.predict(evt[fgeo_col])[0]
        pred = rf.predict(grid)

        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        if north:
            ax.set_rlim(bottom=90, top=0)
        else:
            ax.set_rlim(bottom=-90, top=0)

        img = ax.pcolormesh(self.thtMesh, satlat_mesh, pred.reshape(self.n_lat,self.n_mlt), vmin=vmin, vmax=vmax, cmap=plt.cm.inferno)

        ax.set_title(f"Predicted density at point: {denPred:.2f}")
        if not(hasattr(self, "dial_lbls")):
            self.dial_lbls = [24./360.*float(item.get_text()[:-1]) for item in ax.get_xticklabels()]
        ax.set_xticklabels(self.dial_lbls)
        if (north == (point_satLat >= 0.)):
            ax.scatter(point_theta, point_satLat, s=10, c="red")

        return img

    def plot_shap(self, ax, event, shap_val, shap_name, satlat_mesh, north: bool, vr=[]):
        if len(vr):
            min_shap, max_shap = vr
        else:
            min_shap = shap_val.min()
            max_shap = shap_val.max()

        if isinstance(event, pd.Series):
            point_satLat = event["SatLat"]
            point_theta = np.arctan2(event["sin_SatMagLT"], event["cos_SatMagLT"])
        else:
            point_satLat = event["SatLat"].item()
            point_theta = np.arctan2(event["sin_SatMagLT"].item(), event["cos_SatMagLT"].item())

        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        if north:
            ax.set_rlim(bottom=90, top=0)
        else:
            ax.set_rlim(bottom=-90, top=0)

        mn = shap_val.mean()
        img = ax.pcolormesh(self.thtMesh, satlat_mesh, shap_val.reshape(self.n_lat,self.n_mlt), vmin=min_shap, vmax=max_shap, cmap=plt.cm.inferno)
        ax.scatter(point_theta, point_satLat, s=10, c="cyan")
        ax.set_xticklabels(self.dial_lbls)
        ax.set_title(f"Factor: {shap_name}, Mean: {mn:.2f}")

        return img

    def plot_geo_den(self, ax, rf, event, grid, vr=[0.5,6.5]):
        vmin_den,vmax_den = vr

        if isinstance(event, pd.Series):
            point_satLat = event["SatLat"]
            point_theta = np.arctan2(event["sin_SatMagLT"], event["cos_SatMagLT"])
            evt_date = event['DateTime']
        else:
            point_satLat = event["SatLat"].item()
            point_theta = np.arctan2(event["sin_SatMagLT"].item(), event["cos_SatMagLT"].item())
            evt_date = event['DateTime'].item()

        point_MLT = 24.*point_theta/(2*np.pi)

        # Assuming MLT is relative to GEO subsolar longitude
        date = evt_date.strftime("%Y-%m-%d %H:%M:%S")
        _, evt_sbslon = subsol(evt_date)
        point_satLon = (point_MLT - 12)*15. + evt_sbslon

        predGEO = rf.predict(grid)

        ax.coastlines()
        ax.gridlines(draw_labels=True)
        img0 = ax.pcolormesh(self.geoLon_msh, self.geoLat_msh, predGEO.reshape(self.n_lat*2, self.n_mlt), vmin=vmin_den, vmax=vmax_den, cmap=plt.cm.inferno, alpha=0.95)
        ax.scatter(point_satLon, point_satLat, s=10, c="red")
        ax.set_title(f"Density at: {date}", size=14)
        #ax.axvline(evt_sbslon, ls='-.',lw=1.5,color='#FFC300')
        ax.axvline(evt_sbslon, ls='-.',lw=1.5,color='#3fd7f3')
        return img0

    def plot_geo_night(self, ax, event):
        if isinstance(event, pd.Series):
            evt_date = event['DateTime']
        else:
            evt_date = event['DateTime'].item()
        ax.coastlines()
        ax.gridlines(draw_labels=False)
        ax.add_feature(Nightshade(evt_date, alpha=0.5))

    def gen_plots(self, event, rf, explainer, out_prefix: str = '', write_event: bool = False):
        """
        Generate polar plots for single data event of:
        1. Predicted density
        2-10. Shapley values for RF inputs
        11. Beeswarm plots of Shapley values for full polar grid

        Parameters
        ----------
        event : pandas Series
            Single event to expand over hemisphere
        rf : RandomForestRegressor
            RF that was trained to predict neutral density
        explainer : fasttreeshap.TreeExplainer
            Explainer for 'rf'
        out_prefix : str
            Prefix for output filenames. May include directory name.
        write_event : bool
            Write out Event data to text file?

        Notes
        -----
        Input 'event' must be a single event (pandas Series), e.g.:
        > all_dat = load_data()
        > event = all_data.iloc(4800)
        > polarBear.gen_plots(event, rf, explainer, "./outdir/Event4800")

        This function will run the explainer on the polar grid for 'event', so it
        has to calculate shapley values for NLAT*NMLT points in the grid!
        This may take several minutes.
        """

        point_satLat = event["SatLat"]

        if write_event:
            event.to_frame().transpose().to_csv(out_prefix+'_data.txt', sep=' ', index=False)

        north_qq = point_satLat >= 0.

        grid, satlt_mesh = self.make_grid(event, north_qq)

        figP = plt.Figure()
        axP = figP.add_subplot(111, projection="polar")

        img = self.plot_density(axP, rf, event, grid, satlt_mesh, north_qq)

        figP.colorbar(img)
        figP.tight_layout()
        figP.savefig(out_prefix+'_prediction.png')

        # start
        print("start event mapping")
        tfirst = pytime.perf_counter()
        shap_vals = explainer(grid)
        tnow = pytime.perf_counter()
        print(f"Time taken: {tnow - tfirst}\nContinuing with plots:\n")

        fts.plots.beeswarm(shap_vals, show=False)
        f = plt.gcf()
        f.savefig(out_prefix+"_beeswarm.png", bbox_inches="tight")
        f.clf()

        for i,nme in enumerate(event[fgeo_col].index):
            figP.clf()
            ax = figP.add_subplot(111, projection="polar")
            img = self.plot_shap(ax, event, shap_vals.values[:,i], nme, satlt_mesh, north_qq)
            figP.colorbar(img)
            figP.tight_layout()
            figP.savefig(out_prefix+f"_{nme[:5]}_shap.png")
            del ax

    def movie_plots(self, event, rf, explainer, plot_shap_for = [], out_prefix: str = '', write_event: bool = False, vr: dict = {}, north: bool = None):
        """
        Generate polar plots for single timestep (event), to be used as frames
        in a movie.

        Parameters
        ----------
        event : pandas Series
            Single event to expand over hemisphere
            Must be full data (not just fgeo_col)
        rf : RandomForestRegressor
            RF that was trained to predict neutral density
        explainer : fasttreeshap.TreeExplainer
            Explainer for 'rf'
        out_prefix : str
            Prefix for output filenames. May include directory name.
        write_event : bool
            Write out Event data to text file?
        vr : dict
            Variable ranges, in format {"name": [vmin, vmax]}
            Names must be a valid column name in `module.fgeo_col`
            or "density". Names not in vr will set range to
            min/max of shap value
        plot_shap_for : List[str]
            By default []
            Plot shapley values only for features listed
            Names must be a valid column name in `module.fgeo_col`
            If empty list, do not plot shap maps
            If None, plot shap maps for all features
        north : bool
            By default None
            Plot only northern hemisphere?
            if None, plot whatever hemisphere the data is located on

        Notes
        -----
        module.fgeo_col = ['1300_02', '43000_09', '85550_13', '94400_18',
        'SYM_H index', 'AE', 'SatLat', 'cos_SatMagLT', 'sin_SatMagLT']

        Input 'event' must be a single event (pandas Series), e.g.:
        > all_dat = load_data()
        > event = all_data.iloc(4800)
        > polarBear.gen_plots(event, rf, explainer, "./outdir/Event4800")

        This function will run the explainer on the polar grid for 'event', so it
        has to calculate shapley values for NLAT*NMLT points in the grid!
        This may take several minutes per event.

        NOTE: Default behavior is to not plot shap maps! If you want to plot maps
        for all shap features, pass in "plot_shap_for = None".
        """

        point_satLat = event["SatLat"]
        date = event['DateTime'].strftime("%Y-%m-%d %H:%M:%S")

        if write_event:
            event.to_frame().transpose().to_csv(out_prefix+'_data.txt', sep=' ', index=False)

        north_qq = point_satLat >= 0.
        grid, satlt_mesh = self.make_grid(event, north_qq)

        figP = plt.Figure()
        axP = figP.add_subplot(111, projection="polar")

        vr_den = [0.5,6.5]
        if "density" in vr:
            vr_den = vr["density"]

        img = self.plot_density(axP, rf, event, grid, satlt_mesh, north_qq, vr=vr_den)

        axP.text(0.06, 0.06, f"{date}", transform=figP.transFigure, size=13)
        figP.colorbar(img)
        figP.savefig(out_prefix+'_density.png')

        # start shap
        if plot_shap_for is None:
            shp_list = fgeo_col
        else:
            shp_list = plot_shap_for

        if len(shp_list):
            print("start event mapping")
            tfirst = pytime.perf_counter()
            shap_vals = explainer(grid)
            tnow = pytime.perf_counter()
            print(f"Time taken: {tnow - tfirst}\nContinuing with plots:\n")

        # beeswarm only for full list of features
        if plot_shap_for is None:
            fts.plots.beeswarm(shap_vals, show=False)
            f = plt.gcf()
            f.savefig(out_prefix+"_beeswarm.png", bbox_inches="tight")
            f.clf()


        if len(shp_list):
            for nme in shp_list:
                i = _fgeo_col_dict[nme]
                figP.clf()
                shp = shap_vals.values[:,i]

                if nme in vr:
                    min_shap = vr[nme][0]
                    max_shap = vr[nme][1]
                else:
                    min_shap = shp.min()
                    max_shap = shp.max()

                ax = figP.add_subplot(111, projection="polar")

                img = self.plot_shap(ax, event, shp, nme, satlt_mesh, north_qq, [min_shap, max_shap])
                figP.colorbar(img)

                mn = shap_vals.values[:,i].mean()
                ax.set_title(f"Factor: {nme}, Mean: {mn:.2f}")
                ax.text(0.06, 0.06, f"{date}", transform=figP.transFigure, size=13)

                figP.savefig(out_prefix+f"_{nme[:5]}_shap.png")
                del ax

    # no shap values for this (yet?)
    def full_den_movie_plot(self, event, rf, out_prefix: str = '', write_event: bool = False, vr=[0.5,6.5]):
        if write_event:
            event.to_frame().transpose().to_csv(out_prefix+'_data.txt', sep=' ', index=False)

        # set up grid(s)
        gridN, satlat_mesh_N = self.make_grid(event, north=True)
        gridS, satlat_mesh_S = self.make_grid(event, False)

        gridGEO = self.make_geo_grid(event)

        fig = plt.Figure(figsize=(15,10))
        gs = fig.add_gridspec(3,2, width_ratios=[4,2], height_ratios=[10,10,1])
        ax0 = fig.add_subplot(gs[0,0], projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(gs[1,0], projection=ccrs.PlateCarree())
        ax1 = fig.add_subplot(gs[0,1], projection="polar")
        ax3 = fig.add_subplot(gs[1,1], projection="polar")
        ax4 = fig.add_subplot(gs[2,:])

        # polar density plots
        img1 = self.plot_density(ax1, rf, event, gridN, satlat_mesh_N, north=True, vr=vr)
        _ = self.plot_density(ax3, rf, event, gridS, satlat_mesh_S, north=False, vr=vr)

        ax1.set_title("North", size=14, pad = 30)
        ax3.set_title("South", size=14, pad = 30)

        _ = self.plot_geo_den(ax0, rf, event, gridGEO, vr)
        self.plot_geo_night(ax2, event)

        fig.colorbar(img1, cax=ax4, orientation="horizontal")
        fig.tight_layout()
        fig.savefig(out_prefix+'_density.png')

        del fig, ax0, ax1, ax2, ax3, ax4, gs
