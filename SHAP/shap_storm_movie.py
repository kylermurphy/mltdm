"""
using sorted = test_fgeo.sort_values("DateTime")

and iloc is relative to sorted data

iloc 49817 at 2004-08-07 00:00:00
iloc 49846 (start storm) at 2004-08-07 06:05:00

iloc 50080 (end main phase) at 2004-08-09 20:55:00

iloc 50440 (end recovery) at 2004-08-14 02:55:00
iloc 50464 at 2004-08-14 08:50:00
"""


import shap_analysis as shpa

if __name__ == "__main__":
    outdir = "./single_storm/"
    ist = 49817
    iend = 50465

    rf, explain = shpa.load_tools("FI_GEO_RFdat_AIMFAHR_forest.pkl", n_jobs=8)

    full_data = shpa.load_data(all_cols=True)
    full_data = full_data.sort_values("DateTime") # required for given indices to work
    
    pb = shpa.PolarBear(n_lat = 10, n_mlt = 20)

    for i in range(ist, iend+1, 2):
        pb.movie_plots(full_data.iloc[i], rf, explain, out_prefix=outdir+f"step_{i-ist:04}", write_event=False, plot_shap_for=["cos_SatMagLT", "sin_SatMagLT", "SatLat", "SYM_H index"], vr={"cos_SatMagLT":[-0.6,0.6], "sin_SatMagLT":[-0.6,0.6], "SatLat":[-0.5,0.5], "SYM_H index":[-0.5,1.]}, north=True)
