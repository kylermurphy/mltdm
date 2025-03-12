"""
Similar to shap storm movie, except makes
quad plot of GEO and POLAR density predictions
"""


import shap_analysis as shpa

if __name__ == "__main__":
    outdir = "./single_storm_QUAD/"
    ist = 49817
    iend = 50465

    rf, explain = shpa.load_tools("FI_GEO_RFdat_AIMFAHR_forest.pkl", n_jobs=8)

    full_data = shpa.load_data(all_cols=True)
    full_data = full_data.sort_values("DateTime") # required for given indices to work
    
    pb = shpa.PolarBear(n_lat = 20, n_mlt = 40)

    for i in range(50264, iend, 2):#range(ist,iend+1, 2):
        pb.full_den_movie_plot(full_data.iloc[i], rf, out_prefix=outdir+f"step_{i-ist:04}", write_event=False)
