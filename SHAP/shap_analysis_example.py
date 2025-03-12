import shap_analysis as shpa

if __name__ == "__main__":
    outdir = "./"

    rf, explain = shpa.load_tools("FI_GEO_RFdat_AIMFAHR_forest.pkl", n_jobs=8)

    storm_data = shpa.load_data(storm=True)

    pb = shpa.PolarBear(n_lat = 10, n_mlt = 20)

    pb.gen_plots(storm_data.iloc[4803], rf, explain, out_prefix=outdir+"storm_evt_4803", write_event=True)
