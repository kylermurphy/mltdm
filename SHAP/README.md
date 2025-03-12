## SHAP Value analysis

This folder contains files and python notebooks for the SHAP analysis of the Murphy+2025 Thermospheric Density Random Forest Model [1]_ , specifically FISM2-GEO.

The SHAP analysis could be used for general random forest models, though it is designed specifically for the FISM2-GEO model. For general usage, see the SHAP package (or fasttreeshap for tree-based models). 

Dependencies
------------
pandas
fasttreeshap
numpy
matplotlib
cartopy

sklearn is required to train the random forest (RF) model (see Kyle Murphy notebook); following classes assume that RF model has been saved to pickle with file option 'wb', e.g.

> with open(fname, 'wb') as f:
>     pickle.dump(forest, f)

and that the train-test split of the original satdrag_database*hdf5 files
(as performed in KM's notebook) has been saved to hdf5 in the file
"FI_GEO_RFdat_AIMFAHR_data.h5"

References
----------
.. [1] Murphy, K.M., Halford, A., et al. "Understanding and Modeling ... Neutral Density Using Random Forests", Space Weather, 2025, https://doi.org/10.1029/2024SW003928