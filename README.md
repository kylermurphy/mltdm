# mltdm
Repository for development of the MLTDM Model

## Local Installation

1. Clone repo or download zip
2. In root directory: pip install -e .
3. Create a local directory to hold model data, e.g., mltdm/local_data/
4. Change the configuration file mltdm/mltdm.yaml such that the 'data_dir' variable points to the local directory created in step 3. File paths must be absolute, not relative.
5. Run the model setup (see Notebooks/RF_predict.ipynb):

> import mltdm.den_fx as den_fx
> den_fx.setup()

This will take a few minutes to download the model and initial feature data files.
