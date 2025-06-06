# -*- coding: utf-8 -*-

import yaml
import mltdm

from pathlib import Path
import requests

class config():
    
    def __init__(self):
        
        self.yaml = self.get_config()
        self.dat = self.load_config()

    def get_config(self):
        """Return location of configuration file.
    
        In order
        1. ~/.data_io/dataiorc
        2. The installation folder of data_io
    
        Returns
        -------
        loc : string
            File path of the dataiorc configuration file
        """
        
        config_filename = 'mltdm.yaml'
    
        # Get user configuration location
        home_dir = Path.home()
        config_file_1 = home_dir / config_filename
    
        module_dir = Path(mltdm.__file__)
        config_file_2 = module_dir / '..' / config_filename
        config_file_2 = config_file_2.resolve()
    
        for f in [config_file_1, config_file_2]:
            if f.is_file():
                return str(f) 
            
    def load_config(self):
        """Read in configuration file neccessary for downloading and
        loading data and loading ML models.
    
        Returns
        -------
        config_dic : dict
            Dictionf containing all options from configuration file.
        """
        
        req_keys = ['data_dir','fx_m', 'fxnoAE_m','fism_flare', 'omni', 'zenodo']
        
        with open(self.yaml) as stream:
            try:
                dat = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
        kl = list()
        kd = list(dat.keys())
        if isinstance(dat, dict):
            for k in req_keys:
                kl.append(k in kd)
        
        # the zenodo link points to the most
        # recent release
        # need to resolve the redirect so we
        # can download all the needed files
        if 'zenodo' in dat:
            z_url = requests.get(dat['zenodo'])
            dat['zenodo'] = z_url.url+'/'            
        
        if False in kl:
            print(f'The configuration file {self.yaml}') 
            print('is missing required variables which')
            print('may limit functionality.\n')
            print('Add the missing variables to the config file to')
            print('ensure everything works.\n')
            print('Missing variables are:')  
            for b, k in zip(kl,req_keys):
                if not b:
                    print(f'\t{k}')
                    
                
        return dat
