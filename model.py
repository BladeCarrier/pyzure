# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:31:07 2015

@author: ayanami
"""
import wrapper.configure_browser as configure
import  wrapper.model_wrapper as wr
import pandas as pd
import os
import time

models = wr.models

class Azure_model:
    def __init__(self,login,password,model,verbose = 1,
                 temp_path = "",temp_prefix="_tmp_",
                 driver = None):
        if driver is None:
            driver = configure.firefox()
        self.wrapper = wr.model_wrapper(driver,verbose)
        self.wrapper.sign_in(login,password)
        self.model =model
        self.temp_path = temp_path
        self.temp_prefix = temp_prefix
        self.is_deployed = False
    def fit(self,X,y,existing_dataset=None,
            dataset_folder = "My Datasets",wait_after_publish = True,close_webdriver = True):
        """usage:
            1) existing_dataset is None, X and y are numpy arrays or PANDAS datasets,
                - creates a dataset in temp_path
                - uploads it to the system as temp_prefix+"data", 
                    possibly overriding existing dataset
                - creates an experiment with newly uploaded dataset
                - creates, fits and publishes the trained model
            2) existing_dataset is a valid AzureML dataset name,
                X is "~" or a list of column names
                y is the target column name
                - creates an experiment with an existing dataset
                - [NotImplementedYet] Filters out all columns that are not in X+[y]
                - creates, fits and publishes the trained model
        """
        if existing_dataset is None:
            data = pd.DataFrame(X)
            data["_label"] =y
            data_path = os.path.join(self.temp_path, self.temp_prefix+"data")

            data.to_csv(data_path,index = False)
            del data
            self.wrapper.add_dataset_by_path(data_path,extension=".csv",
                                             overwrite = False,
                                             change_name = True,
                                             new_name= self.temp_prefix+"data",
                                             overwrite_on_fail=True
                                             )
            existing_dataset = self.temp_prefix+"data"
            X = "~"
            y = "_label"
        self.y_name = y
        self.wrapper.create_new_experiment(self.temp_prefix+"experiment")
        self.wrapper.setup_model(existing_dataset,
                                 target_colname=y,
                                 model = self.model,
                                 dataset_folder=dataset_folder
                                 )
        self.wrapper.run_experiment()
        self.wrapper.create_scoring_experiment()
        self.wrapper.run_experiment()
        url,key = self.wrapper.publish_experiment(wait_finished=True)
        self.api_url = url
        self.api_key = key
        if wait_after_publish:
            print "waiting in case Microsoft delays experiment publishing..."
            time.sleep(10)
        if close_webdriver:
            self.wrapper.driver.close()
        self.is_deployed= True
        return self
    def predict(self,X):
        if not self.is_deployed:
            raise ValueError, "Model is not deployed yet. Please fit it first"
        data = pd.DataFrame(X)
        data[self.y_name] =0
        ans = self.wrapper.apply_service(data,self.api_url,self.api_key)
        return ans
        
        
        
        
            
                                            
            
            
            
        
        
        
        
        
        
    