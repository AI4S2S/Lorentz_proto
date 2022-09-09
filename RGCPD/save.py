#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to save forecast outputs i.e. scores and ML models
https://medium.com/analytics-vidhya/save-and-load-your-scikit-learn-models-in-a-minute-21c91a961e9b
"""

import numpy as np
import pickle

# save scores
def save_scores(df_file_name, df):
    df.to_csv(df_file_name)
    print(f"Data is saved in {df_file_name}")

# save ML model
def save_model(pkl_model_filename, sklearn_model):
    pickle.dump(sklearn_model, open(pkl_model_filename, 'wb'))
    print(f"Model is saved in onxx format in {pkl_model_filename}")

