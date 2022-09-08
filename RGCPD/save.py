#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to save forecast outputs i.e. scores and ML models
http://onnx.ai/sklearn-onnx/api_summary.html
"""

import numpy as np
from pathlib import Path
from skl2onnx import to_onnx

# save scores
def save_scores(df_file_name, df):
    df.to_csv(df_file_name)
    print(f"Data is saved in {df_file_name}")

# save ML model
def save_model(onnx_model_filename, sklearn_model):
    X = sklearn_model.X_pred[:1].to_numpy().astype(np.float32)
    onnx_model = to_onnx(sklearn_model, X)
    with open(onnx_model_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model is saved in onxx format in {onnx_model_filename}")

