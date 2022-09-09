from RGCPD import core_pp
import pandas as pd
import numpy as np

from RGCPD.forecasting import func_models as fc_utils
def get_lag_shifted(df_data: pd.DataFrame, target_ts, lags_i: int, labels=None):
    if labels is None:
        labels = df_data.columns[1:-2]
    if type(lags_i) is int:
        lags_i = [lags_i]
    
    list_shifted_dfs = []
    for lag_i in lags_i:
        
        
        splits = df_data.index.levels[0]
        list_training_dfs = [] ; list_target = []
        for s in splits:
            fit_masks = fc_utils.apply_shift_lag(df_data.loc[s].iloc[:,-2:].copy(), lag_i=lag_i)
            x_fit = fit_masks['x_fit']

            # df_temp = df_data.loc[s][labels][x_fit]
            df_temp = df_data.loc[s][labels][x_fit]
            labels_new = [l+f'_{lag_i}' for l in labels]
            df_temp.columns = labels_new
            list_training_dfs.append(df_temp)
            # fix target
            y_fit = fit_masks['y_fit']
            
            list_target.append(target_ts.loc[s][y_fit])
            
        df_shifted = pd.concat(list_training_dfs, keys=splits)
        target_fit = pd.concat(list_target, keys=splits)

        df_shifted.index = target_fit.index
        df_shifted = pd.DataFrame(df_shifted.values, columns=df_shifted.columns, index=target_fit.index)
        list_shifted_dfs.append(df_shifted)
    df_data_shifted = pd.concat(list_shifted_dfs, axis=1)

    # restore train-test mask
    df_splits_adap = df_data.iloc[:,-2:].copy().loc[target_fit.index]
    df_splits_adap.index = df_data_shifted.index
    df_data_shifted = df_data_shifted.merge(df_splits_adap, left_index=True, right_index=True)

    return df_data_shifted, target_fit

