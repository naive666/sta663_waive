import pandas as pd
import numpy as np
import numpy.ma as ma
import torch
import utils
import torch.utils.data as Data
from numpy.lib.stride_tricks import as_strided as stride


# Define some preprocessing functions


class DataTransformer:
    def __init__(self, func, row_window, n_cores, is_drop=True):
        """
        func: function to transform raw data
        row_back: rolling window
        is_drop: is drop the first nan rows
        """
        self.func = func
        self.row_window = row_window
        self.n_cores = n_cores
        self.is_drop = is_drop
    

    def fit_transform(self, X):
        """
        Transform the data from 2d to 3d, with dimension (seq_num, seq_len, input_size), in NLP,
        it corresponds the number of sentence, the number of words in each sentence, the dimension
        of each word
        """
        X_temp = utils.numpy_rolling_chunk(X, self.row_window, len(list(X)), func=self.func, cores=self.n_cores)
        if self.is_drop:
            X_temp = X_temp[self.row_window:]
            return X_temp
        return X_temp
    
    def to_dataset(self, X, y):
        """
        From numpy array to tensor dataset
        """
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        X_tensor = X_tensor.to(torch.float32)
        y_tensor = y_tensor.to(torch.float32)
        dataset = Data.TensorDataset(X_tensor, y_tensor)
        return dataset


    def to_dataloader(self, dataset, batch_size, is_shuffle=True):
        dataloader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle)
        return dataloader
        

def none_transformer(arr):
    return arr

def zscore_transformer(arr, cap=3.5):
    mean_ = ma.mean(ma.masked_invalid(arr), axis=0)
    std_ = ma.std(ma.masked_invalid(arr), axis=0)
    std_ = np.where(std_==0, np.nan, std_)
    std_ = np.where(np.isinf(std_), np.nan, std_)
    result = (arr - mean_) / std_
    result = np.where(result > cap, cap, result)
    result = np.where(result < -cap, -cap, result)
    result = np.where(np.isnan(result), 0, result)
    return result


def min_max_transformer(arr):
    min_ = ma.min(ma.masked_invalid(arr), axis=0)
    max_ = ma.max(ma.masked_invalid(arr), axis=0)
    diff = (max_ - min_)
    diff = np.where(diff==0, np.nan, diff)
    result = (arr - min_) / diff
    mean_ = np.nanmean(result, axis=0)
    inds = np.where(np.isnan(result))
    result[inds] = np.take(mean_, inds[1])

    return result


def data_handler(df):
    # 把最后那几列split一下，按照buy 和 sell的方向
    df_1 = df[df['side'] == 1]
    df_0 = df[df['side'] == 0]
    df_1 = df_1.rename(columns={'TOPLEVELSIZESHRINKAGEAGGR':'TOPLEVELSIZESHRINKAGEAGGR_1', 'TOPLEVELTRADESZSHRINKAGEAGGR':'TOPLEVELTRADESZSHRINKAGEAGGR_1',
                    'TA_HURSTNSECSBACK600':'TA_HURSTNSECSBACK600_1', 'TA_HURSTNSECSBACK1800':'TA_HURSTNSECSBACK1800_1'})
    df_0 = df_0.rename(columns={'TOPLEVELSIZESHRINKAGEAGGR':'TOPLEVELSIZESHRINKAGEAGGR_0', 'TOPLEVELTRADESZSHRINKAGEAGGR':'TOPLEVELTRADESZSHRINKAGEAGGR_0',
                    'TA_HURSTNSECSBACK600':'TA_HURSTNSECSBACK600_0', 'TA_HURSTNSECSBACK1800':'TA_HURSTNSECSBACK1800_0'})

    df_0.drop('side',axis=1,inplace=True)
    df_1.drop('side',axis=1,inplace=True)
    df_0.set_index("featureUpdateMics", drop=True, inplace=True)
    df_1.set_index("featureUpdateMics", drop=True, inplace=True)
    df_1[['TOPLEVELSIZESHRINKAGEAGGR_0', 'TOPLEVELTRADESZSHRINKAGEAGGR_0', 'TA_HURSTNSECSBACK600_0', 'TA_HURSTNSECSBACK1800_0']] = df_0[['TOPLEVELSIZESHRINKAGEAGGR_0', 'TOPLEVELTRADESZSHRINKAGEAGGR_0', 'TA_HURSTNSECSBACK600_0', 'TA_HURSTNSECSBACK1800_0']]
    return df_1

def data_handler_0(df):
    df_0 = df[df['side'] == 0]
    df_0.drop('side',axis=1,inplace=True)
    df_0.set_index("featureUpdateMics", drop=True, inplace=True)
    return df_0

def data_handler_1(df):
    df_1 = df[df['side'] == 1]
    df_1.drop('side',axis=1,inplace=True)
    df_1.set_index("featureUpdateMics", drop=True, inplace=True)
    return df_1



