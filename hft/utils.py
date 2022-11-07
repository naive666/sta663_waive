import numpy.ma as ma
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided as stride
import multiprocessing
from multiprocessing import Pool



def numpy_rolling(df, windows, return_col, preprocessing_func, **kwargs):
    """
    df: dataframe
    windows: rolling windows
    return_col: number of returned columns
    func: callable
    """
    if isinstance(df, pd.Series):
        df_ = pd.DataFrame(df)
    else:
        df_ = df.copy()
    arr = df_.values
    dim0, dim1 = arr.shape
    stride0, stride1 = arr.strides
    new_arr = stride(arr, (dim0-windows+1, windows, dim1), (stride0, stride0, stride1))
    result = np.full((dim0, windows, return_col), np.nan)
    
    for idx, values in enumerate(new_arr, windows-1):
        result[idx,:,:] = preprocessing_func(values, **kwargs)
    return result



def chunck_help(chunk,windows, new_arr, func, cache, **kwargs):
    result_dict = {}
    print(f"{chunk} starts")
    for idx in range(chunk[0], chunk[1]):
        result = func(new_arr[idx], **kwargs)
        result_dict[idx+windows-1] = result
    
    cache.append(result_dict)
    print(f"{chunk} finishes")


def numpy_rolling_chunk(df, windows, return_col, func, cores, **kwargs):
    """
    Parallel computation

    df: dataframe
    windows: rolling windows
    return_col: number of returned columns
    func: callable
    """
    if isinstance(df, pd.Series):
        df_ = pd.DataFrame(df)
    else:
        df_ = df.copy()
    arr = df_.values
    dim0, dim1 = arr.shape
    stride0, stride1 = arr.strides
    new_arr = stride(arr, (dim0-windows+1, windows, dim1), (stride0, stride0, stride1))
    result = np.full((dim0, windows, return_col), np.nan)

    total_number = dim0-windows+1
    chunk_size = total_number // cores
    chunk_list = []
    for i in range(0, total_number+chunk_size, chunk_size):
        if i > total_number:
            break
        end = min(i+chunk_size, total_number)
        chunk_list.append([i, end])


    cache = multiprocessing.Manager().list()
    pool = Pool(cores)
    for chunk in chunk_list:
        pool.apply_async(func=chunck_help, args=(chunk,windows,new_arr,func,cache,), kwds=kwargs)
    pool.close()
    pool.join()
    
    cache_list = list(cache)
    for item in cache_list:
        for idx in item.keys():
            result[idx,:,:] = item[idx]
    
    return result

