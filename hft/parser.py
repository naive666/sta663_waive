import json, os, shutil
from itertools import product

# class Parser:


if __name__ == '__main__':
    save_root_path = '/j/huanming.zhang/hft/checkpoint/second_1'
    row_window_list = [20, 50, 100]
    max_epoch_list = [30, 60]
    hidden_size_list = [4, 8, 16]
    loss_func_list = ['mse', 'mae', 'corr_loss']
    optimizer_name_list = ['adam', 'radam', 'adamw']

    param_comb = product(row_window_list, max_epoch_list, hidden_size_list, loss_func_list, optimizer_name_list)
    for idx, params in enumerate(param_comb):
        save_path = f"{save_root_path}/{idx}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        current_config = {
            "data_path" : "/j/temp/insample.csv.HUOBI_PERP_BTC_USDT.gz",
            "train_ratio" : 0.75,
            "test_ratio" : 1,   
            "save_path" : save_path,
            "data_load_cores" : 15,
            "row_window" : params[0],
            "max_epoch" : params[1],
            "hidden_size" : params[2],
            "fc_layer_num" : 3,
            "fc_layer_param" : [32, 8, 1],
            "loss_func" : params[3],
            "optimizer_name" : params[4]

        }
        with open(save_path+'/config.json', 'w') as w:
            json.dump(current_config, w)
    