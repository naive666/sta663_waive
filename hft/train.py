import pandas as pd
import numpy as np
#  optimizer
from model import LSTM
import torch, os, gc
import matplotlib.pyplot as plt
import json
from optparse import OptionParser
from DataProcessor import DataTransformer, none_transformer, zscore_transformer, min_max_transformer, data_handler, data_handler_0, data_handler_1
from optimizer import AdamW, PlainAdam, RAdam
from loss import BiLinearSimilarity, CosineSimilarityLoss, PearsonCorrLoss, ProjectedDotProductSimilarity


def lstm_backprop(train_loader, dataset_test, max_epoch, model, optimizer, loss_func):
    train_loss_all = []
    valid_loss_all = []
    test_X = dataset_test.tensors[0]
    test_y = dataset_test.tensors[1]
    for epoch in range(max_epoch):
        train_loss = 0
        train_num = 0
        for step, (bx, by) in enumerate(train_loader):
            output = model(bx)
            loss = loss_func(output, by)
            print(f"loss: {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
            train_num += bx.size(0)
        
        test_output = model(test_X)
        valid_loss = loss_func(test_output, test_y)
        valid_loss = valid_loss.item() 
        print(f'Epoch{epoch+1}/{max_epoch}: Loss:{train_loss/train_num}')
        print(f'Epoch{epoch+1}/{max_epoch}: Valid Loss:{valid_loss}')
        train_loss_all.append(train_loss/train_num)
        valid_loss_all.append(valid_loss)
        # save checkpoint
        if (epoch+1) % 2 == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch}
            path_checkpoint = f"{save_root_path}/checkpoint_{epoch}_epoch.pkl"
            torch.save(checkpoint, path_checkpoint)
            # save loss fig
            x = range(len(train_loss_all))
            plt.plot(x, train_loss_all)
            plt.xlabel('epoch')
            plt.ylabel('loss value')
            plt.savefig(f"{save_root_path}/train_loss.jpg")
            plt.close()
            plt.plot(x, valid_loss_all)
            plt.xlabel('epoch')
            plt.ylabel('Valid loss value')
            plt.savefig(f"{save_root_path}/valid_loss.jpg")
            plt.close()

            
    





if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--config', dest='Config', help='config file')
    (options, args) = parser.parse_args()
    config_path = options.Config

    with open(config_path, 'r') as r:
        config = json.load(r)

    data_path = config['data_path']
    train_ratio = config['train_ratio']
    test_ratio = config['test_ratio']
    save_root_path = config['save_path']
    data_load_cores = config['data_load_cores']
    df = pd.read_csv(data_path, compression='gzip', header=0, sep=',')
    # choose data handler
    df = data_handler_1(df)
    data_length = len(df)
    features = list(df)
    features.remove('symbol')
    features.remove('Ret10S')
    features.remove('Ret600S')
    train_split_point = int(data_length * train_ratio)
    test_split_point = int(data_length * test_ratio)
    train_data = df.iloc[:train_split_point]
    test_data = df.iloc[train_split_point:test_split_point]

    row_window = config['row_window']
    batchsize = 10000
    max_epoch = config['max_epoch']

    del df
    gc.collect()

    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path, exist_ok=True)

    
    data_transformer = DataTransformer(zscore_transformer, row_window, n_cores=data_load_cores)

    train_X = train_data[features]
    train_y = train_data['Ret10S'].iloc[row_window:].values.reshape(-1,1)
    test_X = test_data[features]
    test_y = test_data['Ret10S'].iloc[row_window:].values.reshape(-1,1)
    
    print("start data transformer")
    train_X = data_transformer.fit_transform(train_X)
    test_X = data_transformer.fit_transform(test_X)

    dataset_train = data_transformer.to_dataset(train_X, train_y)
    dataloader_train = data_transformer.to_dataloader(dataset_train, batchsize)
    dataset_test = data_transformer.to_dataset(test_X, test_y)
    # dataloader_test = data_transformer.to_dataloader(dataset_test, )

    # delete useless data
    del train_X
    del train_y
    del test_X
    del test_y
    gc.collect()

    # construct model
    input_size = len(features)
    hidden_size = config['hidden_size']
    num_layers = 1
    batch_first = True
    # dropout = 0.43
    fc_layer_num = config['fc_layer_num']
    fc_layer_param = config['fc_layer_param']
    model = LSTM(input_size, hidden_size, num_layers, batchsize, batch_first, fc_layer_num, *fc_layer_param)

    # loss function
    # loss_func = torch.nn.MSELoss()
    func_name = config['loss_func']
    if func_name == 'mse':
        loss_func = torch.nn.MSELoss()
    elif func_name == 'mae':
        loss_func = torch.nn.L1Loss()
    elif func_name == 'corr_loss':
        loss_func = PearsonCorrLoss()
    else:
        raise ValueError("choose right loss")
    
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_name = config['optimizer_name']
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    else:
        raise ValueError("choose right optimizer")

    
    print("start training")
    lstm_backprop(dataloader_train, dataset_test, max_epoch, model, optimizer, loss_func)

    