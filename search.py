import torch
import numpy as np
import argparse
import time
from src.settings import Settings, load_server_config
from src import train_util
from src.trainer import Trainer
from src.model.TrafficForecasting import AutoSTF
from src.DataProcessing import TFdataProcessing
import os
from src.model.mode import Mode, create_mode
import random

import wandb, datetime

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--dataset', type=str, default='PEMS04', help='model dataset')
parser.add_argument('--settings', type=str, default='PEMS04', help='model settings')

parser.add_argument('--scale_num', type=int, default=3, help='number of scale')
parser.add_argument('--in_channels', type=int, default=3, help='number of input feature')
parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of star mlp layer')

parser.add_argument('--mode_name', type=str, default='TWO_PATHS', help='ONE_PATH_FIXED, ONE_PATH_RANDOM, '
                                                                       'TWO_PATHS, ALL_PATHS')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to search')
parser.add_argument('--model_des', type=str, default='anything', help='save model param')
parser.add_argument('--comments', type=str, default='', help='save model param')
args = parser.parse_args()

settings = Settings()
settings.load_settings(args.settings)

server_config = load_server_config()

print(args)
print("settings.data:")
for key, value in settings.data_dict.items():
    print("\t{}: {}".format(key, value))

print("settings.trainer:")
for key, value in settings.trainer_dict.items():
    print("\t{}: {}".format(key, value))

print("settings.model:")
for key, value in settings.model_dict.items():
    if str(key) not in 'candidate_op_profiles':
        print("\t{}: {}".format(key, value))
    else:
        print("\t{}:".format(key))
        for i in value:
            print("\t\t{}: {}".format(i[0], i[1]))
print('\n')
print(args)

device = torch.device(args.device)

scale_list = []
for i in range(args.scale_num):
    scale_list.append(int(settings.data.in_length/args.scale_num))
print(scale_list)

wandb.init(
    # set the wandb project where this run will be logged
    project='AutoSTF-v3',
    config={
        'model_name': 'AutoSTF',
        'tag': 'search',
        'dataset': args.dataset,
        'settings': args.settings,
        'epochs': args.epochs,
        'model_des': args.model_des,

        'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

        'server_name': server_config.server_name,
        'GPU_type': server_config.GPU_type,

        'weight_lr': settings.trainer.weight_lr,
        'weight_lr_decay_ratio': settings.trainer.weight_lr_decay_ratio,
        'weight_decay': settings.trainer.weight_decay,
        'weight_clip_gradient': settings.trainer.weight_clip_gradient,

        'arch_lr': settings.trainer.arch_lr,
        'arch_lr_decay_ratio': settings.trainer.arch_lr_decay_ratio,
        'arch_decay': settings.trainer.arch_decay,
        'arch_clip_gradient': settings.trainer.arch_clip_gradient,

        'batch_size': settings.data.batch_size,
        'early_stop': settings.trainer.early_stop,
        'early_stop_steps': settings.trainer.early_stop_steps,

        'train_prop': float(settings.data.train_prop),
        'valid_prop': float(settings.data.valid_prop),

        'num_sensors': settings.data.num_sensors,
        'in_channels': args.in_channels,
        'out_channels': settings.data.out_channels,
        'in_length': settings.data.in_length,
        'out_length': settings.data.out_length,

        'end_channels': settings.model.end_channels,
        'hidden_channels': settings.model.hidden_channels,

        'num_mlp_layers': args.num_mlp_layers,
        'IsUseLinear': settings.model.IsUseLinear,
        'num_linear_layers': settings.model.num_linear_layers,

        'scale_num': args.scale_num,
        'scale_list': scale_list,

        'num_att_layers': settings.model.num_att_layers,
        'num_hop': settings.model.num_hop,

        'layer_names': settings.model.layer_names,
        'num_temporal_search_node': settings.model.num_temporal_search_node,
        'num_spatial_search_node': settings.model.num_spatial_search_node,
        'temporal_operations': settings.model.temporal_operations,
        'spatial_operations': settings.model.spatial_operations,
    }
)
config = wandb.config


# data processing
TFdata = TFdataProcessing(
    dataset=args.dataset,
    train_prop=settings.data.train_prop,
    valid_prop=settings.data.valid_prop,
    num_sensors=settings.data.num_sensors,
    in_length=settings.data.in_length,
    out_length=settings.data.out_length,
    in_channels=args.in_channels,
    batch_size_per_gpu=settings.data.batch_size
)

scaler = TFdata.scaler
dataloader = TFdata.dataloader
adj_mx_gwn = [torch.tensor(i).to(device) for i in TFdata.adj_mx_gwn]
adj_mx = [torch.tensor(TFdata.adj_mx_dcrnn).to(device), adj_mx_gwn, torch.tensor(TFdata.adj_mx_01).to(device)]

if args.dataset in ['PEMS-BAY', 'METR-LA']:
    mask_support_adj = [torch.tensor(i).to(device) for i in TFdata.adj_mx_gwn]

elif args.dataset in ['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
    mask_support_adj = [torch.tensor(i).to(device) for i in TFdata.adj_mx_01]

else:
    adj_mx_gwn = None
    adj_mx = None
    mask_support_adj = None

mode = create_mode(args.mode_name)


def main(runid):
    seed = runid
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = AutoSTF(
        in_length=settings.data.in_length,
        out_length=settings.data.out_length,

        mask_support_adj=mask_support_adj,
        adj_mx=adj_mx,

        num_sensors=settings.data.num_sensors,
        in_channels=args.in_channels,
        out_channels=settings.data.out_channels,

        hidden_channels=settings.model.hidden_channels,
        end_channels=settings.model.end_channels,

        layer_names=settings.model.layer_names,

        config=config,
        device=device
    )

    num_Params = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', num_Params)
    wandb.log({'num_params': num_Params})

    save_folder = './model_param/'+args.dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_path = os.path.join(save_folder, args.dataset+'_best_search_model_'+args.model_des+'.pt')

    engine = Trainer(model, settings, scaler, device)
    try:
        best_epoch = engine.load(model_path)
        print('load architecture [epoch {}] from {} [done]'.format(best_epoch, model_path))
    except:
        print('load architecture [fail]')
        best_epoch = 0

    his_valid_time = []
    his_train_time = []
    his_valid_loss = []
    min_valid_loss = 1000
    all_start_time = time.time()

    print("start searching...\n", flush=True)
    for epoch in range(best_epoch+1, best_epoch+args.epochs+1):
        epoch_start_time = time.time()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []

        dataloader['search_train_loader'].shuffle()
        train_start_time = time.time()
        for iter, (x, y) in enumerate(dataloader['search_train_loader'].get_iterator()):
            train_x = torch.Tensor(x).to(device)
            train_x = train_x.transpose(1, 3)
            train_y = torch.Tensor(y).to(device)
            train_y = train_y.transpose(1, 3)
            # input x (64, 2, 207, 12)
            # input y (64, 207, 12)
            metrics = engine.train_weight(train_x, train_y[:, 0, :, :], mode=mode)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
        engine.weight_scheduler.step()
        train_end_time = time.time()

        t_time = train_end_time-train_start_time
        his_train_time.append(t_time)

        mean_train_loss = np.mean(train_loss)
        mean_train_mae = np.mean(train_mae)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('\n')
        print(now_time)
        log_loss = '{}, Epoch: {:03d},\n' \
                   'Search Loss: {:.4f}, Search MAE: {:.4f}, Search MAPE: {:.4f}, Search RMSE: {:.4f}, Search Time: {:.4f}'
        print(log_loss.format(args.dataset, epoch,
                              mean_train_loss, mean_train_mae, mean_train_mape, mean_train_rmse, t_time), flush=True)

        # if epoch < 10:
        #     continue

        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []

        valid_start_time = time.time()
        for iter, (x, y) in enumerate(dataloader['search_valid_loader'].get_iterator()):
            val_x = torch.Tensor(x).to(device)
            val_x = val_x.transpose(1, 3)
            val_y = torch.Tensor(y).to(device)
            val_y = val_y.transpose(1, 3)
            # metrics = engine.eval(val_x, val_y[:, 0, :, :])
            metrics = engine.train_arch(val_x, val_y[:, 0, :, :], mode=mode)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])
        engine.arch_scheduler.step()

        valid_end_time = time.time()
        v_time = valid_end_time-valid_start_time
        his_valid_time.append(v_time)

        epoch_time = time.time() - epoch_start_time

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mae = np.mean(valid_mae)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)

        his_valid_loss.append(mean_valid_loss)

        log_loss = 'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid Time: {:.4f}'
        print(log_loss.format(mean_valid_loss, mean_valid_mae, mean_valid_mape, mean_valid_rmse, v_time), flush=True)
        print('Epoch Search Time: {:.4f}'.format(epoch_time))

        if mean_valid_loss < min_valid_loss:
            best_epoch = epoch
            states = {
                'net': engine.model.state_dict(),
                'arch_optimizer': engine.arch_optimizer.state_dict(),
                'arch_scheduler': engine.arch_scheduler.state_dict(),
                'weight_optimizer': engine.weight_optimizer.state_dict(),
                'weight_scheduler': engine.weight_scheduler.state_dict(),
                'best_epoch': best_epoch
            }
            torch.save(obj=states, f=model_path)
            print('\n[save] epoch {}, save parameters to {}'.format(best_epoch, model_path))
            min_valid_loss = mean_valid_loss

        elif settings.trainer.early_stop and epoch - best_epoch > settings.trainer.early_stop_steps:
            print('\n')
            print('-' * 40)
            print('Early Stopped, best search epoch:', best_epoch)
            print('-' * 40)
            break

    all_end_time = time.time()

    print('\n')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    best_id = np.argmin(his_valid_loss)
    print("Searching finished.")
    best_epoch = engine.load(model_path)
    print("The valid loss on best search model is {}, epoch:{}\n".format(str(round(his_valid_loss[best_id], 4)),
                                                                         best_epoch))

    print("All Search Time: {:.4f} secs".format(all_end_time-all_start_time))
    print("Average Search Time: {:.4f} secs/epoch".format(np.mean(his_train_time)))
    print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(his_valid_time)))

    print('\n')
    print("Best Search Model Loaded")

    outputs = []
    true_valid_y = []
    for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()):
        valid_x = torch.Tensor(x).to(device)
        valid_x = valid_x.transpose(1, 3)
        valid_y = torch.Tensor(y).to(device)
        valid_y = valid_y.transpose(1, 3)[:, 0, :, :]

        with torch.no_grad():
            # input (64, 2, 207, 12)
            preds = engine.model(valid_x, mode=Mode.ONE_PATH_FIXED)
            # output (64, 1, 207, 12)

        outputs.append(preds.squeeze(dim=1))
        true_valid_y.append(valid_y)
    valid_yhat = torch.cat(outputs, dim=0)
    true_valid_y = torch.cat(true_valid_y, dim=0)
    # valid_yhat = valid_yhat[:valid_y.size(0), ...]
    valid_pred = scaler.inverse_transform(valid_yhat)
    valid_pred = torch.clamp(valid_pred, min=scaler.min_value, max=scaler.max_value)
    valid_mae, valid_mape, valid_rmse = train_util.metric(valid_pred, true_valid_y)

    log = '12 Average Performance on Valid Data - Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
    print(log.format(valid_mae, valid_mape, valid_rmse))

    outputs = []
    true_test_y = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(device)
        test_x = test_x.transpose(1, 3)
        test_y = torch.Tensor(y).to(device)
        test_y = test_y.transpose(1, 3)[:, 0, :, :]

        with torch.no_grad():
            # input (64, 2, 207, 12)
            preds = engine.model(test_x, mode=Mode.ONE_PATH_FIXED)
            # output (64, 1, 207, 12)

        outputs.append(preds.squeeze(dim=1))
        true_test_y.append(test_y)
    test_yhat = torch.cat(outputs, dim=0)
    true_test_y = torch.cat(true_test_y, dim=0)
    # test_yhat = test_yhat[:test_y.size(0), ...]
    test_pred = scaler.inverse_transform(test_yhat)
    test_pred = torch.clamp(test_pred, min=scaler.min_value, max=scaler.max_value)
    test_mae, test_mape, test_rmse = train_util.metric(test_pred, true_test_y)

    log = '12 Average Performance on Test Data - Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} \n'
    print(log.format(test_mae, test_mape, test_rmse))

    print('Single steps test:')
    mae = []
    mape = []
    rmse = []
    for i in [2, 5, 8, 11]:
        pred_single_step = test_pred[:, :, i]
        real = true_test_y[:, :, i]
        metrics_single = train_util.metric(pred_single_step, real)
        log = 'horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics_single[0], metrics_single[1], metrics_single[2]))

        mae.append(metrics_single[0])
        mape.append(metrics_single[1])
        rmse.append(metrics_single[2])

    print('\nAverage steps test:')
    mae_avg = []
    mape_avg = []
    rmse_avg = []
    for i in [3, 6, 9, 12]:
        pred_avg_step = test_pred[:, :, :i]
        real = true_test_y[:, :, :i]
        metrics_avg = train_util.metric(pred_avg_step, real)
        log = 'average {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i, metrics_avg[0], metrics_avg[1], metrics_avg[2]))
        mae_avg.append(metrics_avg[0])
        mape_avg.append(metrics_avg[1])
        rmse_avg.append(metrics_avg[2])
    wandb.finish()
    return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse, mae, mape, rmse, mae_avg, mape_avg, rmse_avg


if __name__ == "__main__":

    main(1)

    print('Search Finish!\n')

