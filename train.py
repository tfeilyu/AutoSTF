import torch
import numpy as np
import argparse
import time
from src.settings import Settings, load_server_config
from src import train_util
from src.trainer import Trainer
from src.model.TrafficForecasting import AutoSTF
from src.DataProcessing import TFdataProcessing
import os, sys, random
from src.model.mode import Mode

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

parser.add_argument('--epochs', type=int, default=200, help='number of epochs to search')
parser.add_argument('--run_times', type=int, default=1, help='number of run')
parser.add_argument('--model_des', type=str, default='anything', help='save model param')
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
else:
    mask_support_adj = [torch.tensor(i).to(device) for i in TFdata.adj_mx_01]


def main(run_id):
    seed = run_id
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

    save_folder = './model_param/'+args.dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    search_model_path = os.path.join(save_folder, args.dataset+'_best_search_model_'+args.model_des+'.pt')
    train_model_path = os.path.join(save_folder, args.dataset+'_best_train_model_'+args.model_des+'_run_'+str(run_id)+'.pt')

    engine = Trainer(model, settings, scaler, device)
    try:
        best_epoch = engine.load(search_model_path)
        print('load architecture [epoch {}] from {} [done]'.format(best_epoch, search_model_path))
    except:
        print('load search model failed!')
        print('search first...')
        sys.exit()

    his_valid_time = []
    his_train_time = []
    his_valid_loss = []
    min_valid_loss = 1000
    best_epoch = 0
    all_start_time = time.time()

    print("start training...\n", flush=True)
    for epoch in range(best_epoch+1, best_epoch+args.epochs+1):
        epoch_start_time = time.time()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_start_time = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            train_x = torch.Tensor(x).to(device)
            train_x = train_x.transpose(1, 3)
            train_y = torch.Tensor(y).to(device)
            train_y = train_y.transpose(1, 3)
            # input x (64, 2, 207, 12)
            # input y (64, 207, 12)
            train_metrics = engine.train_weight(train_x, train_y[:, 0, :, :], mode=Mode.ONE_PATH_FIXED)
            train_loss.append(train_metrics[0])
            train_mae.append(train_metrics[1])
            train_mape.append(train_metrics[2])
            train_rmse.append(train_metrics[3])
        engine.weight_scheduler.step()

        train_end_time = time.time()
        t_time = train_end_time - train_start_time
        his_train_time.append(t_time)

        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        valid_start_time = time.time()
        for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()):
            val_x = torch.Tensor(x).to(device)
            val_x = val_x.transpose(1, 3)
            val_y = torch.Tensor(y).to(device)
            val_y = val_y.transpose(1, 3)
            val_metrics = engine.eval(val_x, val_y[:, 0, :, :])
            valid_loss.append(val_metrics[0])
            valid_mae.append(val_metrics[1])
            valid_mape.append(val_metrics[2])
            valid_rmse.append(val_metrics[3])
        # engine.arch_scheduler.step()

        valid_end_time = time.time()
        v_time = valid_end_time - valid_start_time
        his_valid_time.append(v_time)

        epoch_time = time.time() - epoch_start_time

        mean_train_loss = np.mean(train_loss)
        mean_train_mae = np.mean(train_mae)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mae = np.mean(valid_mae)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)

        his_valid_loss.append(mean_valid_loss)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time)

        print('{}, Epoch: {:03d}, Epoch Training Time:{:.4f}'.format(args.dataset, epoch, epoch_time))

        log_loss = 'Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train Time: {:.4f}\n' \
                   'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid Time: {:.4f}\n'
        print(log_loss.format(mean_train_loss, mean_train_mae, mean_train_mape, mean_train_rmse, t_time,
                              mean_valid_loss, mean_valid_mae, mean_valid_mape, mean_valid_rmse, v_time),flush=True)

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
            torch.save(obj=states, f=train_model_path)
            print('[eval]\tepoch {}\tsave parameters to {}\n'.format(best_epoch, train_model_path))
            min_valid_loss = mean_valid_loss

        elif settings.trainer.early_stop and epoch - best_epoch > settings.trainer.early_stop_steps:
            print('-' * 40)
            print('Early Stopped, best train epoch:', best_epoch)
            print('-' * 40)
            break

    all_end_time = time.time()

    print('\n')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    best_id = np.argmin(his_valid_loss)
    print("Training finished.")
    best_epoch = engine.load(train_model_path)
    print("The valid loss on best trained model is {}, epoch:{}\n"
          .format(str(round(his_valid_loss[best_id], 4)), best_epoch))

    print("All Training Time: {:.4f} secs".format(all_end_time-all_start_time))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(his_train_time)))
    print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(his_valid_time)))

    print('\n')
    print("Best Train Model Loaded")

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
        pred_singlestep = test_pred[:, :, i]
        real = true_test_y[:, :, i]
        metrics_single = train_util.metric(pred_singlestep, real)
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

    valid_MAE = []
    valid_MAPE = []
    valid_RMSE = []

    test_MAE = []
    teat_MAPE = []
    test_RMSE = []

    MAE = []
    MAPE = []
    RMSE = []

    MAE_avg = []
    MAPE_avg = []
    RMSE_avg = []

    for run_num in range(args.run_times):

        wandb.init(
            # set the wandb project where this run will be logged
            project='AutoSTF-v3',
            config={
                'model_name': 'AutoSTF',
                'tag': 'train',
                'dataset': args.dataset,
                'settings': args.settings,
                'epochs': args.epochs,
                'model_des': args.model_des,

                'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

                'server_name': server_config.server_name,
                'GPU_type': server_config.GPU_type,

                'run_num': run_num,

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

        # track hyperparameters and run metadata
        config = wandb.config
        vm1, vm2, vm3, tm1, tm2, tm3, m1, m2, m3, ma1, ma2, ma3, = main(run_num)

        valid_MAE.append(vm1)
        valid_MAPE.append(vm2)
        valid_RMSE.append(vm3)

        test_MAE.append(tm1)
        teat_MAPE.append(tm2)
        test_RMSE.append(tm3)

        MAE.append(m1)
        MAPE.append(m2)
        RMSE.append(m3)

        MAE_avg.append(ma1)
        MAPE_avg.append(ma2)
        RMSE_avg.append(ma3)

    mae_single = np.mean(np.array(MAE), 0)
    mape_single = np.mean(np.array(MAPE), 0)
    rmse_single = np.mean(np.array(RMSE), 0)

    mae_single_std = np.std(np.array(MAE), 0)
    mape_single_std = np.std(np.array(MAPE), 0)
    rmse_single_std = np.std(np.array(RMSE), 0)

    mae_avg = np.mean(np.array(MAE_avg), 0)
    mape_avg = np.mean(np.array(MAPE_avg), 0)
    rmse_avg = np.mean(np.array(RMSE_avg), 0)

    mae_avg_std = np.std(np.array(MAE_avg), 0)
    mape_avg_std = np.std(np.array(MAPE_avg), 0)
    rmse_avg_std = np.std(np.array(RMSE_avg), 0)

    print('\n')
    print(args.dataset)
    print('\n')

    print('valid\t MAE\t RMSE\t MAPE')
    log = 'mean:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.mean(valid_MAE), np.mean(valid_RMSE), np.mean(valid_MAPE)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(valid_MAE), np.std(valid_RMSE), np.std(valid_MAPE)))
    print('\n')

    print('test\t MAE\t RMSE\t MAPE')
    log = 'mean:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.mean(test_MAE), np.mean(test_RMSE), np.mean(teat_MAPE)))
    log = 'std:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.std(test_MAE), np.std(test_RMSE), np.std(teat_MAPE)))
    print('\n')

    print('single test:')
    print('horizon\t MAE-mean\t RMSE-mean\t MAPE-mean\t MAE-std\t RMSE-std\t MAPE-std')
    for i in range(4):
        log = '{:d}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'
        print(log.format([3, 6, 9, 12][i], mae_single[i], rmse_single[i], mape_single[i], mae_single_std[i],
                         rmse_single_std[i], mape_single_std[i]))

    print('\n')
    print('avg test:')
    print('average\t MAE-mean\t RMSE-mean\t MAPE-mean\t MAE-std\t RMSE-std\t MAPE-std')
    for i in range(4):
        log = '{:d}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'
        print(log.format([3, 6, 9, 12][i], mae_avg[i], rmse_avg[i], mape_avg[i], mae_avg_std[i],
                         rmse_avg_std[i], mape_avg_std[i]))

    print('Train Finish!\n')
