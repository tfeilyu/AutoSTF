import torch
import src.train_util as train_util
from src.model.mode import Mode


class Trainer():
    def __init__(self, model, settings, scaler, device):
        self.model = model
        self.model.to(device)
        self.scaler = scaler
        self.settings = settings

        self.iter = 0
        self.task_level = 1
        self.seq_out_len = settings.data.out_length
        self.max_value = scaler.max_value

        self.loss = train_util.masked_mae

        self.weight_optimizer = torch.optim.Adam(
            self.model.weight_parameters(),
            lr=settings.trainer.weight_lr,
            weight_decay=settings.trainer.weight_decay
        )

        self.arch_optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=settings.trainer.arch_lr,
            weight_decay=settings.trainer.arch_decay
        )

        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.weight_optimizer,
            milestones=settings.trainer.weight_lr_decay_milestones,
            gamma=settings.trainer.weight_lr_decay_ratio
        )
        self.arch_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.arch_optimizer,
            milestones=settings.trainer.arch_lr_decay_milestones,
            gamma=settings.trainer.arch_lr_decay_ratio
        )

    def train_weight(self, input, real_val, mode=Mode.TWO_PATHS):

        self.weight_optimizer.zero_grad()
        self.model.train()

        # input (64, 2, 207, 12)
        output = self.model(input, mode)
        # output (64, 1, 207, 12)

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward(retain_graph=False)

        mae = train_util.masked_mae(predict, real, 0.0).item()
        mape = train_util.masked_mape(predict, real, 0.0).item()
        rmse = train_util.masked_rmse(predict, real, 0.0).item()

        torch.nn.utils.clip_grad_norm_(self.model.weight_parameters(), self.settings.trainer.weight_clip_gradient)
        self.weight_optimizer.step()
        self.weight_optimizer.zero_grad()

        return loss.item(), mae, mape, rmse

    def train_arch(self, input, real_val, mode=Mode.TWO_PATHS):
        self.model.train()
        self.arch_optimizer.zero_grad()

        # input (64, 2, 207, 12)
        output = self.model(input, mode)
        # output (64, 1, 207, 12)

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        mae = train_util.masked_mae(predict, real, 0.0).item()
        mape = train_util.masked_mape(predict, real, 0.0).item()
        rmse = train_util.masked_rmse(predict, real, 0.0).item()

        loss.backward(retain_graph=False)
        torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.settings.trainer.arch_clip_gradient)
        self.arch_optimizer.step()

        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, mode=Mode.ONE_PATH_FIXED)
        # output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        predict = torch.clamp(predict, min=0., max=self.max_value)

        loss = self.loss(predict, real, 0.0)
        mae = train_util.masked_mae(predict, real, 0.0).item()
        mape = train_util.masked_mape(predict, real, 0.0).item()
        rmse = train_util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def infer(self, input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input, mode=Mode.ONE_PATH_FIXED)
        # output = output.transpose(1, 3)

        return output

    def load(self, model_path):
        states = torch.load(model_path, map_location='cuda:0')

        # load net
        self.model.load_state_dict(states['net'])

        # load optimizer
        self.arch_optimizer.load_state_dict(states['arch_optimizer'])
        self.arch_scheduler.load_state_dict(states['arch_scheduler'])

        self.weight_optimizer.load_state_dict(states['weight_optimizer'])
        self.weight_scheduler.load_state_dict(states['weight_scheduler'])

        # load historical records
        return states['best_epoch']

