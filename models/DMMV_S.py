import torch
import torch.nn as nn
from modules.mae_reg.model import MAE_Reg


class moving_avg(nn.Module):
    def __init__(self, kernel_size=24, stride=1):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.history_len = configs.history_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.c_in

        if self.individual:
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Trend.append(nn.Linear(self.history_len, self.pred_len))

        else:
            self.Linear_Trend = nn.Linear(self.history_len, self.pred_len)

        self.period = configs.period
        self.history_len = configs.history_len
        self.pred_len = configs.pred_len

        self.vm = MAE_Reg(
            context_len=configs.history_len,
            pred_len=configs.pred_len,
            periodicity=configs.period,
            norm_const=configs.norm_const,
            align_const=configs.align_const,
            interpolation=configs.interpolation,
            arch=configs.vm_arch,
            finetune_type=configs.ft_type,
            ckpt_dir=configs.ckpt,
            load_ckpt=configs.load_ckpt,
        )

        if configs.trained_MAE_ckpt:
            self.load_checkpoint_from_trained_MAE(configs.trained_MAE_ckpt)

        # One Number Version
        self.gate_weigts = nn.Parameter(
            torch.tensor([[2.0]] * configs.c_in), requires_grad=True
        )

    def load_checkpoint_from_trained_MAE(self, checkpoint_path):
        """
        Load checkpoint for Vision Transformer
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward_seasonal(self, x):
        x = x.permute(0, 2, 1)
        seasonal_output = self.vm.forward(x)
        seasonal_output = seasonal_output[:, -self.pred_len :, :]
        seasonal_output = seasonal_output.permute(0, 2, 1)
        return seasonal_output

    def forward_trend(self, x):
        if self.individual:
            trend_output = torch.zeros(
                [x.size(0), x.size(1), self.pred_len],
                dtype=x.dtype,
            ).to(x.device)
            for i in range(self.channels):
                trend_output[:, i, :] = self.Linear_Trend[i](x[:, i, :])
        else:
            trend_output = self.Linear_Trend(x)

        return trend_output

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        self.seasonal_x, self.trend_x = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )

        self.seasonal_output = self.forward_seasonal(self.seasonal_x)
        self.trend_output = self.forward_trend(self.trend_x)

        seasonal_weights = self.gate_weigts[:, 0].unsqueeze(0).unsqueeze(2)

        # One number version
        trend_weights = 2 - seasonal_weights
        x = (
            self.seasonal_output * seasonal_weights + self.trend_output * trend_weights
        ) / 2

        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x
