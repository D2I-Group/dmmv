import torch
import torch.nn as nn

from modules.mae_reg.model import MAE_Reg_BackCast_Forecast
from modules.PatchTST_backbone import PatchTST_backbone

from modules.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.history_len = configs.history_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        self.individual = configs.individual
        self.channels = configs.c_in
        self.period = configs.period

        self.vm = MAE_Reg_BackCast_Forecast(
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

        # self.revin_layer = RevIN(configs.c_in)
        if self.individual:
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Trend.append(nn.Linear(self.history_len, self.pred_len))

        else:
            self.Linear_Trend = nn.Linear(self.history_len, self.pred_len)

        # Two Number Version
        self.gate_weigts = nn.Parameter(
            torch.tensor([[2.0, 0.0]] * configs.c_in), requires_grad=True
        )

        # One Number Version
        # self.gate_weigts = nn.Parameter(
        #     torch.tensor([[2.0]] * configs.c_in), requires_grad=True
        # )

        if configs.trained_MAE_ckpt:
            self.load_checkpoint_from_trained_MAE(configs.trained_MAE_ckpt)

    def load_checkpoint_from_trained_MAE(self, checkpoint_path):
        """
        Load checkpoint for Vision Transformer
        """
        print("Loading MAE checkpoint from: ", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward_seasonal(self, x):
        x = x.permute(0, 2, 1)
        backcast_output, seasonal_output = self.vm.forward(x)
        backcast_output, seasonal_output = (
            backcast_output.permute(0, 2, 1),
            seasonal_output.permute(0, 2, 1),
        )
        return backcast_output, seasonal_output

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
        x = x.permute(0, 2, 1)
        self.seasonal_x, self.seasonal_output = self.forward_seasonal(x)
        self.trend_x = x - self.seasonal_x

        # self.trend_x = self.trend_x.permute(0, 2, 1)
        # self.trend_x = self.revin_layer(self.trend_x, "norm")
        # self.trend_x = self.trend_x.permute(0, 2, 1)

        self.trend_output = self.forward_trend(self.trend_x)

        # self.trend_output = self.trend_output.permute(0, 2, 1)
        # self.trend_output = self.revin_layer(self.trend_output, "denorm")
        # self.trend_output = self.trend_output.permute(0, 2, 1)

        seasonal_weights = self.gate_weigts[:, 0].unsqueeze(0).unsqueeze(2)

        # One number version
        # trend_weights = 2 - seasonal_weights
        # x = (
        #     self.seasonal_output * seasonal_weights + self.trend_output * trend_weights
        # ) / 2

        # Two number version
        trend_weights = self.gate_weigts[:, 1].unsqueeze(0).unsqueeze(2)
        x = self.seasonal_output * seasonal_weights + self.trend_output * trend_weights

        x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return x
