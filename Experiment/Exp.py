import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from data_provider.data_factory import data_provider
from models.model_list import model_dict
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import metric
from utils.tools import EarlyStopping, test_params_flop, visual


class Exp:
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.exp_info}")
        self.global_step = 0

        self._record_args()
        self._set_seed()
        self._set_device()
        self.device = torch.device("cuda:0")
        self.model = self._build_model().to(self.device)

    def _record_args(self) -> None:
        """
        Record the arguments in the experiment to TensorBoard
        """
        for key, value in self.args.__dict__.items():
            self.writer.add_text(key, str(value))

    def _set_seed(self) -> None:
        """
        Set the seed for the experiment
        """
        random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)

    def _set_device(self) -> None:
        """
        Set the device for the experiment
        """
        self.args.devices_ids = self.args.devices_ids.replace(" ", "")
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices_ids

    def _build_model(self) -> nn.Module:
        """
        Build the model for the experiment
        """
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model)
        return model

    def _get_data(self, flag):
        """
        Get the data for the experiment
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self) -> optim.Optimizer:
        if self.args.optimizer == "Adam":
            model_optim = optim.Adam(
                self.model.parameters(), lr=self.args.learning_rate
            )
        elif self.args.optimizer == "AdamW":
            model_optim = optim.AdamW(
                self.model.parameters(), lr=self.args.learning_rate
            )
        elif self.args.optimizer == "SGD":
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        return model_optim

    def _select_lr_scheduler(
        self, model_optim, train_steps
    ) -> lr_scheduler.LRScheduler:
        if self.args.lr_scheduler == "None":
            scheduler = None
        elif self.args.lr_scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(
                optimizer=model_optim,
                step_size=self.args.step_size,
                gamma=self.args.gamma,
            )
        elif self.args.lr_scheduler == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer=model_optim,
                T_max=self.args.T_max,
                eta_min=self.args.eta_min,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {self.args.lr_scheduler}")
        return scheduler

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        checkpoint_path = os.path.join(self.args.checkpoints_dir, self.args.exp_info)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            delta=self.args.delta,
            save_best_num=self.args.save_best_num,
        )

        model_optim = self._select_optimizer()
        scheduler = self._select_lr_scheduler(model_optim, train_steps)
        criterion = self._select_criterion()

        # Start Training
        if self.args.train_epochs == 0:
            print("Did not train, only test")
            best_model_path = checkpoint_path + "/" + "checkpoint_best.pth"
            torch.save(self.model.state_dict(), best_model_path)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.overlap_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                self.global_step += 1

                if scheduler is not None:
                    scheduler.step()

            epoch_time_cost = time.time() - epoch_time

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.writer.add_scalars(
                "Loss",
                {
                    "train_loss": train_loss,
                    "validation_loss": vali_loss,
                    "test_loss": test_loss,
                },
                epoch,
            )
            self.writer.add_scalar("Train/epoch_time(s)", epoch_time_cost, epoch)
            if scheduler is not None:
                self.writer.add_scalar(
                    "Train/learning_rate", scheduler.get_last_lr()[0], epoch
                )
            else:
                self.writer.add_scalar(
                    "Train/learning_rate", self.args.learning_rate, epoch
                )

            early_stopping(vali_loss, self.model, checkpoint_path)

            if early_stopping.early_stop:
                break

        # Load best model from all epochs with checkpoint_best.pth
        best_model_path = checkpoint_path + "/" + "checkpoint_best.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.overlap_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")

        if test:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        self.args.checkpoints_dir,
                        self.args.exp_info,
                        "checkpoint_best.pth",
                    )
                )
            )

        preds = []
        trues = []
        inputx = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.overlap_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        self.writer.add_text("Test_Metrics", f"mse:{mse}, mae:{mae}, rse:{rse}")

        # np.save(
        #     folder_path + "metrics.npy",
        #     np.array([mae, mse, rmse, mape, mspe, rse, corr]),
        # )
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)
        np.save(folder_path + "x.npy", inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            checkpoint_path = os.path.join(
                self.args.checkpoints_dir, self.args.exp_info
            )
            best_model_path = checkpoint_path + "/" + "checkpoint_best.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = (
                    torch.zeros(
                        [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                    )
                    .float()
                    .to(batch_y.device)
                )
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.overlap_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                outputs = self.model(batch_x)
            pred = outputs.detach().cpu().numpy()  # .squeeze()
            preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return
