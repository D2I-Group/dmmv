import os
import matplotlib.pyplot as plt
import torch

plt.switch_backend("agg")


class EarlyStopping:
    def __init__(self, patience=10, delta=0, save_best_num=3, lower_better=True):
        self.patience = patience
        self.delta = -delta if lower_better else delta
        self.save_best_num = save_best_num
        self.lower_better = lower_better

        self.counter = 0
        self.best_scores = []
        self.early_stop = False

    def __call__(self, score, model, path):
        if not os.path.exists(path):
            os.makedirs(path)

        # Adjust score based on lower_better flag
        score = -score if self.lower_better else score

        # Initialize on first call
        if not self.best_scores:
            self.best_scores.append(score)
            torch.save(model.state_dict(), f"{path}/checkpoint_{abs(score):.4f}.pth")
            torch.save(model.state_dict(), f"{path}/checkpoint_best.pth")
            return

        # If score hasn't improved
        if score < max(self.best_scores) + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        # If score improved
        if len(self.best_scores) >= self.save_best_num:
            old_score = min(self.best_scores)
            old_checkpoint = f"{path}/checkpoint_{abs(old_score):.4f}.pth"
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
            self.best_scores.remove(old_score)

        # Save new checkpoint
        torch.save(model.state_dict(), f"{path}/checkpoint_{abs(score):.4f}.pth")
        self.best_scores.append(score)
        self.best_scores.sort(reverse=True)
        # Update best model
        if score == max(self.best_scores):
            torch.save(model.state_dict(), f"{path}/checkpoint_best.pth")
        self.counter = 0


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print(
            "INFO: Trainable parameter count: {:.2f}M".format(model_params / 1000000.0)
        )
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True
        )
        # print('Flops:' + flops)
        # print('Params:' + params)
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))
