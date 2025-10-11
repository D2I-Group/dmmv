import argparse

import torch
from Experiment.Exp import Exp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Forecasting Experiment")

    # ------------------------------------------------------------------------------------------------
    # Basic Settings
    parser.add_argument(
        "--exp_info",
        type=str,
        required=True,
        help="Experiment Info, will be used as the name of the experiment",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="PatchTST",
        help="Model Name, Options: [PatchTST]",
    )
    parser.add_argument("--random_seed", type=int, default=2021, help="Random Seed")
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="Whether to Train"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/",
        help="Log Directory",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints/",
        help="Checkpoints Directory",
    )

    # ------------------------------------------------------------------------------------------------
    # Forecasting Task Settings
    parser.add_argument(
        "--history_len", type=int, default=24 * 14, help="History Length"
    )
    parser.add_argument(
        "--overlap_len", type=int, default=24 * 4, help="Overlap Length"
    )
    parser.add_argument(
        "--pred_len", type=int, default=24 * 4, help="Prediction Sequence Length"
    )
    parser.add_argument("--period", type=int, default=24 * 4, help="Period of the data")

    # ------------------------------------------------------------------------------------------------
    # Data Settings
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="Root Path of the Data File",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Data File")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="Forecasting Task, Options: [M, S, MS]; M: Multivariate Predict Multivariate, S: Univariate Predict Univariate, MS: Multivariate Predict Univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="Frequency for Time Features Encoding, Options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], You can also use more detailed freq like 15min or 3h",
    )

    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]. timeF: Use time features including month, day, weekday, hour, etc.",
    )

    # ------------------------------------------------------------------------------------------------
    # Model Settings

    ## Decomposition
    parser.add_argument(
        "--decomposition", type=int, default=0, help="decomposition; True 1 False 0"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=25, help="decomposition-kernel"
    )
    parser.add_argument(
        "--individual", type=int, default=0, help="individual head; True 1 False 0"
    )

    ## PatchTST
    # Input / Output
    parser.add_argument("--c_in", type=int, default=7, help="Number of input variables")

    ### Patch
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--padding_patch", default="end", help="None: None; end: padding on the end"
    )

    ### Architecture
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument(
        "--fc_dropout", type=float, default=0.05, help="fully connected dropout"
    )
    parser.add_argument("--head_dropout", type=float, default=0.0, help="head dropout")

    ## RevIN
    parser.add_argument("--revin", type=int, default=1, help="RevIN; True 1 False 0")
    parser.add_argument(
        "--affine", type=int, default=0, help="RevIN-affine; True 1 False 0"
    )
    parser.add_argument(
        "--subtract_last",
        type=int,
        default=0,
        help="0: subtract mean; 1: subtract last",
    )

    ## Formers
    parser.add_argument(
        "--embed_type",
        type=int,
        default=0,
        help="0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding",
    )
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )

    ## Vision Framework
    parser.add_argument(
        "--vit_model", type=str, default="ViT-B_16", help="vit model name"
    )
    parser.add_argument(
        "--train_vit",
        action="store_true",
        help="whether to train ViT parameters",
    )
    parser.add_argument(
        "--adaption_method",
        type=str,
        default="conv",
        help="adaption, options: [conv, self_attention]",
    )
    parser.add_argument(
        "--adaption_dim",
        type=int,
        default=256,
        help="adaption dimension used for attention",
    )
    parser.add_argument(
        "--modality_combine_method",
        type=str,
        default="double_linear_combine",
        help="modality combine method, options: [double_linear_combine, direct_combine, cross_modal_multi_head_attention_combine]",
    )

    ## VisionTS
    parser.add_argument("--norm_const", type=float, default=0.4)
    parser.add_argument("--align_const", type=float, default=0.4)
    parser.add_argument(
        "--interpolation", type=str, default="bilinear", help="interpolation method"
    )
    parser.add_argument(
        "--vm_arch", type=str, default="vit_base", help="vision model arch"
    )
    parser.add_argument("--ft_type", type=str, default="ln", help="fine-tune type")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/")
    parser.add_argument("--load_ckpt", type=bool, default=True)
    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]",
    )
    parser.add_argument(
        "--trained_MAE_ckpt",
        type=str,
        default=None,
        help="pretrained MAE model path",
    )

    parser.add_argument(
        "--trained_PatchTST_ckpt",
        type=str,
        default=None,
        help="pretrained PatchTST model path",
    )

    parser.add_argument(
        "--trained_SimpleVersion_ckpt",
        type=str,
        default=None,
        help="pretrained SimpleVersion model path",
    )

    # ------------------------------------------------------------------------------------------------
    # Training Settings
    parser.add_argument(
        "--num_workers", type=int, default=16, help="data loader num workers"
    )
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch Size of Train Input Data"
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="Optimizer, Options: [Adam, AdamW, SGD]",
    )

    # Early Stopping
    parser.add_argument(
        "--patience", type=int, default=100, help="early stopping patience"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="early stopping delta",
    )
    parser.add_argument(
        "--save_best_num", type=int, default=3, help="number of best scores to save"
    )

    # ------------------------------------------------------------------------------------------------
    # Learning Rate
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--loss", type=str, default="mse", help="loss function")

    # Learning Rate Scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="None",
        help="Scheduler used for learning rate adjustment, Options: [None, StepLR, CosineAnnealingLR, OneCycleLR]",
    )

    # StepLR
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step Size used in StepLR scheduler",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma used in StepLR scheduler",
    )

    # CosineAnnealingLR
    parser.add_argument(
        "--T_max",
        type=int,
        default=100,
        help="T_max used in CosineAnnealingLR scheduler",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=0.0,
        help="Eta Min used in CosineAnnealingLR scheduler",
    )

    # OneCycleLR
    parser.add_argument(
        "--pct_start",
        type=float,
        default=0.3,
        help="Pct Start used in OneCycleLR scheduler",
    )

    # ------------------------------------------------------------------------------------------------
    # GPU Settings
    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU")
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="Use Multiple GPUs",
        default=False,
    )
    parser.add_argument(
        "--devices_ids", type=str, default="0", help="Device IDs of Multiple GPUs"
    )

    parser.add_argument(
        "--test_flop",
        action="store_true",
        default=False,
        help="See utils/tools for usage",
    )

    # ------------------------------------------------------------------------------------------------
    # Build args
    args = parser.parse_args()

    # Show arguments
    print("Args in experiment:")
    print(args)

    # Build experiment class
    if args.is_training:
        # setting record of experiments
        exp = Exp(args)  # set experiments
        print(f">>>>>>>start training : {args.exp_info}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train()

        print(f">>>>>>>testing : {args.exp_info}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(args.exp_info)

        if args.do_predict:
            print(
                f">>>>>>>predicting : {args.exp_info}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            )
            exp.predict(args.exp_info, True)

        torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        print(f">>>>>>>testing : {args.exp_info}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(args.exp_info, test=1)
        torch.cuda.empty_cache()
