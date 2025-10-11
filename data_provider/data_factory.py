from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Pred,
    Dataset_Custom_Split_Variable,
)
from data_provider.data_loader_decompose import Dataset_Custom_Seasonal

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
    "split_variable": Dataset_Custom_Split_Variable,
    "custom_seasonal": Dataset_Custom_Seasonal,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == "train" or flag == "val":
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    elif flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred

    timeenc = 0 if args.embed != "timeF" else 1
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        history_len=args.history_len,
        overlap_len=args.overlap_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
