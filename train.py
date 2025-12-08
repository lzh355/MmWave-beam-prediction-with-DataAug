import argparse
import os
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from model_3Dcov_basic import Model_3D
from dataloader import Dataloader


parser = argparse.ArgumentParser(description="PyTorch Beam Prediction Training")
# Optimization options
parser.add_argument(
    "--epochs", default=4000, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--batch-size", default=160, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.003,
    type=float,
    metavar="LR",
    help="initial learning rate",
)

# Device options
parser.add_argument(
    "--gpu", default="1", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
# Method options
parser.add_argument("--n", type=int, default=640, help="Number of training data")
parser.add_argument("--p", default=0.5, type=float, help="the probability of input augmentation methods")
parser.add_argument("--sigma", default=0.3, type=float, help="the standard variance of noise injection")
parser.add_argument("--Zf", default=8.0, type=float, help="the maximum value of Z")
parser.add_argument("--Z0", default=2.0, type=float,help="the initial value of Z")
parser.add_argument("--k", default=6.0, type=float,help='slope coefficient')
parser.add_argument("--rt", default=5, type=int,help='repeated training time')
parser.add_argument("--aug_type", default="All", type=str, choices=["none", "Cyclic_shift", "Flip", "Noise", "Label","All"], help='type of data augmentation')

args = parser.parse_args()

# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
random_seed = 202511
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if use_cuda:
    torch.cuda.manual_seed_all(random_seed)
    
class CrossEntropy(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        assert torch.any(targets_x<0).sum()==0
        return Lx

@torch.no_grad()
def eval(model, dataset, device, top_k=5):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    out_tensors = []

    # evaluate validation set
    for batch_id in range(dataset.batch_num):
        channel,_,_ = dataset.load_batch(batch_id)
        channel=channel.to(device)
        out_tensor = model(channel)
        out_tensors.append(out_tensor)
        
    out_tensor = torch.cat(out_tensors, dim=1)
    beam_power_nonoise_m,label_nonoise_m = dataset.load_all_set()
    beam_power_nonoise_m=beam_power_nonoise_m.to(device)
    label_nonoise_m=label_nonoise_m.to(device)

    loss = 0
    for loss_count in range(10):
        loss += criterion(torch.squeeze(out_tensor[loss_count, :, :]), label_nonoise_m[:, loss_count])
    
    out_tensor = out_tensor.transpose(1, 0)
    train_sorted = torch.argsort(out_tensor,dim=-1,descending=True)  #[dataset,10,64]
    correct = (train_sorted[:,:,0]== label_nonoise_m).cpu().numpy()

    P = np.sum(correct)
    beam_power_sorted = torch.gather(beam_power_nonoise_m,2,train_sorted[:,:,0 : top_k])
    beam_power_max = torch.max(beam_power_nonoise_m,dim=-1)[0]
    bl_topk = torch.zeros((dataset.dataset_size, 10, top_k))
    for k in range(top_k):
        bl_topk[:,:,k] = (torch.max(beam_power_sorted[:,:,0:k+1],dim=-1).values / beam_power_max) ** 2
    
    BL = torch.squeeze(torch.mean(bl_topk, dim=0)).cpu().numpy()
    acur = P / 10/dataset.dataset_size
    return acur, loss.item(), BL


def main():
    device = torch.device("cuda:0" if use_cuda else "cpu")
    repeat_time = args.rt
    # training epoch
    epoch = args.epochs
    # learning rate
    lr = args.lr
    # batch size
    batch_size = args.batch_size
    info = f"DataAug_3_64beam_{args.n}_{args.aug_type}_lr={lr}"
    # data augment list
    if args.aug_type == "Cyclic_shift":
        DA_list = ["Cyclic_shift"]
    elif args.aug_type == "Flip":
        DA_list = ["Flip"]
    elif args.aug_type == "Noise":
        DA_list = ["Noise"]
        info += f'_sigma={args.sigma}'
    elif args.aug_type == "Label":
        DA_list = ["Label"]
        info += f'_Zf={args.Zf}_Z0={args.Z0}_k={args.k}'
    elif args.aug_type == "All":
        DA_list = ["Cyclic_shift", "Flip", "Noise", "Label"]
        info += f'_Zf={args.Zf}_Z0={args.Z0}_k={args.k}_sigma={args.sigma}'
    else:
        DA_list=[]

    print(info)
    print(f"batch_size:{batch_size}")

    # training set and validation set
    train_dataset = Dataloader(
        path="dataset/data_16Tx_64Tx_training.mat",
        batch_size = batch_size,
        datacount = args.n,
        drop_last = False,
        Shuffle=True,
        p = args.p,
        sigma = args.sigma,
        Z0 = args.Z0, 
        Zf = args.Zf,
        k = args.k, 
        epochs = epoch,
    )
    val_dataset = Dataloader(path="dataset/data_16Tx_64Tx_testing.mat",batch_size=batch_size)
    
    # loss function
    criterion = CrossEntropy() if "Label" in DA_list else nn.CrossEntropyLoss()

    acur_eval = np.zeros((repeat_time, epoch))
    loss_eval = np.zeros((repeat_time, epoch))
    BL_eval = np.zeros((10, 5, repeat_time, epoch))

    for tt in range(repeat_time):
        # model initialization
        model = Model_3D()
        model.to(device)

        # save maximum beampower
        max_BL= 0
        # Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr, betas=(0.9, 0.999)
        )  # use the sum of 10 losses

        for e in tqdm(range(epoch),desc=f'Train {tt} times: '):
            train_dataset.reset()
            running_loss = 0
            for batch_id in range(train_dataset.batch_num):
                input_signal,beam_power_nonoise,labels_nonoise = train_dataset.load_batch(batch_id, DA_list, e)
                input_signal=input_signal.to(device)
                beam_power_nonoise=beam_power_nonoise.to(device)
                labels_nonoise=labels_nonoise.to(device)
                out_tensor = model(input_signal)
                loss = 0
                for loss_count in range(10):
                    loss += criterion(
                            torch.squeeze(out_tensor[loss_count, :, :]),
                            labels_nonoise[:, loss_count, :] if "Label" in DA_list else labels_nonoise[:, loss_count],
                        )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
            losses = running_loss / train_dataset.batch_num
            
            acur, loss_val,  BL = eval(model, val_dataset, device)
            mBL = np.mean(BL[:,0])
            tqdm.write(f"[{e+1}] train_loss: {losses:.3f}  Val_Loss: {loss_val:.3f} Accuracy: {acur:.3f}  BL: {mBL:.3f}")
            acur_eval[tt, e] = np.squeeze(acur)
            loss_eval[tt, e] = loss_val
            BL_eval[:, :, tt, e] = np.squeeze(BL)
            #save the best model
            if mBL > max_BL:
                max_BL = mBL
                model_name = f"result/{info}_{tt}_MODEL.pkl"
                torch.save(model, model_name)
            model.train()
        print(f'Best Acc: {np.max(acur_eval[tt,:]):.4f}  Best BL: {np.max(np.mean(BL_eval[:,0,tt,:],axis=0)):.4f}')

    mat_name = f"result/{info}_evaluation.mat"
    sio.savemat(
        mat_name,
        {"acur_eval": acur_eval,"loss_eval": loss_eval,"BL_eval": BL_eval}
    )
    
    


if __name__ == "__main__":
    main()
