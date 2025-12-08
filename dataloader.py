import numpy as np
import scipy.io as sio
import torch

class Dataloader:
    # initialization
    def __init__(self, path:str, batch_size=160, datacount=40960, drop_last:bool=False, Shuffle:bool=False,p:float=0.5, sigma:float=0.3, Z0:float=2.0, Zf:float=8.0, k:float=6.0, epochs:int=4000):
        self.batch_size = batch_size // 10
        self.dataset_size=datacount//10
        self.batch_num = datacount//batch_size if datacount%batch_size==0 or drop_last else datacount//batch_size+1
        self.DA_fuction = [self.cyclic_shift, self.flip, self.noise_injection, self.LabelAugmentation]
        self.DA_type=["Cyclic_shift", "Flip", "Noise", "Label"]
        self.p = p
        self.sigma= sigma
        self.Z0 = Z0
        self.Zf = Zf
        self.k = k
        self.epochs  =epochs
        data = sio.loadmat(path)
        channel = np.transpose(data["channel_cat"], (1, 2, 0, 3))
        labels = data["max_id_cat"] - 1
        beam_power = data["rsrp_cat"]
        self.channel = channel[0:self.dataset_size,:,:,:]
        self.labels_nonoise = labels[0:self.dataset_size,:]
        self.beam_power_nonoise_m = beam_power[0:self.dataset_size,:,:]
        self.Shuffle = Shuffle
        self.reset()

    def reset(self):
        self.index = np.arange(self.batch_num)
        if self.Shuffle:
            np.random.shuffle(self.index)
    
    def cyclic_shift(self, data, label, beam_power,*args,**kwargs):
        a,b,c,d=data.shape
        to_roll = np.random.rand(a,b)
        to_roll = (to_roll<self.p)
        
        t = np.random.randint(0, d, size=(a,b))
        t = torch.from_numpy(t*to_roll)
        data_arange = torch.arange(d).view(1,1,1,-1).repeat(a,b,c,1)
        data_indices = (data_arange - t.view(a,b,1,1)) % d
        data = torch.gather(data, -1, data_indices)
        label = (label + 4 * t) % 64
        beam_arange = torch.arange(64).view(1,1,-1).repeat(a,b,1)
        beam_indices = (beam_arange - 4*t.unsqueeze(-1)) % 64
        beam_power = torch.gather(beam_power, -1, beam_indices)
        return data, label, beam_power


    def flip(self, data, label, beam_power,*args,**kwargs):
        data = data.reshape(-1,2,16)
        label = label.reshape(-1)
        beam_power = beam_power.reshape(-1,64)
        a,b,c=data.shape
        to_flip = np.random.rand(a)
        arange = torch.arange(a)
        to_flip = arange[to_flip<self.p].tolist()
        data[to_flip,:,:] = torch.flip(data[to_flip,:,:], dims=[-1])        
        data[to_flip,1,:] *= -1
        beam_power[to_flip,:] = torch.flip(beam_power[to_flip,:], dims=[-1])
        label[to_flip] = 63 - label[to_flip]

        data = data.reshape(-1,10,2,16)
        label = label.reshape(-1,10)
        beam_power = beam_power.reshape(-1,10,64)
        return data, label, beam_power
    
    def noise_injection(self, data, label, beam_power,*args,**kwargs):
        a,b,c,d=data.shape
        add_noise = np.random.rand(a,b)
        add_noise = torch.from_numpy(add_noise < self.p).float().view(a,b,1,1)
        data += torch.from_numpy(np.random.randn(a,b,c,d) * self.sigma)*add_noise
        return data, label, beam_power
    
    def LabelAugmentation(self, data, labels, beam_power, epoch):
        if self.Zf == 0:
        # One-hot
            newlabels = (
            torch.zeros(labels.shape[0], 10, 64)
            .scatter_(2, labels.view(-1, 10, 1).long(), 1)
        )
        else:
        # Adaptive power scheduler
            Z = 1.0 if self.epochs==0 else min(self.Z0 + epoch/self.epochs*self.Zf*self.k, self.Zf)
            newlabels = beam_power**(Z/2)
        
        return data, newlabels, beam_power

    def load_all_set(self):
        beam_power = np.float32(self.beam_power_nonoise_m)
        labels = self.labels_nonoise.astype(np.int64)
        labels = torch.from_numpy(labels)
        beam_power = torch.from_numpy(beam_power)
        return (beam_power,labels)
    
    def load_batch(self,idx, DA_list:list=[], epoch:int=0):
        idx = self.index[idx]
        start = idx*self.batch_size
        end = min((idx+1)*self.batch_size,self.dataset_size)
        channel = np.float32(self.channel[start:end,:,:,:])
        beam_power = np.float32(self.beam_power_nonoise_m[start:end,:,:])
        labels = self.labels_nonoise[start:end,:].astype(np.int64)
        channel = torch.from_numpy(channel)
        labels = torch.from_numpy(labels)
        beam_power = torch.from_numpy(beam_power)
        for i,da_type in enumerate(self.DA_type):
            if da_type in DA_list:
                channel,labels,beam_power = self.DA_fuction[i](channel,labels,beam_power,epoch)
        # channel = channel.to(torch.float32)
        # beam_power = channel.to()
        return (channel.transpose(1,2),beam_power,labels)

if __name__ == "__main__":
   dataloader = Dataloader("dataset/data_16Tx_64Tx_training.mat",batch_size=160)
   x_torch,beam,label = dataloader.load_batch(0,[])
   print(x_torch.shape,beam.shape,label.shape)

