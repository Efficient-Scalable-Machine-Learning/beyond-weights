# %%
# all imports for colab
from comet_ml import Experiment

import torch
print(f"PyTorch version: {torch.__version__}")
import pandas as pd
import os

os.nice(19)
device = torch.device('cuda:1')

# %% [markdown]
# # Example for MNIST with the focus on training the delays
# 
# ## The problem:
# Given the MNIST dataset, train a multilayer SNN to classify digits from 0 to 9. 
# The input consists of a temporal tensor of size (N, C, W, H, T) where: 
#  - N is the batch size
#  - C is the number of channels, but for this implementation the classification task will be done through a FC network as for the time being a CNN has not been thought out yet
#  - Width: for the task this will be 1
#  - Height: idem
#  - T: time samples of the input (in this case, as of oct. 24th the input temporal dimension is 300)
# 
# Example of input tensor: 
# ![](notebook_images/8_front.png)
# 
# ![](notebook_images/8_side.png)
# 

# %% [markdown]
# ## Load proper paths for SLAYER Pytorch source modules

# %%
import sys, os
CURRENT_TEST_DIR = os.getcwd()
print(f"Dir: {CURRENT_TEST_DIR}")
sys.path.append(CURRENT_TEST_DIR + "/../../../src") #TODO changed here

# %% [markdown]
# ## Load required modules
# 
# SLAYER modules are available as `snn`
# * The `spike-layer` module will be available as `snn.layer`.
# * The `yaml-parameter` module will be availabe as `snn.params`.
# * The `spike-loss` module will be available as `snn.loss`.
# * The `spike-classifier` module will be available as `snn.predict`.
# * The `spike-IO` module will be available as `snn.io`.
# 

# %%
# imports
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import slayerSNN as snn
from torch.utils.data import Dataset, DataLoader


# %%
# Load net params
netParams = snn.params("MNIST_params.yaml")
simul_time = 10
Ts   = netParams['simulation']['Ts']
Ns   = int(netParams['simulation']['tSample'] / netParams['simulation']['Ts'])
Nin  = int(netParams['layer'][0]['dim'])
Nhid = int(netParams['layer'][1]['dim'])
Nout = int(netParams['layer'][2]['dim'])

# %%

hyper_params = {
    "max_epochs": 500,
    "clamp_delays": True,
	"comment": "Smaller net",
    "learning_rate": 0.01,
	"delays_only_lr": 0.01,
    "steps": 50,
    "batch_size": 'n/a',
	"weights_freeze_epoch": 0,
    "target_spikes": (0, 10),
    "simul_time": simul_time,
    "Nin": Nin,
    "Nhid": Nhid, 
    "Nout": Nout,
    "sample_temporal_information_extension": 4, # if 1 all the image is shown in the first ms, else the image values are quantized as spikes that have delays depending on the intensity
    "delay_init_lower": 0,
    "delay_init_upper": 4
}

epochs = hyper_params['max_epochs']

# %% [markdown]
# ## Define transforms to get MNIST to the temporal domain
# 

# %%
import math
class ToTemporalTensor:
    def __init__(self, simul_time, num_temporal):
        self.simul_time = simul_time
        self.num_temporal = num_temporal

    def __call__(self, nonTemp):
        #simulTime = 30
        #num_temporal = 5
        inShape = nonTemp.shape
        #print("Shape:", inShape)
        temp = torch.reshape(nonTemp, (inShape[0], inShape[1]*inShape[2]))
        # I don't really see a way around this.. 
        # Given an MNIST sample, I will make a tensor of which temporal dimension (simulTime) is given, but the information will be 
        # concentrated in the first dimensions, and the limit limit will be chosen by the iterations of the for loop, I think this 
        # is the fastest way of doing this, but I'm not too sure..
        ranges = list(np.linspace(1, 0, self.num_temporal))
        ranges.append(0)
        tensors_list = []
        #print(temp)
        for i in range(len(ranges)-1):
            tp = temp.clone()
            tp = torch.ceil(tp * (tp < ranges[i]) * (tp >= ranges[i+1]))
            tensors_list.append(tp.clone().reshape(inShape[1]*inShape[2], 1, 1, 1))
        #print(len(tensors_list))
        #print("simul is ", self.simul_time)
        #print("bau ", self.simul_time-len(tensors_list))
        tensors_list.append(torch.zeros((inShape[1]*inShape[2], 1, 1, self.simul_time-len(tensors_list))))
        #print("A scieip ", tensors_list[1].shape)
        out = torch.cat(tensors_list, dim = 3)
        #print("out shape is: ", out.shape)
        return out 

class ToMNISTTemporalGroundTruth:
    def __init__(self, simul_time, spike_times_list: list = None, spike_time_range: tuple = None):
        # Note, suggested use is with time_spike_range for speed reasons
        # I don't check for speed reasons, but of course in time_spikes there cannot be time instances greater than simul time. And the range also has to be in [0, simul_time]
        if spike_times_list == None: 
            self.spike_times_list = math.ceil(simul_time/2)
        

        self.spike_times_list = spike_times_list
        self.spike_time_range = spike_time_range
        self.simul_time = simul_time


    def __call__(self, label):
        '''
        In: 
            - label: correct label (because it's mnist => digit from 0 to 9)
        Out: 
            - (temporal_gt, label): temporal version of the ground truth and original label (for accuracy calculation)
        '''
        temporal = torch.zeros(10, 1, 1, self.simul_time)
        if self.spike_times_list is not None: 
            for spike in self.spike_times_list:
                temporal[label, :, :, spike] = 1
        else: 
            temporal[label, :, :, self.spike_time_range[0]:self.spike_time_range[1]] = 1
        
        return (label, temporal)


# %% [markdown]
# ## Define dataset and dataloader classes

# %%
from torch.utils.data import DataLoader

sample_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ToTemporalTensor(simul_time, hyper_params['sample_temporal_information_extension'])
])

label_transform = ToMNISTTemporalGroundTruth(simul_time, spike_time_range= hyper_params['target_spikes'])

train_data = torchvision.datasets.MNIST('mnist_data', train = True, download = True,  transform=sample_transform, target_transform=label_transform)
test_data = torchvision.datasets.MNIST('mnist_data', train = False, download = True, transform=sample_transform, target_transform=label_transform)

train_data, validation_data = torch.utils.data.random_split(train_data, (48000, 12000))



train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size = 32)
validation_loader = torch.utils.data.DataLoader(validation_data, shuffle=False, batch_size = 32)
test_loader = torch.utils.data.DataLoader(test_data, shuffle = False, batch_size = 32)


# %%
def plot_spike_tensor_input(spike_tensor, sim_time):
    #print("Label: ", sample[0].reshape(28,28,300).shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z,x,y = spike_tensor.reshape(28,28,sim_time).numpy().nonzero()
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.show()

def plot_spike_tensor_label(spike_tensor):
    spikes = spike_tensor.numpy().nonzero()
    #print(spikes, " spikes")

sample_label = next(iter(train_loader))
print(sample_label[0][1].shape)
plot_spike_tensor_input(sample_label[0][1], simul_time)
plot_spike_tensor_label(sample_label[1][1])

# %% [markdown]
# ## Define the network
# The network definition follows similar style as standard PyTorch network definition, but it utilizes snn modules.

# %%
def getClassFromDelayNetOutput_DEPRECATED(net_output):
    batch = net_output.shape[0]
    nz = torch.nonzero(net_output).cpu().detach()
    df = pd.DataFrame(nz)
    out = torch.full((batch,), -1)
    if not df.empty:
        first_spikes = df.loc[df.groupby(0)[4].idxmin()]
        first_spikes[[0,4]]
        spike_times_dict = dict(zip(first_spikes[0], first_spikes[1]))
        for item in spike_times_dict.keys():
            out[item] = spike_times_dict[item]
    
    return out

def getClassFromDelayNetOutputTorch(net_output):
    # given spike train, obtains classes checking which class spikes first. If more than one spike is present at a certain time, the right class is determined as the one with the greatest numer of spikes
    # the case of equal arrival time and equal number of spike is still not considered, possible TODO for the future. 
    batch_size = net_output.shape[0] # get batch size

    classes = []
    for i in range(batch_size):
        nz = torch.nonzero(net_output[i,:,:,:,:]).cpu().detach()
        #print(nz)
        nz_temp = nz[:, 3]
        #print("nz temp: ", nz_temp)
        if nz_temp.nelement() == 0: 
            classes.append(-1)
        else:
            argmin = torch.argmin(nz_temp)
            minim = nz_temp[argmin]
            mins_locs = (nz_temp == minim).nonzero() # these gives the row(s) of the min value found
            num_mins = len(mins_locs)
            
            if num_mins == 1: 
                #print("Class is: ", nz[mins_locs, 0].item())
                classes.append(nz[mins_locs, 0].item())
            else:
                nz_first_col = nz[:, 0]
                #print("pre for: ", nz_first_col)
                bincount = torch.bincount(nz_first_col)
                #print("Bincount : ", bincount)
                #print("Bincount argmax: ", torch.argmax(bincount).item())
                classes.append(torch.argmax(bincount).item())
    #nz = torch.nonzero(net_output).cpu().detach()
    return torch.Tensor(classes).type(torch.int)

class DNetwork(torch.nn.Module):
    def __init__(self, netParams, init_stats = None):
        super(DNetwork, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer

        if init_stats is None: 
            print("HEllo, init stat is NONE")
            self.dfc1 = slayer.vectorizedDelayedDense_constrainedWeights(Nin, Nhid, delay_init_lower = hyper_params['delay_init_lower'], delay_init_upper = hyper_params['delay_init_upper'])
            #self.dfc2_5 = slayer.vectorizedDelayedDense(Nhid, 512, delay_init_lower = hyper_params['delay_init_lower'], delay_init_upper = hyper_params['delay_init_upper'])
            #self.dfc2_6 = slayer.vectorizedDelayedDense(512, 256, delay_init_lower = hyper_params['delay_init_lower'], delay_init_upper = hyper_params['delay_init_upper'])
            #self.dfc2_7 = slayer.vectorizedDelayedDense(256, 128)
            self.dfc3 = slayer.vectorizedDelayedDense_constrainedWeights(256, Nout, delay_init_lower = hyper_params['delay_init_lower'], delay_init_upper = hyper_params['delay_init_upper'])
        else:
            self.dfc1 = slayer.vectorizedDelayedDense_constrainedWeights(Nin, 800, init_stats=init_stats[0], delay_init_lower = hyper_params['delay_init_lower'], delay_init_upper = hyper_params['delay_init_upper'])
            self.dfc3 = slayer.vectorizedDelayedDense_constrainedWeights(800, Nout, init_stats=init_stats[1], delay_init_lower = hyper_params['delay_init_lower'], delay_init_upper = hyper_params['delay_init_upper'])
    
    def turn_off_weights_grad(self):
        self.dfc1.weight.requires_grad = False
        self.dfc3.weight.requires_grad = False

    def turn_off_delays_grad(self):
        self.dfc1.delaysTensor.requires_grad = False
        self.dfc3.delaysTensor.requires_grad = False

    def clamp_delays(self):
        self.dfc1.delaysTensor.data.clamp_(0)
        self.dfc3.delaysTensor.data.clamp_(0)
    
    def forward(self, spikeInput):
        print("Spike input shape: ", spikeInput.shape)
        spikeLayer1 = self.slayer.spike(self.dfc1(self.slayer.psp(spikeInput)))
        spikeLayer3 = self.slayer.spike(self.dfc3(self.slayer.psp(spikeLayer1)))
        return spikeLayer3
    
    def getWeightsNorm(self):
        weightNorm = torch.norm(self.dfc1.weight) + torch.norm(self.dfc3.weight)
        return weightNorm

    def get_min_max_delay(self):
        return (min([torch.min(self.dfc1.delaysTensor), torch.min(self.dfc3.delaysTensor)]), max([torch.max(self.dfc1.delaysTensor), torch.max(self.dfc3.delaysTensor)]) )

# %% [markdown]
# ## Initialize the network

# %%
comet_log = True
# %%
# define the cuda device to run the code on

# create a network instance
init_stats = [
    (0.0571, 0.5458), # (mean, std)
    (-0.5244, 1.0490)
]
init_weights = True
if init_weights:
    dnet = DNetwork(netParams, init_stats).to(device)
else:
    dnet = DNetwork(netParams).to(device)

hyper_params['init_weights'] = init_weights
hyper_params['init_stats'] = init_stats
# create snn loss instance
error = snn.loss(netParams).to(device)

lr = 0.01
optimizer = torch.optim.Adam(dnet.parameters(), lr = lr, amsgrad = True)

# %%
next(iter(train_loader))[0].to(device).shape

# %%
fw = None

next(iter(train_loader))[0].shape

# %%
# %%
if False:
    label
    out[2] = 9
    print(out)
    train_acc = (out == label).float().mean()
    print(train_acc)

# %% [markdown]
# ## Visualize the spike data

# %% [markdown]
# # Look at the parameters

# %%
#i = 0
for parametro in dnet.parameters():
  #print(parametro)
  print(f"Size: {parametro.shape}")
  break
next(iter(train_loader))[0][1,:,:,:,:].unsqueeze(0).shape

# %% [markdown]
# # Run the network
# * Run the network for 10000 epochs
# * `bestNetwork` is stored for inferencing later

# %%

# %%
import time

if comet_log:
    experiment = Experiment(
    api_key = "xX6qWBFbiOreu0W3IrO14b9nB",
    project_name = "with-validation",
    workspace="wedrid"
    )

if comet_log:
	experiment.log_parameters(hyper_params)

# %%
print("Number of minibatches: ", len(train_loader))

# %%


# %%
from pathlib import Path
import datetime
from sklearn.metrics import confusion_matrix

print("Comet log: ", comet_log)

pathname = f"out_spikes_plot/{str(datetime.datetime.now())[0:16]}"
Path(pathname).mkdir(parents=True, exist_ok=True)

optimizer = torch.optim.Adam(dnet.parameters(), lr = lr, amsgrad = True)

loss = torch.tensor(10000)
# dnet.turn_off_delays_grad()
for epoch in range(epochs):
    
    epoch_start = time.time()
    # Reset training stats.
    if epoch == hyper_params["weights_freeze_epoch"]: 
        print("Freezing the weights now.")
        optimizer = torch.optim.Adam(dnet.parameters(), lr = hyper_params['delays_only_lr'], amsgrad = True)
        dnet.turn_off_weights_grad()
    
    if epoch % 5 == 0 and False: 
        torch.save(dnet.state_dict(), f"./full_mnist_models/modello_epoch{epoch}.pt")

    tSt = time.time()
    print(f"Start epoch {epoch} with loss {loss.cpu().data.numpy()}")
    # Training loop.
    train_correct = 0

    for i, (input, truth) in enumerate(train_loader, 0):
        
        # Move the input and target to correct GPU.
        # print(len(truth)) #TODO check
        input  = input.to(device)
        target = truth[1].to(device) 
        label = truth[0]

        # Forward pass of the network.
        output = dnet.forward(input)


        # Calculate loss.
        loss = error.spikeTime(output, target)

        # Reset gradients to zero.
        optimizer.zero_grad()

        # Backward pass of the network.
        loss.backward()

        # Update weights.
        optimizer.step()
        if hyper_params['clamp_delays']:
            print('clamping delays')
            dnet.clamp_delays()

        # Gather training loss stats.

        # Display training stats. (Suitable for normal python implementation)
        print("Epoch: ",epoch, " it: ", i, " of ",  len(train_loader), " it time: ", (time.time() - tSt), " loss: ", loss.cpu().data.item())
    
        pred_label = getClassFromDelayNetOutputTorch(output)
        #print(pred_label)
        #print(getClassFromDelayNetOutput_DEPRECATED(output))
        #pred_label = getClassFromDelayNetOutput_DEPRECATED(output)
        #print("SODIHASODIH ", pred_label)
        train_correct += (pred_label == label).float().mean() 
        if False:
            print("Pred: ", pred_label, "\nLabel: ", label)
            print("Correct in this batch: ", (pred_label == label).float())
        print(" mean ", (pred_label == label).float().mean())
        print("Train correct: ", train_correct)

        if True and comet_log:
            experiment.log_metric("loss", loss.cpu().data.numpy(), step=i)
            experiment.log_metric("train_correct", train_correct)
    train_accuracy = 100 * train_correct / len(train_loader) 

    if comet_log:
        delaysHmp, axs = plt.subplots(2, figsize=(15, 15))
        #delaysHmp = plt.figure(figsize=(12, 10))
        if False:
            axs[0].imshow(dnet.dfc1.delaysTensor.detach().reshape(Nin,Nhid).cpu().numpy(), cmap='Blues', interpolation='nearest')
            axs[1].imshow(dnet.dfc2.delaysTensor.detach().reshape(Nhid,Nout).cpu().numpy(), cmap='Blues', interpolation='nearest')
            experiment.log_figure(figure_name="Delays heatmap", figure=delaysHmp, step=epoch)

        for i in [0, 10]:
            f, axarr = plt.subplots(2,1) 
            axarr[0].imshow(output[i, :, 0, 0, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            axarr[1].imshow(target[i, :, 0, 0, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            experiment.log_figure(figure_name="Target vs output", figure=f, step=epoch)
            mm_delays = dnet.get_min_max_delay()
            experiment.log_metric("Min delay: ", mm_delays[0], step = epoch)
            experiment.log_metric("Max delay: ", mm_delays[1], step = epoch)
            experiment.log_metric("Weights norm: ", dnet.getWeightsNorm(), step = epoch)

    # Update training stats.
    if comet_log:
            experiment.log_metric("train loss epoch", loss.cpu().data.numpy(), step=epoch)
            experiment.log_metric("train accuracy epoch", train_accuracy, step=epoch)

    # Testing loop.
    # Same steps as Training loops except loss backpropagation and weight update.
    test_correct = 0 
    y_pred = []
    y_true = []
    epoch_end = time.time()
    if comet_log:
        experiment.log_metric("epoch train time", epoch_end-epoch_start, step=epoch)
    for i, (input, truth) in enumerate(test_loader, 0):
        input  = input.to(device)
        target = truth[1].to(device) 
        output = dnet.forward(input) 
        if False and (i == 0):
            f, axarr = plt.subplots(2,1) 
            axarr[0].imshow(output[0, :, 0, 0, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            axarr[1].imshow(target[0, :, 0, 0, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            #plt.savefig(f"{pathname}/e{epoch}_{i}.png")
            #plt.show()
        loss = error.spikeTime(output, target)
        label = truth[0]
        # stats.print(epoch, i)
        predicted_classes = getClassFromDelayNetOutputTorch(output)
        
        test_correct += (predicted_classes == label).float().mean()
        y_pred.extend(predicted_classes)
        y_true.extend(label)
    if comet_log:
        experiment.log_confusion_matrix(y_true, y_pred)

    test_accuracy = 100 * test_correct / len(test_loader)
    if comet_log:
        experiment.log_metric("test accuracy epoch", test_accuracy, step=epoch)
        experiment.log_metric("test loss epoch", loss, step=epoch)
    ##validation
    test_correct = 0
    for i, (input, truth) in enumerate(validation_loader, 0):
        input  = input.to(device)
        target = truth[1].to(device) 
        output = dnet.forward(input) 
        if False and (i == 0):
            f, axarr = plt.subplots(2,1) 
            axarr[0].imshow(output[0, :, 0, 0, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            axarr[1].imshow(target[0, :, 0, 0, :].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            #plt.savefig(f"{pathname}/e{epoch}_{i}.png")
            #plt.show()
        loss = error.spikeTime(output, target)
        label = truth[0]
        # stats.print(epoch, i)
        predicted_classes = getClassFromDelayNetOutputTorch(output)
        
        test_correct += (predicted_classes == label).float().mean()
        y_pred.extend(predicted_classes)
        y_true.extend(label)
    if comet_log:
        experiment.log_confusion_matrix(y_true, y_pred)

    test_accuracy = 100 * test_correct / len(validation_loader)
    if comet_log:
        experiment.log_metric("validation accuracy epoch", test_accuracy, step=epoch)
        experiment.log_metric("validation loss epoch", loss, step=epoch)

    #torch.save(dnet.state_dict(), f"./models/model_epoch_{epoch}.pt")
if comet_log:
    experiment.end()


# %%
torch.save(dnet.state_dict(), "./modello.pt")

# %%
sys.exit(0)

# %%


# %% [markdown]
# ## Inference using the best network

# %%
output = bestNet.forward(input)

# %%
cpnet = copy.deepcopy(net)
output = bestNet.forward(input)

# %% [markdown]
# ## Plot the Results

# %%
import numpy as np
plt.figure(1, figsize=(9, 7))
plt.semilogy(losslog)
plt.title('Training Loss')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure(2, figsize=(9, 7))
desAER = np.argwhere(desired.reshape((Nout, Ns)).cpu().data.numpy() > 0)
outAER = np.argwhere(output.reshape((Nout, Ns)).cpu().data.numpy() > 0)
plt.plot(desAER[:, 1], desAER[:, 0], 'o', label='desired')
plt.plot(outAER[:, 1], outAER[:, 0], '.', label='actual')
plt.title('Training Loss')
plt.xlabel('time')
plt.ylabel('neuron ID')
plt.legend()

plt.show()