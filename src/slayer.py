import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
# import slayer_cuda
import slayerCuda
import copy
from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
import einops
import sys
# import matplotlib.pyplot as plt

# # Consider dictionary for easier iteration and better scalability
# class yamlParams(object):
#   '''
#   This class reads yaml parameter file and allows dictionary like access to the members.
    
#   Usage:

#   .. code-block:: python
        
#       import slayerSNN as snn
#       netParams = snn.params('path_to_yaml_file') # OR
#       netParams = slayer.yamlParams('path_to_yaml_file')

#       netParams['training']['learning']['etaW'] = 0.01
#       print('Simulation step size        ', netParams['simulation']['Ts'])
#       print('Spiking neuron time constant', netParams['neuron']['tauSr'])
#       print('Spiking neuron threshold    ', netParams['neuron']['theta'])

#       netParams.save('filename.yaml')
#   '''
#   def __init__(self, parameter_file_path):
#       with open(parameter_file_path, 'r') as param_file:
#           self.parameters = yaml.safe_load(param_file)

#   # Allow dictionary like access
#   def __getitem__(self, key):
#       return self.parameters[key]

#   def __setitem__(self, key, value):
#       self.parameters[key] = value

#   def save(self, filename):
#       with open(filename, 'w') as f:
#           yaml.dump(self.parameters, f)

# class spikeLayer():
class spikeLayer(torch.nn.Module):
    '''
    This class defines the main engine of SLAYER.
    It provides necessary functions for describing a SNN layer.
    The input to output connection can be fully-connected, convolutional, or aggregation (pool)
    It also defines the psp operation and spiking mechanism of a spiking neuron in the layer.

    **Important:** It assumes all the tensors that are being processed are 5 dimensional. 
    (Batch, Channels, Height, Width, Time) or ``NCHWT`` format.
    The user must make sure that an input of correct dimension is supplied.

    *If the layer does not have spatial dimension, the neurons can be distributed along either
    Channel, Height or Width dimension where Channel * Height * Width is equal to number of neurons.
    It is recommended (for speed reasons) to define the neuons in Channels dimension and make Height and Width
    dimension one.*

    Arguments:
        * ``neuronDesc`` (``slayerParams.yamlParams``): spiking neuron descriptor.
            .. code-block:: python

                neuron:
                    type:     SRMALPHA  # neuron type
                    theta:    10    # neuron threshold
                    tauSr:    10.0  # neuron time constant
                    tauRef:   1.0   # neuron refractory time constant
                    scaleRef: 2     # neuron refractory response scaling (relative to theta)
                    tauRho:   1     # spike function derivative time constant (relative to theta)
                    scaleRho: 1     # spike function derivative scale factor
        * ``simulationDesc`` (``slayerParams.yamlParams``): simulation descriptor
            .. code-block:: python

                simulation:
                    Ts: 1.0         # sampling time (ms)
                    tSample: 300    # time length of sample (ms)   
        * ``fullRefKernel`` (``bool``, optional): high resolution refractory kernel (the user shall not use it in practice)  
    
    Usage:

    >>> snnLayer = slayer.spikeLayer(neuronDesc, simulationDesc)
    '''
    def __init__(self, neuronDesc, simulationDesc, fullRefKernel = False):
        super(spikeLayer, self).__init__()
        self.neuron = neuronDesc
        self.simulation = simulationDesc
        self.fullRefKernel = fullRefKernel
        
        # self.srmKernel = self.calculateSrmKernel()
        # self.refKernel = self.calculateRefKernel()
        self.register_buffer('srmKernel', self.calculateSrmKernel())
        self.register_buffer('refKernel', self.calculateRefKernel())
        
    def calculateSrmKernel(self):
        srmKernel = self._calculateAlphaKernel(self.neuron['tauSr'])
        # TODO implement for different types of kernels
        return torch.FloatTensor(srmKernel)
        # return torch.FloatTensor( self._zeroPadAndFlip(srmKernel)) # to be removed later when custom cuda code is implemented
        
    def calculateRefKernel(self):
        if self.fullRefKernel:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'], EPSILON = 0.0001)
            # This gives the high precision refractory kernel as MATLAB implementation, however, it is expensive
        else:
            refKernel = self._calculateAlphaKernel(tau=self.neuron['tauRef'], mult = -self.neuron['scaleRef'] * self.neuron['theta'])
        
        # TODO implement for different types of kernels
        return torch.FloatTensor(refKernel)
        
    def _calculateAlphaKernel(self, tau, mult = 1, EPSILON = 0.01):
        # could be made faster... NOT A PRIORITY NOW
        eps = []
        # tauSr = self.neuron['tauSr']
        for t in np.arange(0, self.simulation['tSample'], self.simulation['Ts']):
            epsVal = mult * t / tau * math.exp(1 - t / tau)
            if abs(epsVal) < EPSILON and t > tau:
                break
            eps.append(epsVal)
        return eps
    
    def _zeroPadAndFlip(self, kernel):
        if (len(kernel)%2) == 0: kernel.append(0)
        prependedZeros = np.zeros((len(kernel) - 1))
        return np.flip( np.concatenate( (prependedZeros, kernel) ) ).tolist()
        
    def psp(self, spike):
        '''
        Applies psp filtering to spikes.
        The output tensor dimension is same as input.

        Arguments:
            * ``spike``: input spike tensor.

        Usage:

        >>> filteredSpike = snnLayer.psp(spike)
        '''
        return _pspFunction.apply(spike, self.srmKernel, self.simulation['Ts'])

    def pspLayer(self):
        '''
        Returns a function that can be called to apply psp filtering to spikes.
        The output tensor dimension is same as input.
        The initial psp filter corresponds to the neuron psp filter.
        The psp filter is learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs correlation operation.
        
        Usage:
        
        >>> pspLayer = snnLayer.pspLayer()
        >>> filteredSpike = pspLayer(spike)
        '''
        return _pspLayer(self.srmKernel, self.simulation['Ts'])

    def pspFilter(self, nFilter, filterLength, filterScale=1):
        '''
        Returns a function that can be called to apply a bank of temporal filters.
        The output tensor is of same dimension as input except the channel dimension is scaled by number of filters.
        The initial filters are initialized using default PyTorch initializaion for conv layer.
        The filter banks are learnable.
        NOTE: the learned psp filter must be reversed because PyTorch performs conrrelation operation.
        
        Arguments:
            * ``nFilter``: number of filters in the filterbank.
            * ``filterLength``: length of filter in number of time bins.
            * ``filterScale``: initial scaling factor for filter banks. Default: 1.

        Usage:
        
        >>> pspFilter = snnLayer.pspFilter()
        >>> filteredSpike = pspFilter(spike)
        '''
        return _pspFilter(nFilter, filterLength, self.simulation['Ts'], filterScale)

    def replicateInTime(self, input, mode='nearest'):
        Ns = int(self.simulation['tSample'] / self.simulation['Ts'])
        N, C, H, W = input.shape
        # output = F.pad(input.reshape(N, C, H, W, 1), pad=(Ns-1, 0, 0, 0, 0, 0), mode='replicate')
        if mode == 'nearest':
            output = F.interpolate(input.reshape(N, C, H, W, 1), size=(H, W, Ns), mode='nearest')
        return output
    
    def dense(self, inFeatures, outFeatures, weightScale=10, preHookFx=None):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _denseLayer(inFeatures, outFeatures, weightScale, preHookFx)

    #conv(inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, preHookFx=None)
    def delayConv2D(self, input_depth = 1, output_depth = 1, kernel_size = None, stride = 1, padding = 0, trainable_weights = False, init_stats = None):
        '''
        Returns a function that can be called to apply a convolution on the temporal tensor, based on delays and not weights. 
        Params: 
            - input_depth: number of input channels
            - output_depth: number of output channels
            - kernel_size: what you'd expect
            - stride: what you'd expect
            - padding: what you'd expect
        Note: the convolution is made with respect to the spatial dimention. 
        '''
        return _delayConv2DLayer(input_depth = input_depth, output_depth = output_depth, kernel_size = kernel_size, stride = stride, padding = padding, trainable_weights = trainable_weights, init_stats = init_stats)

    def delayConv2DFix(self, input_depth = 1, output_depth = 1, kernel_size = None, stride = 1, padding = 0, trainable_weights = False, init_stats = None):
        '''
        Returns a function that can be called to apply a convolution on the temporal tensor, based on delays and not weights. 
        Params: 
            - input_depth: number of input channels
            - output_depth: number of output channels
            - kernel_size: what you'd expect
            - stride: what you'd expect
            - padding: what you'd expect
        Note: the convolution is made with respect to the spatial dimention. 
        '''
        return _delayConv2DLayerFix(input_depth = input_depth, output_depth = output_depth, kernel_size = kernel_size, stride = stride, padding = padding, trainable_weights = trainable_weights, init_stats = init_stats)


    def delayedDense(self, inFeatures, outFeatures, weightScale=10, preHookFx=None):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _delayedDenseLayer(inFeatures, outFeatures, weightScale, preHookFx)

    def vectorizedDelayedDense(self, inFeatures, outFeatures, weightScale=10, preHookFx=None, init_stats = None, delay_init_lower = 0, delay_init_upper = 1):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _vectorizedDelayedDenseLayer(inFeatures, outFeatures, weightScale, preHookFx, init_stats=init_stats, delay_init_lower = delay_init_lower, delay_init_upper = delay_init_upper)

    def vectorizedDelayedDense_constrainedWeights(self, inFeatures, outFeatures, weightScale=10, preHookFx=None, init_stats = None, delay_init_lower = 0, delay_init_upper = 1):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _vectorizedDelayedDenseLayer_constrainedWeights(inFeatures, outFeatures, weightScale, preHookFx, init_stats=init_stats, delay_init_lower = delay_init_lower, delay_init_upper = delay_init_upper)

    
    def vectorizedDelayedDenseStoca(self, inFeatures, outFeatures, weightScale=10, preHookFx=None, init_stats = None, delay_init_lower = 0, delay_init_upper = 1):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _vectorizedDelayedDenseLayerStoca(inFeatures, outFeatures, weightScale, preHookFx, init_stats=init_stats, delay_init_lower = delay_init_lower, delay_init_upper = delay_init_upper)

    def vectorizedAbsDelayedDense(self, inFeatures, outFeatures, weightScale=10, preHookFx=None, init_stats = None):   # default weight scaling of 10
        '''
        Returns a function that can be called to apply dense layer mapping to input tensor per time instance.
        It behaves similar to ``torch.nn.Linear`` applied for each time instance.

        Arguments:
            * ``inFeatures`` (``int``, tuple of two ints, tuple of three ints): 
              dimension of input features (Width, Height, Channel) that represents the number of input neurons.
            * ``outFeatures`` (``int``): number of output neurons.
            * ``weightScale``: sale factor of default initialized weights. Default: 10
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        Usage:
        
        >>> fcl = snnLayer.dense(2048, 512)          # takes (N, 2048, 1, 1, T) tensor
        >>> fcl = snnLayer.dense((128, 128, 2), 512) # takes (N, 2, 128, 128, T) tensor
        >>> output = fcl(input)                      # output will be (N, 512, 1, 1, T) tensor
        '''
        return _vectorizedAbsDelayedDenseLayer(inFeatures, outFeatures, weightScale, preHookFx, init_stats=init_stats)

    def myDense(self, inFeatures, outFeatures, weightScale=10, preHookFx=None):   # default weight scaling of 10
        return _myDense(inFeatures, outFeatures, weightScale, preHookFx)
    
        
    def conv(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, preHookFx=None):    # default weight scaling of 100
        '''
        Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.conv2d`` applied for each time instance.

        Arguments:
            * ``inChannels`` (``int``): number of channels in input
            * ``outChannels`` (``int``): number of channls produced by convoluion
            * ``kernelSize`` (``int`` or tuple of two ints): size of the convolving kernel
            * ``stride`` (``int`` or tuple of two ints): stride of the convolution. Default: 1
            * ``padding`` (``int`` or tuple of two ints):   zero-padding added to both sides of the input. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
            * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
            * ``weightScale``: sale factor of default initialized weights. Default: 100
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> conv = snnLayer.conv(2, 32, 5) # 32C5 flter
        >>> output = conv(input)           # must have 2 channels
        '''
        return _convLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, preHookFx) 
        
    def pool(self, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        '''
        Returns a function that can be called to apply pool layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.``:sum pooling applied for each time instance.

        Arguments:
            * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to pool over
            * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
            * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.
            
        The parameters ``kernelSize``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> pool = snnLayer.pool(4) # 4x4 pooling
        >>> output = pool(input)
        '''
        return _poolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation, preHookFx)

    def convTranspose(self, inChannels, outChannels, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=100, preHookFx=None):
        '''
        Returns a function that can be called to apply conv layer mapping to input tensor per time instance.
        It behaves the same as ``torch.nn.ConvTranspose3d`` applied for each time instance.

        Arguments:
            * ``inChannels`` (``int``): number of channels in input
            * ``outChannels`` (``int``): number of channels produced by transposed convolution
            * ``kernelSize`` (``int`` or tuple of two ints): size of ransposed convolution kernel
            * ``stride`` (``int`` or tuple of two ints): stride of the transposed convolution. Default: 1
            * ``padding`` (``int`` or tuple of two ints): amount of implicit zero-padding added to both sides of the input. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): spacing between kernel elements. Default: 1
            * ``groups`` (``int`` or tuple of two ints): number of blocked connections from input channels to output channels. Default: 1
            * ``weightScale`` : scale factor of default initialized weights. Default: 100
            * ``preHookFx``: a function that operates on weights before applying it. Could be used for quantization etc.
        
        The parameters kernelSize, stride, padding, dilation can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a `tuple` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second is used for the width dimension

        Usage:

        >>> convT = snnLayer.convTranspose(32, 2, 5) # 2T5 flter, the opposite of 32C5 filter
        >>> output = convT(input)
        '''
        return _convTransposeLayer(inChannels, outChannels, kernelSize, stride, padding, dilation, groups, weightScale, preHookFx)

    def unpool(self, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        '''
        Returns a function that can be called to apply unpool layer mapping to input tensor per time instance.
        It behaves same as ``torch.nn.`` unpool layers.

        Arguments:
            * ``kernelSize`` (``int`` or tuple of two ints): the size of the window to unpool over
            * ``stride`` (``int`` or tuple of two ints): stride of the window. Default: `kernelSize`
            * ``padding`` (``int`` or tuple of two ints): implicit zero padding to be added on both sides. Default: 0
            * ``dilation`` (``int`` or tuple of two ints): a parameter that controls the stride of elements in the window. Default: 1
            * ``preHookFx``: a function that operates on weight before applying it. Could be used for quantization etc.

        The parameters ``kernelSize``, ``stride``, ``padding``, ``dialtion`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

        Usage:

        >>> unpool = snnLayer.unpool(2) # 2x2 unpooling
        >>> output = unpool(input)
        '''
        return _unpoolLayer(self.neuron['theta'], kernelSize, stride, padding, dilation, preHookFx)

    def dropout(self, p=0.5, inplace=False):
        '''
        Returns a function that can be called to apply dropout layer to the input tensor.
        It behaves similar to ``torch.nn.Dropout``.
        However, dropout over time dimension is preserved, i.e.
        if a neuron is dropped, it remains dropped for entire time duration.

        Arguments:
            * ``p``: dropout probability.
            * ``inplace`` (``bool``): inplace opeartion flag.

        Usage:

        >>> drop = snnLayer.dropout(0.2)
        >>> output = drop(input)
        '''
        return _dropoutLayer(p, inplace)

    def delayShift(self, input, delay, Ts=1):
        '''
        Applies delay in time dimension (assumed to be the last dimension of the tensor) of the input tensor.
        The autograd backward link is established as well.

        Arguments:
            * ``input``: input Torch tensor.
            * ``delay`` (``float`` or Torch tensor): amount of delay to apply.
              Same delay is applied to all the inputs if ``delay`` is ``float`` or Torch tensor of size 1.
              If the Torch tensor has size more than 1, its dimension  must match the dimension of input tensor except the last dimension.
            * ``Ts``: sampling time of the delay. Default is 1.
        
        Usage:

        >>> delayedInput = slayer.delayShift(input, 5)
        '''
        return _delayFunctionNoGradient.apply(input, delay, Ts)

    def delay(self, inputSize):
        '''
        Returns a function that can be called to apply delay opeartion in time dimension of the input tensor.
        The delay parameter is available as ``delay.delay`` and is initialized uniformly between 0ms  and 1ms.
        The delay parameter is stored as float values, however, it is floored during actual delay applicaiton internally.
        The delay values are not clamped to zero.
        To maintain the causality of the network, one should clamp the delay values explicitly to ensure positive delays.

        Arguments:
            * ``inputSize`` (``int`` or tuple of three ints): spatial shape of the input signal in CHW format (Channel, Height, Width).
              If integer value is supplied, it refers to the number of neurons in channel dimension. Heighe and Width are assumed to be 1.   

        Usage:

        >>> delay = snnLayer.delay((C, H, W))
        >>> delayedSignal = delay(input)

        Always clamp the delay after ``optimizer.step()``.

        >>> optimizer.step()
        >>> delay.delay.data.clamp_(0)  
        '''
        return _delayLayer(inputSize, self.simulation['Ts'])
    
    # def applySpikeFunction(self, membranePotential):
    #   return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

    def spike(self, membranePotential):
        '''
        Applies spike function and refractory response.
        The output tensor dimension is same as input.
        ``membranePotential`` will reflect spike and refractory behaviour as well.

        Arguments:
            * ``membranePotential``: subthreshold membrane potential.

        Usage:

        >>> outSpike = snnLayer.spike(membranePotential)
        '''
        return _spikeFunction.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])
    
    def timeOfFirstSpike(self, membranePotential):
        return _timeOfFirstSpike.apply(membranePotential, self.refKernel, self.neuron, self.simulation['Ts'])

class _denseLayer(nn.Conv3d):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None):
        '''
        '''
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures 
        elif len(inFeatures) == 2:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = 1
        elif len(inFeatures) == 3:
            kernel = (inFeatures[1], inFeatures[0], 1)
            inChannels = inFeatures[2]
        else:
            raise Exception('inFeatures should not be more than 3 dimension. It was: {}'.format(inFeatures.shape))
        # print('Kernel Dimension:', kernel)
        # print('Input Channels  :', inChannels)
        
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        super(_denseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx

    
    def forward(self, input):
        '''
        '''
        #print("Conv init weights are: ", self.weight)
        if self.preHookFx is None:
            return F.conv3d(input, 
                            self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.preHookFx(self.weight), self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)

class _delayConv2DLayer(nn.Module):
    def __init__(self, input_depth = 1, output_depth = 1, kernel_size = None, stride = 1, padding = None, device = torch.device('cuda'), init_stats = None, trainable_weights = False):
        super(_delayConv2DLayer, self).__init__()
        assert (kernel_size is not None)
        # input is of the form (batch, width, height, channels, time)
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        # TODO this module will be moved to the slayer file, so these probably won't be needed. 
        # TODO init statistics to the dense layer
        self.fc_delay_layer = _vectorizedDelayedDenseLayer(kernel_size*kernel_size*input_depth, output_depth, delay_init_lower = 0, delay_init_upper = 5, init_stats = init_stats) #TODO fix delay init property, maybe put it as a global variable? #TODO: I removed the "to(device)" here
        
        if not trainable_weights: 
            print('Switching off weights gradients')
            self.fc_delay_layer.weight.requires_grad = False


    def forward(self, x):
        # 1) the input has to be sliced like a convolution would. 
        #assumes that kernels are square and also stride.
        # TODO: plot the patches
        x = x.to(self.device)
        batch_size = x.shape[0]
        #print("Batch size: ", batch_size)
        x = x.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride) #generates all the "patches"
        #print("Previous patches shape: ", x.shape)
        new_width, new_height = x.shape[1], x.shape[2]
        x = einops.rearrange(x, 'b nw nh d t kw kh -> (b nw nh) (kw kh d) t').unsqueeze(2).unsqueeze(3).to(self.device) #unfortunately I have to do unsequeeze because of how I implemented the delay layer
        #print("As input to FC: ", x.shape)
        #x = self.fc_delay_layer(self.slayer.psp(x)).squeeze() #FIXME: I remove the psp, commented is the original, down is the new version. psp will be called in the network structure
        x = self.fc_delay_layer(x).squeeze()
        #print("After fc layer: ", x.shape)
        x = einops.rearrange(x, '(b nw nh) d t -> b nw nh d t', b = batch_size, nw = new_width, nh = new_height, d = self.output_depth)
        #print("Convolutional output: ", x.shape)
        return x
    
    def clamp_fc_delays(self):
        self.fc_delay_layer.delaysTensor.data.clamp_(0)

    def getWeightsNorm(self):
        return torch.linalg.norm(self.fc_delay_layer.weight)
    
    def getDelaysNorm(self): 
        return torch.linalg.norm(self.fc_delay_layer.delaysTensor)

class _delayConv2DLayerFix(nn.Module):
    def __init__(self, input_depth = 1, output_depth = 1, kernel_size = None, stride = 1, padding = None, device = torch.device('cuda'), init_stats = None, trainable_weights = False):
        super(_delayConv2DLayerFix, self).__init__()
        assert (kernel_size is not None)
        # input is of the form (batch, width, height, channels, time)
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        # TODO this module will be moved to the slayer file, so these probably won't be needed. 
        # TODO init statistics to the dense layer
        self.fc_delay_layer = _vectorizedDelayedDenseLayer(kernel_size*kernel_size*input_depth, output_depth, delay_init_lower = 0, delay_init_upper = 5, init_stats = init_stats, weightScale = 10) #TODO fix delay init property, maybe put it as a global variable? #TODO: I removed the "to(device)" here
        
        if not trainable_weights: 
            print('Switching off weights gradients')
            self.fc_delay_layer.weight.requires_grad = False


    def forward(self, x):
        #### FIX before we expected [N W H D T], now we expect [N D W H T], so I convert the latter to the former
        x = x.permute(0, 2, 3, 1, 4)

        # 1) the input has to be sliced like a convolution would. 
        #assumes that kernels are square and also stride.
        # TODO: plot the patches
        x = x.to(self.device)
        batch_size = x.shape[0]
        #print("Batch size: ", batch_size)
        x = x.unfold(1, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride) #generates all the "patches"
        #print("Previous patches shape: ", x.shape)
        new_width, new_height = x.shape[1], x.shape[2]
        x = einops.rearrange(x, 'b nw nh d t kw kh -> (b nw nh) (kw kh d) t').unsqueeze(2).unsqueeze(3).to(self.device) #unfortunately I have to do unsequeeze because of how I implemented the delay layer
        #print("As input to FC: ", x.shape)
        #x = self.fc_delay_layer(self.slayer.psp(x)).squeeze() #FIXME: I remove the psp, commented is the original, down is the new version. psp will be called in the network structure
        x = self.fc_delay_layer(x).squeeze()
        #print("After fc layer: ", x.shape)
        x = einops.rearrange(x, '(b nw nh) d t -> b nw nh d t', b = batch_size, nw = new_width, nh = new_height, d = self.output_depth)
        #print("Convolutional output: ", x.shape)

        ## FIX: we outputted this [N W H D T], now we want to output this: [N D W H T]
        x = x.permute(0, 3, 1, 2, 4)
        return x
    
    def clamp_fc_delays(self):
        self.fc_delay_layer.delaysTensor.data.clamp_(0)

    def getWeightsNorm(self):
        return torch.linalg.norm(self.fc_delay_layer.weight)
    
    def getDelaysNorm(self): 
        return torch.linalg.norm(self.fc_delay_layer.delaysTensor)

class _vectorizedDelayedDenseLayer(nn.Module):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None, ts = 1, device='cuda', init_stats = None, delay_init_lower = 0, delay_init_upper = 1): #TODO ts can't be fixed like that, and device to change
        '''
        - inFeatures (int, tuple of two ints, tuple of three ints): dimension of input features (Width, Height, Channel) 
            that represents the number of input neurons.
        - outFeatures (int): number of output neurons.
        '''
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures # = the lenght of the input, i.e. the only inFeature
        else:
            raise Exception('Delayed dense layer has for now only been implemented so that it accepts an int as inFeatures, that represents the number of in features (i.e. no channels or other things)')
        '''
        TODO: for now I only accept single-channeled inputs, that should be enough both for the spike train and for nmnist. 
        Example of NMNIST FC:
            torch.Size([512, 2312, 1, 1, 1]) #[out, in, 1, 1, 1]
            torch.Size([10, 512, 1, 1, 1])
        '''
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        #super(_delayedDenseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        super(_vectorizedDelayedDenseLayer, self).__init__()

        #initialize self.variables because I blindly inherited from conv3d initially, but we're not doing a convolution anymore
        self.inFeatures = inChannels
        self.outFeatures = outChannels
        #self.weight = torch.nn.Parameter(torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device), requires_grad = True)
        
        #weight_temp = torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device)
        #torch.nn.init.xavier_uniform_(weight_temp, gain=1.0)
        #self.old_weight = torch.nn.Parameter(weight_temp, requires_grad = True)


        self.Ts = ts

        # so I think that the weight matrix has 5 dimensions because it's easier for the matrix multiplications, but in the case of the delays that's not the case, so it's sufficient a 2D matrix (which must be augmented in the case of different channels, but TODO eventualmente)
        # self.delayMatrix = torch.nn.Parameter(torch.rand(outFeatures, inFeatures, 1, 1), requires_grad=True, ) # added, should match weights matrix size. (idk why outFeatures and inFeatures are inverted but I leave it as is). This cannot be done before call of super
        # TODO it's strange that indexing gives problem, it actually seems to be fully supported by autograd
        # so, I believe that because of reshaping 
        if False:
            delayTensorTemp = torch.empty(self.outFeatures*self.inFeatures, 1, 1).to(device)
            torch.nn.init.xavier_uniform_(delayTensorTemp, gain=1.0)
            self.delaysTensor = torch.nn.Parameter(delayTensorTemp, requires_grad=True)
        else:
            self.delaysTensor = torch.nn.Parameter(((delay_init_lower - delay_init_upper)*torch.rand(self.outFeatures*self.inFeatures, 1, 1)+delay_init_upper).to(device), requires_grad = True)
            #self.delaysTensor = torch.nn.Parameter(torch.rand(self.outFeatures*self.inFeatures, 1, 1).to(device), requires_grad = True)

        if init_stats is None:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.xavier_uniform_(pesi, gain=1.0)
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)
            #self.weight = pesi # new version in which weights are not a trainable parameter
        else:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.normal_(pesi, init_stats[0], init_stats[1]) # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)
            #self.weight = pesi # new version in which weights are not a trainable parameter

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx
        # weights matrix is something like this in colab torch.Size([1024, 200, 1, 1, 1]) where in this case
        # 1024 is the out dimension and 200 is the in dimension
        

    def forward(self, input):

        device = input.device
        shp = input.shape
        #print("The shapee is: ", shp)
        
        #print(f"Previous delays are {len(self.delaysVectors)} of shape {self.delaysVectors[0].shape}")
        input = torch.repeat_interleave(input, self.outFeatures, 1)
        #print("input shape repeat interleave ", input.shape)
        #print(f"Shapes are: \n - {self.delaysTensor.shape[0]} \n - {input.shape[1]}")
        assert ( self.delaysTensor.shape[0] == input.shape[1] )
        #print(f"Input shape of delay is: {repeatedInput.shape}, delay tensor: {self.delaysTensor.shape}")
        #print(f"Before mul: {delayedRepeatedInput}")
        #inp = copy.deepcopy(input.detach())
        
        #print("After weight: ", torch.norm(inp-output))
        #outp = copy.deepcopy(output.detach())
        output = input * self.weight
        output = _delayFunction.apply(output, self.delaysTensor, self.Ts) #TODO check: note, in this case we can apply either delay or weights, the order doesn't matter here because the sum is made after
        #print("After delay: ", torch.norm(outp-output))
        
        #out = torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device)
        # original 
        #index = torch.arange(start=0, end=self.outFeatures).reshape(shp[0], self.outFeatures, 1, 1, 1).repeat((1, self.inFeatures, 1 ,1 ,shp[4])).to(device)
        index = torch.arange(start=0, end=self.outFeatures).reshape(1, self.outFeatures, 1, 1, 1).repeat((shp[0], self.inFeatures, 1 ,1 ,shp[4])).to(device)
        #out.scatter_add_(1, index, output)
        
        output = torch.scatter_add(torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device), 1, index, output)

        return output

class _vectorizedDelayedDenseLayer_constrainedWeights(nn.Module):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None, ts = 1, device='cuda', init_stats = None, delay_init_lower = 0, delay_init_upper = 1): #TODO ts can't be fixed like that, and device to change
        '''
        - inFeatures (int, tuple of two ints, tuple of three ints): dimension of input features (Width, Height, Channel) 
            that represents the number of input neurons.
        - outFeatures (int): number of output neurons.
        '''
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures # = the lenght of the input, i.e. the only inFeature
        else:
            raise Exception('Delayed dense layer has for now only been implemented so that it accepts an int as inFeatures, that represents the number of in features (i.e. no channels or other things)')
        '''
        TODO: for now I only accept single-channeled inputs, that should be enough both for the spike train and for nmnist. 
        Example of NMNIST FC:
            torch.Size([512, 2312, 1, 1, 1]) #[out, in, 1, 1, 1]
            torch.Size([10, 512, 1, 1, 1])
        '''
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        #super(_delayedDenseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        super(_vectorizedDelayedDenseLayer_constrainedWeights, self).__init__()

        #initialize self.variables because I blindly inherited from conv3d initially, but we're not doing a convolution anymore
        self.inFeatures = inChannels
        self.outFeatures = outChannels
        #self.weight = torch.nn.Parameter(torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device), requires_grad = True)
        
        #weight_temp = torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device)
        #torch.nn.init.xavier_uniform_(weight_temp, gain=1.0)
        #self.old_weight = torch.nn.Parameter(weight_temp, requires_grad = True)


        self.Ts = ts

        # so I think that the weight matrix has 5 dimensions because it's easier for the matrix multiplications, but in the case of the delays that's not the case, so it's sufficient a 2D matrix (which must be augmented in the case of different channels, but TODO eventualmente)
        # self.delayMatrix = torch.nn.Parameter(torch.rand(outFeatures, inFeatures, 1, 1), requires_grad=True, ) # added, should match weights matrix size. (idk why outFeatures and inFeatures are inverted but I leave it as is). This cannot be done before call of super
        # TODO it's strange that indexing gives problem, it actually seems to be fully supported by autograd
        # so, I believe that because of reshaping 
        if False:
            delayTensorTemp = torch.empty(self.outFeatures*self.inFeatures, 1, 1).to(device)
            torch.nn.init.xavier_uniform_(delayTensorTemp, gain=1.0)
            self.delaysTensor = torch.nn.Parameter(delayTensorTemp, requires_grad=True)
        else:
            self.delaysTensor = torch.nn.Parameter(((delay_init_lower - delay_init_upper)*torch.rand(self.outFeatures*self.inFeatures, 1, 1)+delay_init_upper).to(device), requires_grad = True)
            #self.delaysTensor = torch.nn.Parameter(torch.rand(self.outFeatures*self.inFeatures, 1, 1).to(device), requires_grad = True)

        if init_stats is None:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.xavier_uniform_(pesi, gain=1.0)
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)
            #self.weight = pesi # new version in which weights are not a trainable parameter
        else:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.normal_(pesi, init_stats[0], init_stats[1]) # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
            pesi = torch.round( pesi ) 
            pesi = torch.clamp(pesi, -1, 1)
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)
            #self.weight = pesi # new version in which weights are not a trainable parameter
            

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)
        self.preHookFx = preHookFx
        # weights matrix is something like this in colab torch.Size([1024, 200, 1, 1, 1]) where in this case
        # 1024 is the out dimension and 200 is the in dimension
        

    def forward(self, input):

        device = input.device
        shp = input.shape
        #print("The shapee is: ", shp)
        
        #print(f"Previous delays are {len(self.delaysVectors)} of shape {self.delaysVectors[0].shape}")
        input = torch.repeat_interleave(input, self.outFeatures, 1)
        #print("input shape repeat interleave ", input.shape)
        #print(f"Shapes are: \n - {self.delaysTensor.shape[0]} \n - {input.shape[1]}")
        assert ( self.delaysTensor.shape[0] == input.shape[1] )
        #print(f"Input shape of delay is: {repeatedInput.shape}, delay tensor: {self.delaysTensor.shape}")
        #print(f"Before mul: {delayedRepeatedInput}")
        #inp = copy.deepcopy(input.detach())
        
        #print("After weight: ", torch.norm(inp-output))
        #outp = copy.deepcopy(output.detach())
        output = input * self.weight
        output = _delayFunction.apply(output, self.delaysTensor, self.Ts) #TODO check: note, in this case we can apply either delay or weights, the order doesn't matter here because the sum is made after
        #print("After delay: ", torch.norm(outp-output))
        
        #out = torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device)
        # original 
        #index = torch.arange(start=0, end=self.outFeatures).reshape(shp[0], self.outFeatures, 1, 1, 1).repeat((1, self.inFeatures, 1 ,1 ,shp[4])).to(device)
        index = torch.arange(start=0, end=self.outFeatures).reshape(1, self.outFeatures, 1, 1, 1).repeat((shp[0], self.inFeatures, 1 ,1 ,shp[4])).to(device)
        #out.scatter_add_(1, index, output)
        
        output = torch.scatter_add(torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device), 1, index, output)

        return output

class _vectorizedDelayedDenseLayerStoca(nn.Module):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None, ts = 1, device='cuda', init_stats = None, delay_init_lower = 0, delay_init_upper = 1): #TODO ts can't be fixed like that, and device to change
        '''
        - inFeatures (int, tuple of two ints, tuple of three ints): dimension of input features (Width, Height, Channel) 
            that represents the number of input neurons.
        - outFeatures (int): number of output neurons.
        '''
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures # = the lenght of the input, i.e. the only inFeature
        else:
            raise Exception('Delayed dense layer has for now only been implemented so that it accepts an int as inFeatures, that represents the number of in features (i.e. no channels or other things)')
        '''
        TODO: for now I only accept single-channeled inputs, that should be enough both for the spike train and for nmnist. 
        Example of NMNIST FC:
            torch.Size([512, 2312, 1, 1, 1]) #[out, in, 1, 1, 1]
            torch.Size([10, 512, 1, 1, 1])
        '''
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        #super(_delayedDenseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        super(_vectorizedDelayedDenseLayerStoca, self).__init__()

        #initialize self.variables because I blindly inherited from conv3d initially, but we're not doing a convolution anymore
        self.inFeatures = inChannels
        self.outFeatures = outChannels
        #self.weight = torch.nn.Parameter(torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device), requires_grad = True)
        
        #weight_temp = torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device)
        #torch.nn.init.xavier_uniform_(weight_temp, gain=1.0)
        #self.old_weight = torch.nn.Parameter(weight_temp, requires_grad = True)


        self.Ts = ts

        # so I think that the weight matrix has 5 dimensions because it's easier for the matrix multiplications, but in the case of the delays that's not the case, so it's sufficient a 2D matrix (which must be augmented in the case of different channels, but TODO eventualmente)
        # self.delayMatrix = torch.nn.Parameter(torch.rand(outFeatures, inFeatures, 1, 1), requires_grad=True, ) # added, should match weights matrix size. (idk why outFeatures and inFeatures are inverted but I leave it as is). This cannot be done before call of super
        # TODO it's strange that indexing gives problem, it actually seems to be fully supported by autograd
        # so, I believe that because of reshaping 
        if False:
            delayTensorTemp = torch.empty(self.outFeatures*self.inFeatures, 1, 1).to(device)
            torch.nn.init.xavier_uniform_(delayTensorTemp, gain=1.0)
            self.delaysTensor = torch.nn.Parameter(delayTensorTemp, requires_grad=True)
        else:
            self.delaysTensor = torch.nn.Parameter(((delay_init_lower - delay_init_upper)*torch.rand(self.outFeatures*self.inFeatures, 1, 1)+delay_init_upper).to(device), requires_grad = True)


        if init_stats is None:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.xavier_uniform_(pesi, gain=1.0)
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)
        else:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.normal_(pesi, init_stats[0], init_stats[1])
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx
        # weights matrix is something like this in colab torch.Size([1024, 200, 1, 1, 1]) where in this case
        # 1024 is the out dimension and 200 is the in dimension
        

    def forward(self, input):

        device = input.device
        shp = input.shape
        #print("The shapee is: ", shp)
        
        #print(f"Previous delays are {len(self.delaysVectors)} of shape {self.delaysVectors[0].shape}")
        input = torch.repeat_interleave(input, self.outFeatures, 1)
        #print("input shape repeat interleave ", input.shape)
        #print(f"Shapes are: \n - {self.delaysTensor.shape[0]} \n - {input.shape[1]}")
        assert ( self.delaysTensor.shape[0] == input.shape[1] )
        #print(f"Input shape of delay is: {repeatedInput.shape}, delay tensor: {self.delaysTensor.shape}")
        #print(f"Before mul: {delayedRepeatedInput}")
        #inp = copy.deepcopy(input.detach())
        
        #print("After weight: ", torch.norm(inp-output))
        #outp = copy.deepcopy(output.detach())
        output = input * self.weight
        output = _delayFunctionStocaGrounding.apply(output, self.delaysTensor, self.Ts) #TODO check: note, in this case we can apply either delay or weights, the order doesn't matter here because the sum is made after
        #print("After delay: ", torch.norm(outp-output))
        
        #out = torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device)
        # original 
        #index = torch.arange(start=0, end=self.outFeatures).reshape(shp[0], self.outFeatures, 1, 1, 1).repeat((1, self.inFeatures, 1 ,1 ,shp[4])).to(device)
        index = torch.arange(start=0, end=self.outFeatures).reshape(1, self.outFeatures, 1, 1, 1).repeat((shp[0], self.inFeatures, 1 ,1 ,shp[4])).to(device)
        #out.scatter_add_(1, index, output)
        
        output = torch.scatter_add(torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device), 1, index, output)

        return output

class _vectorizedAbsDelayedDenseLayer(nn.Module):
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None, ts = 1, device='cuda', init_stats = None): #TODO ts can't be fixed like that, and device to change
        '''
        - inFeatures (int, tuple of two ints, tuple of three ints): dimension of input features (Width, Height, Channel) 
            that represents the number of input neurons.
        - outFeatures (int): number of output neurons.
        '''
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures # = the lenght of the input, i.e. the only inFeature
        else:
            raise Exception('Delayed dense layer has for now only been implemented so that it accepts an int as inFeatures, that represents the number of in features (i.e. no channels or other things)')
        '''
        TODO: for now I only accept single-channeled inputs, that should be enough both for the spike train and for nmnist. 
        Example of NMNIST FC:
            torch.Size([512, 2312, 1, 1, 1]) #[out, in, 1, 1, 1]
            torch.Size([10, 512, 1, 1, 1])
        '''
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        #super(_delayedDenseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        super(_vectorizedAbsDelayedDenseLayer, self).__init__()

        #initialize self.variables because I blindly inherited from conv3d initially, but we're not doing a convolution anymore
        self.inFeatures = inChannels
        self.outFeatures = outChannels
        #self.weight = torch.nn.Parameter(torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device), requires_grad = True)
        
        #weight_temp = torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device)
        #torch.nn.init.xavier_uniform_(weight_temp, gain=1.0)
        #self.old_weight = torch.nn.Parameter(weight_temp, requires_grad = True)


        self.Ts = ts

        # so I think that the weight matrix has 5 dimensions because it's easier for the matrix multiplications, but in the case of the delays that's not the case, so it's sufficient a 2D matrix (which must be augmented in the case of different channels, but TODO eventualmente)
        # self.delayMatrix = torch.nn.Parameter(torch.rand(outFeatures, inFeatures, 1, 1), requires_grad=True, ) # added, should match weights matrix size. (idk why outFeatures and inFeatures are inverted but I leave it as is). This cannot be done before call of super
        # TODO it's strange that indexing gives problem, it actually seems to be fully supported by autograd
        # so, I believe that because of reshaping 
        if False:
            delayTensorTemp = torch.empty(self.outFeatures*self.inFeatures, 1, 1).to(device)
            torch.nn.init.xavier_uniform_(delayTensorTemp, gain=1.0)
            self.delaysTensor = torch.nn.Parameter(delayTensorTemp, requires_grad=True)
        else:
            self.delaysTensor = torch.nn.Parameter(torch.rand(self.outFeatures*self.inFeatures, 1, 1).to(device), requires_grad = True)

        if init_stats is None:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.xavier_uniform_(pesi, gain=1.0)
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)
        else:
            pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
            torch.nn.init.normal_(pesi, init_stats[0], init_stats[1])
            self.weight = torch.nn.Parameter(pesi, requires_grad = True)

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx
        # weights matrix is something like this in colab torch.Size([1024, 200, 1, 1, 1]) where in this case
        # 1024 is the out dimension and 200 is the in dimension
        

    def forward(self, input):

        device = input.device
        shp = input.shape        
        input = torch.repeat_interleave(input, self.outFeatures, 1)
        assert ( self.delaysTensor.shape[0] == input.shape[1] )

        output = input * self.weight

        output = _delayFunction.apply(output, torch.abs(self.delaysTensor), self.Ts) #TODO check: note, in this case we can apply either delay or weights, the order doesn't matter here because the sum is made after
        index = torch.arange(start=0, end=self.outFeatures).reshape(1, self.outFeatures, 1, 1, 1).repeat((shp[0], self.inFeatures, 1 ,1 ,shp[4])).to(device)
        output = torch.scatter_add(torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device), 1, index, output)
        return output

class _delayedDenseLayer(nn.Module): 
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None, ts = 1, device='cuda'): #TODO ts can't be fixed like that, and device to change
        '''
        - inFeatures (int, tuple of two ints, tuple of three ints): dimension of input features (Width, Height, Channel) 
            that represents the number of input neurons.
        - outFeatures (int): number of output neurons.
        '''
        # extract information for kernel and inChannels
        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures # = the lenght of the input, i.e. the only inFeature
        else:
            raise Exception('Delayed dense layer has for now only been implemented so that it accepts an int as inFeatures, that represents the number of in features (i.e. no channels or other things)')
        '''
        TODO: for now I only accept single-channeled inputs, that should be enough both for the spike train and for nmnist. 
        Example of NMNIST FC:
            torch.Size([512, 2312, 1, 1, 1]) #[out, in, 1, 1, 1]
            torch.Size([10, 512, 1, 1, 1])
        '''
        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(outFeatures.shape))
        # print('Output Channels :', outChannels)
        
        #super(_delayedDenseLayer, self).__init__(inChannels, outChannels, kernel, bias=False)
        super(_delayedDenseLayer, self).__init__()

        #initialize self.variables because I blindly inherited from conv3d initially, but we're not doing a convolution anymore
        self.inFeatures = inChannels
        self.outFeatures = outChannels
        #self.weight = torch.nn.Parameter(torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device), requires_grad = True)
        weight_temp = torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device)
        torch.nn.init.xavier_uniform_(weight_temp, gain=1.0)
        self.weight = torch.nn.Parameter(weight_temp, requires_grad = True)


        self.Ts = ts

        # so I think that the weight matrix has 5 dimensions because it's easier for the matrix multiplications, but in the case of the delays that's not the case, so it's sufficient a 2D matrix (which must be augmented in the case of different channels, but TODO eventualmente)
        # self.delayMatrix = torch.nn.Parameter(torch.rand(outFeatures, inFeatures, 1, 1), requires_grad=True, ) # added, should match weights matrix size. (idk why outFeatures and inFeatures are inverted but I leave it as is). This cannot be done before call of super
        # TODO it's strange that indexing gives problem, it actually seems to be fully supported by autograd
        # so, I believe that because of reshaping 
        
        self.delaysVectors = nn.ParameterList([ torch.nn.Parameter(torch.rand(outFeatures, 1, 1).to(device), requires_grad=True) for _ in range(inFeatures) ]) # outfeatures * [..] because we have a vector for each neuron, and each neuron has outFeatures connections
        #delayTensorTemp = torch.rand(self.outFeatures*self.inFeatures, 1, 1).to(device)
        #torch.nn.init.xavier_uniform_(delayTensorTemp, gain=1.0)
        #self.delaysTensor = torch.nn.Parameter(delayTensorTemp, requires_grad=True)
        
        #pesi = torch.empty((self.inFeatures * self.outFeatures,1,1,1)).to(device)
        #torch.nn.init.xavier_uniform_(pesi, gain=1.0)
        #self.weight = pesi
        #print("init weights are: ", {self.weight})

        #assert len(self.delaysVectors) == inFeatures 
        
        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In dense, using weightScale of', weightScale)

        self.preHookFx = preHookFx
        # weights matrix is something like this in colab torch.Size([1024, 200, 1, 1, 1]) where in this case
        # 1024 is the out dimension and 200 is the in dimension
        
        #assert self.weight.shape == self.delayMatrix.shape # not true, because delays don't need to have the same dimensions as the weights.

    
    def forward(self, input):
        '''
        This method will have to apply the pairwise delays. 
        1) For each input x_i we have a delay d_(i,j) that 'moves' the input in time (theoretically only forward) 
        2) Apply the same semantic operation as what is done with the conv3d (i.e. the fc mapping)
        '''

        # in the init I'm probably gonna need a matrix of delays of dimension (inFeatures, outFeatures)
        # so a for loop is very likely not ideal at all.... but I will try to optimize this in another phase
        # but the idea is: 
        # for each x_i, which is a temporal "vector" (or tensor) (i.e. it's a convolution over the input size) do:
        #   apply slayer's delay function to the tensor defined as [x_i  x_i  x_i  x_i ..]'  to obtain \hat{x}
        #   apply weights to \hat{x}, I think this can be done through a multiplication.. the annoying thing is that we're considering each input x_i one by one.. 

        # Thinking about how to go about this, unfortunately (although would have been convenient) it's not possible to inverse the operation, it has to be necessarely first delay and then weights.
        
        #print("You're in the forward! :)")
        #print(f"Input shape is: {input.shape}")

        #get one of the inFeatures trial:
        '''
        With the following prints I checked if the slicing created problem.. apparently it does? List not very efficient, any other way?
        print(f"Delay vector (list): {self.delaysVectors[1].shape} \n Delay vector shape (matrix): {self.delayMatrix[:,1,:,:].shape}")
        print(f"Delay type (list): {type(self.delaysVectors[1])} \n Delay type (matrix): {type(self.delayMatrix[:,1,:,:])}") #delay should not have 5 dimensions but 4, since here batch size is not relevant
        '''
        device = input.device
        #for i in range(self.in_channels):
        #print(f"Weights shape: {self.weight.shape}") #torch.Size([256, 200, 1, 1, 1]) i.e. (out, in, 1, 1, 1)
        
        shp = input.shape
        #print("LA SHAPE: ", shp)
        #print("Original forward weights ", self.weight)
        X_tot = torch.zeros(shp[0], self.outFeatures, shp[2], shp[3], shp[4]).to(device) #FIXME this is for the spike train, what if batch size is different, does it change anything?
        for i in range(self.inFeatures):
            # x_i = input[:,i,:,:,:] # it's a 1D temporal vector e.g. (1, 1900)
            x_i = input[:,i,:,:,:].unsqueeze(1)
            #print("####1", x_i.shape)
            #print(f"After slicing: {x_i.shape}")
            x_i = x_i.repeat(1, self.outFeatures, 1, 1, 1).to(device) # (batch, number of out neurons, ... else), this is a matrix
            #print(f"Delay applied to shapes {x_i.shape} and delay shape {self.delaysVectors[i].shape}")
            x_i = _delayFunction.apply(x_i, self.delaysVectors[i], self.Ts) 
            #print("Original delays vector ", i, self.delaysVectors[i])
            
            
            #print("2",x_i.shape)
            '''
            So this was the first implementation, but slicing I think transforms torch.Parameter to torch.Tensor, and therefore stops working
            _delayFunction.apply(x_i, self.delayMatrix[:,i,:,:], self.Ts)
            '''
            
            #print("3", x_i.shape)
            # let X_i be the matrix
            # multiply each row of X_i by the weight w_(i,j). So actually it's a vector scalar operation, good way to do this vectorizing?
            #x_i = self.weight * self.delaysVectors[i][:, None, None]
            #print("W:", self.weight[:, 1, :, :, :].shape)
            #print(x_i.shape)
            #ACHTUNG print(f"Before mul x_{i} : {x_i}")
            x_i = x_i * self.weight[:, i, :, :, :]
            #ACHTUNG print(f"Multipl x_{i} : {x_i}")
            
            X_tot += x_i
            #print("4", x_i.shape)
        #print(x_i.shape)
        #print("X TOT shape", X_tot.shape)
        #### to be deleted very likely
        # Note: not sure why 3D convolution. ok it's a 3d convolution when there are also channels
        return X_tot

    def vec_old_forward(self, input):
        
        device = input.device
        shp = input.shape
        
        print(f"Previous delays are {len(self.delaysVectors)} of shape {self.delaysVectors[0].shape}")
        listaDelays = []
        for vect in self.delaysVectors: 
            listaDelays.append(vect.data)
            print(vect.data)
        
        delaysTensor = torch.stack(listaDelays, dim=0)
        #print("Now: ", delaysTensor)
        #TODO place delaysTensor in the class
        delaysTensor = torch.reshape(delaysTensor, (delaysTensor.shape[0]*delaysTensor.shape[1], 1, 1)) #this is gonna be a super long delays tensor
        #print("After reshape", delaysTensor)
        print("Vec delays tensor", delaysTensor)

        # make big vector of the input, of which each row is repeated outFeatures time
        # this is actually gonna be done in practice
        repeatedInput = torch.repeat_interleave(input, self.outFeatures, 1)

        #print("Previous input : \n", input)
        #print("Repeated input : \n", repeatedInput)

        assert ( delaysTensor.shape[0] == repeatedInput.shape[1] )

        delayedRepeatedInput = _delayFunction.apply(repeatedInput, delaysTensor, self.Ts)
        #print(f"Non delayed yet input : {repeatedInput.shape}")
        

        print(f"Before reshape weights: {self.weight}")
        reshapedWeights = torch.reshape(self.weight, (self.inFeatures*self.outFeatures,1,1,1))
        print(f"weights reshaped: {reshapedWeights}") #this will be the default shape

        #print("DelayedRepeatedInput", delayedRepeatedInput)
        #print("ReshapedWeights: ", reshapedWeights)
        print(f"Before mul: {delayedRepeatedInput}")

        res = delayedRepeatedInput * reshapedWeights # ok so this simply works
        # Here the only problem is that we need to do that sum, corresponding to the total effect felt by an output neuron
        print(f"Delayed repeated input multipl.: {res}")
        print("res:", res)
        '''print("res shape: ", res.shape)
        print("Input shape was: ", input.shape)
        print("And weights were: ", reshapedWeights) '''


        out = torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device)
        index = torch.arange(start=0, end=self.outFeatures).reshape(shp[0], self.outFeatures, 1, 1, 1).repeat((1, self.inFeatures, 1 ,1 ,shp[4])).to(device)
        #print("Index: ", index)
        out.scatter_add_(1, index, res)
        '''
        X_tot = torch.zeros(1, self.outFeatures, shp[2], shp[3], shp[4]).to(device) 
                for i in range(self.inFeatures):
                    x_i = input[:,i,:,:,:] # it's a 1D temporal vector e.g. (1, 1900)

                    x_i = x_i.repeat(1, self.outFeatures, 1, 1, 1).to(device) 
                    x_i = _delayFunction.apply(x_i, self.delaysVectors[i], self.Ts) 
                    x_i = x_i * self.weight[:, i, :, :, :]
                    print(self.weight[:, i, :, :, :].shape)
                    X_tot += x_i
        return X_tot
        '''
        
        return out

    def vec_forward_deprecated(self, input):
        
        device = input.device
        shp = input.shape
        
        #print(f"Previous delays are {len(self.delaysVectors)} of shape {self.delaysVectors[0].shape}")
        repeatedInput = torch.repeat_interleave(input, self.outFeatures, 1)
        assert ( self.delaysTensor.shape[0] == repeatedInput.shape[1] )
        output = _delayFunction.apply(repeatedInput, self.delaysTensor, self.Ts)
        #print(f"Before mul: {delayedRepeatedInput}")
        output = output * self.weight # ok so this simply works
        out = torch.zeros((shp[0], self.outFeatures, shp[2], shp[3], shp[4])).to(device)
        index = torch.arange(start=0, end=self.outFeatures).reshape(shp[0], self.outFeatures, 1, 1, 1).repeat((1, self.inFeatures, 1 ,1 ,shp[4])).to(device)
        out.scatter_add_(1, index, output)
        return out

class _myDense(nn.Module): 
    def __init__(self, inFeatures, outFeatures, weightScale=1, preHookFx=None, ts = 1, device='cuda'): #TODO ts can't be fixed like that, and device to change

        if type(inFeatures) == int:
            kernel = (1, 1, 1)
            inChannels = inFeatures # = the lenght of the input, i.e. the only inFeature
        else:
            raise Exception('Delayed dense layer has for now only been implemented so that it accepts an int as inFeatures, that represents the number of in features (i.e. no channels or other things)')

        if type(outFeatures) == int:
            outChannels = outFeatures
        else:
            raise Exception('outFeatures should not be more than 1 dimension. It was: {}'.format(outFeatures.shape))

        super(_myDense, self).__init__()

        self.inFeatures = inChannels
        self.outFeatures = outChannels
        weight_temp = torch.empty(outFeatures, inFeatures, 1, 1, 1).to(device)
        torch.nn.init.xavier_uniform_(weight_temp, gain=1.0)
        self.weight = torch.nn.Parameter(weight_temp, requires_grad = True)
        self.Ts = ts

        self.delaysVectors = nn.ParameterList([ torch.nn.Parameter(torch.rand(outFeatures, 1, 1).to(device), requires_grad=True) for _ in range(inFeatures) ]) # outfeatures * [..] because we have a vector for each neuron, and each neuron has outFeatures connections
        # assert len(self.delaysVectors) == inFeatures 
        
        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed

        self.preHookFx = preHookFx
        
    
    def forward(self, input, weightOutside=None):
        '''
        This method will have to apply the pairwise delays. 
        1) For each input x_i we have a delay d_(i,j) that 'moves' the input in time (theoretically only forward) 
        2) Apply the same semantic operation as what is done with the conv3d (i.e. the fc mapping)
        '''

        device = input.device
        shp = input.shape
        #assert weightOutside.shape == self.weight.shape
        print("Init weights: ", self.weight)
        X_tot = torch.zeros(shp[0], self.outFeatures, shp[2], shp[3], shp[4]).to(device) #FIXME this is for the spike train, what if batch size is different, does it change anything?
        print("In features: ", self.inFeatures)
        print("Out features: ", self.outFeatures)
        for i in range(self.inFeatures):
            x_i = input[:,i,:,:,:].unsqueeze(1) # it's a 1D temporal vector e.g. (1, 1900), but actually 1 is the batch size I think.
            print("x_i shape before repeat is ", x_i.shape) 
            x_i = x_i.repeat(1,self.outFeatures, 1, 1, 1).to(device) 
            print("x_i shape after repeat is ", x_i.shape) 
            x_i = _delayFunction.apply(x_i, self.delaysVectors[i], self.Ts) 
            print("****** delay function seemed to work! :)")
            if weightOutside == None:
                #print("BEFORE MATRIX MUL: ", x_i)
                x_i = x_i * self.weight[:, i, :, :, :] #TODO .repeat(shp[0])
                #print("AFTER MATRIX MUL: ", x_i)
            else:
                x_i = x_i * weightOutside[:, i, :, :, :]
            X_tot += x_i
            #print("4", x_i.shape)

        return X_tot
        '''
        So, below the weights are applied directly to the inputs, and this is not what we want. 
        We want to apply the weights to the delayed input. 
        if self.preHookFx is None:
            return F.conv3d(input, 
                            self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.preHookFx(self.weight), self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        '''

class _convLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, preHookFx=None):
        inChannels = inFeatures
        outChannels = outFeatures
        
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # groups
        # no need to check for groups. It can only be int

        # print('inChannels :', inChannels)
        # print('outChannels:', outChannels)
        # print('kernel     :', kernel, kernelSize)
        # print('stride     :', stride)
        # print('padding    :', padding)
        # print('dilation   :', dilation)
        # print('groups     :', groups)

        super(_convLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, dilation, groups, bias=False)

        if weightScale != 1:    
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed
            # print('In conv, using weightScale of', weightScale)

        self.preHookFx = preHookFx

    def forward(self, input):
        '''
        '''
        if self.preHookFx is None:
            return F.conv3d(input, 
                            self.weight, self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)
        else:
            return F.conv3d(input, 
                            self.preHookFx(self.weight), self.bias, 
                            self.stride, self.padding, self.dilation, self.groups)

class _poolLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
        
        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # print('theta      :', theta)
        # print('kernel     :', kernel, kernelSize)
        # print('stride     :', stride)
        # print('padding    :', padding)
        # print('dilation   :', dilation)
        
        super(_poolLayer, self).__init__(1, 1, kernel, stride, padding, dilation, bias=False)   

        # set the weights to 1.1*theta and requires_grad = False
        self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad = False)
        # print('In pool layer, weight =', self.weight.cpu().data.numpy().flatten(), theta)

        self.preHookFx = preHookFx


    def forward(self, input):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        
        # add necessary padding for odd spatial dimension
        # if input.shape[2]%2 != 0:
            # input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], 1, input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        # if input.shape[3]%2 != 0:
            # input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], 1, input.shape[4]), dtype=dtype).to(device)), 3)
        if input.shape[2]%self.weight.shape[2] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]), dtype=dtype).to(device)), 2)
        if input.shape[3]%self.weight.shape[3] != 0:
            input = torch.cat((input, torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]), dtype=dtype).to(device)), 3)

        dataShape = input.shape

        if self.preHookFx is None:
            result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                              self.weight, self.bias, 
                              self.stride, self.padding, self.dilation)
        else:
            result = F.conv3d(input.reshape((dataShape[0], 1, dataShape[1] * dataShape[2], dataShape[3], dataShape[4])), 
                          self.preHooFx(self.weight), self.bias, 
                          self.stride, self.padding, self.dilation)
        # print(result.shape)
        return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))

class _convTransposeLayer(nn.ConvTranspose3d):
    '''
    '''
    def __init__(self, inFeatures, outFeatures, kernelSize, stride=1, padding=0, dilation=1, groups=1, weightScale=1, preHookFx=None):
        inChannels = inFeatures
        outChannels = outFeatures

        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        # groups
        # no need to check for groups. It can only be int

        super(_convTransposeLayer, self).__init__(inChannels, outChannels, kernel, stride, padding, 0, groups, False, dilation)

        if weightScale != 1:
            self.weight = torch.nn.Parameter(weightScale * self.weight) # scale the weight if needed

        self.preHookFx = preHookFx

    def forward(self, input):
        '''
        '''
        if self.preHookFx is None:
            return F.conv_transpose3d(
                input,
                self.weight, self.bias,
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )
        else:
            return F.conv_transpose3d(
                input,
                self.preHookFx(self.weight), self.bias,
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )

class _unpoolLayer(nn.ConvTranspose3d):
    '''
    '''
    def __init__(self, theta, kernelSize, stride=None, padding=0, dilation=1, preHookFx=None):
        # kernel
        if type(kernelSize) == int:
            kernel = (kernelSize, kernelSize, 1)
        elif len(kernelSize) == 2:
            kernel = (kernelSize[0], kernelSize[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernelSize.shape))
        
        # stride
        if stride is None:
            stride = kernel
        elif type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))
        
        super(_unpoolLayer, self).__init__(1, 1, kernel, stride, padding, 0, 1, False, dilation)

        self.weight = torch.nn.Parameter(torch.FloatTensor(1.1 * theta * np.ones((self.weight.shape))).to(self.weight.device), requires_grad=False)

        self.preHookFx = preHookFx

    def forward(self, input):
        '''
        '''
        # device = input.device
        # dtype  = input.dtype
        # # add necessary padding for odd spatial dimension
        # This is not needed as unpool multiplies the spatial dimension, hence it is always fine
        # if input.shape[2]%self.weight.shape[2] != 0:
        #     input = torch.cat(
        #         (
        #             input, 
        #             torch.zeros(
        #                 (input.shape[0], input.shape[1], input.shape[2]%self.weight.shape[2], input.shape[3], input.shape[4]),
        #                 dtype=dtype
        #             ).to(device)
        #         ),
        #         dim=2,
        #     )
        # if input.shape[3]%self.weight.shape[3] != 0:
        #     input = torch.cat(
        #         (
        #             input,
        #             torch.zeros(
        #                 (input.shape[0], input.shape[1], input.shape[2], input.shape[3]%self.weight.shape[3], input.shape[4]),
        #                 dtype=dtype
        #             ),
        #             dim=3,
        #         )
        #     )

        dataShape = input.shape

        if self.preHookFx is None:
            result = F.conv_transpose3d(
                input.reshape((dataShape[0], 1, -1, dataShape[3], dataShape[4])),
                self.weight, self.bias, 
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )
        else:
            result = F.conv_transpose3d(
                input.reshape((dataShape[0], 1, -1, dataShape[3], dataShape[4])),
                self.preHookFx(self.weight), self.bias, 
                self.stride, self.padding, self.output_padding, self.groups, self.dilation,
            )

        return result.reshape((result.shape[0], dataShape[1], -1, result.shape[3], result.shape[4]))

class _dropoutLayer(nn.Dropout3d):
    '''
    '''
    # def __init__(self, p=0.5, inplace=False):
    #   super(_dropoutLayer, self)(p, inplace)

    '''
    '''
    def forward(self, input):
        inputShape = input.shape
        return F.dropout3d(input.reshape((inputShape[0], -1, 1, 1, inputShape[-1])),
                           self.p, self.training, self.inplace).reshape(inputShape)

class _pspLayer(nn.Conv3d):
    '''
    '''
    def __init__(self, filter, Ts):
        inChannels  = 1
        outChannels = 1
        kernel      = (1, 1, torch.numel(filter))

        self.Ts = Ts

        super(_pspLayer, self).__init__(inChannels, outChannels, kernel, bias=False) 

        # print(filter)
        # print(np.flip(filter.cpu().data.numpy()).reshape(self.weight.shape)) 
        # print(torch.FloatTensor(np.flip(filter.cpu().data.numpy()).copy()))

        flippedFilter = torch.FloatTensor(np.flip(filter.cpu().data.numpy()).copy()).reshape(self.weight.shape)

        self.weight = torch.nn.Parameter(flippedFilter.to(self.weight.device), requires_grad = True)

        self.pad = torch.nn.ConstantPad3d(padding=(torch.numel(filter)-1, 0, 0, 0, 0, 0), value=0)

    def forward(self, input):
        '''
        '''
        inShape = input.shape
        inPadded = self.pad(input.reshape((inShape[0], 1, 1, -1, inShape[-1])))
        # print((inShape[0], 1, 1, -1, inShape[-1]))
        # print(input.reshape((inShape[0], 1, 1, -1, inShape[-1])).shape)
        # print(inPadded.shape)
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape(inShape)

class _pspFilter(nn.Conv3d):
    '''
    '''
    def __init__(self, nFilter, filterLength, Ts, filterScale=1):
        inChannels  = 1
        outChannels = nFilter
        kernel      = (1, 1, filterLength)
        
        super(_pspFilter, self).__init__(inChannels, outChannels, kernel, bias=False) 

        self.Ts  = Ts
        self.pad = torch.nn.ConstantPad3d(padding=(filterLength-1, 0, 0, 0, 0, 0), value=0)

        if filterScale != 1:
            self.weight.data *= filterScale

    def forward(self, input):
        '''
        '''
        N, C, H, W, Ns = input.shape
        inPadded = self.pad(input.reshape((N, 1, 1, -1, Ns)))
        output = F.conv3d(inPadded, self.weight) * self.Ts
        return output.reshape((N, -1, H, W, Ns))

class _spikeFunction(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
        '''
        '''
        device = membranePotential.device
        dtype  = membranePotential.dtype
        threshold      = neuron['theta']
        oldDevice = torch.cuda.current_device()

        # if device != oldDevice: torch.cuda.set_device(device)
        # torch.cuda.device(3)

        # spikeTensor = torch.empty_like(membranePotential)

        # print('membranePotential  :', membranePotential .device)
        # print('spikeTensor        :', spikeTensor       .device)
        # print('refractoryResponse :', refractoryResponse.device)
            
        # (membranePotential, spikes) = slayer_cuda.get_spikes_cuda(membranePotential,
        #                                                         torch.empty_like(membranePotential),  # tensor for spikes
        #                                                         refractoryResponse,
        #                                                         threshold,
        #                                                         Ts)
        spikes = slayerCuda.getSpikes(membranePotential.contiguous(), refractoryResponse, threshold, Ts)
        
        pdfScale        = torch.autograd.Variable(torch.tensor(neuron['scaleRho']                 , device=device, dtype=dtype), requires_grad=False)
        # pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho']                   , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        pdfTimeConstant = torch.autograd.Variable(torch.tensor(neuron['tauRho'] * neuron['theta'] , device=device, dtype=dtype), requires_grad=False) # needs to be scaled by theta
        threshold       = torch.autograd.Variable(torch.tensor(neuron['theta']                    , device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)
        # torch.cuda.synchronize()
        
        # if device != oldDevice: torch.cuda.set_device(oldDevice)
        # torch.cuda.device(oldDevice)
        
        return spikes
        
    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
        spikePdf = pdfScale / pdfTimeConstant * torch.exp( - torch.abs(membranePotential - threshold) / pdfTimeConstant)

        # return gradOutput, None, None, None # This seems to work better!
        return gradOutput * spikePdf, None, None, None
        # plt.figure()
        # plt.plot(gradOutput[0,5,0,0,:].cpu().data.numpy())
        # print   (gradOutput[0,0,0,0,:].cpu().data.numpy())
        # plt.plot(membranePotential[0,0,0,0,:].cpu().data.numpy())
        # plt.plot(spikePdf         [0,0,0,0,:].cpu().data.numpy())
        # print   (spikePdf         [0,0,0,0,:].cpu().data.numpy())
        # plt.show()
        # return gradOutput * spikePdf, None, None, None

class _timeOfFirstSpikee(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, neuron, Ts):
        #for now we have out_spike_trains, but let's pretend that that has come from the spike() function

        '''
            Here's the plan: 
                - get the indices of the first spike, maybe we can utilize the old code of ttfs
                - given the indices i, we can get the samples of the membrane potentials mp[i-1] and mp[i]
                - given the two samples we can do a linear interpolation
                - given the linear interpolation we can see where the line intersect theta
                - for the backwards pass, we already have the derivative of the linear interpolation, i.e. il coefficiente angolare
        '''

        out_spike_trains = _spikeFunction.apply(membranePotential, refractoryResponse, neuron, Ts)
        for i in range(out_spike_trains.shape[0]):
            #tmp = out_spike_trains[i] #for every sample of the batch
            memPot = membranePotential[i].squeeze(1).squeeze(1)
            tmp = out_spike_trains[i, :, 0, 0, :]
            idx = torch.arange(tmp.shape[1], 0, -1).to(membranePotential.device)
            tmp2= tmp * idx
            print(tmp2)
            indices = torch.argmax(tmp2, 1, keepdim=True) 
            indices[6][0] = 0 #TODO: remove this, this is only to check correctness
            print("argmax: ",indices)
            # replace any 0 with -1, is this necessary? Not sure, maybe not
            if (indices == 0).any():
                z = (indices == 0).nonzero()[:,0] 
                indices[z] = 1 #FIXME not zero but -1 
            indices_minus_one = indices - 1
            print("Membrane potential shape: ", memPot.shape)
            print("indices: ", indices.shape)
            memPot_i = memPot.gather(dim = 1, index = indices)
            memPot_i_minus_one = memPot.gather(dim = 1, index = indices_minus_one)
            print("membrane potentials")
            print(memPot_i)
            print(memPot_i_minus_one)
            m = memPot_i - memPot_i_minus_one #these are actually the coefficienti angolari perchè si dovrebbe dividere per (x_1 - x_2) ma tale quantità è sempre uno
            # x_diff = indices - indices_minus_one #we don't need this because the difference between x is always 1
            #now we need the b
            b = memPot_i - (m * indices) 
            #now to find the approximation of the exact time we can apply x = (theta-b)/m
            spike_time = (neuron['theta'] - b) / m
            print("Spike times: ", spike_time)

class _pspFunction(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, spike, filter, Ts):
        device = spike.device
        dtype  = spike.dtype
        psp = slayerCuda.conv(spike.contiguous(), filter, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(filter, Ts)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (filter, Ts) = ctx.saved_tensors
        gradInput = slayerCuda.corr(gradOutput.contiguous(), filter, Ts)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass
            
        return gradInput, gradFilter, None

class _delayLayer(nn.Module):
    '''
    '''
    def __init__(self, inputSize, Ts):
        super(_delayLayer, self).__init__()

        if type(inputSize) == int:
            inputChannels = inputSize
            inputHeight   = 1
            inputWidth    = 1
        elif len(inputSize) == 3:
            inputChannels = inputSize[0]
            inputHeight   = inputSize[1]
            inputWidth    = inputSize[2]
        else:
            raise Exception('inputSize can only be 1 or 2 dimension. It was: {}'.format(inputSize.shape))

        self.delay = torch.nn.Parameter(torch.rand((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        # self.delay = torch.nn.Parameter(torch.empty((inputChannels, inputHeight, inputWidth)), requires_grad=True)
        # print('delay:', torch.empty((inputChannels, inputHeight, inputWidth)))
        self.Ts = Ts

    def forward(self, input):
        N, C, H, W, Ns = input.shape 
        if input.numel() != self.delay.numel() * input.shape[-1] * input.shape[0]:
            print('per channel')
            return _delayFunction.apply(input, self.delay.repeat((1, H, W)), self.Ts) # different delay per channel
        else:
            print('per neuron')
            print(f'shape: {input.shape, self.delay.shape, self.Ts}')
            return _delayFunction.apply(input, self.delay, self.Ts) #different delay per neuron # but not pairwise delay

class _delayFunction(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, delay, Ts):
        '''
        '''
        #print(f"##########\n Inside _delayFunction input shape: {input.shape} \n delay shape: {delay.shape} \n Ts: {Ts}")
        #print(f"Type of inputs: \n input: {type(input)} \n delay: {type(delay)} \n {type(Ts)}")
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input.contiguous(), delay.data, Ts) # input.contiguos() is just an efficiency thing, doesn't have anything to do with the maths
        # in reality I don't really see why a cuda implementation was need, as of now.. 
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(output, delay.data, Ts) # I think this is needed because this is a static method, that way the variables are also available in the backwards pass
        return output 

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        # autograd tested and verified
        (output, delay, Ts) = ctx.saved_tensors
        diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
        outputDiff = slayerCuda.conv(output.contiguous(), diffFilter, 1)
        # the conv operation should not be scaled by Ts. 
        # As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
        gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
        # no minus needed here, as it is included in diffFilter which is -1 * [1, -1]

        return slayerCuda.shift(gradOutput.contiguous(), -delay, Ts), gradDelay, None

class _delayFunctionStocaGrounding(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, delay, Ts):
        '''
        '''
        #print(f"##########\n Inside _delayFunction input shape: {input.shape} \n delay shape: {delay.shape} \n Ts: {Ts}")
        #print(f"Type of inputs: \n input: {type(input)} \n delay: {type(delay)} \n {type(Ts)}")
        device = input.device
        dtype  = input.dtype
        cp = fixed_point_quantize(torch.clone(delay.data), wl = 10, fl = 0, rounding = "stochastic")  + 0.1
        output = slayerCuda.shift(input.contiguous(), cp, Ts) # input.contiguos() is just an efficiency thing, doesn't have anything to do with the maths
        # in reality I don't really see why a cuda implementation was need, as of now.. 
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(output, delay.data, Ts) # I think this is needed because this is a static method, that way the variables are also available in the backwards pass
        return output 

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        # autograd tested and verified
        (output, delay, Ts) = ctx.saved_tensors
        diffFilter = torch.tensor([-1, 1], dtype=gradOutput.dtype).to(gradOutput.device) / Ts
        outputDiff = slayerCuda.conv(output.contiguous(), diffFilter, 1)
        # the conv operation should not be scaled by Ts. 
        # As such, the output is -( x[k+1]/Ts - x[k]/Ts ) which is what we want.
        gradDelay  = torch.sum(gradOutput * outputDiff, [0, -1], keepdim=True).reshape(gradOutput.shape[1:-1]) * Ts
        # no minus needed here, as it is included in diffFilter which is -1 * [1, -1]

        return slayerCuda.shift(gradOutput.contiguous(), -delay, Ts), gradDelay, None

class _delayFunctionNoGradient(torch.autograd.Function):
    '''
    '''
    @staticmethod
    def forward(ctx, input, delay, Ts=1):
        '''
        '''
        device = input.device
        dtype  = input.dtype
        output = slayerCuda.shift(input.contiguous(), delay, Ts)
        Ts     = torch.autograd.Variable(torch.tensor(Ts   , device=device, dtype=dtype), requires_grad=False)
        delay  = torch.autograd.Variable(torch.tensor(delay, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(delay, Ts)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (delay, Ts) = ctx.saved_tensors
        return slayerCuda.shift(gradOutput.contiguous(), -delay, Ts), None, None
