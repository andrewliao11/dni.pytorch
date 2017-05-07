# Image classification with synthetic gradient in Pytorch
I implement the ***[Decoupled Neural Interfaces using Synthetic Gradients](http://arxiv.org/abs/1608.05343)*** in **pytorch**. The paper use synthetic gradient to decouple the layers in the network. This is pretty interesting since we won't suffer from **update lock** anymore. I test my model in mnist and archieve similar result as the paper claimed.

## Requirement
- pytorch
- python 2.7
- torchvision

## TODO
- use multi-threading on gpu to analyze the speed

## What's synthetic gradients?
We ofter optimize NN by backpropogation, which is usually implemented in some well-known framework. However, is there another way for the layers in NN to communicate with other layers? Here comes the ***synthetic gradients***! It gives us a way to allow neural networks to communicate, to learn to send messages between themselves, in a decoupled, scalable manner paving the way for multiple neural networks to communicate with each other or improving the long term temporal dependency of recurrent networks.   
The neuron in each layer will automatically produces an error signal(***δa_head***) from synthetic-layers and do the optimzation. And how did the error signal generated? Actually, the network still does the backpropogation. While the error signal(***δa***) from the objective function is not used to optimize the neuron in the network, it is used to optimize the error signal(***δa_head***) produced by the synthetic-layer. The following is the illustration from the paper:
![](https://github.com/andrewliao11/DNI-pytorch/blob/master/misc/dni_illustration.png?raw=true)   

## Usage 
Right now I just implement the FCN version, which is set as the default network structure 
```python
python main_dni.py
```


## Reference
- Deepmind's [post](https://deepmind.com/blog/decoupled-neural-networks-using-synthetic-gradients/) on Decoupled Neural Interfaces Using Synthetic Gradients

