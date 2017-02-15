# PyTorch Highway Networks
[Highway networks](https://arxiv.org/abs/1505.00387) implemented in [PyTorch](http://www.pytorch.org).

![Highway Equation](images/highway.png)

Just the [MNIST example](https://github.com/pytorch/examples/tree/master/mnist) from [PyTorch](http://www.pytorch.org) hacked to work with Highway layers.

## TODO
- ~~make the Highway `nn.Module` reuseable and configurable~~
- Why does softmax work better than sigmoid? This shouldn't be the case...
- Make some training graphs.
- conv highway networks
- recurrent highway networks

## NOTES
- ELU doesn't work better than RELU
- softmax seems to work better than sigmoid for the gate function???
