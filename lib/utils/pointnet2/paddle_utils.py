import paddle.nn as nn
from typing import List,Tuple
# 完成
class SharedMLP(nn.Sequential):
    def __init__(self,args: List[int],
            *,bn: bool = False,
            activation=nn.ReLU(),
            preact: bool = False,
            first: bool = False,
            name: str = "",
            instance_norm: bool = False
    ):
        super().__init__()
        for i in range(len(args)-1):
            self.add_sublayer(
                name + 'layer{}'.format(i),
                    Conv2d(
                        args[i],
                        args[i + 1],
                        bn=(not first or not preact or (i != 0)) and bn,
                        activation=activation
                        if (not first or not preact or (i != 0)) else None,
                        preact=preact,
                        instance_norm=instance_norm
                    )
            )


class _ConvBase(nn.Sequential):
    
    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name="",
            instance_norm=False,
            instance_norm_func=None
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            weight_attr=init,
            bias_attr=nn.initializer.Constant(value=0) if bias else None
        )

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)
        if instance_norm:
            if not preact:
                in_unit = instance_norm_func(out_size)
            else:
                in_unit = instance_norm_func(in_size)

        if preact:
            if bn:
                self.add_sublayer(name + 'bn', bn_unit)

            if activation is not None:
                self.add_sublayer(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_sublayer(name + 'in', in_unit)

        self.add_sublayer(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_sublayer(name + 'bn', bn_unit)

            if activation is not None:
                self.add_sublayer(name + 'activation', activation)

            if not bn and instance_norm:
                self.add_sublayer(name + 'in', in_unit)

class Conv1d(_ConvBase):
    
    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(),
            bn: bool = False,
            init=nn.initializer.KaimingNormal(),
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1D,
            batch_norm=nn.BatchNorm1D,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm1D
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(),
            bn: bool = False,
            init=nn.initializer.KaimingNormal(),
            bias: bool = True,
            preact: bool = False,
            name: str = "",
            instance_norm=False
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2D,
            batch_norm=nn.BatchNorm2D,
            bias=bias,
            preact=preact,
            name=name,
            instance_norm=instance_norm,
            instance_norm_func=nn.InstanceNorm2D
        )



class _BNBase(nn.Sequential):
    
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(num_features=in_size,
        weight_attr=nn.initializer.Constant(1.0),bias_attr=nn.initializer.Constant(0.0)))
    
class BatchNorm1d(_BNBase):
    
    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1D, name=name)

class BatchNorm2d(_BNBase):
    
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2D, name=name)


class FC(nn.Sequential):
    
    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, 
        weight_attr=init() if init is not None else None,
        bias_attr=nn.initializer.Constant(value=0) if not bn else None)


        if preact:
            if bn:
                self.add_sublayer(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_sublayer(name + 'activation', activation)

        self.add_sublayer(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_sublayer(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_sublayer(name + 'activation', activation)

