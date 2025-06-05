import torch
from typing import List, Tuple
from torch import nn
import math
from torch.nn.parameter import Parameter
class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()

        self.weight = Parameter(torch.empty((out_features,in_features),requires_grad=True))

        self.bias = Parameter(torch.empty(out_features,requires_grad=True))
        
    
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        output = input.matmul(self.weight.t())
        output += self.bias
        
        return output

class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)
        
        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """

        layers = []
        num_layers = len(hidden_sizes) 
        layers.append(Linear(in_features=input_size, out_features=hidden_sizes[0]))

        for count in range(1, num_layers):
            layers.append(Linear(in_features=hidden_sizes[count - 1], out_features=hidden_sizes[count]))

        fc_class = Linear(in_features=hidden_sizes[-1], out_features=num_classes)
        
        return nn.ModuleList(layers),fc_class
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """

        if activation == 'relu':
            return torch.maximum(inputs,torch.tensor(0.0))
        elif activation == 'tanh':
            #return (torch.exp(inputs) - torch.exp(torch.neg(inputs)))/(torch.exp(inputs) + torch.exp(torch.neg(inputs)))
            tanh = nn.Tanh()
            return tanh(inputs)
        
        elif activation == 'sigmoid': #claping min and max of input
            return torch.sigmoid(inputs)
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """

        fan_in = module.weight.shape[1]
        fan_out = module.weight.shape[0]

        std = math.sqrt(2.0/float(fan_in+fan_out))

        mean = 0.0

        bias_values = torch.zeros(module.bias.shape)

        with torch.no_grad():
            module.weight.normal_(mean,std)
            module.bias.copy_(bias_values)

        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        
        """

        batch_shape = images.shape[0]

        images = images.reshape(batch_shape,-1)

        for count in range(len(self.hidden_layers)):
            images = self.activation_fn(self.activation,self.hidden_layers[count](images))
        out_put = self.output_layer(images)

        return out_put

