import numpy as np
from mytorch import tensor
from mytorch.autograd_engine import Function

class ReLU(Function):
    @staticmethod
    def forward(ctx,Z):
        if not(type(Z).__name__ == 'Tensor'):
            raise Exception("Arg for relu must be tensor")
        ctx.save_for_backward(Z)
        requires_grad = Z.requires_grad

        A = tensor.Tensor(np.maximum(0.0, Z.data),requires_grad = requires_grad, is_leaf = not requires_grad)
        return A

    @staticmethod
    def backward(ctx,grad_output):
        Z  = ctx.saved_tensors[0]
        dZ = (Z.data > 0)*1.0
        return [tensor.Tensor(dZ*grad_output.data),None]


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return [tensor.Tensor(grad_output.data.T),None]


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return [tensor.Tensor(grad_output.data.reshape(ctx.shape)),None]

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return [tensor.Tensor(grad_output.data / a.data),None]



class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Exp must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        #print("grad_output.data \n",grad_output.data)
        #print("np.exp(a.data)\n",np.exp(a.data))
        return [tensor.Tensor(grad_output.data * np.exp(a.data)),None]


class Matmul(Function):
    """ W.T@ A in a linear layer """
    @staticmethod
    def forward(ctx,W , A):
        """ static method for tensor product of a and b"""
        # Check that both args are tensors
        if not (type(W).__name__ == 'Tensor' and type(A).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(W).__name__, type(A).__name__))
        ctx.save_for_backward(W,A)
        requires_grad = W.requires_grad or A.requires_grad
        c = tensor.Tensor(np.dot(W.data,A.data), requires_grad =requires_grad,is_leaf = not requires_grad )
        return c


    @staticmethod   
    def backward(ctx,grad_output):
        W,A= ctx.saved_tensors
        dW  = tensor.Tensor(np.dot(grad_output.data, A.data.T))
        dA  = tensor.Tensor(np.dot(W.data.T,grad_output.data))

        assert (dA.shape == A.shape)
        assert (dW.shape == W.shape)
        return [dW,dA]



class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return [grad_a, grad_b]


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a,b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = -np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return [grad_a, grad_b]



class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return [tensor.Tensor(grad)]


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.multiply(a.data,b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.multiply(grad_output.data,b.data)
        # dL/db = dout/db * dL/dout
        grad_b = np.multiply(a.data, grad_output.data)

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return [grad_a, grad_b]



class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.true_divide(a.data,b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da  / b
        grad_a = np.true_divide(grad_output.data, b.data)
        # dL/db = -a* dL/dout / b^2 
        grad_b = -a.data*np.true_divide(grad_output.data,b.data**2)
        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return [grad_a, grad_b]



def LogSoftmax(X):
    """ Args:
        X (Tensor): (N, C) 
    Returns:
         LogSoftmax : (N,C)
    """
    Max_vals_across_C = tensor.Tensor(X.data.max(axis=1,keepdims=True),requires_grad=False) # (N,1)
    Shifted_X_vals    = X - Max_vals_across_C # (N,C) -(N,1)
    Scales            = Shifted_X_vals.exp().sum(axis=1,keepdims=True).log() + Max_vals_across_C # (N,1)
    Scales            = Scales.exp()
    softmaxs          = X.exp()/Scales
    LogSoftmaxs        = softmaxs.log()

    return LogSoftmaxs




def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.
    batch_size, num_classes = predicted.shape
    Max_vals_across_C = tensor.Tensor(predicted.data.max(axis=1,keepdims=True),requires_grad=False) # (N,1)
    Shifted_X_vals    = predicted - Max_vals_across_C # (N,C) -(N,1)
    Scales            = Shifted_X_vals.exp().sum(axis=1,keepdims=True).log() + Max_vals_across_C # (N,1)
    Scales            = Scales.exp()
    softmaxs          = predicted.exp()/Scales
    P                 = softmaxs.log()

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    targets = to_one_hot(target,num_classes)
    loss = (P*targets).sum(axis=1).sum(axis=0)
    loss = loss/tensor.Tensor(-batch_size,requires_grad=False)
    return loss



def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)


"""This file contains new code for hw2 that you should copy+paste to the appropriate files.

We'll tell you where each method/class belongs."""


# ---------------------------------
# nn/functional.py
# ---------------------------------

def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    out = (input_size - kernel_size)//stride +1
  
    return out

    


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x,weight,bias)
        ctx.stride = stride # saving for backward

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        # output_size = get_conv1d_output_size(None, None, None)
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        # out = np.zeros(())

        out = np.zeros((batch_size,out_channel,output_size))

        requires_grad = weight.requires_grad or bias.requires_grad or x.requires_grad

        k = 0
        for i in range(0,input_size-kernel_size+1,stride):

            segment = x.data[:,:,i:i+kernel_size] #(batch_size, in_channel, kernel_size)

            out[:,:,k] = np.tensordot(segment, weight.data, axes=([1,2],[1,2]))+bias.data

            k +=1

        return tensor.Tensor(out,requires_grad=requires_grad, is_leaf=not requires_grad)


    @staticmethod
    def backward(ctx, grad_output):
    
        x,weight,bias = ctx.saved_tensors
        stride        = ctx.stride

        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        
        grad_weight = np.zeros(weight.shape)
        grad_bias   = np.zeros(bias.shape)
        grad_x      = np.zeros(x.shape)
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
   
            # x (Tensor): (batch_size, in_channel, input_size) input data
            # weight (Tensor): (out_channel, in_channel, kernel_size)
            # bias (Tensor): (out_channel,)
            # stride (int): Stride of the convolution
            # grad_output: (batch_size, out_channel, output_size) output data

            # segment      : (batch_size, in_channel, kernel_size)
            # grad_out    : (batch_size, out_channel, output_size)
            # grad_weight : (out_channel, in_channel, kernel_size)

        # dX
        """
        for i_c in range(in_channel):
            for o_s in range(output_size):
                for k_s in range(kernel_size):
                    grad_x[:,i_c,stride*o_s+k_s] += np.tensordot(grad_output.data[:,:,o_s],weight.data[:,i_c,k_s],axes=([1],[0]))
        
        
        """
        # dX 
        for o_s in range(output_size):
            for k_s in range(kernel_size):
                grad_x[:,:,stride*o_s+k_s] += np.tensordot(grad_output.data[:,:,o_s],weight.data[:,:,k_s],axes=([1],[0]))
        

        # dW
        k = 0
        for i in range(0,input_size-kernel_size+1,stride):
            segment = x.data[:,:,i:i+kernel_size] 
            grad_weight += np.tensordot(grad_output.data[:,:,k], segment, axes =([0],[0])) # sum across mini_batches
            k +=1

        # db 
        grad_bias += np.einsum('ijk->j',grad_output.data) # sum across batches and output_size
        return[tensor.Tensor(grad_x), tensor.Tensor(grad_weight),tensor.Tensor(grad_bias)]
 




class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return [tensor.Tensor(grad),None]
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return [tensor.Tensor(grad),None]


# ---------------------------------
# nn/functional.py : HW3
# ---------------------------------
class Pow(Function):
    @staticmethod
    def forward(ctx,a,exponent):
        ctx.exponent = exponent
        ctx.save_for_backward(a)
        out = tensor.Tensor(a.data**exponent, requires_grad=a.requires_grad, is_leaf= not a.requires_grad)
        return out

    @staticmethod
    def backward(ctx,grad_output):
        exponent = ctx.exponent 
        a = ctx.saved_tensors[0]
        grad = grad_output.data*exponent*a.data**(exponent-1)
        return [tensor.Tensor(grad),None]
        


class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        #raise NotImplementedError('Implemented Slice.forward')
        ctx.save_for_backward(x)
        ctx.indices = indices
        out = tensor.Tensor(x.data[indices],requires_grad= x.requires_grad,is_leaf = not x.requires_grad)
        return out

    @staticmethod
    def backward(ctx,grad_output):
        indices = ctx.indices
        x  = ctx.saved_tensors[0]
        grad = np.zeros(x.shape)
        grad[indices] = grad_output.data
        return [tensor.Tensor(grad),None]
        #raise NotImplementedError('Implemented Slice.backward')



################### THE CONCATENATION #####################

### ALTERNATE 1

# This should work for everyone. The non-tesnor argument is brought to the end so that this become similar to the other functions such as Reshape in terms of what you return from backward and how you handle this in your Autograd engine

# ---------------------------------
# nn/functional.py
# ---------------------------------

class Cat(Function):

    @staticmethod
    #def forward(ctx,dim,*seq):
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int)
         is the dimension along which we want to concatenate 
        '''
        *seq, dim = args
        indices = [x.shape[dim] for x in list(seq)]
        my_list = [x.requires_grad for x in list(seq)]
        requires_grad = True if True in my_list else False
        ctx.indices =np.cumsum(indices)
        ctx.dim = dim
        out = np.concatenate([a.data for a in list(seq)],dim)
        return tensor.Tensor(out,requires_grad=requires_grad, is_leaf = not requires_grad)

    @staticmethod
    def backward(ctx,grad_output):
        indices = ctx.indices
        dim     = ctx.dim
        grad_seq = np.split(grad_output.data,indices,axis=dim)[:-1]
        return [tensor.Tensor(grad) for grad in grad_seq]



