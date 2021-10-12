import numpy as np

import mytorch.autograd_engine as autograd_engine
from mytorch.nn import functional as F
from mytorch.autograd_engine import AccumulateGrad


class Tensor:
    """Tensor object, similar to `torch.Tensor`
    A wrapper around a NumPy array that help it interact with MyTorch.

    Args:
        data (np.array): the actual data of the tensor
        requires_grad (boolean): If true, accumulate gradient in `.grad`
        is_leaf (boolean): If true, this is a leaf tensor; see writeup.
        is_parameter (boolean): If true, data contains trainable params
    """
    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None # Set during forward pass
        self.grad = None
        self.is_parameter = is_parameter

    # ------------------------------------
    # [Not important] For printing tensors
    # ------------------------------------
    def __str__(self):
        return "{}{}".format(
            str(self.data),
            ", grad_fn={}".format(self.grad_fn.__class__.__name__) if self.grad_fn is not None else ""
        )

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------
    # Tensor Operations (NOT part of comp graph)
    # ------------------------------------------
    @property
    def shape(self):
        """Returns the shape of the data array in a tuple.
        >>> a = Tensor(np.array([3,2])).shape
        (2,)
        """
        return self.data.shape

    def fill_(self, fill_value):
        """In-place operation, replaces data with repeated value"""
        self.data.fill(fill_value)
        return self

    def copy(self):
        """Returns copy of this tensor
        Note: after copying, you may need to set params like `is_parameter` manually"""
        return Tensor(self.data)

    # Below methods can be used WITHOUT creating a tensor first
    # (For example, we can call Tensor.zeros(3,2) directly)

    @staticmethod
    def zeros(*shape):
        """Creates new tensor filled with 0's
        Args:
            shape: comma separated ints i.e. Tensor.zeros(3,4,5)
        Returns:
            Tensor: filled w/ 0's
        """
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(*shape):
        """Creates new tensor filled with 1's
        Note: if you look up "asterik args python", you'll see this function is
        called as follows: ones(1, 2, 3), not: ones((1, 2, 3))
        """
        return Tensor(np.ones(shape))

    @staticmethod
    def arange(*interval):
        """Creates new tensor filled by `np.arange()`"""
        return Tensor(np.arange(*interval))

    @staticmethod
    def randn(*shape):
        """Creates new tensor filled by normal distribution (mu=0, sigma=1)"""
        return Tensor(np.random.normal(0, 1, shape))

    @staticmethod
    def empty(*shape):
        """Creates an tensor with uninitialized data (NOT with 0's).

        >>> Tensor.empty(1,)
        [6.95058141e-310]
        """
        return Tensor(np.empty(shape))

    # ----------------------
    # Autograd backward init
    # ----------------------
    def backward(self):
        """Kicks off autograd backward (see writeup for hints)"""
        #raise Exception("TODO: Kick off `autograd_engine.backward()``")
        output = Tensor.ones(*self.shape)
        #self.grad = output
        autograd_engine.backward(self.grad_fn,output)

    # ------------------------------------------
    # Tensor Operations (ARE part of comp graph)
    # ------------------------------------------
    def T(self):
        """Transposes data (for 2d data ONLY)

        >>> Tensor(np.array([[1,2,3],[4,5,6]])).T()
        [[1, 4],
         [2, 5],
         [3, 6]]
        """
        return F.Transpose.apply(self)

    def reshape(self, *shape):
        """Makes new tensor of input shape, containing same data
        (NOT in-place operation)

        >>> Tensor(np.array([[1,2,3],[4,5,6]])).reshape(3,2)
        [[1, 2],
         [3, 4],
         [5, 6]]
        """
        return F.Reshape.apply(self, shape)

    def log(self):
        """Element-wise log of this tensor, adding to comp graph"""
        return F.Log.apply(self)

    def exp(self):
        """Element-wise exp of this tensor, adding to comp graph"""
        return F.Exp.apply(self)

    def __add__(self, other):
        """Links "+" to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after adding
        """
        return F.Add.apply(self, other)

    def __sub__(self, other):
        """Links "-" to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after elmentwise-substraction
        """
        return F.Sub.apply(self,other)


    def __mul__(self,other):
        """Links "*" to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after elementwise-product
        """
        return F.Mul.apply(self,other)

    def __truediv__(self,other):
        """Links "/" to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after elementwise-division
        """
        return F.Div.apply(self,other)


    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis, keepdims)


    def __matmul__(self,other):
        """Links " " to the comp. graph
        Args:
            other (Tensor): other tensor to add
        Returns:
            Tensor: result after matrix product
        """
        return F.Matmul.apply(self,other)



    # ---------------------------------
    # tensor.py : HW3
    # ---------------------------------
    def __pow__(self,other):
        """ computes the Self**other
        Args: 
            other: (integer): the exponent
        Returns: 
            Tensor : result after the elementwise power
        """
        return F.Pow.apply(self,other)


    # NOTE: This is a METHOD to be implmented in tensor.Tensor class.

    def __len__(self,):
        return len(self.data)

    
    # ---------------------------------
    # tensor.py
    # ---------------------------------

    # NOTE: This is a METHOD to be implmented in tensor.Tensor class.
    def __getitem__(self, key):
        # TODO: Implement the __getitem__ operation. Simply invoke the appropriate function from functional.py
        #raise NotImplementedError('TODO: Implement functional.Slice')
        #print("key: ",key)
        return F.Slice.apply(self,key)



    # ---------------------------------
    # tensor.py
    # ---------------------------------

    # NOTE: This is a METHOD to be implemented in tensor.Tensor class.
    def unsqueeze(self,dim=0):
        """ 
        Returns a new tensor with a dimension of size one inserted at the specified position. 
        
        NOTE: If you are not sure what this operation does, please revisit Recitation 0.
        
        Example:
            a
            [[1 2 3]
             [4 5 6]]
            
            a.unsqueeze(0)
            [[[1 2 3]
              [4 5 6]]]
            
            a.unsqueeze(1)
            [[[1 2 3]]
            
             [[4 5 6]]]
            
            a.unsqueeze(2)
            [[[1]
              [2]
              [3]]
            
             [[4]
              [5]
              [6]]]
        """
        # TODO: Implement the unsqueeze operation
        #raise NotImplementedError('Use existing functions in functional.py to implement this operation!')
        shape = list(self.shape)
        shape.insert(dim,1)
        return F.Reshape.apply(self,shape)

    # tensor.py
    # ---------------------------------

    # NOTE: This is an indepedent function and NOT A METHOD in tensor.Tensor class 
def cat(seq,dim=0):
    '''
    Concatenates the given sequence of seq tensors in the given dimension.
    All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    
    NOTE: If you are not sure what this operation does, please revisit Recitation 0.

    Args:
        seq (list of Tensors) - List of interegers to concatenate
        dim (int) - The dimension along which to concatenate
    Returns:
        Tensor - Concatenated tensor

    Example:

        seq
        [[[3 3 4 1]
          [0 3 1 4]],
         [[4 2 0 0]
          [3 0 4 0]
          [1 4 4 3]],
         [[3 2 3 1]]]
        
        tensor.cat(seq,0)
        [[3 3 4 1]
         [0 3 1 4]
         [4 2 0 0]
         [3 0 4 0]
         [1 4 4 3]
         [3 2 3 1]]
    '''
    # TODO: invoke the appropriate function from functional.py. One-liner; don't overthink
    #raise NotImplementedError("TODO: Complete functional.Cat!")
    #print("in Tensor : Seq: {}, dim : {} ".format(seq,dim))

    return F.Cat.apply(*seq,dim)


    