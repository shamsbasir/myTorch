from mytorch import tensor
from mytorch.tensor import Tensor,cat
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of 
    samples in the batch for which the PackedSequence is being constructed.
    PLEASE read the description carefully to avoid confusion. 
    The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K)
         where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape 
        ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # TODO: INSTRUCTIONS
    # 1) Find the sorted indices based on number of time steps in each sample
    size = [(index,x.shape[0]) for index ,x in enumerate(sequence)]
    size.sort(key=lambda x : x[1],reverse=True)
    sorted_indices =[]
    for item in size:
        index,_ = item 
        sorted_indices.append(index)

    time_dim_sizes = [x.shape[0] for x in sequence]

    # 2) Extract slices from each sample and properly order them for the construction of the 
    # packed tensor. __getitem__ you defined for Tensor class will come in handy
    
    data = []
    batch_sizes=[]
    for time in range(max(time_dim_sizes)):
        count = 0
        for index in sorted_indices:
            if time > time_dim_sizes[index]-1:
                continue 
            element = sequence[index][time,:]
            element = element.unsqueeze(0)
            #print("New element.shape :{}".format(element.shape))
            count +=1
            data.append(element)
        batch_sizes.append(count)

    # 3)  Use the tensor.cat function to create a single tensor from the re-ordered segements
    data = cat(data,dim=0)
    batch_sizes = np.array(batch_sizes)
    sorted_indices = np.array(sorted_indices)
    # 4)Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.

    return PackedSequence(data,sorted_indices,batch_sizes)
    #raise NotImplementedError('Implement pack_Sequence!')

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list
    # (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices
    
    list_of_tensor = []
    for i in range(max(ps.batch_sizes)):
        row = []
        count = 0
        for b in ps.batch_sizes:
            for j in range(b):
                if i==j:
                    element =ps.data[count].unsqueeze(0)
                    row.append(element)
                count +=1
        row = cat(row,dim=0)
        list_of_tensor.append(row)
    
    count = 0
    l = []
    while (count < len(list_of_tensor)):
        for index,item in enumerate(ps.sorted_indices):
            if item == count:
                l.append(list_of_tensor[index])
                count +=1

    return l
    #raise NotI mplementedError('Implement unpack_sequence')
