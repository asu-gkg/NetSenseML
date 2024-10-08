class Compressor():
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)
    
    

class Memory():
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass

