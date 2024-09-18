import torch 

class SimpleDGCCompressor:
    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio

    def _sparsify(self, tensor):
        # Flatten the tensor
        tensor = tensor.view(-1)
        numel = tensor.numel()
        num_selects = max(1, int(numel * self.compress_ratio))

        # Get the importance (absolute values)
        importance = tensor.abs()

        # Find the top k% elements directly using topk
        threshold, indices = torch.topk(importance, num_selects, sorted=False)

        # Extract the values at these indices
        values = tensor[indices]
        return values, indices, numel

    def compress(self, tensor):
        # Perform sparsification
        values, indices, numel = self._sparsify(tensor)
        
        # Return the compressed representation and the original tensor size
        return (values, indices, numel)

    def decompress(self, compressed_tensor, original_size):
        values, indices, numel = compressed_tensor
        # Create an empty tensor of the original size
        decompressed_tensor = torch.zeros(original_size, device=values.device)
        # Put the values back to their original positions
        decompressed_tensor.index_put_([indices], values)
        # Reshape back to the original shape
        return decompressed_tensor.view(original_size)
