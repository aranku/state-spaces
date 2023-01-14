from src.utils.config import instantiate
import torch
from torch import nn
from src.models.sequence.base import SequenceModule
from src.utils import registry

class CompressionModel(SequenceModule):
    def __init__(self, d_model, transposed=False, chunk_len=199, num_special_tokens=1, model=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.model = instantiate(registry.model, model)
        self.chunk_len = chunk_len
        self.transposed = transposed
        self.num_special_tokens = num_special_tokens
        self.special_token = nn.Parameter(torch.zeros(num_special_tokens,d_model))

    def forward(self, x, *args, state=None, **kwargs):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        special_token_batched = self.special_token.repeat(batch_size,1,1)

        chunked_x = [x[:,i:i + self.chunk_len,:] for i in range(0, seq_length, self.chunk_len)]

        prefix = []
        outputs = []
        for i,chunk in enumerate(chunked_x):
            output, state = self.model(torch.cat(prefix + [chunk, special_token_batched], dim=1))
            prefix.append(output[:,-self.num_special_tokens:,:])
            outputs.append(output[:,i*self.num_special_tokens:-self.num_special_tokens,:])

        result = torch.cat(outputs,dim=1)
        torch._assert(x.shape == result.shape, "Input and output shapes do not match")
        return result, [state]