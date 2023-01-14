from src.utils.config import instantiate
import torch
from src.models.sequence.base import SequenceModule
from src.utils import registry

class WindowModel(SequenceModule):
    def __init__(self, d_model, transposed=False, chunk_len=199, model=None, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.model = instantiate(registry.model, model)
        self.chunk_len = chunk_len
        self.transposed = transposed

    def forward(self, x, state=None):
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        chunked_x = [x[:,i:i + self.chunk_len,:] for i in range(0, seq_length, self.chunk_len)]
        outputs = []
        for i,chunk in enumerate(chunked_x):
            output, state = self.model(chunk)
            outputs.append(chunk)

        x = torch.cat(outputs,dim=1)
        return x, [state]