import torch as t
import time
from torch.nn import Module

class BasicModule(t.nn.Module):
    """封装了nn.modu;e 主要提供save和Load两种方法"""
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))#模型默认的方法

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None,counter=0):
        if name is None:
            prefix = 'D:/workspace/code/pytorch/DogCat/models/checkpoints/' + self.model_name + '_'+str(counter)+'.pth'
            # name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        t.save(self.state_dict(), prefix)
        return name

