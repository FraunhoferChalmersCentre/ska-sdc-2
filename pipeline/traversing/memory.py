import nvsmi
import torch
import numpy as np
import os


def max_batch_size(model, dim, gpu_memory_mib):
    torch.cuda.empty_cache()
    pid = os.getpid()
    model.eval()
    input_one = torch.ones(1, 1, *dim).to('cuda')
    try:
        model(input_one)
    except:
        raise ValueError('Too little memory for batch size = 1')
    gpus = nvsmi.get_gpu_processes()
    pids = [int(str(gpu).split(' ')[1]) for gpu in gpus]
    index = np.argwhere(np.array(pids) == pid)[0][0]
    mem_one = float(str(gpus[index]).split(' ')[-1].split('MB')[0])
    del input_one
    torch.cuda.empty_cache()

    input_two = torch.ones(2, 1, *dim).to('cuda')
    try:
        model(input_two)
    except:
        raise ValueError('Too little memory for batch size = 2')
    gpus = nvsmi.get_gpu_processes()
    mem_two = float(str(gpus[index]).split(' ')[-1].split('MB')[0])
    del input_two
    torch.cuda.empty_cache()

    k = mem_two - mem_one
    m = mem_one - k
    batch_size = int((gpu_memory_mib - m)/k)
    input_bs = torch.ones(batch_size, 1, *dim).to('cuda')
    try:
        model(input_bs)
    except:
        raise ValueError('Too little memory for batch size = ' + str(batch_size))
    del input_bs
    torch.cuda.empty_cache()
    return batch_size
