import os
import sys
import timeit
import argparse
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def main(
        args: Dict[str, Any]) -> None:

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = 'str', choices = ['training', 'inference'],
                        help = 'Test mode: "training" or "inference"')
    parser.add_argument('--base-path', type = str,
                        help = 'Base path')
    parser.add_argument('--fp16', action = 'store_true',
                        help = 'Use FP16')
    parser.add_argument('--accelerator-type', type = int, choices = ['cpu', 'cuda', 'ipu'],
                        help = 'Accelerator type: "cpu", "cuda", or "ipu"')
    parser.add_argument('--num-accelerator', type = int,
                        help = '# of accelerator')
    parser.add_argument('--accelerator-list', type = str, nargs = '+',
                        help = 'Accelerator list')
    parser.add_argument('--omp-num-threads', type = int,
                        default = 1,
                        help = 'OMP_NUM_THREADS option')
    parser.add_argument('--debug', action = 'store_true',
                        help = 'debug mode')
    args = vars(parser.parse_args())

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(opt['accelerator_list'])    
    os.environ['OMP_NUM_THREADS'] = str(opt['omp_num_threads'])

    main(args)
