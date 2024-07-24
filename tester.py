import os
import sys
import timeit
import argparse
from typing import List, Dict, Tuple, Set, Union, Optional, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTester(object):

    def __init__(self) -> None:

        pass

    def prepare_model(self) -> None:

        pass

    def test_training(self) -> None:

        pass

    def test_inference(self) -> None:

        pass
