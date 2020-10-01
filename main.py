# -*- coding: utf-8 -*-
"""
@author: anubhav sachan
"""
'''
torch.normal(mean, std, *, generator=None, out=None) → Tensor
'''
import sys
import time
import logging
import torch

from utils import get_parser, init_logging_handler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)

    init_logging_handler(args.log_dir)
    logging.debug(str(args))

    if args.train:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train {}'.format(current_time))