# -*- coding: utf-8 -*-
"""
@author: anubhav sachan
"""
import time
import logging
import os
import numpy as np
import argparse
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='log', help='Logging directory')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory to store model')
    parser.add_argument('--load', type=str, default='', help='File name to load trained model')
    parser.add_argument('--train', type=bool, default=True, help='Set to train mode')
    parser.add_argument('--test_case', type=int, default=1000, help='Number of test cases')
    parser.add_argument('--save_per_epoch', type=int, default=4, help="Save model every XXX epoches")
    parser.add_argument('--print_per_batch', type=int, default=200, help="Print log every XXX batches")

    parser.add_argument('--epoch', type=int, default=48, help='Max number of epoch')
    parser.add_argument('--process', type=int, default=8, help='Process number')
    parser.add_argument('--batchsz', type=int, default=32, help='Batch size')
    parser.add_argument('--batchsz_traj', type=int, default=512, help='Batch size to collect trajectories')
    parser.add_argument('--policy_weight_sys', type=float, default=2.5, help='Pos weight on system policy pretraining')
    parser.add_argument('--policy_weight_usr', type=float, default=4, help='Pos weight on user policy pretraining')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of the policy')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discounted factor')
    parser.add_argument('--clip', type=float, default=10, help='Gradient clipping')
    parser.add_argument('--interval', type=int, default=400, help='Update interval of target network')

    parser.add_argument('--act_mu_en', type=float, default=0.05, help='The mean of the actual distribution')
    parser.add_argument('--act_sig_en', type=float, default=0.0667, help='The std deviation of the actual distribution')
    parser.add_argument('--act_mu_ch', type=float, default=0.5, help='The mean of the channel distribution')
    parser.add_argument('--act_sig_ch', type=float, default=0.165, help='The std deviation of the channel distribution')
    parser.add_argument('--N_0', type=float, default=4e-15, help='The noise variance')

    # Lower and Upper bounds of state space
    parser.add_argument('--en_min', type=float, default=0, help='Minimum level of the energy')
    parser.add_argument('--en_max', type=float, default=0.25, help='Maximum level of the energy')
    parser.add_argument('--ch_min', type=float, default=0, help='Minimum level of the channel')
    parser.add_argument('--ch_max', type=float, default=1.0, help='Maximum level of the channel')
    parser.add_argument('--min_bat', type=float, default=0.0, help='Minimum level of the Battery')
    parser.add_argument('--max_bat', type=float, default=3.0, help='Maximum level of the Battery')

    return parser


def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
