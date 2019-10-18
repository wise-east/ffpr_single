# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket

import torch

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__file__)
RANDOM_STATE = 42 

def get_dataset(args, tokenizer): 

    """ Get custom data"""
    data_path = args.data_path
    data_cache = args.data_cache + '_' + type(tokenizer).__name__  # For avoiding using GPT cache for GPT-2 and vice-versa
    if data_cache and os.path.isfile(data_cache):
        logger.info("Load tokenized dataset from cache at %s", data_cache)
        dataset = torch.load(data_cache)
        return dataset 
    
    logger.info("No cache file found. Load dataset from %s", data_path)
    logging.info("Loading data and tokenizing...")
    with open(data_path, 'r') as f: 
        data = json.load(f) 

    train, valid = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    reformatted_data = {'train': train, 'valid': valid}

    def tokenize(obj): 
        if isinstance(obj, str): 
            return tokenizer.encode(obj) 
        if isinstance(obj, dict): 
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    dataset = tokenize(reformatted_data)  
    torch.save(dataset, data_cache)
    
    return dataset 


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_logdir(model_name: str):
    """Create unique path to save results and checkpoints, e.g. runs/Sep22_19-45-59_gpu-7_gpt2"""
    # Code copied from ignite repo
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return logdir
