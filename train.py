import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import trainer as Trainer
import trainer.loss as Loss
import metric as Metric
import model as Model
import model.tokenizer as Tokenizer
from parse_config import ConfigParser
from utils import prepare_device

import pdb
debug = pdb.set_trace

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config:ConfigParser):
    logger = config.get_logger('train')

    # build tokenizer
    tokenizer = config.init_obj('tokenizer', Tokenizer)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data, tokenizer=tokenizer)
    valid_data_loader = data_loader.split_validation()
    
    # build model architecture, then print to console
    model = config.init_obj('arch', Model)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(Loss, config['loss'])
    metrics = [getattr(Metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = config.init_obj(
        'trainer', Trainer, 
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        tokenizer=tokenizer,
    )
    # trainer = Trainer()

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Automated Audio Captioning for Reinforcement Learning Final Project.')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
