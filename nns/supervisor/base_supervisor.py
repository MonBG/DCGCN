import os
import yaml
import time
import math
import torch
import json
import matplotlib.pyplot as plt
import copy
from torch.utils.tensorboard import SummaryWriter
from abc import ABCMeta, abstractmethod
from tqdm import tqdm

from utils.data_utils import load_data
from utils.logger import get_logger
from utils.train_inits import print_model_parameters, init_seed
from nns.config import device

DEBUG_MODE = True

""" Supervisor config list
base_dir: D:/Code_Files/Pycharm/Traffic
log_dir: data/model_logs/***
log_level: DEBUG
save_and_log: false

data:
  batch_size: 64
  dataset_dir: data/METR-LA-***

model:
  description: ***
  special_id: ***
  *model specific*: ***

loss:
  *loss specific*: ***

train:
  epoch: 0
  epochs: 100
  log_step: 10
  tensorboard_dir: runs/***
  plot: true
  optimizer: adam
  epsilon: 1.0e-3
  lr_type: MultiStepLR
  base_lr: 0.001
  lr_milestones: [20, 30, 40, 50]
  lr_decay_ratio: 0.1
  max_grad_norm: 5
  early_stop: 10
"""


def load_checkpoint(log_dir, Supervisor, base_dir=None, save_and_log=False, save_tb=False,
                    load_data=True, batch_size=None, **kwargs):
    check_point = torch.load(os.path.join(log_dir, 'best_model.pth'))
    state_dict = check_point['state_dict']
    with open(os.path.join(log_dir, 'config.txt'), 'r') as f:
        saved_kwargs = json.load(f)

    if base_dir is not None:
        saved_kwargs['base_dir'] = base_dir
    saved_kwargs['pre_trained'] = True
    saved_kwargs['log_dir'] = log_dir  # set to current log_dir
    saved_kwargs['save_and_log'] = save_and_log
    saved_kwargs['save_tb'] = save_tb
    saved_kwargs['load_data'] = load_data
    if load_data and batch_size is not None:
        saved_kwargs['data']['batch_size'] = batch_size
    for key, value in kwargs.items():  # additional params to changed
        saved_kwargs[key] = value

    supervisor = Supervisor(**saved_kwargs)
    supervisor.model.load_state_dict(state_dict)
    return supervisor


class BaseSupervisor(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.base_dir = kwargs.get('base_dir')
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._loss_kwargs = kwargs.get('loss')

        # set seed
        init_seed(self._train_kwargs.get('seed', 1))

        # supervisor state
        self._save_and_log = self._kwargs.get('save_and_log', True)
        self._save_tb = self._kwargs.get('save_tb', True)

        # logging
        self._run_id = self._gen_run_id()
        log_level = self._kwargs.get('log_level', 'INFO')
        if self._kwargs.get('pre_trained', False):
            self.log_dir = self._get_log_dir(kwargs, None)  # use log_dir directly
        else:
            self.log_dir = self._get_log_dir(kwargs, self._run_id)
        if self._save_and_log:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        self.logger = get_logger(self.log_dir, __name__, 'info.log', level=log_level, write_file=self._save_and_log)
        if self._save_tb:
            self.tb_dir = os.path.join(self.base_dir,
                                       self._train_kwargs.get('tensorboard_dir', 'runs'),
                                       self._run_id)
            self._writer = SummaryWriter(self.tb_dir)

        # data set
        if self._model_kwargs.get('load_data', True):
            self._data = load_data(os.path.join(self.base_dir, self._data_kwargs['dataset_dir']),
                                   self._data_kwargs['batch_size'],
                                   sample_ratio=self._data_kwargs.get('sample_ratio', 1.))
            self.standard_scaler = self._data['scaler']
            self.train_loader = self._data['train_loader']
            self.val_loader = self._data['val_loader']
            self.train_per_epoch = len(self.train_loader)
            self.val_per_epoch = len(self.val_loader)
            self.test_loader = self._data['test_loader']

        # model
        self.model = self._get_model()
        self.model = self.model.to(device)
        self.logger.info("Model created")
        self.logger.info(f"Model Desc: {self._model_kwargs['description']}")

        # training
        self.train_iter = 0  # starts from 1 when training
        self.cl_learn = self._train_kwargs.get('cl_learn', False)
        if self.cl_learn:
            self.cl_step = self._train_kwargs.get('cl_step', 100)
            self.logger.info(f"Train model with curriculum learning, step {self.cl_step}")
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.optimizer = None
        self.lr_scheduler = None
        self.reset_optimizer_and_scheduler()

        # model saving
        self._save_model_dir = self.log_dir
        self.save_config()

    def reset_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self._train_kwargs.get('base_lr', 0.001),
                                          eps=self._train_kwargs.get('epsilon', 1e-3),
                                          weight_decay=self._train_kwargs.get('weight_decay', 0))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                 milestones=self._train_kwargs.get('lr_milestones',
                                                                                                   [20, 30, 40, 50]),
                                                                 gamma=self._train_kwargs.get('lr_decay_ratio', 0.1))

    def save_checkpoint(self, epoch):
        if not self._save_and_log:
            self.logger.info("Save_and log is false. No model is saved")
            return

        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self._kwargs
        }
        torch.save(state, self._save_model_dir + f'/best_model.pth')
        self.logger.info("Saved current best model at {}".format(epoch))

    def save_config(self):
        if self._save_and_log:
            with open(os.path.join(self.log_dir, 'config.txt'), 'w') as f:
                json.dump(self._kwargs, f, indent=2)

    def train(self):
        self._train()

    def _train(self):
        self.train_iter = 0
        if self._train_kwargs.get('debug', False):
            self.logger.info("Debug Mode")
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []

        # print_model_parameters(self.model)
        start_time = time.time()
        for epoch in tqdm(range(1, self._train_kwargs['epochs'] + 1)):
            train_epoch_loss = self.train_epoch(epoch)
            if math.isnan(train_epoch_loss):
                self.logger('Found nan loss, training loop break!')
                break
            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.val_loader is None:
                val_epoch_loss = 0
            else:
                val_epoch_loss = self.val_epoch(epoch)
            self.logger.info(f"LR: {self.optimizer.param_groups[0]['lr']}")
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            # if train_epoch_loss > 1e5:
            #     self.logger.warning('Gradient explosion detected. Ending...')
            #     break

            if self.val_loader is None:
                val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                if epoch < self._train_kwargs['min_epochs']:
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                    print(f"not_improved_count: {not_improved_count}")
                best_state = False
            # early stop
            if not_improved_count == self._train_kwargs['early_stop']:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self._train_kwargs['early_stop']))
                break

            # save to tensorboard
            if self._save_tb:
                for name, weight in self.model.named_parameters():
                    name_split = name.split(".")
                    save_name = '/'.join(name_split)
                    self._writer.add_histogram(save_name, weight, epoch)
            # save the best state
            if best_state:
                self.save_checkpoint(epoch)
                best_model = copy.deepcopy(self.model.state_dict())
            # plot loss figure
            if self._train_kwargs['plot'] and self._save_and_log:
                self._plot_line_figure([train_loss_list, val_loss_list], path=self._save_model_dir,
                                       warmup=self._train_kwargs.get('plot_warmup', 5))
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min".format(training_time / 60))
        self.model.load_state_dict(best_model)

    @staticmethod
    def _get_log_dir(kwargs, run_id=None):
        log_dir = kwargs.get('log_dir')
        base_dir = kwargs.get('base_dir')
        if log_dir is None:
            log_dir = base_dir
        if run_id is not None:
            log_dir = os.path.join(base_dir, log_dir, run_id)
        else:
            log_dir = os.path.join(base_dir, log_dir)

        return log_dir

    @staticmethod
    def _plot_line_figure(losses, path, warmup=5):
        # whole loss
        train_loss = losses[0]
        val_loss = losses[1]
        plt.style.use('ggplot')
        epochs = list(range(1, len(train_loss) + 1))
        plt.plot(epochs, train_loss, 'r-o')
        plt.plot(epochs, val_loss, 'b-o')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(os.path.join(path, 'loss.png'), bbox_inches="tight")
        plt.cla()
        plt.close("all")

        # loss with warmup
        train_loss = losses[0][(warmup - 1):]
        val_loss = losses[1][(warmup - 1):]
        plt.style.use('ggplot')
        epochs = list(range(1, len(train_loss) + 1))
        plt.plot(epochs, train_loss, 'r-o')
        plt.plot(epochs, val_loss, 'b-o')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(os.path.join(path, 'loss_warmup.png'), bbox_inches="tight")
        plt.cla()
        plt.close("all")

    @abstractmethod
    def _gen_run_id(self):
        pass

    @abstractmethod
    def _get_model(self):
        pass

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    @abstractmethod
    def val_epoch(self, epoch):
        pass
