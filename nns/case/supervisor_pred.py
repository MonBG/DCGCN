import time
import os
import json
import math

import numpy as np
import torch
from tqdm import tqdm

from nns.case.pred_blocks import CascadeGCNv2d4
from nns.case.dag_gen_gru import DagGenGRUg4s2v1
from nns.supervisor.loss import masked_mae_loss, masked_mse_loss
from utils.graph_utils import spectral_norm_tensor, spatial_norm_tensor
from nns.supervisor.base_supervisor import BaseSupervisor
from nns.config import device

from utils.plot_lib import display_seq_pred
from utils.path_utils import base_dir
from utils.data_utils import load_graph
from utils.metrics_lib import All_Metrics, All_Metrics2


class CASECausalPred(BaseSupervisor):
    def __init__(self, **kwargs):
        super(CASECausalPred, self).__init__(**kwargs)

        self.loss_type = self._loss_kwargs.get('type', 'mae')
        if self.loss_type == 'mae':
            self.masked_pred_loss = masked_mae_loss(self.standard_scaler, 0.1)
        elif self.loss_type == 'mse':
            self.masked_pred_loss = masked_mse_loss(self.standard_scaler, 0.1)
        else:
            raise Exception
        self.dag_gen_version = 'g4s2v1'
        # norm_type: 0, ori; 1, spectral; 2, spatial; 10, spectral without self-loop; 20, spatial without self-loop
        self.norm_type = int(self._model_kwargs.get('norm_type', 0))
        self.sym_graph = self._model_kwargs.get('sym_graph', False)
        self.dag_gen_model = self.load_dag_gen_model()
        self.split_num = int(self._data_kwargs.get('init_split', 1))
        # cl_learning
        self.task_level = 1

    def train(self):
        if self.num_nodes > 50:
            self.logger.info(f"Generate causal graph before training and save it to the drive. "
                             f"Split num {self.split_num}")
            self.init_graphs(self.train_loader, save=True, cat='train')
            self.init_graphs(self.val_loader, save=True, cat='val')
            self.init_graphs(self.test_loader, save=True, cat='test')
        else:
            self.logger.info("Generate causal graph before training.")
            self.graphs_train = self.init_graphs(self.train_loader)
            self.graphs_val = self.init_graphs(self.val_loader)
        self._train()

    def load_adj_mx(self):
        adj_mat = load_graph(os.path.join(self.base_dir, "data/sensor_graph/adj_mx.pkl"))
        if adj_mat.shape[0] > self.num_nodes:
            adj_mat = adj_mat[:self.num_nodes, :self.num_nodes]
        return adj_mat

    def load_dag_gen_model(self):
        log_dir = self._model_kwargs.get('dag_gen_log_dir')
        check_point = torch.load(os.path.join(log_dir, 'best_model.pth'))
        state_dict = check_point['state_dict']
        saved_kwargs = check_point['config']
        dag_gen_model = globals()['DagGenGRU' + self.dag_gen_version]

        hidden_dim = int(saved_kwargs['model'].get('hidden_dim', 512))
        z_dim = int(saved_kwargs['model'].get('z_dim', 8))
        num_layers = int(saved_kwargs['model'].get('num_layers', 6))
        num_heads = int(saved_kwargs['model'].get('num_heads', 8))
        feats_layers = int(saved_kwargs['model'].get('feats_layers', 3))
        agg_feats = saved_kwargs['model'].get('agg_feats', 'ori')
        node_norm = saved_kwargs['model'].get('node_norm', False)
        use_norm = saved_kwargs['model'].get('use_norm', False)
        use_pp = saved_kwargs['model'].get('use_pp', False)
        step_pri = saved_kwargs['model'].get('step_pri', 0.01)
        step_dual = saved_kwargs['model'].get('step_dual', 0.01)
        reg_sp_intra = saved_kwargs['model'].get('reg_sp_intra', 2e-3)
        num_intra_pp_iters = saved_kwargs['model'].get('num_intra_pp_iters', 1000)
        dist_adj = self.load_adj_mx()

        model = dag_gen_model(self.num_nodes, self.in_feats_dim, self.out_feats_dim,
                              hidden_dim=hidden_dim, num_layers=num_layers, dist_adj=dist_adj,
                              feats_layers=feats_layers, agg_feats=agg_feats, node_norm=node_norm,
                              use_norm=use_norm, use_pp=use_pp, step_pri=step_pri, step_dual=step_dual,
                              reg_sp_intra=reg_sp_intra, num_intra_pp_iters=num_intra_pp_iters,
                              num_heads=num_heads, z_dim=z_dim)
        model = model.to(device)
        model.load_state_dict(state_dict)
        self.logger.info(f"load dag_gen_attn_{self.dag_gen_version} model successfully in {log_dir}")
        return model

    def gen_causal_graph(self, data):
        self.dag_gen_model.eval()
        with torch.no_grad():
            if self.split_num == 1.:
                graphs, _ = self.dag_gen_model(data, gen_graph_only=True)  # (T, B, 2, N, N)
            else:
                batch_size = data.shape[1]
                split_bs = math.ceil(batch_size / self.split_num)
                graphs = []
                for i in range(self.split_num):
                    graphs_i, _ = self.dag_gen_model(data[:, i * split_bs:(i + 1) * split_bs, ...], gen_graph_only=True)
                    graphs.append(graphs_i)
                graphs = torch.concat(graphs, dim=1)

            if self.sym_graph:
                graphs = graphs + graphs.transpose(-1, -2)
            if self.norm_type == 1:
                graphs = spectral_norm_tensor(graphs, add_self_loops=True)
            elif self.norm_type == 10:
                graphs = spectral_norm_tensor(graphs, add_self_loops=False)
            elif self.norm_type == 2:
                graphs = spatial_norm_tensor(graphs, add_self_loops=True)
            elif self.norm_type == 20:
                graphs = spatial_norm_tensor(graphs, add_self_loops=False)
        return graphs

    def save_init_graphs(self, graphs, cat, batch_idx):
        dataset_name = os.path.basename(self._data_kwargs['dataset_dir'])
        sample_ratio = self._data_kwargs['sample_ratio']
        seed = self._train_kwargs.get('seed', 1)
        batch_size = self._data_kwargs['batch_size']
        save_name = f"{cat}_bi_{batch_idx}"
        log_dir = self._model_kwargs.get('dag_gen_log_dir')
        save_dir = os.path.join(log_dir, 'init', f'{dataset_name}_s{seed}_sr{sample_ratio}_bs{batch_size}')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(graphs.cpu(), os.path.join(save_dir, f"{save_name}.pt"))

    def check_init_graphs(self, cat, batch_idx):
        dataset_name = os.path.basename(self._data_kwargs['dataset_dir'])
        sample_ratio = self._data_kwargs['sample_ratio']
        seed = self._train_kwargs.get('seed', 1)
        batch_size = self._data_kwargs['batch_size']
        save_name = f"{cat}_bi_{batch_idx}"
        log_dir = self._model_kwargs.get('dag_gen_log_dir')
        save_dir = os.path.join(log_dir, 'init', f'{dataset_name}_s{seed}_sr{sample_ratio}_bs{batch_size}')
        if os.path.exists(os.path.join(save_dir, f"{save_name}.pt")):
            return True
        else:
            return False

    def load_init_graphs(self, cat, batch_idx):
        dataset_name = os.path.basename(self._data_kwargs['dataset_dir'])
        sample_ratio = self._data_kwargs['sample_ratio']
        seed = self._train_kwargs.get('seed', 1)
        batch_size = self._data_kwargs['batch_size']
        save_name = f"{cat}_bi_{batch_idx}"
        log_dir = self._model_kwargs.get('dag_gen_log_dir')
        save_dir = os.path.join(log_dir, 'init', f'{dataset_name}_s{seed}_sr{sample_ratio}_bs{batch_size}')

        if not os.path.exists(os.path.join(save_dir, f"{save_name}.pt")):
            self.logger.info(f"Can not found init_graph of {save_name}")
            raise Exception
        else:
            graphs = torch.load(os.path.join(save_dir, f"{save_name}.pt")).to(device)
            return graphs

    def init_graphs(self, data_loader, save=False, cat='train'):
        print("fast mode, initiating causal_graphs...")
        causal_graphs = []
        pbar = tqdm(total=len(data_loader))
        for batch_idx, (data, target) in enumerate(data_loader):
            if save and self.check_init_graphs(cat, batch_idx):
                pbar.update()
                continue
            data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                             out_feats_dim=self.out_feats_dim)
            graph = self.gen_causal_graph(data)
            pbar.update()
            if save:
                self.save_init_graphs(graph, cat, batch_idx)
                continue
            causal_graphs.append(graph)
        return causal_graphs

    def _get_model(self):
        self.model_version = 'CascadeGCNv2d4'
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.in_feats_dim = int(self._model_kwargs.get('in_feats_dim', 1))
        self.out_feats_dim = int(self._model_kwargs.get('out_feats_dim', 1))
        self.hidden_dim = int(self._model_kwargs.get('hidden_dim', 64))
        self.num_layers = int(self._model_kwargs.get('num_layers', 2))
        self.seq_len = int(self._model_kwargs.get('seq_len', 11))
        self.horizon = int(self._model_kwargs.get('horizon', 11))
        self.dist_adj = self.load_adj_mx()

        self.logger.info(f"Pred Model Version {self.model_version}")
        pred_model = globals()[self.model_version]

        self.logger.info(f"pred model {self.model_version}")
        model = pred_model(self.num_nodes, self.in_feats_dim, self.out_feats_dim,
                           hidden_dim=self.hidden_dim, dist_adj=self.dist_adj,
                           num_layers=self.num_layers, seq_len=self.seq_len, horizon=self.horizon)
        return model

    def _gen_run_id(self):
        batch_size = self._data_kwargs.get('batch_size')

        special_id = self._model_kwargs.get('special_id')
        dag_gen_version = self._model_kwargs.get('dag_gen_version', 'm1v1')
        model_version = 'CascadeGCNv2d4'
        num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        in_feats_dim = int(self._model_kwargs.get('in_feats_dim', 1))
        out_feats_dim = int(self._model_kwargs.get('out_feats_dim', 1))
        hidden_dim = int(self._model_kwargs.get('hidden_dim', 1))
        num_layers = int(self._model_kwargs.get('num_layers', 1))
        seq_len = int(self._model_kwargs.get('seq_len', 11))
        horizon = int(self._model_kwargs.get('horizon', 11))
        norm_type = int(self._model_kwargs.get('norm_type', 0))
        sym_graph = 1 if self._model_kwargs.get('sym_graph', False) else 0

        learning_rate = self._train_kwargs.get('base_lr')
        run_time = time.strftime('%m%d%H%M%S')

        run_id = f'{run_time}_case_pred_{special_id}_{model_version}_{dag_gen_version}' \
                 f'_bs_{batch_size}_N_{num_nodes}_seq_{seq_len}_hor_{horizon}' \
                 f'_in_{in_feats_dim}_out_{out_feats_dim}_h_{hidden_dim}_nl_{num_layers}' \
                 f'_norm_{norm_type}_sym_{sym_graph}_lr_{learning_rate}'
        return run_id

    def loss(self, y_true, y_pred):
        """
        :param y_true: (T, B, N, D)
        :param y_pred: (T, B, N, D)
        :return:
        """
        pred_loss = self.masked_pred_loss(y_pred, y_true)

        if torch.isnan(pred_loss).item():
            self.logger.info('nan occur in loss computation')

        return pred_loss, {
            "pred_loss": pred_loss,
        }

    @staticmethod
    def _get_x_y_correct_dim(x, y):
        """
        :param x: shape (batch_size, seq_len + 1, num_sensor, input_dim)
        :param y: shape (batch_size, seq_len, num_sensor, input_dim)
        :returns x shape (seq_len + 1, batch_size, num_sensor, input_dim)
                 y shape (seq_len, batch_size, num_sensor, input_dim)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def prepare_data(self, x, y, num_nodes=None, out_feats_dim=None):
        """ split x
        :param x: shape (B, T + 1, N, D)
        :param y: shape (B, T, N, F)
        :return:
        """
        if num_nodes is not None:
            x = x[..., :num_nodes, :]
            y = y[..., :num_nodes, :]
        x, y = self._get_x_y_correct_dim(x, y)  # (T or T+1, B, N, D)
        if out_feats_dim is not None:
            y = y[..., :out_feats_dim]
        return x.to(device), y.to(device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.train_iter += 1
            data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                             out_feats_dim=self.out_feats_dim)
            self.optimizer.zero_grad()

            if self._train_kwargs.get('debug', False):
                torch.autograd.set_detect_anomaly(True)

            if self.num_nodes > 50:
                graphs = self.load_init_graphs('train', batch_idx)
            else:
                graphs = self.graphs_train[batch_idx]
            tgt_pred = self.model(data[1:, ...], graphs)

            if self.cl_learn:
                if self.train_iter % self.cl_step == 0 and self.task_level < self.horizon:
                    self.task_level += 1
                    self.logger.info(f"update task_level: {self.task_level}")
                loss, _ = self.loss(target[:self.task_level, ...], tgt_pred[:self.task_level, ...])
            else:
                loss, _ = self.loss(target, tgt_pred)

            if self._train_kwargs.get('debug', False):
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._train_kwargs['max_grad_norm'])

            self.optimizer.step()
            total_loss += loss.item()

            # log information
            if batch_idx % self._train_kwargs['log_step'] == 0:
                self.logger.info(f'Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.3f}')
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info(f'**********Train Epoch {epoch}: averaged Loss: mae {train_epoch_loss:.3f}')
        return train_epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                                 out_feats_dim=self.out_feats_dim)
                if self.num_nodes > 50:
                    graphs = self.load_init_graphs('val', batch_idx)
                else:
                    graphs = self.graphs_val[batch_idx]
                tgt_pred = self.model(data[1:, ...], graphs)
                loss, _ = self.loss(target, tgt_pred)

                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info(f'**********Val Epoch {epoch}: average Loss: {val_loss:.6f}')
        return val_loss

    def plot_pred(self, figname, save_dir):
        print("Testing...")
        self.model.eval()
        y_preds = []
        labels = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                                 out_feats_dim=self.out_feats_dim)
                if self.num_nodes > 50:
                    graphs = self.load_init_graphs('test', batch_idx)
                else:
                    graphs = self.gen_causal_graph(data)
                y_pred = self.model(data[1:, ...], graphs)  # (T, B, N, F)

                y_preds.append(y_pred)
                labels.append(target)

        y_preds = self.standard_scaler.inverse_transform(torch.cat(y_preds, dim=1)).detach().cpu()  # (T, B, N, F)
        labels = self.standard_scaler.inverse_transform(torch.cat(labels, dim=1)).detach().cpu()

        mae, rmse, mape = All_Metrics(y_preds, labels, 0.1, 0.1)
        display_seq_pred(y_preds[0, ...], labels[0, ...],
                         title=f"mae {mae:.2f}, mape {mape * 100: .2f}%, rmse {rmse: .2f}, ",
                         figname=figname, save_path=save_dir)
        print('finished!')

    def test(self):
        print("Testing...")
        self.model.eval()
        y_preds = []
        labels = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                                 out_feats_dim=self.out_feats_dim)
                if self.num_nodes > 50:
                    graphs = self.load_init_graphs('test', batch_idx)
                else:
                    graphs = self.gen_causal_graph(data)
                y_pred = self.model(data[1:, ...], graphs)  # (T, B, N, F)

                y_preds.append(y_pred)
                labels.append(target)

        y_preds = self.standard_scaler.inverse_transform(torch.cat(y_preds, dim=1)).detach().cpu()  # (T, B, N, F)
        labels = self.standard_scaler.inverse_transform(torch.cat(labels, dim=1)).detach().cpu()

        loss_dict = {}
        for t in range(12):
            mae, rmse, mape = All_Metrics(y_preds[t, ...], labels[t, ...], 0.1, 0.1)
            loss_dict[f'hor {t + 1}'] = (mae.item(), mape.item(), rmse.item())
            self.logger.info(f"Horizon {t + 1}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")

        for t in [3, 6, 9, 12]:
            mae, rmse, mape = All_Metrics(y_preds[:t, ...], labels[:t, ...], 0.1, 0.1)
            self.logger.info(f"Average Horizon {t}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")
            loss_dict[f'Avg hor {t}'] = (mae.item(), mape.item(), rmse.item())

        return {
            'y_pred': y_preds,
            'label': labels,
            'loss': loss_dict
        }

    def test_and_log_hparms(self, **kwargs):
        loss_dict = self.test()['loss']

        save_dict = {
            'params': {
                'base_lr': self._train_kwargs['base_lr'],
                'cl_step': self._train_kwargs['cl_step'],
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers
            },
            'rets': loss_dict
        }

        if self._save_and_log:
            with open(os.path.join(self.log_dir, 'test_record.txt'), 'w') as f:
                json.dump(save_dict, f, indent=2)
