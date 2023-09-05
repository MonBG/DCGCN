import time
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from nns.case.dag_gen_gru import DagGenGRUg4s2v1
from nns.supervisor.loss import graph_sparsity_loss, kl_div_gauss, acyclic_loss, graph_segment_change_loss
from nns.supervisor.loss import masked_mae_loss, masked_mse_loss
from nns.post_processing_blocks import constraint2
from nns.supervisor.base_supervisor import BaseSupervisor
from nns.config import device

from utils.graph_utils import plot_multi_slice_graphs
from utils.path_utils import base_dir
from utils.data_utils import load_graph
from utils.plot_lib import heatmap


marker_styles = ["v", "^", "<", ">", "1", "2", "3", "8", "s", "p", "P", "*", "h", "+", "x",
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


class CASEDagGenSupervisor(BaseSupervisor):
    def __init__(self, **kwargs):
        super(CASEDagGenSupervisor, self).__init__(**kwargs)

        self.reg_sp = self._loss_kwargs.get('reg_sp', 2e-3)
        self.reg_sg = self._loss_kwargs.get('reg_sg', 2e-3)
        self.masked_mae_loss = masked_mae_loss(self.standard_scaler, 0.1)
        self.masked_mse_loss = masked_mse_loss(self.standard_scaler, 0.1)

    def load_adj_mx(self):
        adj_mat = load_graph(os.path.join(base_dir, "data/sensor_graph/adj_mx.pkl"))
        if adj_mat.shape[0] > self.num_nodes:
            adj_mat = adj_mat[:self.num_nodes, :self.num_nodes]
        return adj_mat

    def _get_model(self):
        self.model_version = 'g4s2v1'

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.in_feats_dim = int(self._model_kwargs.get('in_feats_dim', 1))
        self.out_feats_dim = int(self._model_kwargs.get('out_feats_dim', 1))
        self.hidden_dim = int(self._model_kwargs.get('hidden_dim', 512))
        self.z_dim = int(self._model_kwargs.get('z_dim', 8))
        self.num_layers = int(self._model_kwargs.get('num_layers', 6))
        self.num_heads = int(self._model_kwargs.get('num_heads', 8))
        self.feats_layers = int(self._model_kwargs.get('feats_layers', 3))
        self.agg_feats = self._model_kwargs.get('agg_feats', 'ori')
        self.node_norm = self._model_kwargs.get('node_norm', False)
        self.use_norm = self._model_kwargs.get('use_norm', False)
        self.use_pp = self._model_kwargs.get('use_pp', False)
        self.step_pri = self._model_kwargs.get('step_pri', 0.01)
        self.step_dual = self._model_kwargs.get('step_dual', 0.01)
        self.reg_sp_intra = self._model_kwargs.get('reg_sp_intra', 2e-3)
        self.num_intra_pp_iters = self._model_kwargs.get('num_intra_pp_iters', 1000)

        self.dist_adj = self.load_adj_mx()

        self.logger.info(f"DagGen Model Version {self.model_version}")
        dag_gen_model = globals()['DagGenGRU' + self.model_version]

        model = dag_gen_model(self.num_nodes, self.in_feats_dim, self.out_feats_dim,
                              hidden_dim=self.hidden_dim, num_layers=self.num_layers, dist_adj=self.dist_adj,
                              feats_layers=self.feats_layers, agg_feats=self.agg_feats, node_norm=self.node_norm,
                              use_norm=self.use_norm, use_pp=self.use_pp, step_pri=self.step_pri,
                              step_dual=self.step_dual, reg_sp_intra=self.reg_sp_intra,
                              num_intra_pp_iters=self.num_intra_pp_iters,
                              num_heads=self.num_heads, z_dim=self.z_dim)
        return model

    def _gen_run_id(self):
        seed = self._train_kwargs.get('seed', 1)
        batch_size = self._data_kwargs.get('batch_size')

        special_id = self._model_kwargs.get('special_id')
        model_version = 'g4s2v1'
        seq_len = int(self._model_kwargs.get('seq_len', 12))

        num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        in_feats_dim = int(self._model_kwargs.get('in_feats_dim', 1))
        out_feats_dim = int(self._model_kwargs.get('out_feats_dim', 1))
        hidden_dim = int(self._model_kwargs.get('hidden_dim', 512))
        z_dim = int(self._model_kwargs.get('z_dim', 512))
        num_layers = int(self._model_kwargs.get('num_layers', 6))
        num_heads = int(self._model_kwargs.get('num_heads', 8))
        feats_layers = int(self._model_kwargs.get('feats_layers', 3))
        agg_feats = self._model_kwargs.get('agg_feats', 'ori')
        node_norm = 1 if self._model_kwargs.get('node_norm', False) else 0
        use_norm = 1 if self._model_kwargs.get('use_norm', False) else 0
        use_pp = 1 if self._model_kwargs.get('use_pp', False) else 0
        step_pri = self._model_kwargs.get('step_pri', 0.01)
        step_dual = self._model_kwargs.get('step_dual', 0.01)
        reg_sp_intra = self._model_kwargs.get('reg_sp_intra', 2e-3)
        num_intra_pp_iters = self._model_kwargs.get('num_intra_pp_iters', 1000)

        reg_sp = self._loss_kwargs.get('reg_sp', 0)
        reg_sg = self._loss_kwargs.get('reg_sg', 0)
        learning_rate = self._train_kwargs.get('base_lr')
        run_time = time.strftime('%m%d%H%M%S')

        run_id = f'{run_time}_case_dag_gen_{special_id}_{model_version}_seed_{seed}_bs_{batch_size}' \
                 f'_N_{num_nodes}_seq_{seq_len}_in_{in_feats_dim}_out_{out_feats_dim}_h_{hidden_dim}_z_{z_dim}' \
                 f'_nl_{num_layers}_nh_{num_heads}_f_{agg_feats}_fl_{feats_layers}' \
                 f'_nn_{node_norm}_un_{use_norm}_upp_{use_pp}' \
                 f'_sp0_{reg_sp_intra}_npp0_{num_intra_pp_iters}' \
                 f'_lr_{learning_rate}_sp_{reg_sp}_sg_{reg_sg}'
        return run_id

    def loss(self, x, graphs, x_reconst, max_loop=30):
        """
        :param x: (T, B, N, D)
        :param graphs: (T, B, m, N, N)
        :param x_reconst: (T, B, N, D)
        :return:
        """
        intra_graph = graphs[:, :, -1, :, :]

        acyclic = constraint2(intra_graph, max_loop).abs().mean()
        sp_loss = graph_sparsity_loss(graphs)
        sg_loss = graph_segment_change_loss(graphs)

        if 's3' in self.model_version or 's4' in self.model_version:
            dec_mean_seq, prior_mean_seq, prior_log_var_seq, enc_mean_seq, enc_log_var_seq = x_reconst
            kl_div = kl_div_gauss(enc_mean_seq, enc_log_var_seq, prior_mean_seq, prior_log_var_seq, reduction='sum')
            nll_loss = self.masked_mse_loss(dec_mean_seq, x[..., :self.out_feats_dim])
            reconst_loss = kl_div + nll_loss
        else:
            reconst_loss = self.masked_mae_loss(x_reconst, x[..., :self.out_feats_dim])

        total_loss = reconst_loss + self.reg_sp * sp_loss + self.reg_sg * sg_loss

        if not self.use_pp:
            augmented_acyclic = self.rho * acyclic.square() / 2 + self.alpha * acyclic
            total_loss += augmented_acyclic

        if torch.isnan(total_loss).item():
            self.logger.info('nan occur in loss computation')
            assert not torch.isnan(reconst_loss).item(), 'nan occur in reconst_loss'

        return total_loss, {
            "reconst_loss": reconst_loss,
            "sp": sp_loss,
            "acyclic": acyclic,
            "sg": sg_loss
        }

    @staticmethod
    def _get_x_y_correct_dim(x, y):
        """
        :param x: shape (batch_size, seq_len + 1, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len + 1, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
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
        detail_losses = {
            "reconst_loss": 0,
            "sp": 0,
            "sg": 0,
            "acyclic": 0
        }
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                             out_feats_dim=self.out_feats_dim)
            self.optimizer.zero_grad()

            if self._train_kwargs.get('debug', False):
                torch.autograd.set_detect_anomaly(True)

            graphs, x_reconst = self.model(data)
            loss, details = self.loss(data[1:, ...], graphs, x_reconst, max_loop=self._compute_max_loop(epoch))
            assert torch.isnan(loss).sum() == 0, print(loss)

            if self._train_kwargs.get('debug', False):
                with torch.autograd.detect_anomaly():
                    loss.backward()
            else:
                loss.backward()

            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._train_kwargs['max_grad_norm'])
            for name, param in self.model.named_parameters():
                assert torch.isnan(param).sum() == 0, print(name)
            self.optimizer.step()
            for name, param in self.model.named_parameters():
                assert torch.isnan(param).sum() == 0, print(name)
                if param.grad is not None:
                    assert torch.isnan(param.grad).sum() == 0, print(name)

            total_loss += loss.item()
            for key in detail_losses.keys():
                detail_losses[key] += details[key].item()
            # log information
            if batch_idx % self._train_kwargs['log_step'] == 0:
                self.logger.info(f'Train Epoch {epoch}: {batch_idx}/{self.train_per_epoch} Loss: {loss.item():.3f}')
        train_epoch_loss = total_loss / self.train_per_epoch
        train_ac_loss = detail_losses["acyclic"] / self.train_per_epoch
        train_sp_loss = detail_losses["sp"] / self.train_per_epoch
        train_sg_loss = detail_losses["sg"] / self.train_per_epoch
        train_reconst_loss = detail_losses["reconst_loss"] / self.train_per_epoch
        self.logger.info(f'**********Train Epoch {epoch}: averaged Loss: {train_epoch_loss:.3f}, '
                         f'max_loop {self._compute_max_loop(epoch)}')
        self.logger.info(f'**********Train Epoch {epoch}: detailed Loss: '
                         f'reconst {train_reconst_loss:.3f}, sp {train_sp_loss: .3f}, '
                         f'sg {train_sg_loss: .3f}, ac {train_ac_loss: .3f}')
        return train_epoch_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = self.prepare_data(data, target, num_nodes=self.num_nodes,
                                                 out_feats_dim=self.out_feats_dim)
                graphs, x_reconst = self.model(data)
                loss, _ = self.loss(data[1:, ...], graphs, x_reconst, self.num_nodes)
                total_val_loss += loss.item()
        val_loss = total_val_loss / self.val_per_epoch
        self.logger.info(f'**********Val Epoch {epoch}: average Loss: {val_loss:.6f}')
        return val_loss

    def plot_example_test_graph(self, sample_id=0):
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, _ = self.prepare_data(data, target, num_nodes=self.num_nodes, out_feats_dim=self.out_feats_dim)
                graphs, _ = self.model(data, gen_graph_only=True)  # (T, B, 2, N, N)
                save_dir = os.path.join(self.log_dir, f'eg_n{sample_id}')
                sp_loss = graph_sparsity_loss(graphs)
                sg_loss = graph_segment_change_loss(graphs)
                ac_loss = acyclic_loss(graphs[:, :, -1, :, :], approx=False)
                ac_loss_2 = constraint2(graphs[:, :, -1, :, :], max_loop=self.num_nodes).mean()
                self.logger.info(f"Example_graphs: exact ac_loss {ac_loss: .4f}, "
                                 f"approx ac_loss {ac_loss_2:.4f}, sp_loss {sp_loss: .4f}, sg_loss {sg_loss: .4f}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plot_multi_slice_graphs(graphs[:, sample_id, ...], plot_dir=save_dir)

                # heatmap of traffic speed
                # plot_data = self.standard_scaler.inverse_transform(data[:, sample_id, :, 0]).cpu().numpy()
                # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                # heatmap(plot_data, ax=ax, row_labels=[str(i) for i in range(plot_data.shape[0])],
                #         col_labels=[str(i) for i in range(plot_data.shape[1])], cmap='RdYlGn',
                #         cbar_kw={'fraction': 0.06}, show_grid=False)
                # plt.savefig(os.path.join(save_dir, 'heat_plot.png'))
                #
                # # time series of traffic data
                # fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                # for i in range(self.num_nodes):
                #     ax.plot(np.arange(0, plot_data.shape[0]),
                #             plot_data[:, i], label=f'N{i}', marker=marker_styles[i])
                # ax.legend()
                # plt.savefig(os.path.join(save_dir, 'time_series.png'))
                break

    def eval_train_ac_loss(self):
        ac_total_loss = 0.
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, _ = self.prepare_data(data, target, num_nodes=self.num_nodes, out_feats_dim=self.out_feats_dim)
                graphs, _ = self.model(data)
                ac_loss = constraint2(graphs[:, :, -1, :, :], max_loop=self.num_nodes).abs().mean()
                ac_total_loss += ac_loss.item()
        ac_avg_loss = ac_total_loss / self.train_per_epoch
        return ac_avg_loss

    def reset_optimizer_and_scheduler_with_constant_lr(self, base_lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=base_lr,
                                          weight_decay=self._train_kwargs.get('weight_decay', 0),
                                          eps=self._train_kwargs.get('epsilon', 1e-5))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=[100])

    def train_lagrangian(self):
        # initialization of Lagrangian multipliers
        self.rho = np.float32(1e-3)
        self.alpha = np.float32(0)
        self.eta = self._train_kwargs.get('lagrangian_eta', 10)
        self.gamma = self._train_kwargs.get('lagrangian_gamma', 0.5)
        self.stop_criterion = self._train_kwargs.get('lagrangian_eps', 1e-6)
        self.train_ac_avg_loss = float('inf')
        base_lr = self._train_kwargs.get('base_lr', 0.1)
        milestones = self._train_kwargs.get('lagrangian_milestones', [3, 6, 10, 15])

        # Method of multipliers
        for outer_epoch in range(1, 100):
            # optimize over sub-problem
            self._kwargs['epoch'] = 0  # reset sub-problem epoch
            self.logger.info(f"----------Lagrangian Method Epoch {outer_epoch}: sub-problem training")
            if outer_epoch in milestones:
                base_lr *= self._train_kwargs.get('lr_decay_ratio', 0.1)
            self.reset_optimizer_and_scheduler_with_constant_lr(base_lr)
            self.train()
            if outer_epoch == 1:
                # change early stopping criterion to encourage fast convergence
                self._train_kwargs['min_epochs'] = 0
                self._train_kwargs['early_stop'] = self._train_kwargs.get('lagrangian_early_stop', 5)
                self._train_kwargs['epochs'] = self._train_kwargs.get('lagrangian_epochs', 50)

            # stopping criterion
            ac_avg_loss = self.eval_train_ac_loss()
            if ac_avg_loss < self.stop_criterion:
                self.logger.info(f"----------Lagrangian Method Converged at Epoch {outer_epoch} "
                                 f"with ac_loss {ac_avg_loss: .5f}")
                break

            # update Lagrangian multipliers
            self.alpha = self.alpha + self.rho * ac_avg_loss
            if outer_epoch > 0:
                self.rho = self.eta * self.rho if ac_avg_loss > self.gamma * self.train_ac_avg_loss else self.rho
            self.train_ac_avg_loss = ac_avg_loss
            self.logger.info(f"----------Lagrangian Method Epoch {outer_epoch}: "
                             f"primal residual {ac_avg_loss:.4f}, update rho {self.rho:.3f}, alpha {self.alpha: .3f}")
            if self.rho > 5*1e5:
                self.logger.info(f"----------Lagrangian Method Stop due to large rho")
                break

    def train_dag_gen(self):
        if self.use_pp:
            self.logger.info("Train with PP")
            self.train()
        else:
            self.logger.info("Train with Dual Lagrangian")
            self.train_lagrangian()

    def _compute_max_loop(self, epoch):
        if self.rho >= 1e-2 or self.num_nodes <= 50:
            return self.num_nodes
        else:
            if epoch <= 3:
                return min(30, self.num_nodes)
            elif epoch <= 6:
                return min(60, self.num_nodes)
            elif epoch <= 10:
                return min(100, self.num_nodes)
            else:
                return self.num_nodes
