import os
import torch

from utils.metrics_lib import All_Metrics
from utils.data_utils import load_graph
from utils.plot_lib import display_seq_pred, align_seq2seq_output
from utils.train_inits import init_seed
from dcrnn.DCRNNModel import DCRNNModel
from dcrnn.lib.graph_laplacian import calculate_scaled_laplacian
from nns.dbgcn.dynamic_dbgcn import DynamicDBGCN
from nns.supervisor.loss import masked_mae_loss
from nns.dbgcn.dynamic_dbgcn_trainer import DynamicDBGCNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pre_trained_dcrnn(log_dir):
    saved_state = torch.load(os.path.join(log_dir, 'best_model.pth'))
    args = saved_state['config']

    adj_mat = load_graph(os.path.join(args['base_dir'], args['graph_pkl_filename']), args['num_nodes'])
    supports = [calculate_scaled_laplacian(adj_mat, lambda_max=None)]
    supports = [torch.tensor(i).to(args['device']) for i in supports]

    model = DCRNNModel(supports, num_node=args['num_nodes'], input_dim=args['input_dim'],
                       hidden_dim=args['rnn_units'], out_dim=args['output_dim'],
                       order=args['diffusion_step'], num_layers=args['num_rnn_layers'])
    model.to(args['device'])
    model.load_state_dict(saved_state['state_dict'])
    return model


def test_seq2seq_model_t_in(model, data_loader, scaler, t_in=12, logger=None,
                            plot_seq=False, fig_name='', plot_path=None, mask_value=0.):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            label = target[..., :model.output_dim]
            data = data[:, -t_in:, ...]
            output = model(data, target, teacher_forcing_ratio=0)
            y_true.append(label)
            y_pred.append(output)
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
    for t in range(y_true.shape[1]):
        mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], mask_value, mask_value)
        message = "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100)
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    mae, rmse, mape = All_Metrics(y_pred, y_true, mask_value, mask_value)
    message = "Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100)
    if logger is not None:
        logger.info(message)
    else:
        print(message)

    if plot_seq:
        # horizon 1
        t = 0
        mae, rmse, mape = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], 0.1, 0.1)
        display_seq_pred(y_pred[:, t, ...], y_true[:, t, ...],
                         title=f"mae {mae:.3f}, mape {mape * 100: .3f}%, rmse {rmse: .3f}, ",
                         figname=fig_name, save_path=plot_path)

    return y_true, y_pred


def test_tv_graph_generator_with_dbgcn(data, graph_generator, args, trainer_save_dir,
                                       seed=1, pre_trained=False, mode='train',
                                       plot_seq=False, fig_name='', plot_path=None):
    init_seed(seed)
    net = DynamicDBGCN(args['num_nodes'], args['input_dim'], args['output_dim'],
                       args['in_seq_len'], args['out_seq_len']).to(device)
    loss_criterion = masked_mae_loss(data['scaler'], 0.1)
    trainer = DynamicDBGCNTrainer(net, graph_generator, loss_criterion,
                                  data['train_loader'], data['val_loader'], data['scaler'],
                                  args, save_dir=trainer_save_dir)
    if pre_trained:
        trainer.load_checkpoint(trainer.best_path)
    if mode == 'train':
        trainer.train()

    y_true, y_pred = trainer.test(trainer.model, trainer.graph_generator, data[f'test_loader'],
                                  data['scaler'], normed=args['normed'])
    aligned_y_ture = align_seq2seq_output(y_true, args['data_format'])  # (seq_len, num_nodes, dim)
    aligned_y_pred = align_seq2seq_output(y_pred, args['data_format'])
    mae, rmse, mape = All_Metrics(aligned_y_pred, aligned_y_ture, 0.1, 0.1)
    print(f"Horizon 1 (Align to Seq2Seq), MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape * 100:.4f}%")
    if plot_seq:
        display_seq_pred(aligned_y_pred, aligned_y_ture,
                         title=f"mae {mae:.2f}, mape {mape * 100: .4f}%, rmse {rmse: .2f}, ",
                         figname=fig_name, save_path=plot_path)

    return mae, mape, rmse
