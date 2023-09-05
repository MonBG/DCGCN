import os

base_dir = 'D:/Code_Files/Pycharm/Traffic_CASE'


def get_model_config_path(model_name, model_dir_name='model_logs', dataset='la'):
    return os.path.join(base_dir, f'data/{model_dir_name}/',
                        model_name, f'{model_name}_{dataset}.yaml')
