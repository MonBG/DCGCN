import yaml

from nns.case.supervisor_pred import CASECausalPred
from utils.path_utils import get_model_config_path, base_dir


with open(get_model_config_path('pred', model_dir_name='models')) as f:
    supervisor_config = yaml.safe_load(f)


def model_training():
    supervisor_config['base_dir'] = base_dir
    supervisor = CASECausalPred(**supervisor_config)
    supervisor.train()
    supervisor.test_and_log_hparms()
    print("Train finished")


if __name__ == "__main__":
    model_training()
