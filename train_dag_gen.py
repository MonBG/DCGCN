import yaml

from nns.case.supervisor_dag_gen import CASEDagGenSupervisor
from utils.path_utils import get_model_config_path, base_dir


with open(get_model_config_path('dag_gen', model_dir_name='models')) as f:
    supervisor_config = yaml.safe_load(f)


def model_training():
    supervisor_config['base_dir'] = base_dir
    supervisor = CASEDagGenSupervisor(**supervisor_config)
    supervisor.train_dag_gen()
    # supervisor.plot_example_test_graph(sample_id=26)
    print("Train finished")


if __name__ == "__main__":
    model_training()
