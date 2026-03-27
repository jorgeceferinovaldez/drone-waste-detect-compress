import yaml
from pyprojroot import here
from pathlib import Path
from typing import Union, Callable, Iterable

import os
print("Current working directory:", os.getcwd())


# Cargar configuración desde config.yaml
def load_config(config_path: str = "config.yaml") -> dict:
    abs_config_path = os.path.join(os.path.dirname(__file__), config_path)
    print(f"Loading configuration from {abs_config_path}")
    with open(abs_config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Generar función de directorio
def make_dir_function(dir_name: Union[str, Iterable[str]]) -> Callable[..., Path]:
    def dir_path(*args) -> Path:
        if isinstance(dir_name, str):
            return here().joinpath(dir_name, *args)
        else:
            return here().joinpath(*dir_name, *args)
    return dir_path

# Cargar configuración
config = load_config()

# Definir funciones de directorios basadas en la configuración
project_dir = make_dir_function("")

data_dir = make_dir_function("data")
raw_data_dir = make_dir_function(config['data']['raw_data_dir'])
interim_data_dir = make_dir_function(config['data']['interim_data_dir'])
processed_data_dir = make_dir_function(config['data']['processed_data_dir'])
external_data_dir = make_dir_function(config['data']['external_data_dir'])

models_dir = make_dir_function(config['models']['trained_models_dir'])
pretrained_dir = make_dir_function(config['models']['pretrained_dir'])
predictions_dir = make_dir_function(config['models']['predictions_dir'])
summaries_dir = make_dir_function(config['models']['summaries_dir'])

notebooks_dir = make_dir_function(config['notebooks_dir'])
references_dir = make_dir_function(config['references_dir'])
reports_dir = make_dir_function(config['reports_dir'])
figures_dir = make_dir_function(config['figures_dir'])

docs_dir = make_dir_function(config['docs_dir'])
requirements_file = make_dir_function(config['requirements_file'])
setup_file = make_dir_function(config['setup_file'])
tox_file = make_dir_function(config['tox_file'])

# Ejemplo de uso
if __name__ == "__main__":
    print(f"Raw data directory: {raw_data_dir()}")
    print(f"Processed data directory: {processed_data_dir()}")
    print(f"Trained models directory: {models_dir()}")
    print(f"Reports directory: {reports_dir()}")
    print(f"Figures directory: {figures_dir()}")
    print(f"Documentation directory: {docs_dir()}")
    print(f"Requirements file: {requirements_file()}")
    print(f"Setup file: {setup_file()}")
    print(f"Tox file: {tox_file()}")
