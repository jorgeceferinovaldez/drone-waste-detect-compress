import os
import yaml
from pathlib import Path
from typing import Union, Callable, Iterable

from pyprojroot import here


# ---------------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    abs_config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(abs_config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Fábrica de rutas
# ---------------------------------------------------------------------------

def make_dir_function(dir_name: Union[str, Iterable[str]]) -> Callable[..., Path]:
    def dir_path(*args) -> Path:
        if isinstance(dir_name, str):
            return here().joinpath(dir_name, *args)
        else:
            return here().joinpath(*dir_name, *args)
    return dir_path


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

config = load_config()

# Raíz del proyecto
project_dir = make_dir_function("")

# Datos
data_dir            = make_dir_function("data")
raw_data_dir        = make_dir_function(config["data"]["raw_data_dir"])
interim_data_dir    = make_dir_function(config["data"]["interim_data_dir"])
processed_data_dir  = make_dir_function(config["data"]["processed_data_dir"])
external_data_dir   = make_dir_function(config["data"]["external_data_dir"])

# Modelos
models_dir    = make_dir_function(config["models"]["trained_models_dir"])
pretrained_dir = make_dir_function(config["models"]["pretrained_dir"])

# Pesos cheng2020-anchor por calidad (q=1..6)
cheng2020_model_paths = {
    q: make_dir_function(config["models"]["cheng2020"][f"q{q}"])
    for q in config["compressai"]["qualities"]
}

# Pesos YOLO por variante
yolo_model_paths = {
    variant: make_dir_function(path)
    for variant, path in config["models"]["yolo"].items()
}

# Logs TensorBoard — cheng2020 por calidad
cheng2020_log_dirs = {
    q: make_dir_function(config["logs"]["cheng2020"][f"q{q}"])
    for q in config["compressai"]["qualities"]
}

# Logs TensorBoard — YOLO por variante
yolo_log_dirs = {
    variant: make_dir_function(log_dir)
    for variant, log_dir in config["logs"]["yolo"].items()
}

# Informes
notebooks_dir  = make_dir_function(config["notebooks_dir"])
references_dir = make_dir_function(config["references_dir"])
reports_dir    = make_dir_function(config["reports_dir"])
figures_dir    = make_dir_function(config["figures_dir"])
docs_dir       = make_dir_function(config["docs_dir"])

# Archivos de proyecto
requirements_file = make_dir_function(config["requirements_file"])
setup_file        = make_dir_function(config["setup_file"])
tox_file          = make_dir_function(config["tox_file"])

# ---------------------------------------------------------------------------
# Accesos directos a parámetros de entrenamiento
# ---------------------------------------------------------------------------

compressai_params  = config["compressai"]
yolo_params        = config["yolo"]
throughput_params  = config["throughput"]


# ---------------------------------------------------------------------------
# Utilidad: devuelve el log dir de cheng2020 para una calidad dada
# ---------------------------------------------------------------------------

def get_cheng2020_log_dir(quality: int) -> Path:
    if quality not in config["compressai"]["qualities"]:
        raise ValueError(f"Calidad {quality} no válida. Opciones: {config['compressai']['qualities']}")
    return cheng2020_log_dirs[quality]()


def get_yolo_log_dir(variant: str) -> Path:
    if variant not in yolo_log_dirs:
        raise ValueError(f"Variante '{variant}' no válida. Opciones: {list(yolo_log_dirs.keys())}")
    return yolo_log_dirs[variant]()


# ---------------------------------------------------------------------------
# Verificación rápida al ejecutar directamente
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Proyecto:          {project_dir()}")
    print(f"Datos raw:         {raw_data_dir()}")
    print(f"Datos procesados:  {processed_data_dir()}")
    print(f"Modelos:           {models_dir()}")
    print(f"Pretrained:        {pretrained_dir()}")
    print()
    for q in config["compressai"]["qualities"]:
        print(f"  cheng2020 q={q}: {cheng2020_model_paths[q]()}")
    print()
    for variant in config["yolo"]["variants"]:
        print(f"  YOLO {variant}: {yolo_model_paths[variant]()}")
    print()
    for q in config["compressai"]["qualities"]:
        print(f"  log cheng2020 q={q}: {get_cheng2020_log_dir(q)}")
    for variant in config["yolo"]["variants"]:
        print(f"  log YOLO {variant}: {get_yolo_log_dir(variant)}")
    print()
    print(f"Throughput output: {reports_dir(config['throughput']['output_csv'])}")
