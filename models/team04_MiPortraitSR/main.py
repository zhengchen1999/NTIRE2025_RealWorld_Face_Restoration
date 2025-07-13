from argparse import ArgumentParser, Namespace
import yaml
import torch
from .diffbir.inference import BFRInferenceLoop
from accelerate.utils import set_seed
import os

class ArgsFromYaml:
    def __init__(self, cfg):
        for key, value in cfg.items():
            if isinstance(value, dict):
                setattr(self, key, ArgsFromYaml(value))
            else:
                setattr(self, key, value)

def MPSR(model_dir, input_path, output_path, device):
    current_path = os.path.dirname(os.path.abspath(__file__))

    test_yaml_path = os.path.join(current_path, "configs/inference/test.yml")
    with open(test_yaml_path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['diffbir']['model_dir'] = model_dir
    yaml_data['diffbir']['input'] = input_path
    yaml_data['diffbir']['output'] = output_path
    # yaml_data['diffbir']['device'] = device
    args = ArgsFromYaml(yaml_data)
    set_seed(args.diffbir.seed)
    restorer = BFRInferenceLoop(args)
    restorer.run()

if __name__ == "__main__":
    pass