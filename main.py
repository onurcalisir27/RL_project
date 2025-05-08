import numpy as np
import argparse
import hydra
from omegaconf import DictConfig
from train import Train
from eval import Evaluate

def omegaconf_to_dict(d: DictConfig) -> dict:
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret

@hydra.main(version_base="1.2", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):
    config = omegaconf_to_dict(cfg)

    task = config['task']['name']
    if task == 'PickPlace':
        object = config['object']
        if object not in ['bread', 'can', 'milk']:
            raise Exception('Unknown object type. \n Available objects: bread, can, milk')

    if config['train']:
        Train(config)

    if config['test']:
        Evaluate(config)

if __name__ == '__main__':
    main()