import yaml
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(ROOT_DIR + '/config.yaml') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

if config['path']['data'][0] == '.':
    config['path']['data'] = ROOT_DIR + config['path']['data'][1:]

config['path']['data'] = os.path.realpath(config['path']['data'])
