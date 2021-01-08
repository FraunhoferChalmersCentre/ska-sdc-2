import yaml
import os
import socket

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(ROOT_DIR + '/config.yaml') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

if socket.gethostname().split('.')[0] == 'nuthatch':
    config['path']['data'] = '/scratch/ska/data/'
else:
    if config['path']['data'][0] == '.':
        config['path']['data'] = ROOT_DIR + config['path']['data'][1:]

    config['path']['data'] = os.path.realpath(config['path']['data'])
