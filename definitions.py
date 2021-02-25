import yaml
import os
import socket
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(ROOT_DIR + '/config.yaml') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

for k, v in config.get('path').items():
    if socket.gethostname().split('.')[0] == 'nuthatch':
        edited_value = v.replace('.', '/scratch/ska/data/')
    else:
        edited_value = v.replace('.', ROOT_DIR)
        edited_value = edited_value.replace('~', os.path.expanduser("~"))

    edited_value = os.path.realpath(edited_value)
    config['path'][k] = edited_value


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)