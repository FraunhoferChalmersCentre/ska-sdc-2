import os
import argparse

import gdown
from dataprep import filename
from definitions import config, ROOT_DIR

types = ['dev_s', 'dev_l', 'eval']
parser = argparse.ArgumentParser()
parser.add_argument('--type', metavar='T', nargs='*', default=types,  help='types to download (dev_s/dev_l/eval)')

args = parser.parse_args()
os.chdir(ROOT_DIR)
if not os.path.exists(config['path']['data']):
    os.makedirs(config['path']['data'])


truth_cat_urls = {'dev_s': 'https://drive.google.com/uc?id=1acvwFHEtrlFaDXz4xAbEd_TiU7iEClMT',
                  'dev_l': 'https://drive.google.com/uc?id=1Ga0qIP4UXCC-AWPSTdKcxpZNBb-XPTjQ',
                  'eval': ''}

sky_urls = {'dev_s': 'https://drive.google.com/uc?id=1vL3hc4pWpeSAJZKtjYyVCqsMlYb7LcgA',
            'dev_l': 'https://drive.google.com/uc?id=1K3kSJjuAjnNoZ044S1bTsMPiqf-SCUOo',
            'eval': 'https://drive.google.com/uc?id=1aNZMTW1sHISU9DWZ6Dr-APmyeMrMxLpo'}

cont_urls = {'dev_s': 'https://drive.google.com/uc?id=10hkkoAaH-gbDragXp90u_yqW-FPKszf5',
             'dev_l': 'https://drive.google.com/uc?id=1e7dlcEsdPVjt1t_hEHKnWxpc9azMKHaw',
             'eval': 'https://drive.google.com/uc?id=1PJ3QSMZyh_QvHVcJH9o28sg9rdOV7KVc'}

readme_urls = {'dev_s': 'https://drive.google.com/uc?id=11Bu0g99Jf3Qfz_tEQv7D8tahHlKMU9Hh',
               'dev_l': 'https://drive.google.com/uc?id=1UFdRDz1kbS9ukpEV_UMK6nhPN_rMHtlq',
               'eval': 'https://drive.google.com/uc?id=1Bljk1MvA-lZEqlpAQjavuWdoLSQtkLBl'}

for t in args.type:
    if t != 'eval':
        print('Downloading truth catalogue for {}'.format(t))
        gdown.download(truth_cat_urls[t], filename.true_data(t), quiet=True)

    print('Downloading sky fits for {}'.format(t))
    gdown.download(sky_urls[t], filename.sky_data(t), quiet=True)

    print('Downloading radio continuum counterpart fits for {}'.format(t))
    gdown.download(cont_urls[t], filename.cont_data(t), quiet=True)

    print('Downloading README file for {}'.format(t))
    gdown.download(readme_urls[t], filename.readme(t), quiet=True)

print('Finished downloading')
