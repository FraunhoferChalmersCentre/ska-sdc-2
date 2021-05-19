import os
import argparse

import wget
from utils import filename
from definitions import config, ROOT_DIR

types = ['dev_s', 'dev_l', 'eval']
parser = argparse.ArgumentParser()
parser.add_argument('--type', metavar='T', nargs='*', default=types,  help='types to download (dev_s/dev_l/eval)')

args = parser.parse_args()
os.chdir(ROOT_DIR)
if not os.path.exists(config['path']['data']):
    os.makedirs(config['path']['data'])

truth_cat_urls = {'dev_s': 'https://owncloud.ia2.inaf.it/index.php/s/fUXSO5MdK2QT9eh/download?path=%2F&files=sky_dev_truthcat_v2.txt',
                  'dev_l': 'https://owncloud.ia2.inaf.it/index.php/s/hinxyghvmqdP4kh/download?path=%2F&files=sky_ldev_truthcat_v2.txt',
                  'eval': ''}

sky_urls = {'dev_s': 'https://owncloud.ia2.inaf.it/index.php/s/fUXSO5MdK2QT9eh/download?path=%2F&files=sky_dev_v2.fits',
            'dev_l': 'https://owncloud.ia2.inaf.it/index.php/s/hinxyghvmqdP4kh/download?path=%2F&files=sky_ldev_v2.fits',
            'eval': 'https://owncloud.ia2.inaf.it/index.php/s/IEC7eOO4Qdaoi2L/download?path=%2F&files=sky_eval.fits'}

cont_urls = {'dev_s': 'https://owncloud.ia2.inaf.it/index.php/s/fUXSO5MdK2QT9eh/download?path=%2F&files=cont_dev.fits',
             'dev_l': 'https://owncloud.ia2.inaf.it/index.php/s/hinxyghvmqdP4kh/download?path=%2F&files=cont_ldev.fits',
             'eval': 'https://owncloud.ia2.inaf.it/index.php/s/IEC7eOO4Qdaoi2L/download?path=%2F&files=cont_eval.fits'}

readme_urls = {'dev_s': 'https://owncloud.ia2.inaf.it/index.php/s/fUXSO5MdK2QT9eh/download?path=%2F&files=README_dev.txt',
               'dev_l': 'https://owncloud.ia2.inaf.it/index.php/s/hinxyghvmqdP4kh/download?path=%2F&files=README_ldev.txt',
               'eval': 'https://owncloud.ia2.inaf.it/index.php/s/IEC7eOO4Qdaoi2L/download?path=%2F&files=README_eval.txt'}

for t in args.type:
    if t != 'eval':
        print('Downloading truth catalogue for {}'.format(t))
        wget.download(truth_cat_urls[t], out=filename.data.true(t))

    print('Downloading sky fits for {}'.format(t))
    wget.download(sky_urls[t], out=filename.data.sky(t))

    print('Downloading radio continuum counterpart fits for {}'.format(t))
    wget.download(cont_urls[t], out=filename.data.cont(t))

    print('Downloading README file for {}'.format(t))
    wget.download(readme_urls[t], out=filename.data.readme(t))

print('Finished downloading')
