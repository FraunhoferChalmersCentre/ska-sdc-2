import os
from definitions import config


def prepare_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


class DirectoryFileName:
    def __init__(self, directory):
        self.directory = directory
        prepare_dir(directory)

    def get_path(self, filename):
        return os.path.join(self.directory, filename)

    def _eval_dev_name(self, types, eval_name, dev_name):
        if types == 'eval':
            return self.get_path(eval_name)
        elif types[:3] == 'dev':
            return self.get_path(dev_name.format(types[-1].replace('s', '')))
        raise ValueError('Unknown type')


class ProcessedFileName(DirectoryFileName):
    def __init__(self):
        super().__init__(config['path']['processed'])

    def segmentmap(self, types):
        return self._eval_dev_name(types, 'segmap_eval.npz', 'segmap_{}dev.npz')

    def allocation_dict(self, types):
        return self._eval_dev_name(types, 'segmap_eval.npz', 'allocation_{}dev.pb')


class DataFileName(DirectoryFileName):
    def __init__(self):
        super().__init__(config['path']['data'])

    def true(self, types):
        return self.get_path('sky_dev_{}truthcat_v1.1.txt'.format(types[-1].replace('s', '')))

    def sky(self, types):
        return self._eval_dev_name(types, 'sky_eval.fits', 'sky_{}dev.fits')

    def cont(self, types):
        return self._eval_dev_name(types, 'cont_eval.fits', 'cont_{}dev.fits')

    def readme(self, types):
        return self._eval_dev_name(types, 'README_eval.txt', 'README_{}dev.txt')
    
    def transformed(self, types, transform):
        return self._eval_dev_name(types, 'sky_eval.fits', 'sky_{}dev_' + '{}_transform.fits'.format(transform))


processed = ProcessedFileName()
data = DataFileName()
