import os
from definitions import config


def prepare_dir(directory):
    splitted = directory[1:].split('/')
    for i in range(1, len(splitted) + 1):
        d = '/' + '/'.join(splitted[:i])
        if not os.path.exists(d):
            os.mkdir(d)


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


class ModelsFileName(DirectoryFileName):
    def __init__(self):
        super().__init__(config['path']['models'])

    def new_model(self):
        return self.by_id(self.new_id)

    def new_id(self):
        return len([name for name in os.listdir(self.directory) if os.path.isfile(name)])

    def by_id(self, id):
        return self.get_path('{}.pt'.format(id))


class ProcessedFileName(DirectoryFileName):
    def __init__(self):
        super().__init__(config['path']['processed'])

    def segmentmap(self, types):
        return self._eval_dev_name(types, 'segmap_eval.npz', 'segmap_{}dev.npz')

    def allocation_dict(self, types):
        return self._eval_dev_name(types, 'segmap_eval.npz', 'allocation_{}dev.pb')

    def dataset(self, size: str):
        name = 'dataset_{}'.format(size)
        path = self.get_path(name)
        prepare_dir(path)
        return path

    def validation_dataset(self, size: str, reduction):
        name = 'validation_dataset_{}_reduction_{}'.format(size, int(reduction))
        path = self.get_path(name)
        prepare_dir(path)
        return path

    def hyperopt_dataset(self, size: str, reduction, checkpoint: str):
        name = 'hyperopt_dataset_{}_reduction_{}_{}'.format(size, int(reduction), checkpoint)
        path = self.get_path(name)
        prepare_dir(path)
        return path

    def test_dataset(self, checkpoint):
        name = f'test_{checkpoint}'
        path = self.get_path(name)
        prepare_dir(path)
        return path


class DataFileName(DirectoryFileName):
    def __init__(self):
        super().__init__(config['path']['data'])

    def true(self, types):
        return self.get_path('sky_dev_{}truthcat_v2.txt'.format(types[-1].replace('s', '')))

    def test_sky(self):
        return self.get_path('sky_test.fits')

    def test_true(self):
        return self.get_path('sky_full_truthcat_v2.txt')

    def sky(self, types):
        return self._eval_dev_name(types, 'sky_eval.fits', 'sky_{}dev.fits')

    def cont(self, types):
        return self._eval_dev_name(types, 'cont_eval.fits', 'cont_{}dev.fits')

    def readme(self, types):
        return self._eval_dev_name(types, 'README_eval.txt', 'README_{}dev.txt')

    def transformed(self, types, transform):
        return self._eval_dev_name(types, 'sky_eval.fits', 'sky_{}dev_' + '{}_transform.fits'.format(transform))


models = ModelsFileName()
processed = ProcessedFileName()
data = DataFileName()
