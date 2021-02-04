import numpy as np
from utils.data.splitting import split


def test_split():
    n_units = 10

    data = {'unit_id': [], 'a': [], 'b': [], 'c': [], 'd': []}
    for i, unit in enumerate(range(n_units)):
        for j in range(i + 1):
            data['a'].append(np.random.randn(1))
            data['b'].append(np.random.randn(3))
            data['c'].append(np.random.randn(3, 3))
            data['d'].append(np.random.randn(3, 3, 3))
            data['unit_id'].append(i)

    data = {k: np.array(v) for k, v in data.items()}
    left_fraction = .8
    left, right = split(data, left_fraction, 'unit_id')
    assert len(np.unique(left['unit_id'])) == 8
    assert len(np.intersect1d(left['unit_id'], right['unit_id'])) == 0
    assert left['a'].shape[1:] == (1,)
    assert left['b'].shape[1:] == (3,)
    assert left['c'].shape[1:] == (3, 3)
    assert left['d'].shape[1:] == (3, 3, 3)
