import pandas as pd
from astropy.io.fits import getheader

from pipeline.data.segmentmap import prepare_df
from definitions import config
from pipeline.segmentation.utils import generate_validation_segmentmap, generate_validation_input_cube
from pipeline.common import filename

val_dataset_path = filename.processed.validation_dataset(config['segmentation']['size'],
                                                         100 * config['segmentation']['validation']['reduction'])

fits_file = f'{val_dataset_path}/input_cube.fits'

df = pd.read_csv(filename.data.true(config['segmentation']['size']), sep=' ', index_col='id')

generate_validation_input_cube(val_dataset_path)

header = getheader(fits_file)

segmentmap = generate_validation_segmentmap(val_dataset_path, header, df.copy())

df_true = prepare_df(df, header)
df_true.to_csv(f'{val_dataset_path}/df.txt', sep=' ', index_label='id')
