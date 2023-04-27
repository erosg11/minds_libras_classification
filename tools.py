import pandas as pd
from pathlib import Path

BASE_PATH = Path('/media/eros/BackupMae/datasets/Minds')
OUT_PATH = Path('./Outs')

def open_meta_df():
    return pd.read_csv(OUT_PATH / 'dataset_metadata.csv.gz', 
                      dtype={
                          'filename': 'string',
                          'stem': 'string',
                          'pose_id': 'int64',
                          'word': 'category',
                          'sinalizador': 'int64',
                          'repetition': 'int64',
                      }, index_col=0)