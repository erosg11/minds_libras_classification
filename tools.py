import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from os import getenv
import numpy as np

load_dotenv()

BASE_PATH = Path(getenv('VIDEO_ROOT'))
OUT_PATH = Path(getenv('OUT_FOLDER'))

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

USE_POINTS = np.array([int(x) for x in getenv('USE_POINTS').split(',')]) if getenv('USE_POINTS') else np.arange(33)