# %%
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from os import getenv
from loguru import logger

load_dotenv()

# %%
BASE_PATH = Path(getenv('VIDEO_ROOT'))
OUT_PATH = Path(getenv('OUT_FOLDER'))

logger.info('Incializado com BASE_PATH: {!r} e OUT_PATH: {!r}', BASE_PATH, OUT_PATH)

# %%
videos = list(BASE_PATH.glob('**/*.mp4'))
logger.info('Encontrados {} vídeos', len(videos))

# %%
df = pd.DataFrame([[str(x), x.stem] for x in videos], columns=['filename', 'stem'])
logger.info('Incializado df: {}', df.head())

# %%
df[['pose_id', 'word', 'sinalizador', 'repetition']] = df['stem'].str.extract(r'^(\d+)(\w+)Sinalizador(\d+)-(\d+)$')
logger.info('Adicionadas features ao df: {}', df.head())

# %%
nulls = (pd.isnull(df) | pd.isna(df)).any()
assert not nulls.any()
logger.info('Verificação de integridade de null: {}', nulls)

# %%
df['pose_id'] = pd.to_numeric(df['pose_id'])
df['sinalizador'] = pd.to_numeric(df['sinalizador'])
df['repetition'] = pd.to_numeric(df['repetition'])
df['word'] = df['word'].astype('category')
df['filename'] = df['filename'].astype('string')
df['stem'] = df['stem'].astype('string')
logger.success('df final: {}', df.info())

# %%
df.to_csv(OUT_PATH / 'dataset_metadata.csv.gz')

logger.success('df escrito em disco')

# %%



