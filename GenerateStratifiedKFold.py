# %%
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tools import BASE_PATH, OUT_PATH, open_meta_df

# %%
fold_path = OUT_PATH / 'Folds'
meta_df = open_meta_df()
meta_df.head()

# %%
X = np.array(meta_df.index).reshape((-1, 1))

# %%
y = meta_df['pose_id'].values

# %%
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    fold_index = fold_path / str(i)
    fold_index.mkdir(exist_ok=True, parents=True)
    np.save(fold_index / f'test', test_index)
    np.save(fold_index / f'train', train_index)

# %%



