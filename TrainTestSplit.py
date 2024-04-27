# %%
from sklearn.model_selection import train_test_split
from tools import BASE_PATH, OUT_PATH, open_meta_df
import numpy as np

# %%
meta_df = open_meta_df()

# %%
idx_train, idx_test = train_test_split(meta_df.index, train_size=0.75, random_state=42, shuffle=True, stratify=meta_df['pose_id'])

# %%
np.save(OUT_PATH / 'train_idx.npy', idx_train.to_numpy())
np.save(OUT_PATH / 'test_idx.npy', idx_test.to_numpy())

# %%



