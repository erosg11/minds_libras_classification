# %%
from mediapipe.python.solutions import pose as mp_pose
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import cv2
import numpy as np
from dotenv import load_dotenv
from os import getenv
from loguru import logger

load_dotenv()

# %%
BASE_PATH = Path(getenv('VIDEO_ROOT'))
OUT_PATH = Path(getenv('OUT_FOLDER'))
meta_df = pd.read_csv(OUT_PATH / 'dataset_metadata.csv.gz', 
                      dtype={
                          'filename': 'string',
                          'stem': 'string',
                          'pose_id': 'int64',
                          'word': 'category',
                          'sinalizador': 'int64',
                          'repetition': 'int64',
                      }, index_col=0)
logger('Lido meta df: {}', meta_df.head())

# %%
videos_landmarks = []
append_video_landmarks = videos_landmarks.append

for video in tqdm(list(meta_df.itertuples(index=True))):
    base_row = [video.Index]
    cap = cv2.VideoCapture(video.filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tq = tqdm(total=frame_count)
    with mp_pose.Pose() as pose_tracker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Falha ao ler o frame do vídeo {!r}, pulando o vídeo", video)
                break
            result = pose_tracker.process(image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_landmarks = result.pose_landmarks
            if pose_landmarks is not None:
                landmarks = pose_landmarks.landmark
                assert len(landmarks) == 33, f'Unexpected number of predicted pose landmarks: {len(landmarks)}'
                list_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in landmarks]
                append_video_landmarks(base_row + list_landmarks)
            tq.update(1)
logger.info('Lidos {} videos', len(videos_landmarks))

# %%
video_id = np.array([x[0] for x in videos_landmarks])
final_landmarks = np.array([x[1:] for x in videos_landmarks])
logger.info('Formato final dos landmarks: {!r}', final_landmarks.shape)

# %%
np.save(str(OUT_PATH / 'video_id'), video_id)
np.save(str(OUT_PATH / 'landmarks'), final_landmarks)
