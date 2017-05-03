import os

from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd

BLENDING_DIR = 'blending/'


def make_submission(predictions, root, title, estimator=None, params={}, score=None, add_to_blending=False):
    submission = pd.DataFrame({'Id': np.arange(len(predictions)),
                               'Prediction': predictions})

    datetime_now = datetime.now()

    dir_name = title
    dir_name += '[score={:.5}]'.format(score) if score is not None else ''
    dir_name += '[{}]'.format(datetime_now.strftime('%d-%m-%Y %H:%M:%S'))
    dir_path = os.path.join(root, dir_name)
    os.makedirs(dir_path)

    # submission.csv
    submission.to_csv(os.path.join(dir_path, 'submission.csv'), index=False)

    # params.json
    with open(os.path.join(dir_path, 'params.json'), 'w') as fout:
        json.dump(params, fout, indent=4)

    # estimator.pkl
    if estimator is not None:
        with open(os.path.join(dir_path, 'estimator.pkl'), 'wb') as fout:
            try:
                pickle.dump(estimator, fout)
            except:
                print('Failed to pickle model!')

    if add_to_blending:
        blending_params = {
            'predictions': list(predictions),
            'score': score,
            'params': params
        }

        if not os.path.exists(BLENDING_DIR):
            os.makedirs(BLENDING_DIR)
        with open(os.path.join(BLENDING_DIR, '{}.json'.format(dir_name)), 'w') as fout:
            json.dump(blending_params, fout, indent=4)