from lib.core.base_trainer.net_work import Train
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train_config import config as cfg
from lib.core.kaggle.kl import score
import setproctitle

setproctitle.setproctitle("comp")


def main():
    n_fold = 5

    data = pd.read_csv(cfg.DATA.data_file)
    data['primary_label'] = data['primary_label'].astype(str)

    extra_data_file = cfg.DATA.get('extra_data_file', None)
    if extra_data_file and os.path.exists(extra_data_file):
        extra = pd.read_csv(extra_data_file)
        extra['primary_label'] = extra['primary_label'].astype(str)
        data = pd.concat([data, extra], ignore_index=True)
        print(f'Extra data loaded: {len(extra)} rows from {extra_data_file}, total: {len(data)}')

    soundscape = pd.read_csv(cfg.DATA.soundscape_file)
    soundscape['primary_label'] = soundscape['primary_label'].astype(str)

    sample_submission = pd.read_csv(cfg.DATA.sample_submission_file, nrows=1)
    labels = [str(col) for col in sample_submission.columns if col != 'row_id']
    nm2cls = {label: idx for idx, label in enumerate(labels)}
    
    
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=cfg.SEED)
    data['fold'] = -1

    label_counts = data['primary_label'].value_counts()
    rare_mask = data['primary_label'].map(label_counts) < n_fold
    data_common = data[~rare_mask].copy()
    data_rare = data[rare_mask].copy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_common, data_common['primary_label'])):
        data_common.iloc[val_idx, data_common.columns.get_loc('fold')] = fold_idx
    data.loc[data_common.index, 'fold'] = data_common['fold']
    data.loc[data_rare.index, 'fold'] = -1

    all_oof = []
    all_true = []
    all_weight = []
    for fold in range(n_fold):

        train_data = data[(data['fold'] != fold) | (data['fold'] == -1)].copy()
        train_data = pd.concat([train_data, soundscape], ignore_index=True)
        val_data = data[data['fold'] == fold].copy()

        pse_base = cfg.DATA.get('pse_data_file', None)
        if pse_base and '{fold}' in str(pse_base):
            cfg.DATA.pse_data_file = pse_base.replace('{fold}', str(fold))

        trainer = Train(train_df=train_data,
                        val_df=val_data,
                        fold=fold,
                        nm2cls=nm2cls)

        ### train
        trainer.custom_loop()

        all_oof.append(trainer.oof_pre)
        all_true.append(trainer.oof_gt)


    all_oof = np.concatenate(all_oof)
    all_true = np.concatenate(all_true)


    oof = pd.DataFrame(all_oof.copy())
    oof['id'] = np.arange(len(oof))

    true = pd.DataFrame(all_true.copy())
    true['id'] = np.arange(len(true))

    cv = score(solution=true, submission=oof, row_id_column_name='id')
    print('CV Score  for EfficientNetB2 =', cv)


if __name__ == '__main__':
    main()
