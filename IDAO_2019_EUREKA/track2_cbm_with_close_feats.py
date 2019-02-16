import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import utils, scoring
import catboost as cb


def main():
    np.random.seed(0)
    data = pd.concat([pd.concat([
    pd.read_csv(f'../data/train_part_{i}_v2.csv.gz', usecols=utils.SIMPLE_FEATURE_COLUMNS + ['label', 'weight'], na_values=-9999)
    for i in (1, 2)], ignore_index=True), pd.read_csv('../data/train_closest_hits_features.csv').drop(columns='Unnamed: 0')], axis=1)
    print('\n\ndata shape:', data.shape)
    train_ds, valid_ds = train_test_split(data, test_size=.05, shuffle=True)
    print('train data shape:', train_ds.shape)
    print('valid data shape:', valid_ds.shape)

    model = cb.CatBoostClassifier(iterations=1500, verbose=5, eval_metric='AUC')
    print('params:', model.get_params())
    dtrain = cb.Pool(train_ds.drop(columns=['label', 'weight']).values, train_ds.label.values, weight=train_ds.weight.clip(0).values)
    dvalid = cb.Pool(valid_ds.drop(columns=['label', 'weight']).values, valid_ds.label.values, weight=valid_ds.weight.clip(0).values)
    model.fit(dtrain, eval_set=[dtrain, dvalid], early_stopping_rounds=50)
    model.save_model('../models/cbm_with_close_feats_90p.cbm')
    valid_preds = model.predict_proba(dvalid)
    print('Valid rejection90:', scoring.rejection90(valid_ds.label.values, valid_preds[:, 1], valid_ds.weight.values))

    test_ds = pd.concat([pd.read_csv('../data/test_public_v2.csv.gz', usecols=utils.SIMPLE_FEATURE_COLUMNS, na_values=-9999), pd.read_csv('../data/test_closest_hits_features.csv').drop(columns='Unnamed: 0')], axis=1)
    print('test data shape:', test_ds.shape)
    test_preds = model.predict_proba(test_ds.values)[:, 1]
    pd.DataFrame({'id': test_ds.index, 'prediction': test_preds}, columns=['id', 'prediction']
               ).to_csv('../submissions/cbm_with_close_feats_90p.csv', index=False)

if __name__ == "__main__":
    main()

