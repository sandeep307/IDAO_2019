import pandas as pd
import swifter
import utils

def main():
    train_data = utils.load_train_hdf('../data/')
    close_feats = train_data.swifter.apply(utils.find_closest_hit_per_station, result_type='expand', axis=1)
    close_feats.to_csv('../data/train_closest_hits_features.csv')

    test_data = pd.read_hdf('../data/test_public_v2.hdf')
    close_feats = test_data.swifter.apply(utils.find_closest_hit_per_station, result_type='expand', axis=1)
    close_feats.to_csv('../data/test_closest_hits_features.csv')
    
    private_test_data = pd.read_hdf('../data/test_private_v2_track_1.hdf')
    close_feats = private_test_data.swifter.apply(utils.find_closest_hit_per_station, result_type='expand', axis=1)
    close_feats.to_csv('../data/private_test_closest_hits_features.csv')

if __name__ == "__main__":
    main()