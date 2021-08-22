import pandas as pd

from FEMTOBearing._save_load_pkl import save_pkl, load_pkl
from feature_engineering import feature_generation as fg


# Input preprocessed dataset
train_dict = load_pkl('Bearing1_1_preprocess_dict')
test_dict = load_pkl('Bearing1_3_preprocess_dict')

sampling_freq = 25600
n_top_freq = 5

# Get feature
train_feature_dict = pd.DataFrame()
for f in train_dict:
    train_feature_dict = pd.concat([train_feature_dict,
                                    fg.get_featured_df(train_dict[f].iloc[:, 0:2],
                                                       sampling_freq,
                                                       n_top_freq)], 
                                   axis=0, ignore_index=True)
test_feature_dict = pd.DataFrame()
for f in test_dict:  
    test_feature_dict = pd.concat([test_feature_dict,
                                   fg.get_featured_df(test_dict[f],
                                                      sampling_freq,
                                                      n_top_freq)], 
                                  axis=0, ignore_index=True)

save_pkl(train_feature_dict, 'Bearing1_1_feature_dict')
save_pkl(test_feature_dict, 'Bearing1_3_feature_dict')
    
