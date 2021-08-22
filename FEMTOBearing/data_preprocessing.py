import os
import pandas as pd
import pickle


class FEMTOBearingDataSet:
    def __init__(self):
        self.path
        self.file_name
        
    # Read dataset
    def read_data(self, column_name):
        df_dict = {}
        path_to_folder = self.path
        file_location, file_name = [], []
        for i in os.listdir(path_to_folder):
            if i.startswith('acc'): # import file with acc
                file_name.append(i[:-len('.csv')])
                file_location.append(path_to_folder+'/'+i)
        for name, loc in zip(file_name, file_location):
            df_dict[name] = pd.read_csv(loc, names=column_name)
        return df_dict
    
    # Drop useless time
    def drop_time(self, df_dict):
        drop_df_dict = {}
        for f in df_dict:
            drop_df_dict[f] = df_dict[f].drop(['Hour', 'Minute', 
                                               'Second', 'MicroSecond'],
                                              axis=1)
        return drop_df_dict

    # Add RUL to each sample
    def add_RUL(self, df_dict):
        pass
    
    # Creat and output pkl
    def output_data(self, df_dict):
        file = open(f'pkl/{self.file_name}_preprocess_dict.pkl', 'wb')
        pickle.dump(df_dict, file)        
        file.close()
    
    
class Learning_FEMTOBearingDataSet(FEMTOBearingDataSet):
    def __init__(self, file_name):
        self.path = f'Data/Learning_set/{file_name}'
        self.file_name = file_name
        
    # Add RUL to each sample
    def add_RUL(self, df_dict):
        for n in range(1, len(df_dict)+1):
            df_dict[f'acc_{n:05d}']['RUL'] = len(df_dict)-n
        return df_dict
    
    
class Test_FEMTOBearingDataSet(FEMTOBearingDataSet):
    def __init__(self, file_name, actual_RUL):
        self.path = f'Data/Test_set/{file_name}'
        self.file_name = file_name
        self.actual_RUL = actual_RUL
    
    # Add RUL to each sample
    def add_RUL(self, df_dict):
        for n in range(1, len(df_dict)+1):
            df_dict[f'acc_{n:05d}']['RUL'] = self.actual_RUL-n
        return df_dict


def get_preprocess_data(FEMTO_data):
    original_df = FEMTO_data.read_data(['Hour', 'Minute', 'Second', 'MicroSecond', 
                                        'HAcc', 'Vacc'])

    droptime_df = FEMTO_data.drop_time(original_df)
    # addRUL_df = FEMTO_data.add_RUL(droptime_df)
    FEMTO_data.output_data(droptime_df)


learn_data_name = ['Bearing1_1', 'Bearing1_2']
for file_name in learn_data_name:
    learning_data = Learning_FEMTOBearingDataSet(file_name)
    get_preprocess_data(learning_data)


test_data_name = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']
test_data_RUL = [339, 1610, 1460, 7570]
for file_name, RUL in zip(test_data_name, test_data_RUL):
    test_data = Test_FEMTOBearingDataSet(file_name, RUL)
    get_preprocess_data(test_data) 