import pickle

def save_pkl(dict_pkl, file_name):
    file = open(f'FEMTOBearing/pkl/{file_name}.pkl', 'wb')
    pickle.dump(dict_pkl, file) 
    file.close()
    
def load_pkl(file_name):
    file = open(f'FEMTOBearing/pkl/{file_name}.pkl', 'rb')
    dict_pkl = pickle.load(file)
    file.close()
    return dict_pkl