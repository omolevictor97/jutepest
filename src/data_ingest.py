import os
from src.constant import * #TRAIN_DIR_PATH, VAL_DIR_PATH, TEST_DIR_PATH, DATA_FOLDER_PATH
import sys


class DataIngest:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.train_dir_path = TRAIN_DIR_PATH
        self.val_dir_path = VAL_DIR_PATH
        self.test_dir_path = TEST_DIR_PATH

    
    def read_data_path(self):
        try:
            if os.path.exists(self.data_folder):
                train_path = os.path.join(self.data_folder, self.train_dir_path)
                val_path   = os.path.join(self.data_folder, self.val_dir_path)
                test_path  = os.path.join(self.data_folder, self.test_dir_path)
                
                # check if all these paths exist
                if os.path.isdir(os.path.normpath(train_path)) and os.path.isdir(os.path.normpath(val_path)) \
                and os.path.isdir(os.path.normpath(test_path)):
                    return train_path, val_path, test_path
            else:
                return f'{self.data_folder} is not a valid folder'
        except FileNotFoundError as FE:
            return F'Error: {FE}'
        
