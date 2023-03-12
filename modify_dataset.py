import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt


def create_dataset(phase):
  csv_path = 'raw_data/'+ phase +'/labels_'+ phase + '/' + phase + '_gtruth.csv'
  img_path = 'raw_data/'+ phase +'/data/'
  df = pd.read_csv(csv_path)
  np_array = df.values
  cwd = os.getcwd()
  # cwd = '/mnt/c/Users/souga/Documents/biometric_fusion/ChaLearnLAP'
  no_rows = len(np_array[:,0])
  for i in range(no_rows):
      name_of_img = str(np_array[i,0]) + '.jpg'
      class_name = str(np_array[i,1])
      dest_folder_path = cwd + '/' + 'modified_data/'+ phase + '/' + class_name
      # The exist_ok=True argument tells the function not to raise an error if the folder already exists.
      os.makedirs(dest_folder_path, exist_ok=True)
      # now copy the file from dir "raw_data" to dir "modified_data"
      src_file_path = cwd + '/' + 'raw_data/'+ phase + '/data/' + name_of_img
      shutil.copy(src_file_path, dest_folder_path) 

def main():
  phase = 'train'
  create_dataset(phase)
  phase = 'test'
  create_dataset(phase)
  phase = 'val'
  create_dataset(phase)

if __name__ == '__main__':
    main()
