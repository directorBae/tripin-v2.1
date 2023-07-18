import numpy as np
import pandas as pd

class tripin:
  def init(self, inputData, data, tensor, df):
    self.dfs = []
    self.inp = inputData
    self.plc_data = data
    self.word_v_df = df
    self.word_v_tensor = tensor
    for i in range(len(self.plc_data)):
      valsum = sum(self.plc_data['tag'].iloc[i].values())
      temp_data = pd.DataFrame(index = self.word_v_df.index.tolist(), columns = ['value']).fillna(0.0)
      for word in list(self.plc_data['tag'].iloc[i].keys()):
        if word in temp_data.index:
          temp_data.loc[word] = self.plc_data['tag'].iloc[i][word] / valsum
      self.dfs.append(temp_data.values)
  
  def calculate_tensor(self, mat1, mat2, mat3):
    return np.matmul(np.matmul(mat1, mat3.T), mat2)

  def run(self):
    ret_mats = []
    for plc_mat in self.dfs:
      ret_mats.append(self.calculate_tensor(self.inp, plc_mat, self.word_v_tensor))
    return ret_mats