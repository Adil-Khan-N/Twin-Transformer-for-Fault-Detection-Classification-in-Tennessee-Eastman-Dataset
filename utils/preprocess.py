
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

class Preprocessor:
    def __init__(self, data_dir="datasets", batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.feature_cols = None
        self.scaler = StandardScaler()

    def load_csv_chunks(self, filename, chunksize=10000):
        chunks = []
        for chunk in pd.read_csv(os.path.join(self.data_dir, filename), chunksize=chunksize):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)

    def load_data(self):
        faulty_train = self.load_csv_chunks("faulty_training.csv")
        faulty_test = self.load_csv_chunks("faulty_testing.csv")
        normal_train = self.load_csv_chunks("fault_free_training.csv")
        normal_test = self.load_csv_chunks("fault_free_testing.csv")

        normal_train['label'] = 0
        normal_test['label'] = 0
        for fault_num in range(1, 21):
            faulty_train.loc[faulty_train['faultNumber'] == fault_num, 'label'] = fault_num
            faulty_test.loc[faulty_test['faultNumber'] == fault_num, 'label'] = fault_num

        train_data = pd.concat([normal_train, faulty_train], ignore_index=True)
        test_data = pd.concat([normal_test, faulty_test], ignore_index=True)
        return train_data, test_data

    def scale_features(self, train_data, test_data):
        self.feature_cols = [col for col in train_data.columns if col.startswith('xmeas_') or col.startswith('xmv_')]
        train_data[self.feature_cols] = self.scaler.fit_transform(train_data[self.feature_cols]).astype(np.float32)
        test_data[self.feature_cols] = self.scaler.transform(test_data[self.feature_cols]).astype(np.float32)
        return train_data, test_data

    def reshape_as_timeseries(self, df):
        X, y = [], []
        lengths = df.groupby(['simulationRun', 'label']).size()
        expected_length = lengths.max()
        for (sim_id, label), group in df.groupby(['simulationRun', 'label']):
            if len(group) == expected_length:
                X.append(group[self.feature_cols].values.astype(np.float32))
                y.append(label)
        return np.array(X, dtype=np.float32), np.array(y)

    def get_dataloaders(self):
        train_data, test_data = self.load_data()
        train_data, test_data = self.scale_features(train_data, test_data)

        X_train, y_train = self.reshape_as_timeseries(train_data)
        X_test, y_test = self.reshape_as_timeseries(test_data)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader













# # preprocess.py
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import os

# class Preprocessor:
#     def __init__(self, data_dir="C:/Users/Adil/Desktop/TwinTransformer/datasets", batch_size=64):
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.feature_cols = None
#         self.scaler = StandardScaler()

#     def load_data(self):
#         faulty_train = pd.read_csv(f"{self.data_dir}/faulty_training.csv")
#         faulty_test = pd.read_csv(f"{self.data_dir}/faulty_testing.csv")
#         normal_train = pd.read_csv(f"{self.data_dir}/fault_free_training.csv")
#         normal_test = pd.read_csv(f"{self.data_dir}/fault_free_testing.csv")

#         normal_train['label'] = 0
#         normal_test['label'] = 0
#         for fault_num in range(1, 21):
#             faulty_train.loc[faulty_train['faultNumber'] == fault_num, 'label'] = fault_num
#             faulty_test.loc[faulty_test['faultNumber'] == fault_num, 'label'] = fault_num

#         train_data = pd.concat([normal_train, faulty_train])
#         test_data = pd.concat([normal_test, faulty_test])
#         return train_data, test_data

#     def scale_features(self, train_data, test_data):
#         self.feature_cols = [col for col in train_data.columns if col.startswith('xmeas_') or col.startswith('xmv_')]
#         train_data[self.feature_cols] = self.scaler.fit_transform(train_data[self.feature_cols])
#         test_data[self.feature_cols] = self.scaler.transform(test_data[self.feature_cols])
#         return train_data, test_data

#     def reshape_as_timeseries(self, df):
#         X, y = [], []
#         lengths = df.groupby(['simulationRun', 'label']).size()
#         expected_length = lengths.max()
#         for (sim_id, label), group in df.groupby(['simulationRun', 'label']):
#             if len(group) == expected_length:
#                 X.append(group[self.feature_cols].values)
#                 y.append(label)
#         return np.array(X, dtype=np.float32), np.array(y)

#     def get_dataloaders(self):
#         train_data, test_data = self.load_data()
#         train_data, test_data = self.scale_features(train_data, test_data)

#         X_train, y_train = self.reshape_as_timeseries(train_data)
#         X_test, y_test = self.reshape_as_timeseries(test_data)

#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
#         )

#         train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
#         val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
#         test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
#         val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
#         test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

#         return train_loader, val_loader, test_loader


# # import numpy as np
# # import pandas as pd
# # import torch
# # from torch.utils.data import DataLoader, TensorDataset
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split

# # import os
# # print("Current working dir:", os.getcwd())
# # print("Looking for:", os.path.abspath("../dataset/faulty_training.csv"))

# # #Loading the data
# # faulty_train = pd.read_csv("../dataset/faulty_training.csv")
# # faulty_test = pd.read_csv("../dataset/faulty_testing.csv")
# # normal_train = pd.read_csv("../dataset/fault_free_training.csv")
# # normal_test = pd.read_csv("../dataset/fault_free_testing.csv")

# # normal_train['label'] = 0;
# # normal_test['label'] = 0;

# # #.loc -> 
# # for fault_num in range(1, 21):
    
# #     faulty_train.loc[faulty_train['faultNumber'] == fault_num, 'label'] = fault_num
    
# #     faulty_test.loc[faulty_test['faultNumber'] == fault_num, 'label'] = fault_num

# # train_data = pd.concat([normal_train, faulty_train])
# # test_data = pd.concat([normal_test, faulty_test])

# # feature_cols = [col for col in train_data.columns if col.startswith('xmeas_') or col.startswith('xmv_')]
# # scaler = StandardScaler()
# # train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
# # test_data[feature_cols] = scaler.transform(test_data[feature_cols])

# # #to make it into Timesteps*52 variables
# # def reshape_as_timeseries_shape(df, feature_cols):
# #     X, y = [], []
# #     lengths = df.groupby(['simulationRun', 'label']).size()
# #     expected_length = lengths.max()
    
# #     for (sim_id, label), group in df.groupby(['simulationRun', 'label']):
# #         if len(group) == expected_length:
# #             X.append(group[feature_cols].values)
# #             y.append(label)
# #         else:
# #             print(f"Dropping run {sim_id} (length {len(group)})")
    
# #     return np.array(X, dtype=np.float32), np.array(y)
    
# # X_train, y_train = reshape_as_timeseries_shape(train_data, feature_cols)
# # X_test, y_test = reshape_as_timeseries_shape(test_data, feature_cols)

# # X_train,  X_val, y_train, y_val = train_test_split(
# #     X_train,y_train,
# #     test_size = 0.2,
# #     random_state = 42,
# #     stratify = y_train
# # )

# # X_train = torch.tensor(X_train, dtype=torch.float32)
# # y_train = torch.tensor(y_train, dtype=torch.int64)
# # X_val = torch.tensor(X_val, dtype=torch.float32)
# # y_val = torch.tensor(y_val, dtype=torch.int64)
# # X_test = torch.tensor(X_test, dtype=torch.float32)
# # y_test = torch.tensor(y_test, dtype=torch.int64)


# # # Output dataset shapes
# # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# # print(f"X_valid shape: {X_val.shape}, y_valid shape: {y_val.shape}")
# # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# # train_dataset = TensorDataset(X_train, y_train)
# # valid_dataset = TensorDataset(X_val, y_val)
# # test_dataset = TensorDataset(X_test, y_test)

# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
# # valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, drop_last=True)
# # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

# # n_classes = len(torch.unique(y_train))
# # print(f"Number of classes: {n_classes}")

