import torch
import torch.nn as nn
import torch.utils.data

import os
import yaml
import sys

import pandas as pd

from PIL import Image
from torchvision.transforms import ToTensor

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class GeoLifeDataset(torch.utils.data.Dataset):
    """
    This is a custom dataset to load the data from the mount directory
    It takes in input the root directory and if there needs to be a transformation to the features
    Also we need to only take a part of the dataset because it is BIG, and for testing/debugging we can't use all the data.
    The features: 
        - Location (lat, long)
        - RGB image .jpeg
        - Near-IR image .jpeg
        - Altitude map .tif
        - Land cover map .tif
        - 19 bioclimatic features (in the pre-extracted/environmental_vectors.csv)
        - 8 Pedologic data features (in the pre-extracted/environmental_vectors.csv)
    """
    def __init__(self, file_path, file_type="train", country="fr", transform=None, data_portion=.05):
        self.file_path = file_path
        self.transform = transform
        self.default_transform = A.Compose([A.Resize(256,256), ToTensorV2()])

        df_obs_fr = pd.read_csv(f"{file_path}/observations/observations_{country}_{file_type}.csv", sep=";")
        
        
        if country == "all":
            df_obs_us = pd.read_csv(f"{file_path}/observations/observations_fr_{file_type}.csv", sep=";")
            df_obs_us = pd.read_csv(f"{file_path}/observations/observations_us_{file_type}.csv", sep=";")
            df_obs = pd.concat((df_obs_fr, df_obs_us))
        
        df_obs = df_obs.sample(frac=data_portion, replace=False, random_state=1) #carfull, the spiecies repartition is unbalanced, so we might want to take a better subsample
        self.data_length = len(self.data_set)

        #Add to the df the bioclimatic and pedologic data
        df_rasters = pd.read_csv(f"{file_path}/pre-extracted/environmental_vectors.csv")
        df_obs = df_obs.merge(df_rasters, left_on="observation_id", right_on="observation_id", suffixes=('', ''))
        
        #Create column for the path of each images
        df_obs["rgb_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == 1 else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_rgb.jpg", axis=1)
        df_obs["altitude_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == 1 else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_altitude.tif", axis=1)
        df_obs["landcover_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == 1 else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_landcover.jpg", axis=1)
        df_obs["near_ir_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == 1 else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_near_ir.jpg", axis=1)

        self.data_set = df_obs

        self.list_of_features = df_obs.columns
        self.list_of_features.remove('species_id', 'observation_id', 'subset')
        try:
            self.list_of_features.remove('subset')
        except ValueError:
            pass


    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        label = self.data_set.iloc[idx]["species_id"]
        
        #Pillow works for png and tif
        image = Image.open(self.data_set.iloc[idx]["rgb_image"]).convert("RGB")

        if self.transform:
            image = self.transform(image=image)['image']
        else :
            image = self.default_transform(image=image)['image']

        #get features from df
        features = torch.tensor(self.data_set.iloc[idx][self.list_of_features], dtype=torch.float32)

        # combine image(s) and df data
        combined_data = torch.cat((image, features), dim=0)

        return combined_data, label
    
    def classes(self):
        # Optional function but usfull to get the list of classes
        return self.data_set.species_id.unique().tolist()
    
    def tensor_size(self):
        # Also optionnal but could be usfull
        return None


def test_dataset(config):
    train_set = GeoLifeDataset(config['data']['trainpath'], file_type="train", country="all", data_portion=.01)
    print(f"Dataset has {len(train_set)} samples")
    print(f"Dataset has {len(train_set.classes)} classes")
    print(f"Dataset has {train_set.tensor_size} as tensor size")
    print("first index tensor and label is : ")
    print(train_set[0])
    print("dataset well made")

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", "r"))
    test_dataset(config)