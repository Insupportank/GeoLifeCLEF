import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

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
    def __init__(self, file_path, file_type="train", country="fr", transform=None, data_portion=.2):
        self.file_path = file_path
        self.transform = transform
        self.default_transform = A.Compose([
            A.Resize(256,256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
            ])
      
        if country == "all":
            df_obs_fr = pd.read_csv(f"{file_path}/observations/observations_fr_{file_type}.csv", sep=";")
            df_obs_us = pd.read_csv(f"{file_path}/observations/observations_us_{file_type}.csv", sep=";")
            df_obs = pd.concat((df_obs_fr, df_obs_us))
        
        else:
            df_obs = pd.read_csv(f"{file_path}/observations/observations_{country}_{file_type}.csv", sep=";")
        
        self.categories = df_obs.species_id.unique().tolist()
        df_obs = df_obs.sample(frac=data_portion, replace=False, random_state=1) #carfull, the spiecies repartition is unbalanced, so we might want to take a better subsample

        #Add to the df the bioclimatic and pedologic data
        df_rasters = pd.read_csv(f"{file_path}/pre-extracted/environmental_vectors.csv", sep =";")
        df_obs = df_obs.merge(df_rasters, left_on="observation_id", right_on="observation_id", suffixes=('', ''))

        #Create column for the path of each images
        df_obs["rgb_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if str(x['observation_id'])[0] == '1' else 'us'}/{str(x['observation_id'])[-2:]}/{str(x['observation_id'])[-4:-2]}/{x['observation_id']}_rgb.jpg", axis=1)
        df_obs["altitude_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if str(x['observation_id'])[0] == '1' else 'us'}/{str(x['observation_id'])[-2:]}/{str(x['observation_id'])[-4:-2]}/{x['observation_id']}_altitude.tif", axis=1)
        df_obs["landcover_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if str(x['observation_id'])[0] == '1' else 'us'}/{str(x['observation_id'])[-2:]}/{str(x['observation_id'])[-4:-2]}/{x['observation_id']}_landcover.jpg", axis=1)
        df_obs["near_ir_image"] = df_obs.apply(lambda x: f"{self.file_path}/patches-{'fr' if str(x['observation_id'])[0] == '1' else 'us'}/{str(x['observation_id'])[-2:]}/{str(x['observation_id'])[-4:-2]}/{x['observation_id']}_near_ir.jpg", axis=1)

        self.data_set = df_obs

        self.list_of_features = df_obs.columns
        self.list_of_features = self.list_of_features.drop(['species_id', 'observation_id', 'subset', 'rgb_image', 'altitude_image', 'landcover_image', 'near_ir_image'])

        
        features = self.data_set[self.list_of_features]
        # I don't know why, but bio_19 has some values that are not float Or NaN...
        features.bio_19 = pd.to_numeric(features['bio_19'], errors='coerce')
        
        ### /!\ Carefull to not fill na with mean if we split data for test/train (its fine for the validation sso we still do it)
        features = features.fillna(features.mean())
        epsilon = 1e-7
        self.normalized_features = (epsilon + features-features.min())/(features.max()-features.min())

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        label = self.data_set.iloc[idx]["species_id"]
        #Pillow works for png and tif
        image_rgb = np.array(Image.open(self.data_set.iloc[idx]["rgb_image"]).convert("RGB"))
        image_altitude = np.array(Image.open(self.data_set.iloc[idx]["altitude_image"]))
        image_landcover = np.array(Image.open(self.data_set.iloc[idx]["altitude_image"]))         
        image_near_ir = np.array(Image.open(self.data_set.iloc[idx]["altitude_image"]))


        

        if self.transform:
            transformed_rgb = self.transform(image=image_rgb)
            transformed_altitude = self.transform(image=image_altitude)
            transformed_landcover = self.transform(image=image_landcover)
            transformed_near_ir = self.transform(image=image_near_ir)
        else :
            transformed = self.default_transform(image = np.asarray(image))

        image = transformed['image'] #3,256,256 if RGB like it is now
        image = image.to(torch.float32)

        # #get features from df
        features = torch.tensor(self.normalized_features.iloc[idx][self.list_of_features], dtype=torch.float32)

        return {"image": image, "features": features}, label


def test_dataset(config):
    train_set = GeoLifeDataset(config['data']['trainpath'], file_type="train", country="all", data_portion=.01)
    print(f"Dataset has {len(train_set)} samples")
    print(f"Dataset has {len(train_set.categories)} classes")
    print(f"Dataset has {tuple(train_set[0][0]['image'].shape)} as image tensor size; {tuple(train_set[0][0]['features'].shape)} as features tensor size.")
    print("first index tensor and label is : ")
    print(train_set[0])
    print("dataset well made")

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    test_dataset(config)