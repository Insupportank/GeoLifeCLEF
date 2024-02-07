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
    def __init__(self, file_path, file_type="train", country="all", transform=None, data_portion=.2):
        self.file_path = file_path
        self.transform = transform
        self.file_type = file_type
        self.default_transform_rgb = A.Compose([
            A.Resize(256,256),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
            ])
        self.default_transform_non_rgb = A.Compose([
            A.Resize(256,256),
            A.Normalize(
                mean=0.456,
                std=0.22
                ),
            ToTensorV2()
            ])
      
        if country == "all":
            df_obs_fr = pd.read_csv(f"{file_path}/observations/observations_fr_{file_type}.csv", sep=";")
            df_obs_us = pd.read_csv(f"{file_path}/observations/observations_us_{file_type}.csv", sep=";")
            df_obs = pd.concat((df_obs_fr, df_obs_us))
        
        else:
            df_obs = pd.read_csv(f"{file_path}/observations/observations_{country}_{file_type}.csv", sep=";")
        
        if file_type == "train":
            self.categories = df_obs.species_id.unique().tolist()
        df_obs = df_obs.sample(frac=data_portion, replace=False, random_state=1) #carfull, the spiecies repartition is unbalanced, so we might want to take a better subsample

        #Add to the df the bioclimatic and pedologic data
        df_rasters = pd.read_csv(f"{file_path}/pre-extracted/environmental_vectors.csv", sep =";")
        self.data_set = df_obs.merge(df_rasters, left_on="observation_id", right_on="observation_id", suffixes=('', '')).astype({'observation_id': 'int'})
        self.data_set.observation_id = self.data_set.observation_id.astype(str)

        #Create column for the path of each images
        self.data_set["rgb_image"] = self.data_set.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == '1' else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_rgb.jpg", axis=1)
        self.data_set["altitude_image"] = self.data_set.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == '1' else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_altitude.tif", axis=1)
        self.data_set["landcover_image"] = self.data_set.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == '1' else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_landcover.jpg", axis=1)
        self.data_set["near_ir_image"] = self.data_set.apply(lambda x: f"{self.file_path}/patches-{'fr' if x['observation_id'][0] == '1' else 'us'}/{x['observation_id'][-2:]}/{x['observation_id'][-4:-2]}/{x['observation_id']}_near_ir.jpg", axis=1)

        self.list_of_features = self.data_set.columns
        error = 'raise' if self.file_type == 'train' else 'ignore' #to ignore the drop of species_id
        self.list_of_features = self.list_of_features.drop(['species_id', 'observation_id', 'subset', 'rgb_image', 'altitude_image', 'landcover_image', 'near_ir_image'], errors=error)

        
        features = self.data_set[self.list_of_features]
        # I don't know why, but bio_19 has some values that are not float Or NaN...
        features.bio_19 = pd.to_numeric(features['bio_19'], errors='coerce')
        
        ### /!\ Carefull to not fill NA with mean if we split data for test/train (its fine for the validation so we still do it)
        features = features.fillna(features.mean())
        # Adding epsilon let us not having 0 as min, and help the crossentropyloss not have log(0)
        epsilon = 1e-7
        self.normalized_features = (epsilon + features-features.min())/(features.max()-features.min())

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        if self.file_type == "train":
            label = self.data_set.iloc[idx]["species_id"]
        else:
            # We need the observation ids to make a submission
            label = self.data_set.iloc[idx]["observation_id"]
        #Pillow works for png and tif
        image_rgb = np.array(Image.open(self.data_set.iloc[idx]["rgb_image"]).convert("RGB"))
        #image_near_ir = np.array(Image.open(self.data_set.iloc[idx]["near_ir_image"]))


        

        if self.transform:
            transformed_rgb = self.transform(image=image_rgb)
            #transformed_near_ir = self.transform(image=image_near_ir)
        else :
            transformed_rgb = self.default_transform_rgb(image=image_rgb)
            #transformed_near_ir = self.default_transform_non_rgb(image=image_near_ir)

        image = transformed_rgb['image'] #3,256,256 if RGB like it is now
        # image_near_ir=transformed_near_ir["image"]
        # image = torch.cat((image_near_ir,image_rgb[1:,:,:]))

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