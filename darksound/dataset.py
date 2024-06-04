"""
Copyright [2023] [Poutaraud]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Optional, Tuple

import urllib.request
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, list_dir, list_files

import maad
import bambird
import librosa

class Darksound(VisionDataset):
    """`Darksound <https://zenodo.org/records/10512103>` Dataset.

    Args:
        root (str): Root directory of dataset where directory ``data`` exists.
        split (str, optional): If "train", creates dataset from the train set, if "test" creates from the test set.
        hpss (bool, optional): Whether to apply Harmonic Percussive Source Separation (HPSS)
            on the spectrogram. Defaults to True.
        tfr (str, optional): Type of time-frequency representation. Possible to choose
            between 'spec', 'cqt' or 'mel'. Defaults to 'mel'.
        n_classes (int, optional): Determines the number of classes (1 <> 30) to use for the test set. 
            This parameter does not work when split='train'. Defaults to None.
        n_samples (int, optional): Determines the number of samples for each species in the train set. 
            This parameter does not work when split='test'. Defaults to None.
        remove_background: Whether to remove background from the spectrogram. 
            Defaults to True.
        config (str, optional): Path to the configuration file that is used for downloading the train set.
        transform (callable, optional): A function/transform that takes in an HPD image 
            and returns a transformed version.
        data (str, optional): Whether to download data from 'zenodo' or 'xc' (Xeno-Canto). Defaults to 'zenodo' which
            corresponds to the data used in the research paper.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    
    folder = "data"
    
    target_species = [
                       # TINAMOUS
                       'Great Tinamou',
                       'Rusty Tinamou',
                       'Little Tinamou',
                       'Cinereous Tinamou',
                       'Variegated Tinamou',
                       # POTOOS
                       'Great Potoo',
                       'Common Potoo',
                       'Rufous Potoo',
                       'Long-tailed Potoo',
                       'White-winged Potoo', 
                       # FALCONS
                       'Barred Forest Falcon',
                       'Lined Forest Falcon',
                       'Slaty-backed Forest Falcon',
                       'Collared Forest Falcon', 
                       # OWLS
                       'Foothill Screech Owl',
                       'Tawny-bellied Screech Owl', 
                       'Crested Owl',
                       'Spectacled Owl',
                       'Mottled Owl',
                       'Black-banded Owl',
                       'Amazonian Pygmy Owl',
                       # NIGHTJARS
                       'Short-tailed Nighthawk',
                       'Spot-tailed Nightjar',
                       'White-tailed Nightjar',
                       'Band-tailed Nighthawk',
                       'Lesser Nighthawk',
                       'Nacunda Nighthawk',
                       'Blackish Nightjar', 
                       'Pauraque',
                       'Ladder-tailed Nightjar',
                      ]

    def __init__(
        self,
        root: str = os.getcwd(),
        split: str = 'test',
        hpss: bool = True,
        tfr: str = 'mel',
        n_classes: int = None,
        n_samples: int = None,
        remove_background: bool = False,
        config: str = "config.yaml",
        transform: Optional[Callable] = None,
        data: str = 'zenodo',
        download: bool = False) -> None:

        super().__init__(os.path.join(root, self.folder), transform=transform)
        self.split = split
        self.hpss = hpss
        self.tfr = tfr
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.remove_background = remove_background
        self.data = data
        self.params = bambird.load_config(os.path.join(root, config))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it.")

        self.target_folder = os.path.join(self.root, self.split)
        
        if self.split == 'train':
            if self.n_classes != None:
                print("Number of classes is only possible with split='test'.")
                return
            
            df_cluster = pd.read_csv(os.path.join(self.target_folder, 'cluster.csv'), sep=';') 
            self.df = df_cluster[df_cluster.auto_label == 1.0] # keep only rois pseudo-labeled as signal
            species_fullname = self.df.fullfilename_ts.str.split(os.sep) 
            self.df['species'] = species_fullname.str[-2] # add column with species full name
            
            if self.n_samples != None:
                # Keep categories with more than n_samples
                self.df = self.df.groupby(['species']).filter(lambda group: len(group) > self.n_samples)
                # Iterate over each unique species
                for i in list(self.df.species.unique()):
                    category = self.df[self.df.species == i]
                    # Get the number of ROIs in the category
                    n_rois = len(category)
                    if n_rois > self.n_samples:
                        # Drop ROIs above the number of shots per category
                        self.df = self.df.drop(category.sample(n=n_rois - self.n_samples).index)

#                 ################### /!\ HACK FOR CHECKING DIFFERENCE BETWEEN NUMBER OF CLASSES /!\ ################### 
#                 # 720 classes with 50 shots VS 120 classes with 300 shots (n_samples=50 VS 300)
#                 random_species = random.sample(list(self.df.species.unique()), int(36000 / self.n_samples)) 
#                 self.df = self.df[self.df.species.isin(random_species)]
#                 ######################################################################################################
                
            self._species = list(self.df.species.unique())
            self._species_files = [
                [(audio, idx) for audio in self.df[self.df.species == species].filename_ts] 
                for idx, species in enumerate(self._species)
            ]
            self._flat_species_files: List[Tuple[str, int]] = sum(self._species_files, [])
                
        if self.split == 'test':
            if self.n_samples != None:
                print("Number of samples is only possible with split='train'.")
                return
            
            self._species = list_dir(self.target_folder)
            self._species_files = [
                [(audio, idx) for audio in list_files(os.path.join(self.target_folder, species), ".wav")]
                for idx, species in enumerate(self._species)
            ]
            self._flat_species_files: List[Tuple[str, int]] = sum(self._species_files, [])
                
            if self.n_classes != None:
                labels = [instance[1] for instance in self._flat_species_files]
                unique_labels = np.unique(labels)
                if self.n_classes > len(unique_labels):
                    print(f"Dataset contains only {len(unique_labels)} classes. Choose a number between 2 and {len(unique_labels)}.")
                    return
                else:
                    classes = random.sample(list(unique_labels), k=self.n_classes)
                    self._flat_species_files: List[Tuple[str, int]] = [i for i in self._flat_species_files if i[1] in classes]            

    def __len__(self) -> int:
        return len(self._flat_species_files)
    
    def __getlabel__(self) -> list:
        labels = [instance[1] for instance in self._flat_species_files]
        return labels

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where label is index of the label species class.
        """
        # Load audio file
        audio, label = self._flat_species_files[index]
        audio_path = os.path.join(self.target_folder, self._species[label], audio)
        y, sr = self._load_audio(audio_path, pad=True)

        # Compute spectrogram
        Y = torch.flip(self._get_spectrogram(y, sr, tfr=self.tfr), dims=[0, 1])
        
        # Remove background 
        if self.remove_background:
            Y, _, _ = maad.sound.remove_background(Y.numpy()[0])
            # convert array to tensor
            Y = torch.from_numpy(Y).unsqueeze_(dim=0)
            
        # Compute Harmonic Percussive Source Separation
        if self.hpss:
            Y = self._source_separation(Y.numpy()[0])
        else:
            Y = Y.repeat(3, 1, 1)

        if self.transform:
            Y = self.transform(Y)
            
        return Y, label
    
    def _load_audio(self, path, pad=False, seconds=3):  
        # Load audio file and normalize it using torch
        y, sr = torchaudio.load(path, normalize=True)  
        # Fade in and out to avoid aliasing from window effects
        fade = T.Fade(fade_in_len=int(sr/10), fade_out_len=int(sr/10), fade_shape='half_sine')
        y = fade(y)

        if pad: # Pad audio file to a fix length in seconds
            samples = sr * seconds
            if y.shape[1] >= (samples):
                y.resize_(1, samples)
            else:
                diff = (samples) - y.shape[1]
                pad = torch.nn.ConstantPad1d((int(np.ceil(diff/2)), int(np.floor(diff/2))), 0)
                y = pad(y)
        return y, sr
    
    def _get_spectrogram(self, y, sr, tfr='mel', n_fft=2048, hop_length=512, n_mels=256):
        # FT spectrogram
        if tfr == 'spec':
            spectrogram = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0)
        
        # CQT spectrogram
        if tfr == 'cqt':
            cqt = np.abs(librosa.cqt(y.numpy(), sr=sr))
            return torch.Tensor(cqt)

        # Mel spectrogram
        if tfr == 'mel':
            spectrogram = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm='slaney',
                n_mels=n_mels)
            
        return spectrogram(y)
    
    def _scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled
    
    def _source_separation(self, Y, margin=(1.0,5.0)):
        # Compute Harmonic Percussive Source Separation
        H, P = librosa.decompose.hpss(Y, margin=margin)
        # Convert amplitude spectrogram to dB-scaled 
        Y = librosa.amplitude_to_db(Y, ref=np.max)
        H = librosa.amplitude_to_db(H, ref=np.max)
        P = librosa.amplitude_to_db(P, ref=np.max)

        # Compute delta features
        D = librosa.feature.delta(Y)
        
        # Normalize spectrogram
        H = self._scale_minmax(H)
        P = self._scale_minmax(P) 
        D = self._scale_minmax(D)
        
        HPD = np.nan_to_num(np.transpose(np.asarray(np.dstack((D,H,P))), (2,0,1)))
        
        return torch.from_numpy(HPD)

    def _check_integrity(self) -> bool:
        if not os.path.isdir(os.path.join(self.root, self.split)):
            return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return        
        elif self.split == 'train':
            if self.data == 'zenodo':
                # Download data from Zenodo
                print('Request recordings on Zenodo for building pseudo-labeled set...')
                url = "https://zenodo.org/records/10512103/files/train.zip?download=1"
                download_and_extract_archive(url, self.root, filename=self.split + ".zip", md5="800286a48fffcd0d812999be1405c22b")
                df_xc = pd.read_csv(os.path.join(os.path.join(self.folder, "darksound"), "xc_metadata.csv"), sep=";")
            
            if self.data == 'xc':
            # Download data from Xeno-Canto
                print('Request recordings on Xeno-Canto database for building pseudo-labeled set...')
                query_xc        = self.params['PARAMS_XC']['PARAM_XC_LIST']
                numPages        = 1
                page            = 1
                df_dataset      = pd.DataFrame()

                while page < numPages + 1:
                    url         = 'https://www.xeno-canto.org/api/2/recordings?query={0}&page={1}'.format('%20'.join(query_xc), page)
                    jsonPage    = urllib.request.urlopen(url)
                    jsondata    = json.loads(jsonPage.read().decode('utf-8'))
                    # Check number of pages
                    numPages    = jsondata['numPages']
                    # Concatenate pandas dataframe of records & convert to .csv file
                    df_dataset  = pd.concat([df_dataset, pd.DataFrame(jsondata['recordings'])], ignore_index=True)
                    # Increment the current page
                    page += 1

                # Remove target species from dataset 
                df_dataset = df_dataset[~df_dataset.en.isin(self.target_species)]
                # Set the number of files to download for each species
                for i in df_dataset.en.value_counts()[df_dataset.en.value_counts() > self.params['PARAMS_XC']['NUM_FILES']].index.to_list():
                    rows = df_dataset[df_dataset.en == i]
                    to_drop = rows.sample(n=self.params['PARAMS_XC']['NUM_FILES']).index.to_list()
                    df_dataset = df_dataset.drop(to_drop)
                    
                # Download data from Xeno-Canto
                df_xc, csv_xc   = bambird.download_xc(
                                    df_dataset      = df_dataset,
                                    rootdir         = self.folder, 
                                    dataset_name    = Path('darksound'), 
                                    csv_filename    = self.params['PARAMS_XC']['CSV_XC_FILE'],
                                    overwrite       = True,
                                    verbose         = True)

            # Extract the ROIS of the full dataset
            df_rois, csv_rois = bambird.multicpu_extract_rois(
                                    dataset     = df_xc,
                                    params      = self.params['PARAMS_EXTRACT'],
                                    save_path   = self.folder / Path(self.split),
                                    overwrite   = True,
                                    verbose     = False)

            # Compute the features of the full dataset       
            df_features, csv_features = bambird.multicpu_compute_features(
                                    dataset     = df_rois,
                                    params      = self.params['PARAMS_FEATURES'],
                                    save_path   = self.folder / Path(self.split),
                                    overwrite   = True,
                                    verbose     = True)

            ################### /!\ HACK TO REMOVE IN THE FUTURE /!\ ###################### 
            for i in df_features.categories.unique(): # unique categories
                cat = df_features[df_features.categories == i]
                if len(cat) < 3: # Remove categories with less than 3 ROIs
                    df_features = df_features.drop(cat.index)
            ###############################################################################

            # Cluster the data and assign pseudo-labels
            df_cluster, csv_cluster = bambird.find_cluster(
                                      dataset   = df_features,
                                      params    = self.params['PARAMS_CLUSTER'],
                                      save_path = self.folder / Path(self.split),
                                      display   = False,
                                      verbose   = False)

        elif self.split == 'test':
            url = "https://zenodo.org/records/10512103/files/test.zip?download=1"
            download_and_extract_archive(url, self.root, filename=self.split + ".zip", md5="00676bede612d3d7b8933ac03bc07a85")
        else:
            print("Dataset not found or corrupted. Enter split='test'.")
