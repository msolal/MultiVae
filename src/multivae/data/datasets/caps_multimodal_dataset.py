from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from multivae.data.datasets.base import DatasetOutput, MultimodalBaseDataset

from clinicadl.utils.loading import nifti_to_tensor, pt_to_tensor

from torch.nn.functional import one_hot

PARTICIPANT_ID = "participant_id"
SESSION_ID = "session_id"

# in the future, this class should probably inherit CapsDataset from clinicadl as well

class CapsMultimodalDataset(MultimodalBaseDataset):
    def __init__(
        self,
        caps_directory: Path,
        tsv_label: Path,
        img_modalities: List[str],
        txt_modalities: Optional[List[str]] = [],
    ):
        
        self.caps_directory = caps_directory

        self.img_modalities = img_modalities
        self.txt_modalities = txt_modalities

        self.df = pd.read_csv(tsv_label, sep="\t")
        mandatory_col = {
            "participant_id",
            "session_id",
        }
        
        if self.txt_modalities is not None:
            for txt_modality in self.txt_modalities:
                mandatory_col.add(txt_modality)

        if not mandatory_col.issubset(set(self.df.columns)):
            raise Exception(
                f"The tsv file is not in the correct format."
                f"Columns should include {mandatory_col}"
            )
            
        for txt_modality in self.txt_modalities:
            self._transform_tabular_data_(txt_modality)  


    def __getitem__(self, index):

        #TODO add to DatasetOutput participant_id, session_id and image_path

        participant, session = self._get_meta_data(index)
        
        X = dict()
        
        for img_modality in self.img_modalities:
            image = self._get_full_image(img_modality, participant, session)
            X[img_modality] = image
            
        for txt_modality in self.txt_modalities:
            txt_data = self._get_tabular_data(txt_modality, index)
            X[txt_modality] = txt_data

        return DatasetOutput(
            data=X,
        )

    def __len__(self):
        return len(self.df)
    
    def _get_pt_image_path(self, img_modality, participant, session) -> Path:
        pt_image_path = Path(
            self.caps_directory, 
            "subjects", 
            participant, 
            session,
            "deeplearning_prepare_data", 
            "image_based",
            img_modality, 
            f"{participant}_{session}_{img_modality}.pt"
        )
        return pt_image_path
    
    def _get_nifti_image_path(self, img_modality, participant, session) -> Path:
        nifti_image_path = Path(
            self.caps_directory, 
            "subjects", 
            participant, 
            session,
            img_modality, 
            f"{participant}_{session}_{img_modality}.nii.gz"
        )
        return nifti_image_path

    def _get_full_image(self, img_modality, participant, session) -> torch.Tensor:

        pt_image_path = self._get_pt_image_path(img_modality, participant, session)
        if pt_image_path.is_file():
            return pt_to_tensor(pt_image_path)

        nifti_image_path = self._get_nifti_image_path(img_modality, participant, session)        
        return nifti_to_tensor(nifti_image_path)
    
    def _transform_tabular_data_(self, txt_modality) -> None:
        
        if txt_modality == "sex": 
            self.df[txt_modality+"_"] = self.df[txt_modality].map({"M": torch.tensor([[0, 1]]), "F": torch.tensor([[1, 0]])})
                    
        elif txt_modality == "age":
            num_classes = 5
            self.df[txt_modality+"_"] = pd.cut(self.df[txt_modality], bins=num_classes, labels=list(range(num_classes)))
            self.df[txt_modality+"_"] = self.df[txt_modality+"_"].apply(lambda x: one_hot(torch.tensor(x), num_classes=num_classes).float())
            
        else: 
            raise Exception(f"txt_modality {txt_modality} not supported")
        
        
    def _get_tabular_data(self, txt_modality, index) -> torch.Tensor:
        return torch.tensor(self.df.at[index, txt_modality+"_"]).float()

    # adapted from clinicadl, no img_index and sample_index
    def _get_meta_data(self, index) -> Tuple[str, str]:
        participant = self._get_participant(index)
        session = self._get_session(index)

        return participant, session

    # straight from clinicadl
    def _get_participant(self, index) -> str:
        return self.df.at[index, PARTICIPANT_ID]

    # straight from clinicadl
    def _get_session(self, index) -> str:
        return self.df.at[index, SESSION_ID]
