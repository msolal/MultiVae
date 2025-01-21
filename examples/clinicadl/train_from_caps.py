from multivae.models import MVTCAE, MVTCAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig

from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    TrainingCallback,
    WandbCallback,
)

from pathlib import Path

from multivae.data.datasets.caps_multimodal_dataset import CapsMultimodalDataset

import torch
from torch.utils.data import random_split

from multivae.metrics import LikelihoodsEvaluator


dataset = CapsMultimodalDataset(
    caps_directory=Path("/Users/maelys.solal/Documents/projects/MultiVae/examples/clinicadl/dummy_caps"),
    tsv_label=Path("/Users/maelys.solal/Documents/projects/MultiVae/examples/clinicadl/dummy_caps/subjects.tsv"),
    img_modalities=["pet_linear", "t1_linear"],
    txt_modalities=['age', 'sex'],
    # txt_modalities=['sex'],
)

train_data, val_data, test_data = random_split(
    dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42)
)

model_config = MVTCAEConfig(
    # n_modalities=2,
    n_modalities=4,
    latent_dim=20,
    
    input_dims= {
        'pet_linear': (1,28,28), 
        't1_linear': (1,28,28), 
        'age': (1, 5),
        'sex': (1, 2),
    },
    decoders_dist= {
        'pet_linear' : 'normal', 
        't1_linear' : 'normal', 
        'age' : 'categorical', 
        'sex' : 'categorical', 
    },
    
    alpha=2./3.,
    beta=2.5,
)

model = MVTCAE(model_config = model_config)

trainer_config = BaseTrainerConfig(
    num_epochs=2,
    learning_rate=1e-2,
)

# wandb_cb = WandbCallback()
# wandb_cb.setup(trainer_config, model_config, project_name="mmnist")

# callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    training_config=trainer_config,
    train_dataset=train_data, 
    eval_dataset=val_data, 
)

trainer.train()

ll_module = LikelihoodsEvaluator(
    model=model,
    output='./metrics',# where to log the metrics
    test_dataset=test_data
)
ll_module.eval()