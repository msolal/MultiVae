import importlib
import inspect
import logging
import os
import shutil
import sys
import tempfile
import warnings
from copy import deepcopy
from http.cookiejar import LoadError
from typing import Union

import cloudpickle
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from pythae.models.base.base_utils import CPU_Unpickler, ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from torch.nn import functional as F

from ...data.datasets.base import MultimodalBaseDataset
from ..auto_model import AutoConfig
from ..nn.default_architectures import BaseDictDecoders, BaseDictEncoders
from .base_config import BaseMultiVAEConfig, EnvironmentConfig
from .base_model import BaseModel
from .base_utils import hf_hub_is_available, model_card_template

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseMultiVAE(BaseModel):
    """Base class for Multimodal VAE models.

    Args:
        model_config (BaseMultiVAEConfig): An instance of BaseMultiVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.


    """

    def __init__(
        self,
        model_config: BaseMultiVAEConfig,
        encoders: dict = None,
        decoders: dict = None,
    ):
        nn.Module.__init__(self)

        self.model_name = "BaseMultiVAE"
        self.model_config = model_config
        self.n_modalities = model_config.n_modalities
        self.input_dims = model_config.input_dims
        self.multiple_latent_spaces = False  # Default value, this field must be changes
        # in models using multiple latent spaces

        if encoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide encoders or input dims for the modalities in the model_config."
                )
            else:
                if len(self.input_dims.keys()) != self.n_modalities:
                    raise AttributeError(
                        f"The provided number of input_dims {len(self.input_dims.keys())} doesn't"
                        f"match the number of modalities ({self.n_modalities} in model config "
                    )
                encoders = self.default_encoders(model_config)
        else:
            self.model_config.custom_architectures.append("encoders")

        if decoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide decoders or input dims for the modalities in the model_config."
                )
            else:
                if len(self.input_dims.keys()) != self.n_modalities:
                    raise AttributeError(
                        f"The provided number of input_dims {len(self.input_dims.keys())} doesn't"
                        f"match the number of modalities ({self.n_modalities} in model config "
                    )
                decoders = self.default_decoders(model_config)
        else:
            self.model_config.custom_architectures.append("decoders")

        self.sanity_check(encoders, decoders)

        self.latent_dim = model_config.latent_dim
        self.model_config = model_config
        self.device = None

        self.set_decoders(decoders)
        self.set_encoders(encoders)

        self.modalities_name = list(self.decoders.keys())

        # Check that the modalities' name are coherent
        if self.input_dims is not None:
            if self.input_dims.keys() != self.encoders.keys():
                raise KeyError(
                    f"Warning! : The modalities names in model_config.input_dims : {list(self.input_dims.keys())}"
                    f" does not match the modalities names in encoders : {list(self.encoders.keys())}"
                )

        self.use_likelihood_rescaling = model_config.uses_likelihood_rescaling
        if self.use_likelihood_rescaling:
            if self.model_config.rescale_factors is not None:
                self.rescale_factors = model_config.rescale_factors
            elif self.input_dims is None:
                raise AttributeError(
                    " inputs_dim = None but (use_likelihood_rescaling = True"
                    " in model_config)"
                    " To compute likelihood rescalings we need the input dimensions."
                    " Please provide a valid dictionary for input_dims."
                )
            else:
                max_dim = max(*[np.prod(self.input_dims[k]) for k in self.input_dims])
                self.rescale_factors = {
                    k: max_dim / np.prod(self.input_dims[k]) for k in self.input_dims
                }
        else:
            self.rescale_factors = {k: 1 for k in self.encoders}
            # above, we take the modalities keys in self.encoders as input_dims may be None

        # Set the reconstruction losses
        if model_config.decoders_dist is None:
            model_config.decoders_dist = {k: "normal" for k in self.encoders}
        if model_config.decoder_dist_params is None:
            model_config.decoder_dist_params = {}
        self.set_decoders_dist(
            model_config.decoders_dist, deepcopy(model_config.decoder_dist_params)
        )

    def set_decoders_dist(self, recon_dict, dist_params_dict):
        """Set the reconstruction losses functions decoders_dist
        and the log_probabilites functions recon_log_probs.
        recon_log_probs is the normalized negative version of recon_loss and is used only for
        likelihood estimation.
        """
        self.recon_log_probs = {}

        for k in recon_dict:
            if recon_dict[k] == "normal":
                params_mod = dist_params_dict.pop(k, {})
                scale = params_mod.pop("scale", 1.0)
                self.recon_log_probs[k] = lambda input, target: dist.Normal(
                    input, scale
                ).log_prob(target)

            elif recon_dict[k] == "bernoulli":
                self.recon_log_probs[k] = lambda input, target: dist.Bernoulli(
                    logits=input
                ).log_prob(target)

            elif recon_dict[k] == "laplace":
                params_mod = dist_params_dict.pop(k, {})
                scale = params_mod.pop("scale", 1.0)
                self.recon_log_probs[k] = lambda input, target: dist.Laplace(
                    input, scale
                ).log_prob(target)

            elif recon_dict[k] == "categorical":
                self.recon_log_probs[k] = lambda input, target: cross_entropy(
                    input, target
                )

        # TODO : add the possibility to provide custom reconstruction loss and in that case use the negative
        # reconstruction loss as the log probability.

    def sanity_check(self, encoders, decoders):
        if self.n_modalities != len(encoders.keys()):
            raise AttributeError(
                f"The provided number of encoders {len(encoders.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )

        if self.n_modalities != len(decoders.keys()):
            raise AttributeError(
                f"The provided number of decoders {len(decoders.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )

        if encoders.keys() != decoders.keys():
            raise AttributeError(
                "The names of the modalities in the encoders dict doesn't match the names of the modalities"
                " in the decoders dict."
            )

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ) -> ModelOutput:
        """
        Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.

        """

        # If the input cond_mod is a string : convert it to a list
        if type(cond_mod) == str:
            if cond_mod == "all":
                cond_mod = list(self.encoders.keys())
            elif cond_mod in self.encoders.keys():
                cond_mod = [cond_mod]
            else:
                raise AttributeError(
                    'If cond_mod is a string, it must either be "all" or a modality name'
                    f" The provided string {cond_mod} is neither."
                )

        ignore_incomplete = kwargs.pop("ignore_incomplete", False)
        # Deal with incomplete datasets
        if hasattr(inputs, "masks") and not ignore_incomplete:
            # Check that all modalities in cond_mod are available for all samples points.
            mods_avail = torch.tensor(True)
            for m in cond_mod:
                mods_avail = torch.logical_and(mods_avail, inputs.masks[m])
            if not torch.all(mods_avail):
                raise AttributeError(
                    "You tried to encode a incomplete dataset conditioning on",
                    f"modalities {cond_mod}, but some samples are not available"
                    "in all those modalities.",
                )
        return ModelOutput(cond_mod=cond_mod)

    def decode(self, embedding: ModelOutput, modalities: Union[list, str] = "all"):
        """Decode a latent variable z in all modalities specified in modalities.

        Args:
            embedding (ModelOutput): contains the latent variables. It must have the same format as the
                output of the encode function.
            modalities (Union(List, str), Optional): the modalities to decode from z. Default to 'all'.
        Return
            ModelOutput : containing a tensor per modality name.
        """
        self.eval()
        if modalities == "all":
            modalities = list(self.decoders.keys())
        elif type(modalities) == str:
            modalities = [modalities]

        if embedding.one_latent_space:
            z = embedding.z
            outputs = ModelOutput()
            if len(z.shape) == 3:
                N, bs, ldim = z.shape
                z = z.view(N * bs, ldim)
                for m in modalities:
                    recon = self.decoders[m](z).reconstruction
                    outputs[m] = recon.reshape(N, bs, *recon.shape[1:])
            else:
                for m in modalities:
                    outputs[m] = self.decoders[m](z).reconstruction
            return outputs
        else:
            z_content = embedding.z
            outputs = ModelOutput()
            for m in modalities:
                z = torch.cat([z_content, embedding.modalities_z[m]], dim=-1)
                if len(z.shape) == 3:
                    N, bs, ldim = z.shape
                    recon = self.decoders[m](z.view(N * bs, ldim)).reconstruction
                    outputs[m] = recon.reshape(N, bs, *recon.shape[1:])
                else:
                    outputs[m] = self.decoders[m](z).reconstruction
            return outputs

    def predict(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        gen_mod: Union[list, str] = "all",
        N: int = 1,
        flatten: bool = False,
        **kwargs,
    ):
        """Generate in all modalities conditioning on a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to condition on. It must contain at least the modalities
                contained in cond_mod.
            cond_mod (Union[list, str], optional): The modalities to condition on. Defaults to 'all'.
            gen_mod (Union[list, str], optional): The modalities to generate. Defaults to 'all'.
            N (int) : Number of samples to generate. Default to 1.
            flatten (int) : If N>1 and flatten is False, the returned samples have dimensions (N,len(inputs),...).
                Otherwise, the returned samples have dimensions (len(inputs)*N, ...)

        Returns:
            ~pythae.models.base.base_utils.ModelOutput

        ..codeblock :
            >>> predictions = model.predict(test_set, cond_mod = ['modality1', 'modality2'], gen_mod='modality3')
            >>> predictions.modality3


        """
        self.eval()
        ignore_incomplete = kwargs.pop("ignore_incomplete", False)
        z = self.encode(
            inputs,
            cond_mod,
            N=N,
            flatten=flatten,
            ignore_incomplete=ignore_incomplete,
            **kwargs,
        )
        output = self.decode(z, gen_mod)
        # n_data = len(z.z) // N
        # if not flatten and N > 1:
        #     for m in output.keys():
        #         output[m] = output[m].reshape(N, n_data, *output[m].shape[1:])
        return output

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """
        Main forward pass outputing the VAE outputs
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs

        Args:
            inputs (BaseDataset): The training data with labels, masks etc...

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.

        .. note::
            The loss must be computed in this forward pass and accessed through
            ``loss = model_output.loss``
        """
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch)

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    def default_encoders(self, model_config) -> nn.ModuleDict:
        return BaseDictEncoders(self.input_dims, model_config.latent_dim)

    def default_decoders(self, model_config) -> nn.ModuleDict:
        return BaseDictDecoders(self.input_dims, model_config.latent_dim)

    def set_encoders(self, encoders: dict) -> None:
        """Set the encoders of the model"""
        self.encoders = nn.ModuleDict()
        for modality in encoders:
            encoder = encoders[modality]
            if not issubclass(type(encoder), BaseEncoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, encoder must inherit from BaseEncoder class from "
                        "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                    )
                )

            self.encoders[modality] = encoder

    def set_decoders(self, decoders: dict) -> None:
        """Set the decoders of the model"""
        self.decoders = nn.ModuleDict()
        for modality in decoders:
            decoder = decoders[modality]
            if not issubclass(type(decoder), BaseDecoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, decoder must inherit from BaseDecoder class from "
                        "pythae.models.base_architectures.BaseDecoder. Refer to documentation."
                    )
                )
            self.decoders[modality] = decoder

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        raise NotImplementedError

    def compute_cond_nll(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: str,
        pred_mods: list,
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Compute the conditional likelihoods ln p(x|y) , ln p(y|x) with MonteCarlo Sampling and the approximation :

                ln p(x|y) = \sum_{z ~ q(z|y)} ln p(x|z)

        Args:
            inputs (MultimodalBaseDataset): the data to compute the likelihood on.
            cond_mod (str): the modality to condition on
            gen_mod (str): the modality to condition on
            K (int, optional): number of samples per batch. Defaults to 1000.
            batch_size_K (int, optional): _description_. Defaults to 100.

        Returns:
            dict: _description_
        """

        # Compute K samples for each datapoint
        o = self.encode(inputs, cond_mod, N=K)

        # Compute the negative recon_log_prob for each datapoint
        ll = {k: [] for k in pred_mods}

        n_data = len(inputs.data[list(inputs.data.keys())[0]])
        for i in range(n_data):
            start_idx, stop_index = 0, batch_size_K
            lnpxs = {k: [] for k in pred_mods}

            while stop_index <= K:
                # Encode with the conditional VAE
                latents = o.z[start_idx:stop_index]

                # Decode with the opposite decoder
                for k in pred_mods:
                    target = inputs.data[k][i]
                    recon = self.decoders[k](
                        latents
                    ).reconstruction  # (batch_size,*recon_shape)
                    # Compute lnp(y|z)
                    lpxz = (
                        self.recon_log_probs[k](target, recon)
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )
                    lnpxs[k].append(torch.logsumexp(lpxz, dim=0))

                # next batch
                start_idx += batch_size_K
                stop_index += batch_size_K
            for k in pred_mods:
                ll[k].append(torch.logsumexp(torch.tensor(lnpxs[k]), dim=0) - np.log(K))

        results = {}
        for k in pred_mods:
            results["ll_" + cond_mod + "_" + k] = torch.sum(torch.tensor(ll[k])) / len(
                ll[k]
            )

        return ModelOutput(**results)

    def generate_from_prior(self, n_samples, **kwargs):
        """
        Generate latent samples from the prior distribution.
        This is the base class in which we consider a static standard Normal Prior.
        This may be overwritten in subclasses.
        """
        sample_shape = (
            [n_samples, self.latent_dim] if n_samples > 1 else [self.latent_dim]
        )
        z = dist.Normal(0, 1).rsample(sample_shape).to(self.device)
        return ModelOutput(z=z, one_latent_space=True)

    def cond_nll_from_subset(
        self,
        inputs: MultimodalBaseDataset,
        subset: Union[list, tuple],
        pred_mods: Union[list, tuple],
        K=1000,
        batch_size_k=100,
    ):
        """
        Compute the negative log-likelihoods of the conditional generative models : encoding
        from one / a subset of modalities to generate others.
        log p(x_i|x_j,x_l) = log 1/K \sum_k p(x_i|z_k).


        Args:
            inputs (MultimodalBaseDataset): The inputs to use for the estimation.
            subset (list, tuple): The modalities to take as inputs.
            pred_mods (list, tuple): The modalities to take into account in the prediction.
            K (int, optional): The number of samples. Defaults to 1000.
            batch_size_k (int, optional): Batch size to use. Defaults to 100.
        """

        cnll = {m: [] for m in pred_mods}

        # Encode using the subset modalities
        nb_computed_samples = 0

        while nb_computed_samples < K:
            n_samples = min(batch_size_k, K - nb_computed_samples)
            encode_output = self.encode(inputs, subset, N=n_samples)
            if encode_output.one_latent_space:
                decode_output = self.decode(encode_output, pred_mods)

                for mod in pred_mods:
                    recon = decode_output[mod]  # (batch_size_k, n_data, *recon_size )
                    target = inputs.data[mod]
                    lpxz = self.recon_log_probs[mod](recon, target)
                    cnll[mod].append(torch.logsumexp(lpxz, dim=0))

            nb_computed_samples += n_samples

        cnll = {m: torch.logsumexp(torch.stack(cnll[mod]), dim=0) for m in cnll}
        return cnll
