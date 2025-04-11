# Paths, libraries, and options ================================================

# Libraries & paths
import os

import sys
sys.path.append(f"../../..")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import Tensor
from einops import rearrange
from typing import Callable, List, Optional, Union
from params import Params, ROOT
from generators import TrainGenerator_MultiSpecies, ValGenerator_MultiSpecies, TestGenerator_SingleSpecies
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss
from math import floor

# gReLU "Genomics" Zoo!
import grelu.resources

from grelu.model.blocks import ChannelTransformBlock, LinearBlock
from grelu.model.layers import AdaptivePool
from grelu.model.layers import (
    Activation,
    Attention,
    ChannelTransform,
    Crop,
    Dropout,
    Norm,
    Pool,
)

# Model Declarations ==========================================================

class ConvHead(nn.Module):
    """
    A 1x1 Conv layer that transforms the the number of channels in the input and then
    optionally pools along the length axis.

    Args:
        n_tasks: Number of tasks (output channels)
        in_channels: Number of channels in the input
        norm: If True, batch normalization will be included.
        act_func: Activation function for the convolutional layer
        pool_func: Pooling function.
        dtype: Data type for the layers.
        device: Device for the layers.
    """

    def __init__(
        self,
        n_tasks: int,
        in_channels: int,
        act_func: Optional[str] = None,
        pool_func: Optional[str] = None,
        norm: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        # Save all params
        self.n_tasks = n_tasks
        self.in_channels = in_channels
        self.act_func = act_func
        self.pool_func = pool_func
        self.norm = norm

        # Create layers
        self.channel_transform = ChannelTransformBlock(
            self.in_channels,
            self.n_tasks,
            act_func=self.act_func,
            norm=self.norm#,
            # dtype=dtype,
            # device=device
        )
        self.pool = AdaptivePool(self.pool_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input data.
        """
        x = self.channel_transform(x)
        x = self.pool(x)
        return x

class LinearBlock(nn.Module):
    """
    Linear layer followed by optional normalization,
    activation and dropout.

    Args:
        in_len: Length of input
        out_len: Length of output
        act_func: Name of activation function
        dropout: Dropout probability
        norm: If True, apply layer normalization
        bias: If True, include bias term.
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        in_len: int,
        out_len: int,
        act_func: str = "relu",
        dropout: float = 0.0,
        norm: bool = False,
        bias: bool = True,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.norm = Norm(
            func="layer" if norm else None, in_dim=in_len, dtype=dtype, device=device
        )
        self.linear = nn.Linear(in_len, out_len, bias=bias, dtype=dtype, device=device)
        self.dropout = Dropout(dropout)
        self.act = Activation(act_func)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.act(x)
        return x
    
class FeedForwardBlock(nn.Module):
    """
    2-layer feed-forward network. Can be used to follow layers such as GRU and attention.

    Args:
        in_len: Length of the input tensor
        dropout: Dropout probability
        act_func: Name of the activation function
        kwargs: Additional arguments to be passed to the linear layers
    """

    def __init__(
        self,
        in_len: int,
        dropout: float = 0.0,
        act_func: str = "relu",
        **kwargs,
    ) -> None:
        super().__init__()
        self.dense1 = LinearBlock(
            in_len,
            in_len * 2,
            norm=True,
            dropout=dropout,
            act_func=act_func,
            bias=True,
            **kwargs,
        )
        self.dense2 = LinearBlock(
            in_len * 2,
            in_len,
            norm=False,
            dropout=dropout,
            act_func=None,
            bias=True,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
class GRUBlock(nn.Module):
    """
    Stacked bidirectional GRU layers followed by a feed-forward network.

    Args:
        in_channels: The number of channels in the input
        n_layers: The number of GRU layers
        gru_hidden_size: Number of hidden elements in GRU layers
        dropout: Dropout probability
        act_func: Name of the activation function for feed-forward network
        norm: If True, include layer normalization in feed-forward network.
        dtype: Data type of the weights
        device: Device on which to store the weights
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 1,
        dropout: float = 0.0,
        act_func: str = "relu",
        norm: bool = False,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=in_channels,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
            num_layers=n_layers,
            dtype=dtype,
            device=device,
        )
        self.ffn = FeedForwardBlock(
            in_len=in_channels,
            dropout=dropout,
            act_func=act_func,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        Args:
            x : Input tensor of shape (N, C, L)

        Returns:
            Output tensor
        """
        x = rearrange(x, "b t l -> b l t")
        x = self.gru(x)[0]
        # Combine output of forward and reverse GRU
        x = x[:, :, : self.gru.hidden_size] + x[:, :, self.gru.hidden_size :]
        x = self.ffn(x)
        x = rearrange(x, "b l t -> b t l")
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, params):
        super(FeatureExtractor, self).__init__()
        self.conv1          = nn.Conv1d(in_channels=4, out_channels=params.convfilters, kernel_size=params.filtersize, padding="same")
        self.pool           = nn.MaxPool1d(kernel_size=params.pool_size+1, stride=params.strides+1, padding=params.pool_size // 2)
        self.gru_tower      = GRUBlock(
            in_channels=params.convfilters,
            n_layers=1,
            dropout= 0.0,
            act_func="relu",
            norm=False,
            device=None,
            dtype=None
        )
        self.pooled_embedding=ConvHead(
            n_tasks=(params.lstmnodes*2)-1,
            in_channels=params.convfilters,
            pool_func="avg",
            act_func=None,
            norm=False,
            dtype=None,
            device=None
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)                   # -> [B, C, S]
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        embedding = self.gru_tower(x)
        pooled_embedding = self.pooled_embedding(embedding)
        return pooled_embedding

class Classifier(nn.Module):
    def __init__(self, params):
        super(Classifier, self).__init__()
        self.head=ConvHead(
            n_tasks=1,
            in_channels=(params.lstmnodes*2)-1,
            pool_func="avg",
            act_func=None,
            norm=False,
            dtype=None,
            device=None
        )
    
    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.head(x))
        return x
    
# Function Declarations =======================================================

def get_model_files(tf, test_species, num_holdout):
    # This function returns the filepath where the model for a given
    # TF, training species, and run is saved.
    # By default, the file for the best model across all training epochs
    # is returned and we grab the latest trained model.

    model_path = ROOT + "/".join(["/models", tf, test_species + "_tested", f"EvoPS-{num_holdout}/"])
    classifier_suffix = ".classifier.pt"
    feature_extractor_suffix = ".feature_extractor.pt"
    
    # get all files that match the prefix and suffix
    classifier_files        = [f for f in os.listdir(model_path) if f.endswith(classifier_suffix)]
    feature_extractor_files = [f for f in os.listdir(model_path) if f.endswith(feature_extractor_suffix)]
    
    # sort files and return the one that is most recent
    latest_classifier_file          = max([model_path + f for f in classifier_files], key=os.path.getctime)
    latest_feature_extractor_file   = max([model_path + f for f in feature_extractor_files], key=os.path.getctime)

    return latest_classifier_file, latest_feature_extractor_file

def get_preds_file(tf, test_species, num_holdout):
    preds_root = ROOT + "/model_out"
    os.makedirs(preds_root, exist_ok=True)
    return f"{preds_root}/EvoPS-{num_holdout}_{tf}_{test_species}-tested.preds"

def get_labels_file(tf, test_species, num_holdout):
    preds_root = ROOT + "/model_out"
    os.makedirs(preds_root, exist_ok=True)
    return f"{preds_root}/EvoPS-{num_holdout}_{tf}_{test_species}-tested.labels"

def test_collate(batch):
    data    = torch.cat([dict_item['sequence'] for dict_item in batch]).float()
    label   = torch.stack([dict_item['label'] for dict_item in batch]).int()
    
    return {
        "sequence": data,
        "label": label
    }

# Main =========================================================================

if __name__ == "__main__":

    SAVE = True

    seed = 1182024

    torch.manual_seed(seed)
    
    np.random.seed(seed)

	#--- (0) Set up device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nUsing device: {device}")

	#--- (1) Set up parameters and dataloders

    test_file, tf, target, _lambda, match_mean, holdout = sys.argv

    # Convert hyperparameters to appropriate types
    _lambda     = float(_lambda)
    match_mean  = bool(int(match_mean))
    holdout     = int(holdout)

    # Define relatedness
    params  = Params(args = [f"EvoPS-{holdout}", tf, target], verbose=True)

    # Modulate the number of source species based on relatedness
    if params.target_species == "hg38":
        evolutionary_relatedness    = ["rheMac10", "mm10", "rn7", "canFam6"]
    elif params.target_species == "rheMac10":
        evolutionary_relatedness    = ["hg38", "mm10", "rn7", "canFam6"]
    elif params.target_species == "mm10":
        evolutionary_relatedness    = ["rn7", "rheMac10", "hg38", "canFam6"]
    elif params.target_species == "rn7":
        evolutionary_relatedness    = ["mm10", "rheMac10", "hg38", "canFam6"]
    elif params.target_species == "canFam6":
        evolutionary_relatedness    = ["hg38", "rheMac10", "mm10", "rn7"]

    # Modulate the number of source species based on relatedness
    print(f"\nHolding out {evolutionary_relatedness[holdout]} from the source species.")

    params.source_species       = list(filter(lambda x: x not in evolutionary_relatedness[holdout], params.source_species))

    print(f"Source species: {params.source_species}")

    print("\n---------------------------------------\n")

    # Get both model files
    classifier_file, feature_extractor_file = get_model_files(tf, target, holdout)
    preds_file                              = get_preds_file(tf, target, holdout)
    labels_file                             = get_labels_file(tf, target, holdout)

	#---  (2) Define model, optimizer and load weights

    # Feature Extractor
    feature_extractor = FeatureExtractor(params)
    feature_extractor = feature_extractor.to(device)
    feature_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"Feature Extractor Architecture:\n{feature_extractor}\n")

    # Classifier
    classifier = Classifier(params)
    classifier = classifier.to(device)
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier Architecture:\n{classifier}\n")

    # Params count
    total_params = feature_params + classifier_params
    print(f"Total number of parameters: {total_params}")

    # Load weights
    print(f"Loading feature extractor from {feature_extractor_file}\n")
    feature_extractor.load_state_dict(torch.load(feature_extractor_file))

    print(f"Loading classifier from {classifier_file}\n")
    classifier.load_state_dict(torch.load(classifier_file))

    #--- (3) Set up dataloader

    teg_ss      = TestGenerator_SingleSpecies(params=params, percent_to_batch=1.0)

    test_loader = DataLoader(
        dataset=teg_ss,
        batch_size=10000,
        pin_memory=True,
        collate_fn=test_collate
    )

    #--- (4) Generate predictions

    test_bar    = tqdm(enumerate(test_loader), total=len(test_loader))

    with torch.no_grad():

        print(f"Generating predictions for {target}-on-{target}.\n")

        # Turn on evaluation mode
        feature_extractor.eval()
        classifier.eval()

        # A list to store all the probabilities
        test_results = {"probs": [], "labels":[]}

        for batch_idx, data in test_bar:
                curr_data = torch.tensor(data['sequence'], dtype=torch.float32).to(device)
                test_results['probs'].extend(classifier(feature_extractor(curr_data)).detach().cpu().numpy().tolist())
                test_results['labels'].extend(data['label'].detach().cpu().numpy().tolist())

        if SAVE:
            print(f"Saving predictions to {preds_file}")
            
            preds = np.array(test_results['probs'], dtype=np.float32).T
            np.save(preds_file, preds)
            
            labels = np.array(test_results['labels'], dtype=np.int32).T
            np.save(labels_file, labels)