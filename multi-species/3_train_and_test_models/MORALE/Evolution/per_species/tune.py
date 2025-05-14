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
from generators import TrainGenerator_MultiSpecies, ValGenerator_SingleSpecies, TestGenerator_SingleSpecies
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

#===============================================================================

def print_confusion_matrix(y, probs, threshold = 0.5):
	''' This method prints a confusion matrix to stdout, as well as
		the precision and recall when the predicted probabilities
		output by the model are threshold-ed at 0.5. Because this
		threshold is somewhat arbitrary, other metrics of performance
		such as the auPRC of the loss are more informative single
		quantities, but the confusion matrix can be useful to look at
		to show relative false positive and false negative rates.
	'''
	npthresh = np.vectorize(lambda t: 1 if t >= threshold else 0)
	pred = npthresh(probs)
	conf_matrix = confusion_matrix(y, pred)
	print("Confusion Matrix at t = 0.5:\n", conf_matrix)
	# precision = TP / (TP + FP)
	print("Precision at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1]))
	# recall = TP / (TP + FN)
	print("Recall at t = 0.5:\t", conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0]), "\n")

def print_val_metrics(params, val_results):

	print(f"\n==== Target ({params.target_species}) Validation ====")
	labels = np.array(val_results['labels'])
	probs = np.array(val_results['probs'])
	probs = probs.squeeze()

	assert labels.shape == probs.shape, (labels.shape, probs.shape)

	print("AUC:\t", roc_auc_score(labels, probs))

	auPRC = average_precision_score(labels, probs)
	print("auPRC:\t", auPRC)

	loss = log_loss(labels, probs)  # this is binary cross-entropy
	print("Loss:\t", loss)

	print_confusion_matrix(labels, probs)

	return auPRC

def moment_alignment(Xs, Xt, _match_mean=False, _lambda=1.0):
    """
    https://github.com/antoinedemathelin/adapt/blob/master/adapt/feature_based/_deepcoral.py
    Converted from TensorFlow to PyTorch.
    """
    
    EPS = torch.finfo(torch.float32).eps

    batch_size = float(Xs.size(0))  # Use float for consistency with original factors

    factor_1 = 1. / (batch_size - 1. + EPS)
    factor_2 = 1. / batch_size

    sum_src = torch.sum(Xs, dim=0)
    sum_src_row = sum_src.unsqueeze(0)  # Reshape to (1, -1)
    sum_src_col = sum_src.unsqueeze(1)  # Reshape to (-1, 1)

    cov_src = factor_1 * (
        torch.matmul(Xs.transpose(0, 1), Xs) -
        factor_2 * torch.matmul(sum_src_col, sum_src_row)
    )

    sum_tgt = torch.sum(Xt, dim=0)
    sum_tgt_row = sum_tgt.unsqueeze(0)  # Reshape to (1, -1)
    sum_tgt_col = sum_tgt.unsqueeze(1)  # Reshape to (-1, 1)

    cov_tgt = factor_1 * (
        torch.matmul(Xt.transpose(0, 1), Xt) -
        factor_2 * torch.matmul(sum_tgt_col, sum_tgt_row)
    )

    mean_src = torch.mean(Xs, dim=0)
    mean_tgt = torch.mean(Xt, dim=0)

    disc_loss_cov = 0.25 * torch.square(cov_src - cov_tgt)
    disc_loss_mean = torch.square(mean_src - mean_tgt)

    disc_loss_cov = torch.mean(disc_loss_cov)
    disc_loss_mean = torch.mean(disc_loss_mean)
    disc_loss = _lambda * (disc_loss_cov + _match_mean * disc_loss_mean)
    
    return disc_loss

def train_collate(batch):
    data_binding_positive   = torch.cat([item['sequence']['binding']['positive'] for item in batch], dim=0).float()
    data_binding_negative   = torch.cat([item['sequence']['binding']['negative'] for item in batch], dim=0).float()
    data_background         = torch.cat([item['sequence']['background'] for item in batch], dim=0).float()

    label_binding_positive  = torch.from_numpy(np.array([item['label']['binding']['positive'] for item in batch])).flatten().int()
    label_binding_negative  = torch.from_numpy(np.array([item['label']['binding']['negative'] for item in batch])).flatten().int()
    label_background        = torch.from_numpy(np.array([item['label']['background'] for item in batch])).flatten().int()

    index_binding_positive  = torch.from_numpy(np.array([item['index']['binding']['positive'] for item in batch])).flatten().int()
    index_binding_negative  = torch.from_numpy(np.array([item['index']['binding']['negative'] for item in batch])).flatten().int()
    index_background        = torch.from_numpy(np.array([item['index']['background'] for item in batch])).flatten().int()

    return {
        "sequence": {
            "binding": {
                "positive": data_binding_positive,
                "negative": data_binding_negative
            },
            "background": data_background
        },
        "label": {
            "binding": {
                "positive": label_binding_positive,
                "negative": label_binding_negative
            },
            'background': label_background
        },
        "index": {
            "binding": {
                "positive": index_binding_positive,
                "negative": index_binding_negative
            },
            'background': index_background
        }
    }

def val_collate(batch):
    data    = torch.cat([dict_item['sequence'] for dict_item in batch]).float()
    label   = torch.stack([dict_item['label'] for dict_item in batch]).int()
    
    return {
        "sequence": data,
        "label": label
    }

#===============================================================================

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
    
#===============================================================================

if __name__ == "__main__":

    SAVE = False

    seed = 1182024

    torch.manual_seed(seed)
    
    np.random.seed(seed)

    #--- (0) Set up device

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nUsing device: {device}")

    #--- (1) Set up parameters and dataloders

    tune_file, tf, target, _lambda, match_mean, holdout = sys.argv

    # Convert hyperparameters to appropriate types
    _lambda     = float(_lambda)
    match_mean  = bool(int(match_mean))
    holdout     = int(holdout)

    # Construct the parameters
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

    tg_ms			= TrainGenerator_MultiSpecies(params=params, percent_to_batch=0.10)
    train_loader	= DataLoader(
        dataset=tg_ms,
        batch_size=floor(200 / len(params.source_species)),
        pin_memory=True,
        collate_fn=train_collate
    )

    vg_ss				= ValGenerator_SingleSpecies(params=params, percent_to_batch=0.50)
    validation_loader	= DataLoader(
        dataset=vg_ss,
        batch_size=10000,
        pin_memory=True,
        collate_fn=val_collate
    )

	#---  (2) Define model and the optimizer

    # Instantiate the models
    feature_extractor = FeatureExtractor(params)
    feature_extractor = feature_extractor.to(device)
    feature_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"Feature Extractor Architecture:\n{feature_extractor}\n")

    classifier = Classifier(params)
    classifier = classifier.to(device)
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(f"Classifier Architecture:\n{classifier}\n")

    total_params = feature_params + classifier_params
    print(f"Total number of parameters: {total_params}")

    optimizer = optim.Adam(
        (
            {'params': feature_extractor.parameters()},
            {'params': classifier.parameters()}
        ),
        lr=1e-3
    )

    #--- (3) Train loop

    patience                = 2             # Number of epochs to wait for improvement before stopping
    epochs_no_improve       = 0             # Counter for epochs without improvement
    early_stop_triggered    = False         # Flag to indicate if early stopping happened
    auprcs                  = []
    num_epochs              = params.epochs # 15

    print(f"Starting training for {num_epochs} epochs with early stopping patience={patience}")

    for epoch in range(0, num_epochs):

        print(f"\n--- Training (on epoch {epoch}) ---\n")

        # Turn on training mode
        feature_extractor.train()
        classifier.train()

        # Set the range we are looking at based on the epoch
        list(map(lambda gen: gen.set_epoch(epoch), train_loader.dataset.source_datasets))
        train_loader.dataset.target_dataset.set_epoch(epoch)

        # Lets keep track with tqdm
        train_bar  = tqdm(enumerate(train_loader), total=len(train_loader))
        val_bar    = tqdm(enumerate(validation_loader), total=len(validation_loader))

        for batch_idx, data in train_bar:

            # (1) Process data
            seq = torch.cat([
                data['sequence']['binding']['positive'].to(device),
                data['sequence']['binding']['negative'].to(device),
                data['sequence']['background'].to(device)
            ], dim=0)

            binding_label = torch.cat([
                data['label']['binding']['positive'].to(device),
                data['label']['binding']['negative'].to(device)
            ], dim=0)

            background_label = data['label']['background'].to(device)

            background_index = data['index']['background'].to(device)

            # (2) Run through the feature extractor
            seq_embedding = feature_extractor(seq)

            # (3) Split the features into binding and background
            binding_len             = seq.shape[0] - background_label.shape[0]
            binding_embedding       = seq_embedding[0:binding_len, :]
            background_embedding    = seq_embedding[binding_len:, :]

            assert binding_embedding.shape[0] == binding_label.shape[0], (binding_embedding.shape, binding_label.shape[0])
            assert background_embedding.shape[0] == background_label.shape[0], (background_embedding.shape, background_label.shape[0])

            # Extract the source and target information
            loss_morale = 0
            for i in torch.unique(background_index):
                for j in torch.unique(background_index):

                    if i == j: # same domain
                        continue

                    # Get the mask
                    mask_i = background_index == i
                    mask_j = background_index == j

                    # Define the source and target features
                    source_features = background_embedding[mask_i]
                    target_features = background_embedding[mask_j]

                    # Calculate the moment alignment loss
                    loss_morale += moment_alignment(
                        Xs=source_features.squeeze(),
                        Xt=target_features.squeeze(),
                        _match_mean=False,
                        _lambda=1.0
                    )

            # Get the predictions on the source domain
            source_pred = torch.squeeze(classifier(binding_embedding))

            # Measure loss with binary cross entropy and CORAL
            loss_pred = nn.BCELoss()(source_pred, binding_label.float())

            loss = loss_pred + loss_morale

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 500 == 0:
                print(f"Step: {batch_idx} BCE Loss: {loss_pred.item()}, MORALE Loss:  {loss_morale.item()}, Total loss: {loss.item()}")

        #--- Validation

        print(f"\n--- Validation (on epoch {epoch}) ---\n")
        
        with torch.no_grad():

            # Turn on evaluation mode
            feature_extractor.eval()
            classifier.eval()

            # A list to store all the probabilities
            val_results = {"probs": [], "labels":[]}

            for batch_idx, data in val_bar:
                curr_data = torch.tensor(data['sequence'], dtype=torch.float32).to(device)
                val_results['probs'].extend(classifier(feature_extractor(curr_data)).detach().cpu().numpy().tolist())
                val_results['labels'].extend(data['label'].detach().cpu().numpy().tolist())

            target_auprc    = print_val_metrics(params, val_results)
            current_auprcs  = auprcs

            if len(current_auprcs) == 0 or target_auprc > max(current_auprcs):
                print("Best model so far! (target species) validation auPRC = ", target_auprc)            
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Validation AUPRC did not improve for {epochs_no_improve} epoch(s). Best: {max(current_auprcs):.4f}")

            # Check for early stopping
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                early_stop_triggered = True
                break # Exit the main training loop

            current_auprcs.append(target_auprc)
            auprcs = current_auprcs

    # Report at end of training
    if not early_stop_triggered:
        print(f"\nTraining finished after {num_epochs} epochs.")
    else:
        print(f"\nTraining stopped early.")

    print(f"Best Validation AUPRC achieved: {max(current_auprcs):.4f}")
    print("Validation AUPRC history:", [f"{x:.4f}" for x in current_auprcs])