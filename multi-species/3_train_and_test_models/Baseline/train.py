#===============================================================================

# Libraries & paths
import sys
sys.path.append(f"..")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from params import Params, ROOT
from torch.utils.data import DataLoader
from tqdm import tqdm
from generators import TrainGenerator_SingleSpecies, ValGenerator_SingleSpecies
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, log_loss

# For bidirectional GRU
from torch import Tensor
from einops import rearrange
from typing import List, Optional
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

# Constants
SAVE        = True

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

def print_val_metrics(val_results):

	print("\n==== Baseline Validation ====")
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

def train_collate(batch):
    data_binding_positive   = torch.cat([item['sequence']['binding']['positive'].reshape(1, 1000, 4) for item in batch], dim=0).float()
    data_binding_negative   = torch.cat([item['sequence']['binding']['negative'].reshape(1, 1000, 4) for item in batch], dim=0).float()
    data_background         = torch.cat([item['sequence']['background'].reshape(1, 1000, 4) for item in batch], dim=0).float()

    label_binding_positive  = torch.from_numpy(np.array([item['label']['binding']['positive'] for item in batch])).int()
    label_binding_negative  = torch.from_numpy(np.array([item['label']['binding']['negative'] for item in batch])).int()
    label_background        = torch.from_numpy(np.array([item['label']['background'] for item in batch])).int()

    index_binding_positive  = torch.from_numpy(np.array([item['index']['binding']['positive'] for item in batch])).int()
    index_binding_negative  = torch.from_numpy(np.array([item['index']['binding']['negative'] for item in batch])).int()
    index_background        = torch.from_numpy(np.array([item['index']['background'] for item in batch])).int()
    
    return {
        "sequence": {
            "binding": {
                "positive": data_binding_positive.float(),
                "negative": data_binding_negative.float()
            },
            "background": data_background.float()
        },
        "label": {
            "binding": {
                "positive": label_binding_positive.int(),
                "negative": label_binding_negative.int()
            },
            'background': label_background.int()
        },
        "index": {
            "binding": {
                "positive": index_binding_positive.int(),
                "negative": index_binding_negative.int()
            },
            'background': index_background.int()
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
    
class BasicModel(nn.Module):
    def __init__(self, params):
        super(BasicModel, self).__init__()
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
        x = x.transpose(1, 2) # -> [B, C, S]
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        embedding = self.gru_tower(x) # this is the embedding!
        pooled_embedding = self.pooled_embedding(embedding)
        x = torch.nn.functional.sigmoid(self.head(pooled_embedding))
        return x
	
#===============================================================================

if __name__ == "__main__":

	#--- (0) Set up device

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	print(f"\nUsing device: {device}")

	#--- (1) Set up parameters
	
	train_file, tf, target_species = sys.argv
	params  = Params(args = ["Baseline", tf, target_species], verbose=True)
	print("\n---------------------------------------\n")

	#--- (2) Specify model, optimizer and peak parameter count

	basic_model = BasicModel(params)
	basic_model = basic_model.to(device)

	optimizer = optim.Adam(
		[
			{'params': basic_model.parameters()}
		],
		lr=1e-3
	)

	# We print the model
	print("\nModel:")
	print(basic_model)

	total_params = sum(p.numel() for p in basic_model.parameters())
	print(f"Total number of parameters: {total_params}")

	#--- (3) Set up data loaders
 
	tg_ss			= TrainGenerator_SingleSpecies(params=params, percent_to_batch=0.25)
	train_loader	= DataLoader(
		dataset=tg_ss,
		batch_size=200,
		pin_memory=True,
		collate_fn=train_collate
	)

	vg_ss				= ValGenerator_SingleSpecies(params=params, percent_to_batch=0.5)
	validation_loader	= DataLoader(
		dataset=vg_ss,
		batch_size=10000,
		pin_memory=True,
		collate_fn=val_collate
	)

	#--- (4) Now start our training loop

	auprcs      = []
	num_epochs  = params.epochs
	seed        = 1182024

	torch.manual_seed(seed)

	np.random.seed(seed)

	for epoch in range(0, num_epochs):

		print(f"\nEpoch: {epoch+1}\n")

		#--- Prepare model for training

		basic_model.train()
		
		train_loader.dataset.data.set_epoch(epoch) # We look at different subsets of data based on the epoch!
		train_bar  = tqdm(enumerate(train_loader), total=len(train_loader))

		val_bar    = tqdm(enumerate(validation_loader), total=len(validation_loader))
		
		for batch_idx, data in train_bar:

			# (1) Process data
			seq_binding             = torch.cat([
				data['sequence']['binding']['positive'].to(device),
				data['sequence']['binding']['negative'].to(device)
			], dim=0)

			label_binding           = torch.cat([
				data['label']['binding']['positive'].to(device),
				data['label']['binding']['negative'].to(device)
			], dim=0)

			# (2) Run through the feature extractor

			binding_pred = torch.squeeze(basic_model(seq_binding))
			
			assert binding_pred.shape[0] == seq_binding.shape[0], (binding_pred.shape, seq_binding.shape[0])

			loss_pred	= nn.BCELoss()(binding_pred, label_binding.float())
			loss		= loss_pred

			# (3) Update information
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if batch_idx % 500 == 0:
				print(f"Step: {batch_idx} BCE Loss: {loss_pred.item()} Total loss: {loss.item()}")

			# # Explicitly delete tensors after they're used
			# del seq_binding, label_binding, binding_pred, loss_pred, loss
			# torch.cuda.empty_cache() # Crucial:  Empty the CUDA cache

		#--- Prepare model for validation

		with torch.no_grad():

			basic_model.eval()

			# A list to store all the probabilities
			val_results = {"probs": [], "labels":[]}

			for batch_idx, data in val_bar:
				curr_data = torch.tensor(data['sequence'], dtype=torch.float32).to(device)

				val_results['probs'].extend(basic_model(curr_data).cpu().numpy().tolist())
				val_results['labels'].extend(data['label'].cpu().numpy().tolist())

				# del curr_data # Delete after use!
				# torch.cuda.empty_cache()

			target_auprc = print_val_metrics(val_results)

			current_auprcs = auprcs

			if len(current_auprcs) == 0 or target_auprc > max(current_auprcs):
				print("Best model so far! (target species) validation auPRC = ", target_auprc)
				if SAVE:
					torch.save(basic_model.state_dict(), f"{params.modelfile}.baseline.pt")

			current_auprcs.append(target_auprc)
			auprcs = current_auprcs