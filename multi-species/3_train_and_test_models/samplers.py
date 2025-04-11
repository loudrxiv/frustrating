import random

from torch.utils.data import Sampler

class EfficientSequentialSampler(Sampler):
    """
    A sequential sampler that allows controlling the number of steps per epoch
    by specifying the desired number of samples based on batch size.
    Assumes the dataset is shuffled beforehand.
    """
    def __init__(self, data_source, batch_size, total_samples=None):
        """
        Initialize the sampler.

        Args:
            data_source: Dataset to sample from
            batch_size: Size of each batch
            total_samples: Desired total number of samples to iterate through (if None, uses all samples)
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size

        # If total_samples is not specified, use the entire dataset
        if total_samples is None:
            self.total_samples = len(self.data_source)
        else:
            self.total_samples = min(total_samples, len(self.data_source))  # Ensure total_samples doesn't exceed dataset size

        # Calculate steps per epoch
        self.steps_per_epoch = self.total_samples // self.batch_size

    def __iter__(self):
        """Return an iterator over the indices."""
        indices = list(range(len(self.data_source)))

        # Limit to total_samples
        subset_indices = indices[:self.total_samples]

        # No need to shuffle inside __iter__ if the dataset is pre-shuffled
        return iter(subset_indices)

    def __len__(self) -> int:
        """Return the total number of samples that will be used per epoch."""
        return self.total_samples
    
class EfficientSequentialCyclicalSampler(Sampler):
    """
    A sequential sampler that allows controlling the number of steps per epoch
    by specifying the desired number of samples based on batch size.
    Implements cyclical sampling to improve data representativeness across epochs.
    """
    def __init__(self, data_source, batch_size, total_samples=None, start_epoch: int = 0):
        """
        Initialize the sampler.

        Args:
            data_source: Dataset to sample from
            batch_size: Size of each batch
            total_samples: Desired total number of samples to iterate through (if None, uses all samples)
            start_epoch: The epoch number to start from (default is 0)
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.epoch = start_epoch

        # If total_samples is not specified, use the entire dataset
        if total_samples is None:
            self.total_samples = len(self.data_source)
        else:
            self.total_samples = min(total_samples, len(self.data_source))  # Ensure total_samples doesn't exceed dataset size

        # Calculate steps per epoch
        self.steps_per_epoch = self.total_samples // self.batch_size

    def __iter__(self):
        """Return an iterator over the indices, implementing cyclical sampling."""
        indices = list(range(len(self.data_source)))

        # Calculate the starting offset based on the current epoch
        start_index = (self.epoch * self.total_samples) % len(self.data_source)

        # Get the subset of indices for the current epoch with wrap-around
        subset_indices = indices[start_index:] + indices[:start_index]
        subset_indices = subset_indices[:self.total_samples]  # Limit to total_samples

        random.shuffle(subset_indices)
        self.epoch += 1
        print(f"\n--- Train Sampler will be at epoch {self.epoch} next ---\n")
        return iter(subset_indices)

    def __len__(self) -> int:
        """Return the total number of samples that will be used per epoch."""
        return self.total_samples

    def set_epoch(self, epoch: int):
        """
        Set the current epoch number. Useful for resuming training from a checkpoint.

        Args:
            epoch: The epoch number to set.
        """
        self.epoch = epoch

class EpochBasedSampler(Sampler):
    def __init__(self, data_source, indices_per_epoch, current_epoch=0):
        self.data_source        = data_source
        self.indices_per_epoch  = indices_per_epoch
        self.current_epoch      = current_epoch # Track the current epoch
        self.total_length       = len(data_source)

    def set_epoch(self, epoch):
        """Set the current epoch, it starts at 0"""
        self.current_epoch = epoch

    def __iter__(self):
        start_index = self.current_epoch * self.indices_per_epoch
        end_index = min((self.current_epoch + 1) * self.indices_per_epoch, self.total_length) # Handle case at the end of the dataset
        
        indices = list(range(start_index, end_index))
        
        if self.shuffle:
          indices = torch.randperm(len(indices)).tolist() # Shuffle within the range
        return iter(indices)


    def __len__(self):
      """Returns how many index will be in the sample"""
      start_index = self.current_epoch * self.indices_per_epoch
      end_index = min((self.current_epoch + 1) * self.indices_per_epoch, self.total_length) # Handle case at the end of the dataset
      return end_index - start_index