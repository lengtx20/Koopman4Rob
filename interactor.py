from abc import ABC, abstractmethod


class InteractorBasis(ABC):
    @abstractmethod
    def add_config(self, config):
        """Add the global shared configuration to the interactor."""

    @abstractmethod
    def add_first_batch(self, batch):
        """Add the first batch of data from the data loader to the interactor."""

    @abstractmethod
    def update(self, prediction, batch):
        """Update with the model prediction or batch data."""

    @abstractmethod
    def get_model_input(self, last_prediction, batch):
        """Get the model input for the next iteration."""

    @abstractmethod
    def shutdown(self):
        """Shutdown the interactor and environment."""
