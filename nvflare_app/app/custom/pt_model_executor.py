# NVFlare Custom Model Executor (local task execution)
import torch
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.fl_constant import ReturnCode

from src.utils import get_model, get_loss, get_data_centralized
from src.pt_constants import PATH_TO_DATA_CENTRALIZED_DIR
from src.eval_utils import evaluate_accuracy



class PtModelExecutor(Executor):

    def __init__(self, model_name: str, lr=0.01, epochs=1, batch_size=32, early_stopping_rounds=1,
                 loss_fn_name='CrossEntropyLoss',
                 train_task_name=AppConstants.TASK_TRAIN, submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL):
        """
        Initialize the model executor with the model architecture and parameters.
        :param model_name: Name of the model. Can be one of the following: 'CIFAR10'.
        """
        super().__init__()
        # Parameters
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._early_stopping_rounds = early_stopping_rounds

        # Task names
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name

        # Training setup
        # Use GPU if available
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Get the model
        self._model = get_model(model_name)
        self._model.to(self._device)
        # Loss function
        self._criterion = get_loss(loss_fn_name)
        # Optimizer
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr, momentum=0.9)

        # get training and validation data (will be replaced with federated data in the future)
        self._train_loader, self._val_loader, _ = get_data_centralized(dataset_name=model_name,
                                                                       batch_size=self._batch_size,
                                                                       root_dir=PATH_TO_DATA_CENTRALIZED_DIR
                                                                       )

    def _train_model(self, fl_ctx: FLContext, weights: dict):

        self.log_info(fl_ctx, f"Training the model for {self._epochs} epochs.")
        # Set model weights and set into training mode
        self._model.load_state_dict(weights)
        self._model.train()

        # Execute training loop for the specified number of epochs
        for epoch in range(self._epochs):
            for i, data in enumerate(self._train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()

    def _evaluate_model(self, fl_ctx: FLContext, weights: dict):
        """
        Evaluate the model on the validation data.
        """
        self._model.load_state_dict(weights)
        self._model.eval()
        val_accuracy = evaluate_accuracy(self._model, self._val_loader, self._device)
        self.log_info(fl_ctx, f"Validation accuracy of client {fl_ctx.get_identity_name()}: {val_accuracy:.4f}")
        return val_accuracy

    def share_weights(self, fl_ctx: FLContext) -> Shareable:
        """
        Share the model weights with the server.
        """
        weights = {k: v.cpu().detach().numpy() for k, v in self._model.state_dict().items()}
        self.log_info(fl_ctx, f"Client {fl_ctx.get_identity_name()} shares updated weights with the server.")

        outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights,
                           meta={"num_steps_cur_round": len(self._train_loader)})
        return outgoing_dxo.to_shareable()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        if task_name == self._train_task_name:
            self.log_info(fl_ctx, "Received train task.")
            # Exctract dxo from the shareable
            incoming_dxo = from_shareable(shareable)

            # Ensure data kind is weights
            if incoming_dxo.data_kind != DataKind.WEIGHTS:
                self.log_error(fl_ctx, f"Expected data kind to be WEIGHTS but got {incoming_dxo.data_kind}.")
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Convert weights to tensor and run training
            torch_weights = {k: torch.as_tensor(v) for k, v in incoming_dxo.data.items()}
            self._train_model(fl_ctx, torch_weights) # this updates the model weights

            # share the updated weights
            return self.share_weights(fl_ctx)
        elif task_name == self._submit_model_task_name:
            # Submit the model
            self.share_weights(fl_ctx)
        elif task_name == 'get_weights':
            return self.share_weights(fl_ctx)
        elif task_name == 'validate':
            # Extract weights
            incoming_dxo = from_shareable(shareable)
            if not incoming_dxo.data_kind == DataKind.WEIGHTS:
                self.log_exception(fl_ctx, f"DXO is of type {incoming_dxo.data_kind} but expected type WEIGHTS.")
                return make_reply(ReturnCode.BAD_TASK_DATA)
            weights = {k: torch.as_tensor(v, device=self._device) for k, v in incoming_dxo.data.items()}
            # Evaluate the model
            val_accuracy = self._evaluate_model(fl_ctx, weights)

            outgoing_dxo = DXO(data_kind=DataKind.METRICS, data={"val_accuracy": val_accuracy})
            return outgoing_dxo.to_shareable()


        else:
            raise ValueError(f"Unknown task name: {task_name}")

        return Shareable()
