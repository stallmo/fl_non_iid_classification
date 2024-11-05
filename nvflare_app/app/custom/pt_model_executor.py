# NVFlare Custom Model Executor (local task execution
import torch
from nvflare.apis.dxo import from_shareable, DataKind
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.fl_constant import ReturnCode

from src.utils import get_model, get_loss


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

    def _train_model(self, fl_ctx: FLContext, weights: dict):

        # Set model weights and set into training mode
        self._model.load_state_dict(weights)
        self._model.train()

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
        elif task_name == self._submit_model_task_name:
            # Submit the model
            self._submit_model(fl_ctx)
        else:
            raise ValueError(f"Unknown task name: {task_name}")

        return Shareable()
