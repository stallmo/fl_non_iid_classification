import torch

import argparse
import os

from nvflare_app.app.custom.src.utils import create_logger, get_model, get_data_centralized, save_pt_model
from nvflare_app.app.custom.src.eval_utils import evaluate_accuracy


if __name__ == "__main__":
    # initialize logging
    logger = create_logger(file_name=os.path.abspath('./centralized_baseline.log'))

    # Define command line arguments
    parser = argparse.ArgumentParser(description='PyTorch centralized learning baseline')
    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Dataset used')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--early_stopping_rounds', type=int, default=10,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='Number of batches to wait before logging training status')

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    # Read arguments
    args = parser.parse_args()
    dataset_name = args.dataset
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed
    early_stopping_rounds = args.early_stopping_rounds
    log_interval = args.log_interval

    logger.info(f"Dataset parsed: {dataset_name}")
    logger.info(f"Number of epochs parsed: {n_epochs}")
    logger.info(f"Learning rate parsed: {learning_rate}")
    logger.info(f"Batch size parsed: {batch_size}")
    logger.info(f"Seed parsed: {seed}")
    logger.info(f"Early stopping rounds parsed: {early_stopping_rounds}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    # Create model and get data
    model = get_model(dataset_name)
    model.to(device)

    # logger.info(f'Summary of the model: {torchsummary.summary(model, (3, 32, 32))}')
    train_loader, val_loader, test_loader = get_data_centralized(dataset_name, batch_size,
                                                                 root_dir=os.path.abspath('./data/centralized'))

    # define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # use adam optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    cur_val_accuracy = 0
    n_epochs_without_improvement = 0
    for epoch in range(n_epochs):
        logger.info(f"Starting epoch {epoch + 1}")
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % log_interval == log_interval - 1:
                train_accuracy = evaluate_accuracy(model, train_loader, device)
                val_accuracy = evaluate_accuracy(model, val_loader, device)
                logger.info(
                    f"[{epoch + 1}, {i + 1}] loss: {loss.item()} train accuracy: {train_accuracy:.4f} val accuracy: {val_accuracy:.4f}")
        # Implement early stopping here if needed
        new_val_accuracy = evaluate_accuracy(model, val_loader, device)
        if new_val_accuracy > cur_val_accuracy:
            cur_val_accuracy = new_val_accuracy
            n_epochs_without_improvement = 0
            logger.info(f"Validation accuracy improved to {cur_val_accuracy}")
        else:
            n_epochs_without_improvement += 1
            logger.info(
                f"Validation accuracy did not improve. Number of epochs without improvement: {n_epochs_without_improvement}")
            if n_epochs_without_improvement >= args.early_stopping_rounds:
                logger.info(f"Early stopping after epoch {epoch + 1}")
                break

    logger.info("Finished Training")
    # Test the model on the test data
    test_accuracy = evaluate_accuracy(model, test_loader, device)
    logger.info(f"Accuracy of the network on the test images: {100 * test_accuracy}%")

    # save model
    save_pt_model(model, os.path.abspath(f'./models/{dataset_name}/centralized_baseline.pt'))
    logger.info(f"Model saved at {os.path.abspath(f'./models/{dataset_name}/centralized_baseline.pt')}")
    logger.info("Finished")
