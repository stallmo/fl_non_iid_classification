import logging
from nvflare_app.app.custom.src.utils import get_data_federated
from nvflare_app.app.custom.src.pt_constants import PATH_TO_DATA_CENTRALIZED_DIR


if __name__=='__main__':

    logging.basicConfig(level=logging.INFO)

    # Test the get_data_federated function
    dataset_name = 'CIFAR10'
    batch_size = 2
    root_dir = PATH_TO_DATA_CENTRALIZED_DIR
    client_id = 0
    alpha = 0.5
    val_ratio = 0.2
    for client_id in range(10):
        trainloader, testloader = get_data_federated(dataset_name, batch_size, root_dir, client_id, alpha, val_ratio)
        train_iter = iter(trainloader)
        test_iter = iter(testloader)

        # Check if the data is loaded correctly
        train_data, train_labels = next(train_iter)
        test_data, test_labels = next(test_iter)

        logging.info(
            f"Client {client_id}: Training data shape: {train_data.shape}, Training labels shape: {train_labels.shape}")
        logging.info(f"Client {client_id}: Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")
        logging.info(f"Client {client_id}: First training labels: {train_labels}")

        all_labels = [label for _, label in trainloader.dataset]
        logging.info(f"Client {client_id}: Number of labels: {len(all_labels)}")
        logging.info(f"Client {client_id}: Number of classes: {len(set(all_labels))}")
        # count occurrences of each label
        label_counts = {label: all_labels.count(label) for label in set(all_labels)}

        logging.info(f"Client {client_id}: Count per label: {label_counts}")