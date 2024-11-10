import pickle
import os
import argparse

import numpy as np

from nvflare_app.app.custom.src.utils import create_logger, get_torch_datasets
from nvflare_app.app.custom.src.fl_distribution_utils import synthesize_equal_dirichlet_client_data_distribution, create_plot_of_distributions

if __name__=='__main__':

    # create logger
    logger = create_logger(file_name=os.path.abspath('./fl_data_distribution_utils.log'))

    # Define commmand line arguments
    parser = argparse.ArgumentParser(description='Create (non-IID) client data distribution for federated learning')
    parser.add_argument('--n_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Name of the dataset')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet distribution parameter')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio for each client')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--create_plot', type=bool, default=True, help='Create a plot of the client data distribution')

    # Read arguments
    args = parser.parse_args()
    n_clients = args.n_clients
    dataset_name = args.dataset_name
    alpha = args.alpha
    val_ratio = args.val_ratio
    seed = args.seed
    create_plot = args.create_plot

    logger.info(f"Number of clients parsed: {n_clients}")
    logger.info(f"Dataset name parsed: {dataset_name}")
    logger.info(f"Alpha parsed: {alpha}")
    logger.info(f"Validation ratio parsed: {val_ratio}")
    logger.info(f"Seed parsed: {seed}")
    logger.info(f"Create plot parsed: {create_plot}")

    # Load the dataset to be distributed
    trainset, testset = get_torch_datasets(dataset_name, root_dir=os.path.abspath('./data/centralized'), apply_transforms=True)
    labels = np.array(trainset.targets)
    logger.info(f"Number of labels: {len(labels)}")
    logger.info(f"Number of classes: {len(set(labels))}")
    logger.info(f"First ten labels: {labels[:10]}")

    # Create the client data distribution based on the Dirichlet distribution of the labels
    client_data_distributions = synthesize_equal_dirichlet_client_data_distribution(labels, n_clients, alpha, val_ratio=val_ratio)

    # Save the client data distribution
    save_path = os.path.abspath(f'./data/distributed/{dataset_name}/alpha={alpha}_val_ratio={val_ratio}/client_data_distributions.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(client_data_distributions, f)
    logger.info(f"Client data distributions saved at {save_path}")

    if create_plot:
        # test if we can load the data
        with open(save_path, 'rb') as f:
            client_data_distributions = pickle.load(f)
        logger.info(f"Client data distributions loaded from {save_path}")
        logger.info(f"Number of clients: {len(client_data_distributions)}")
        logger.info(f"First client data distribution keys: {client_data_distributions[0].keys()}")
        logger.info(f"First client data distribution values: {client_data_distributions[0]['train'][:10]}")
        # Create a plot of the client data distribution
        plot_save_path = os.path.abspath(f'./data/distributed/{dataset_name}/alpha={alpha}_val_ratio={val_ratio}/client_data_distributions.png')
        create_plot_of_distributions(client_data_distributions, labels, plot_save_path)
        logger.info(f"Plot of client data distributions saved at {plot_save_path}")






