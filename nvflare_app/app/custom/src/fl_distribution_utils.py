import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging


def synthesize_equal_dirichlet_client_data_distribution(labels, n_clients: int, alpha: float,
                                                        val_ratio: float = 0.1) -> dict:
    """
    Synthesize a client data distribution based on the Dirichlet distribution. Each client has an equal number of points.
    The labels will be distributed according to Dirichlet distribution with concentration parameter alpha.
    Higher alpha means more uniform distribution (closer to IID setting), whereas lower alpha means more non-IID data.
    :param labels: List-like object with the labels. The labels must be integers starting from 0, and the i-th label corresponds to the i-th data point.
    :param n_clients: Number of clients.
    :param alpha: Concentration parameter of the Dirichlet distribution.
    :param val_ratio: Ratio of the data points to be used for validation.
    :return: Dictionary with client IDs as keys and the corresponding indices.
    """
    labels = np.array(labels)
    n_classes = len(np.unique(labels))
    logging.debug(f"Number of classes: {n_classes}")
    n_samples_per_client = int(len(labels) / n_clients)
    logging.debug(f"Number of samples per client: {n_samples_per_client}")

    # create dictionary mapping list of indices to labels
    label_to_indices = {cur_label: np.argwhere([label == cur_label for label in labels]) for cur_label in
                        np.unique(labels)}
    logging.debug(
        f"Labels: {label_to_indices.keys()}. Number of samples per label: {[len(label_to_indices[cur_label]) for cur_label in label_to_indices.keys()]}")
    # shuffle the indices as we will be drawing them one by one
    for indices in label_to_indices.values():
        np.random.shuffle(indices)
    # create a counter dictionary to keep track of how many samples per label have been assigned already
    label_counter = {cur_label: 0 for cur_label in label_to_indices.keys()}

    # first, determine the proportion of labels for each client (will be input to multinomial to determine labels)
    # by sampling from a Dirichlet distribution
    multinomial_vals_per_client = np.random.dirichlet([alpha] * n_classes, n_clients)

    client_data_distributions = {}

    # for each client, sample the labels according to the multinomial distribution
    for client_id in range(n_clients):
        cur_client_inds = []
        # we will draw the samples one by one to make sure we don't exceed the number of samples or the number of points per labels
        for _ in range(n_samples_per_client):
            # sample the label
            label = np.random.multinomial(1, multinomial_vals_per_client[client_id]).argmax()
            # get the index of the next label to assign
            label_idx = label_to_indices[label][label_counter[label]][0]
            logging.debug(f"Client {client_id}: Assigning index {label_idx} of label {label} to client {client_id}.")
            # add the index to the client data distribution
            cur_client_inds.append(label_idx)
            # increment the counter
            label_counter[label] += 1

            # if the counter has reached the end of the indices, recalculate the multinomial
            if label_counter[label] >= len(label_to_indices[label]):
                # remove the label from the multinomial values for all clients
                for _client_id in range(n_clients):
                    multinomial_vals_per_client[_client_id][label] = 0
                    # renormalize the multinomial values
                    multinomial_vals_per_client[_client_id] /= multinomial_vals_per_client[_client_id].sum()

            # Lastly, we need to split the data into training and validation
            # Note that the data is already shuffled
            n_train = int((1 - val_ratio) * n_samples_per_client)
            logging.debug(
                f"Client {client_id}: {n_train} training samples, {n_samples_per_client - n_train} validation samples")
            train_indices = cur_client_inds[:n_train]
            val_indices = cur_client_inds[n_train:]
            client_dict = {'train': train_indices, 'val': val_indices}

            client_data_distributions[client_id] = client_dict

    return client_data_distributions

def create_plot_of_distributions(client_data_distributions, labels, save_path=None):
    """
    Create a barplot of the data distribution by mapping the assigned indices back to the labels.
    :param client_data_distributions: Dictionary with client IDs as keys and the corresponding indices.
    :param labels: List-like object with the labels. The labels must be integers starting from 0, and the i-th label corresponds to the i-th data point.
    :return:
    """
    data = []
    for client in client_data_distributions.keys():
        train_labels = labels[client_data_distributions[client]['train']]
        val_labels = labels[client_data_distributions[client]['val']]
        train_label_counts = {cur_label: np.sum(train_labels == cur_label) for cur_label in np.unique(labels)}
        val_label_counts = {cur_label: np.sum(val_labels == cur_label) for cur_label in np.unique(labels)}
        data.append({'client': client, 'split': 'train', **train_label_counts})
        data.append({'client': client, 'split': 'validation', **val_label_counts})
    df = pd.DataFrame(data)
    df = df.set_index(['client', 'split'])
    df = df.stack().reset_index()
    df.columns = ['client', 'split', 'class', 'count']
    sns.set_style('whitegrid')
    g = sns.catplot(x='client', y='count', hue='class', col='class', row='split', data=df, kind='bar', sharey=False)

    if save_path is not None:
        g.savefig(save_path)
    else:
        plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    labels = np.random.randint(0, 10, 1000)
    print(f"Type of labels: {type(labels)}")
    n_clients = 10
    alpha = 0.1  # high alpha means more uniform distribution (closer to IID)
    val_ratio = 0.1
    client_data_distributions = synthesize_equal_dirichlet_client_data_distribution(labels, n_clients, alpha, val_ratio)
    print(client_data_distributions)
    print("All clients: ", client_data_distributions.keys())
    for client in client_data_distributions.keys():
        print(
            f"Client {client}: {len(client_data_distributions[client]['train'])} training samples, {len(client_data_distributions[client]['val'])} validation samples")
        print(f"Client {client} training samples: {client_data_distributions[client]['train']}")

    # create a barplot of the data distribution by mapping the assigned indices back to the labels
    create_plot_of_distributions(client_data_distributions, labels, save_path='client_data_distribution_test.png')