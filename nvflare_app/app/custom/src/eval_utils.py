import logging
import torch


def evaluate_accuracy(model, testloader, device):
    """
    Evaluate the accuracy of the model on the test set.
    :param model: Pytorch model.
    :param testloader: Dataloader for the test set.
    :return: Accuracy of the model on the test set.
    """

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            features, labels = data
            features, labels = features.to(device), labels.to(device)
            model_output = model(features)
            _, predicted = torch.max(model_output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info(f"Accuracy of the network on the test images: {100 * correct / total}%")
    logging.info(f"Correct: {correct}, Total: {total}.")

    return correct / total
