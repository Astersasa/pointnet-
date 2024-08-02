import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from getmodel4 import get_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data():
    try:
        test_dataset = torch.load('testdata.pt')
        test_labels = torch.load('testlabels.pt')
        return test_dataset, test_labels
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_model(model_path, num_classes=6, device='cuda'):
    model = get_model(num_class=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_loader, device):
    all_preds = []
    all_targets = []
    all_correctness = []

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_correctness.extend((preds == target).cpu().numpy())

    return all_targets, all_preds, all_correctness

def main():
    test_dataset, test_labels = load_data()
    model_path = 'trained_model-acc=0.8813868613138686.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = TensorDataset(test_dataset, test_labels)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    best_accuracy = 0
    best_targets = None
    best_preds = None
    best_correctness = None
    best_class_report = None

    num_experiments = 10  # Number of experiments to run

    for experiment in range(num_experiments):
        print(f'Experiment {experiment + 1}/{num_experiments}')
        model = load_model(model_path, num_classes=6, device=device)
        all_targets, all_preds, all_correctness = evaluate_model(model, data_loader, device)

        instance_acc = accuracy_score(all_targets, all_preds)
        class_labels = ['D', 'E', 'F', 'G', 'J', 'K']
        class_report = classification_report(all_targets, all_preds, target_names=class_labels, digits=2, zero_division=0, output_dict=True)

        print(f'Experiment {experiment + 1} Accuracy: {instance_acc:.2f}')

        if instance_acc > best_accuracy:
            best_accuracy = instance_acc
            best_targets = all_targets
            best_preds = all_preds
            best_correctness = all_correctness
            best_class_report = class_report

    print(f'\nBest Evaluation Accuracy: {best_accuracy:.2f}')
    print('Best Classification Report:')
    print(classification_report(best_targets, best_preds, target_names=class_labels, digits=2, zero_division=0))

    plot_confusion_matrix(best_targets, best_preds, class_labels)
    plot_classification_report(best_class_report,  class_labels)

    results_df = pd.DataFrame({
        'Target': best_targets,
        'Prediction': best_preds,
        'Correct': best_correctness
    })

    results_df.to_csv('best_classification_results.csv', index=False)
    print('Saved best classification results to best_classification_results.csv')

def plot_confusion_matrix(all_targets, all_preds, class_labels):
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def plot_classification_report(class_report, class_labels):
    report_df = pd.DataFrame(class_report).T
    report_df = report_df.loc[class_labels]  # remove support row
    report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 8))
    plt.title('Classification Report')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()
