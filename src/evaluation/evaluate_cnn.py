import torch
import torch
from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_data
from model import build_model
import matplotlib.pyplot as plt
from PIL import Image

def main():
    # Load the best model
    model = build_model(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load('saved_models/cnn_best_model.pth', map_location='cpu'))

    # Set model to evaluation mode
    model.eval()

    # Load test data
    _, _, test_loader, test_dataset, class_names = load_data()

    # Evaluate on test set
    all_preds = []
    all_labels = []
    misclassified_inclusion_paths = []

    image_paths = [path for path, label in test_dataset.dataset.samples]

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

            # Save image paths for misclassified 'inclusion' samples
            for i in range(len(labels)):
                if class_names[labels[i]] == 'inclusion' and preds[i] != labels[i]:
                    sample_idx = batch_idx * test_loader.batch_size + i
                    misclassified_inclusion_paths.append((image_paths[test_dataset.indices[sample_idx]], class_names[preds[i]]))

    # Visualize misclassified 'inclusion' samples
    plt.figure(figsize=(15, 5))
    for i, (path, pred_class) in enumerate(misclassified_inclusion_paths):
        img = Image.open(path).convert('RGB')
        plt.subplot(2, 6, i + 1)
        plt.imshow(img)
        plt.title(f"Pred: {pred_class}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    for path, _ in misclassified_inclusion_paths:
        print(f"Misclassified 'inclusion' sample: {path}")

    test_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()
