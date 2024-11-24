import matplotlib.pyplot as plt
import numpy as np
import random

def visualize_misclassified_samples(X_test, y_test, y_pred):
    misclassified_indices = np.where(y_test != y_pred)[0]
    random_indices = random.sample(list(misclassified_indices), min(5, len(misclassified_indices)))

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}")
        plt.axis('off')
    plt.suptitle("Misclassified Samples", fontsize=16)
    plt.show()
