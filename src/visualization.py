import matplotlib.pyplot as plt

def visualize_samples(X, y, samples=10):
    plt.figure(figsize=(15, 4))
    for i in range(samples):
        plt.subplot(1, samples, i + 1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.suptitle("Sample Digits from Training Data", fontsize=16)
    plt.show()
