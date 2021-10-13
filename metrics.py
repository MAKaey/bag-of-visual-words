from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list
    ) -> None:
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
def display_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list
    ) -> None:
    
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

def plot_visual_word_vocabulary(
        features: np.ndarray,
        num_clusters: int
    ) -> None:
    
    x_axis = np.arange(num_clusters)
    y_axis = np.absolute(np.sum(features, axis=0, dtype=np.int32))

    figure = plt.figure(figsize=(8,5))
    plt.title("Visual Word Vocabulary")
    plt.bar(x_axis, y_axis)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, num_clusters+5, 5), np.arange(0, num_clusters+5, 5))
    plt.show()