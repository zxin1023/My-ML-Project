import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_loss_curve(train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()


def plot_accuracy_curve(train_accs, val_accs, save_path=None):
    """
    绘制训练集和验证集准确率随 epoch 变化的曲线图。

    参数:
        train_accs (list or array-like): 每个 epoch 对应的训练准确率。
        val_accs (list or array-like): 每个 epoch 对应的验证准确率。
        save_path (str, optional): 如果提供，将图像保存到指定路径；否则直接显示图像。

    返回:
        None
    """
    plt.figure()
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Accuracy curve saved to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵图。

    参数:
        y_true (list or array-like): 真实标签。
        y_pred (list or array-like): 预测标签。
        class_names (list): 类别名称列表。
        save_path (str, optional): 如果提供，将图像保存到指定路径；否则直接显示图像。

    返回:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Reds", colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()
