"""
Useful functions for Mechine Learning

@author: Rui Zhu 
@creation time: 2024-10-26
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

__all__ = [
    'score_for_each_class',
    'plot_confusion_matrix'
]

def score_for_each_class(y_true, y_pred, label_encoder):
    """
    从分类结果中获得每个类别的评估指标

    Parameters
    ----------
    y_true: np.ndarray
        真实标签, 编码到整数
    y_pred: np.ndarray
        预测标签, 编码到整数
    label_encoder: sklearn.preprocessing.LabelEncoder
        标签编码器
    """
    res = {}
    label_names = label_encoder.classes_
    label_ids = label_encoder.transform(label_names)
    for label_id, label_name in zip(label_ids, label_names):
        label_id = int(label_id)
        res[label_id] = {}
        res[label_id]['label'] = label_name
        # 将多分类转换成2分类
        y_true_binary = (y_true == label_id).astype(int)
        y_pred_binary = (y_pred == label_id).astype(int)
        # 计算分类评估指标
        res[label_id]['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        res[label_id]['precision'] = precision_score(y_true_binary, y_pred_binary).item()
        res[label_id]['recall'] = recall_score(y_true_binary, y_pred_binary).item()
        res[label_id]['f1'] = f1_score(y_true_binary, y_pred_binary).item()
    return res

def plot_confusion_matrix(y_true, y_pred, labels, percentage=True, 
                          fig_title=None, fontsize=15, cmap=plt.cm.Blues, 
                          ax=None):
    """
    绘制混淆矩阵, 可以选择是否显示百分比(即, 每个类别的Recall)
    """
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    exist_true_label = cm_counts.sum(axis=1) != 0
    cm_counts = cm_counts[exist_true_label]
    # 计算每个类别的占比
    cm_percentage = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
    if percentage:
        cm = cm_percentage
    else:
        cm = cm_counts

    # 绘制混淆矩阵
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    else:
        ax = ax
    cax = ax.matshow(cm_percentage, cmap=cmap)

    # 设置轴标签
    true_labels = labels[exist_true_label]
    pred_labels = labels
    ax.set_xticks(range(len(pred_labels)))
    ax.set_yticks(range(len(true_labels)))
    ax.set_xticklabels(pred_labels, rotation=45)
    ax.set_yticklabels(true_labels)

    # 在每个格子中间写上百分比数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            precent = cm_percentage[i, j]
            if precent >= 0.5:
                color = 'white'
            else:
                color = 'black'
            if percentage:
                content = f'{cm[i, j]:.2%}'
            else:
                content = f'{cm[i, j]}'
            ax.text(j, i, content, ha='center', va='center', size=fontsize, color=color)

    # 隐藏下坐标轴的刻度线
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Predicted label', size=15)
    ax.set_ylabel('True label', size=15)

    ax.set_title(fig_title, size=15)
    
    return None