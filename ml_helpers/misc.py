from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# for printing DFs to the console
def pprint(dataframe, showindex=False):
    print(tabulate(dataframe, headers='keys', showindex=showindex))

def train_test_split(dataframe, frac=0.7):
    train = dataframe.sample(frac=frac)
    test = dataframe.drop(train.index)
    return train, test


def plot_binary_classification_results(y_true, y_pred, y_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs, pos_label=1)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    prec = report['1.0']['precision']
    rec = report['1.0']['recall']
    f1 = report['1.0']['f1-score']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_title(f'ROC Curve: {model_name}')
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    stats_text = f'Class 1:\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1-Score: {f1:.3f}'
    ax1.text(0.6, 0.2, stats_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_title(f'Confusion Matrix: {model_name}')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    ax2.set_xticklabels(['0', '1'])
    ax2.set_yticklabels(['0', '1'])
    plt.tight_layout()
    plt.show()




