import json

from print_util import print_dict
from preproc_util import preprocess_dataset
from CONST import RAW_DATASET_PATH, COL_NAMES
from kneed import KneeLocator

import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

_scoring = ['accuracy', 'precision', 'recall', 'f1']

def eval_stratifiedKFold(model, _X, _y, _cv=5):
    cv = RepeatedStratifiedKFold(n_splits=_cv, n_repeats=1, random_state=1)
    results = cross_validate(estimator=model,
                                X=_X,
                                y=_y,
                                cv=cv, 
                                scoring=_scoring,
                                return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],  
            "Mean Training Accuracy": results['train_accuracy'].mean()*100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


def cross_validation(model, _X, _y, _cv=5):
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }

def decision_tree_eval(X, encoded_y):
    decision_tree_model = DecisionTreeClassifier(criterion="entropy", random_state=0)
    #decision_tree_result = cross_validation(decision_tree_model, X, encoded_y, 5)
    decision_tree_result = eval_stratifiedKFold(decision_tree_model, X, encoded_y, 5)
    return decision_tree_result

def svm_eval(X, encoded_y):
    svm_model = svm.SVC(kernel='rbf')
    #svm_result = cross_validation(svm_model, X, encoded_y, 5)
    svm_result = eval_stratifiedKFold(svm_model, X, encoded_y, 5)
    return svm_result

def kneigbors_eval(X, encoded_y):
    kn = KNeighborsClassifier(n_neighbors=3)
    #kn_result = cross_validation(kn, X, encoded_y)
    kn_result = eval_stratifiedKFold(kn, X, encoded_y)
    return kn_result

def eval_bs(X, encoded_y):
    decision_tree_report = decision_tree_eval(X, encoded_y)
    svm_report = svm_eval(X, encoded_y)
    kn_report = kneigbors_eval(X, encoded_y) 
    
    print("Decision tree:")
    print_dict(decision_tree_report)

    print("SVM:")
    print_dict(svm_report)

    print("K Neigbors:")
    print_dict(kn_report)

def driven_func():
    X, y = preprocess_dataset(RAW_DATASET_PATH, COL_NAMES)
    eval_bs(X, y)

if __name__ == "__main__":
    driven_func()
