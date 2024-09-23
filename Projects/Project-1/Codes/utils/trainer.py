"""
trainer.py
"""

import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


class ModelTrainer:
    def __init__(self, model, dest_dir):
        self.model = model
        self.dest_dir = dest_dir

    def fit(self, images, labels, verbose=False):
        self.model.fit(images, labels, verbose=verbose)
        filename = f"{self.dest_dir}/model.joblib"
        joblib.dump(self.model, filename)

    def get_train_acc(self, images, labels):
        y_pred = self.model.predict(images)
        self.train_acc = accuracy_score(labels, y_pred)
        print(f"Train Accuracy: {self.train_acc:.4f}")
        return self.train_acc


class ModelCrossValidator:
    def __init__(self, model, dest_dir, n_splits=5, random_state=42):
        self.model = model
        self.dest_dir = dest_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.acc_scores = []
        self.skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

    def fit(self, images, labels, verbose=False):
        for fold, (train_index, val_index) in enumerate(self.skf.split(images, labels)):
            X_train, X_val = images[train_index], images[val_index]
            y_train, y_val = labels[train_index], labels[val_index]

            fold_model = self.model
            fold_model.fit(X_train, y_train, verbose=verbose)
            y_pred_val = fold_model.predict(X_val)
            acc = accuracy_score(y_val, y_pred_val)
            self.acc_scores.append(acc)
            print(f"Fold {fold} Accuracy: {acc:.4f}")

            filename = f"{self.dest_dir}/model_fold_{fold}.joblib"
            joblib.dump(fold_model, filename)

        mean_accuracy = np.mean(self.acc_scores)
        print(f"Mean Cross-Validation Accuracy: {mean_accuracy:.4f}")
        return mean_accuracy
