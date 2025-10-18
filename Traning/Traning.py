#!/usr/bin/env python
"""
train_image_classifiers.py

Train KNN, GaussianNB, SVM, and SGDClassifier on a 5-class image dataset.
Images are binary (grayscale) PNGs of varying dimensions. This script:
  - Recursively loads images from class-named folders
  - Resizes to fixed size
  - Flattens pixels as features
  - Splits into train/test
  - Trains and evaluates four models

Usage:
    python train_image_classifiers.py \
        --dataset_dir ~/Dataset \
        --image_size 64 \
        --test_size 0.2
"""

import os
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CLASS_FOLDERS = {
    0: "Google_Docs",
    1: "MSOffice",
    2: "Python",
    3: "LaTeX",
    4: "LibreOffice"
}

def load_images(dataset_dir, image_size):
    """
    Load images and labels.
    Returns X (n_samples, image_size*image_size) and y (n_samples,).
    """
    X, y = [], []
    for label, folder in CLASS_FOLDERS.items():
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            path = os.path.join(folder_path, fname)
            try:
                img = Image.open(path).convert('L')
                img = img.resize((image_size, image_size), Image.ANTIALIAS)
                arr = np.asarray(img, dtype=np.uint8).flatten()
                X.append(arr)
                y.append(label)
            except Exception as e:
                print("Warning: could not load", path, ":", e)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=int)
    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s

def train_knn(X, y):
    param_grid = {'n_neighbors': [3,5,7]}
    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
    clf.fit(X, y)
    print("KNN best params:", clf.best_params_)
    return clf.best_estimator_

def train_nb(X, y):
    clf = GaussianNB()
    clf.fit(X, y)
    return clf

def train_svm(X, y):
    param_grid = {'C': [0.1,1,10], 'gamma': ['scale',0.01,0.1]}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
    clf.fit(X, y)
    print("SVM best params:", clf.best_params_)
    return clf.best_estimator_

def train_sgd(X, y):
    param_grid = {
        'alpha': [1e-4,1e-3],
        'loss': ['hinge','log'],
        'penalty': ['l2','l1']
    }
    clf = GridSearchCV(
        SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        param_grid, cv=5, n_jobs=-1
    )
    clf.fit(X, y)
    print("SGD best params:", clf.best_params_)
    return clf.best_estimator_

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True,
                        help='Root dataset folder containing class subfolders')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Resize images to image_size x image_size')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion for test split')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    print("Loading images...")
    X, y = load_images(args.dataset_dir, args.image_size)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    X_train_s, X_test_s = scale_features(X_train, X_test)

    print("Training KNN...")
    knn = train_knn(X_train_s, y_train)
    print("Training Naive Bayes...")
    nb = train_nb(X_train_s, y_train)
    print("Training SVM...")
    svm = train_svm(X_train_s, y_train)
    print("Training SGD...")
    sgd = train_sgd(X_train_s, y_train)

    evaluate(knn, X_test_s, y_test, "K-Nearest Neighbors")
    evaluate(nb,  X_test_s, y_test, "Gaussian Naive Bayes")
    evaluate(svm, X_test_s, y_test, "RBF SVM")
    evaluate(sgd, X_test_s, y_test, "SGD Classifier")

if __name__ == '__main__':
    main()