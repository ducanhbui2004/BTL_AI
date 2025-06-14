import os
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    # Lấy tên mô hình và thư mục từ tham số dòng lệnh
    parser = argparse.ArgumentParser("Training model")
    parser.add_argument("--model_name", help="Name of the model", type=str, default="model")
    parser.add_argument("--dir", help="Location of the model", type=str, default="models")
    args = parser.parse_args()

    # Đọc dữ liệu từ thư mục data
    X, y, mapping = [], [], dict()
    for current_class_index, pose_file in enumerate(os.scandir("data")):
        file_path = f"data/{pose_file.name}"
        pose_data = np.load(file_path)
        X.append(pose_data)
        y += [current_class_index] * pose_data.shape[0]
        mapping[current_class_index] = pose_file.name.split(".")[0]

    X, y = np.vstack(X), np.array(y)

    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Huấn luyện mô hình SVM
    model = SVC(decision_function_shape='ovo', kernel='rbf')
    model.fit(X_train, y_train)

    # Tính accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # In kết quả
    print(f"Training examples: {X.shape[0]}. Num classes: {len(mapping)}")
    print(f"Train accuracy: {round(train_accuracy * 100, 2)}% - Test accuracy: {round(test_accuracy * 100, 2)}%")

    # Lưu mô hình
    model_path = os.path.join(args.dir, f"{args.model_name}.pkl")
    with open(model_path, "wb") as file:
        pickle.dump((model, mapping), file)
    print(f"Saved model to {model_path}")

    # -------- VẼ MA TRẬN NHẦM LẪN --------
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Tên lớp
    labels = [mapping[i] for i in range(len(mapping))]

    # Vẽ biểu đồ heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
