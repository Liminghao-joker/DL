import os
import numpy as np
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

# MNIST dataset parameters
total_num = 1797
img_dim = (8, 8)
img_size = 64
valid_ratio = 0.2
test_ratio = 0.2

#TODO: create train_test_split and train_valid_split on my own without sklearn
# Reference:
# def train_valid_split(data_set, val_ratio, seed):
#     valid_data_size = int(len(data_set) * val_ratio)
#     train_data_size = len(data_set) - valid_data_size
#     train_set, valid_set = random_split(data_set, [train_data_size, valid_data_size], torch.Generator().manual_seed(seed))
#     return np.array(train_set), np.array(valid_set)

def init_sklearn_mnist():
    """
    download sklearn digits dataset and save it as a pickle file
    """
    print("Loading sklearn digits dataset...")
    digits = load_digits()
    X = digits.data
    y = digits.target

    print(f"原数据集大小: X = {X.shape}, y = {y.shape}") # (1797, 64), (1797,)

    # split the dataset into train, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = valid_ratio, random_state=42)

    dataset = {
        'train_img': X_train,
        'train_label': y_train,
        'valid_img': X_val,
        'valid_label': y_val,
        'test_img': X_test,
        'test_label': y_test,
    }

    print("Creating pickle file...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)
    print("Pickle file successfully created!")
    print(f"训练集大小: X_train = {X_train.shape}, y_train = {y_train.shape}") # (1149, 64), (1149,)
    print(f"验证集大小: X_val = {X_val.shape}, y_val = {y_val.shape}") # (288, 64), (288,)
    print(f"测试集大小: X_test = {X_test.shape}, y_test = {y_test.shape}") # (360, 64), (360,)

def load_sklearn_mnist(normalize=True, flatten=True, one_hot_label=False):
    """
    load sklearn digits dataset from pickle file

    Parameters:
        normalize: normalize the image data to [0, 1]
        flatten: flatten the image data to 1D array (8, 8) -> (64,)
        one_hot_label: convert the labels to one-hot encoding

    Returns:
        (train_img, train_label), (valid_img, valid_label), (test_img, test_label)
    """
    if not os.path.exists(save_file):
        init_sklearn_mnist()
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'valid_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 16.0  # pixel values are from 0 to 16

    if one_hot_label:
        for key in ('train_label', 'valid_label', 'test_label'):
            dataset[key] = one_hot_encode(dataset[key], num_classes=10)

    if not flatten:
        for key in ('train_img', 'valid_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 8, 8)
    return ((dataset['train_img'], dataset['train_label']),
            (dataset['valid_img'], dataset['valid_label']),
            (dataset['test_img'], dataset['test_label']))

#? Why use one_hot_encode here?
def one_hot_encode(labels, num_classes=10):
    """
    convert labels to one-hot encoding
    """
    one_hot_labels = np.zeros((labels.shape[0], num_classes))
    for idx, label in enumerate(labels):
        one_hot_labels[idx, label] = 1
    return one_hot_labels

def get_data():
    (x_train, t_train), (x_val, t_val), (x_test, t_test) = load_sklearn_mnist(flatten=True, normalize=True, one_hot_label=True)
    return (x_train, t_train), (x_test, t_test)

if __name__ == "__main__":
    (x_train, t_train), (x_val, t_val), (x_test, t_test) = load_sklearn_mnist()






