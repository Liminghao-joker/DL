# mnist.py
import numpy as np
from PIL import Image
from two_layer_net import TwoLayerNet
from load_mnist import get_data
from optimizer import *

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img * 16))  # scale to [0, 255]
    pil_img.show()


def test():
    (x_train, t_train), (x_test, t_test) = get_data()

    iters_num = 10000
    batch_size = 32
    learning_rate = 0.01
    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    network = TwoLayerNet(input_size=64, hidden_size=50, output_size=10)
    optimizer = {
        'SGD': SGD(learning_rate),
        'Momentum': Momentum(learning_rate),
        'AdaGrad': AdaGrad(learning_rate),
        'Adam': Adam(learning_rate)
    }

    results = {}

    for opt_name, optimizer in optimizer.items():
        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        for i in range(iters_num):
            # mini batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # compute gradient
            grad = network.gradient(x_batch, t_batch)

            params = network.params
            # update parameters
            optimizer.update(params, grad)

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            # record learning process
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                # print(f"train acc, test acc | {train_acc}, {test_acc}")

        results[opt_name] = {
            'train_acc': train_acc_list[-1],
            'test_acc': test_acc_list[-1]
        }

    print('Optimizer comparison results:')
    for opt_name, accs in results.items():
        print(f"{opt_name}: Train Accuracy = {accs['train_acc']:.4f}, Test Accuracy = {accs['test_acc']:.4f}")

if __name__ == "__main__":
    test()
