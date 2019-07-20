import numpy as np
import random


def get_sample_data(sample_size):
    Maxnum = 10
    Minnum = 1
    x_data = np.random.randint(Minnum, Maxnum, sample_size)
    y_data = np.random.randint(Minnum, Maxnum, sample_size)
    return x_data, y_data


def eval_loss(x_data,y_data,w,b):
    avg_loss = sum([(w * x + b - y) ** 2 for x, y in zip(x_data, y_data)]) / (2 * len(x_data))
    return avg_loss


def cal_step_gradient(x_data, gt_y_data, w, b, lr):
    avg_dw = sum([( w * x + b - y) * x for x, y in zip(x_data, gt_y_data)]) / len(x_data)
    avg_db = sum([(w * x + b - y) for x, y in zip(x_data, gt_y_data)]) / len(x_data)
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


def train(x_data, gt_y_data, batch_size, lr, max_iter):
    w = 0
    b = 0
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_data), batch_size)
        batch_x = [x_data[j] for j in batch_idxs]
        batch_y = [gt_y_data[j] for j in batch_idxs]
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(x_data, gt_y_data, w, b)))


def run():
    x_data, y_data = get_sample_data(100)
    lr = 0.001
    max_iter = 10
    train(x_data, y_data, 10, lr, max_iter)


if __name__ == '__main__':
    run()
