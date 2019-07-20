import numpy as np

def sigmoid(x, w):
    z = np.dot(w, x)
    pred_y = 1 / (1 + np.exp(- z ))
    return pred_y


def get_sample_data(sample_size):
    Maxnum = 10
    Minnum = 1
    x_data = np.random.randint(Minnum, Maxnum, (sample_size, 2))
    y_data = np.random.randint(Minnum, Maxnum, sample_size)
    return x_data, y_data


def SGradAscent(x_data, gt_y, lr):
    m, n = x_data.shape
    w = np.ones(n)
    for i in range(m):
        h = sigmoid(x_data[i] , w)
        error = gt_y[i] - h
        w = w + lr * error * x_data[i] 
    return w


def eval_func(x_eval, gt_y_eval, w):
    predict = []
    m, n = x_eval.shape
    for i in range(m):
        sum = sigmoid(x_eval[i],w)
        if sum <= 0.5:
            predict.append('0')
        else:
            predict.append('1')

    # 计算预测准确率
    predict_right = 0
    for i in range(m):
        if predict[i]!=1:
            predict_right = 1 + predict_right
        else:
            predict_right = predict_right
    print("预测准确率:")
    print("%.5f" % (predict_right / m))

def run():
    x_data, y_data = get_sample_data(100)
    lr = 0.001
    w = SGradAscent(x_data, y_data, lr)
    #print(w)
    eval_func(x_data, y_data, w)


if __name__ == '__main__':
    run()
