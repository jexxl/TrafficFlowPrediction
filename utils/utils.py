import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

mae = metrics.mean_absolute_error


def rmse(y_true, y_pred): return np.sqrt(
    metrics.mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    mask = y_true != 0
    return (np.fabs(y_true - y_pred) / y_true)[mask].mean() * 100


# 根据编号获取流量序列
def get_flow_series_by_idx(idx, only_eight_hours=False):
    path = 'data/approach_%d.csv' % idx
    df = pd.read_csv(path, usecols=[1, 2], parse_dates=[0], index_col=[0])
    if only_eight_hours:
        df = df[df.index.hour.isin([7, 8, 11, 12, 17, 18, 20, 21])]
    return df.iloc[:, 0]


# 根据编号获取路段车道数
def get_num_lanes_by_idx(idx):
    path = 'data/approach_%d.csv' % idx
    with open(path, 'r') as f:
        f.readline()
        second_line = f.readline()
    return int(second_line[-2])


# 滑动窗口获取x和y
def get_x_y_by_sliding_window(ts, in_steps=6, out_steps=1, step=1):
    if isinstance(ts, pd.Series):
        arr = ts.tolist()
    elif isinstance(ts, np.ndarray):
        arr = ts.reshape(-1).tolist()
    else:
        raise Exception('The first argument must be pd.Series or np.ndarray. ')
    x_lst, y_lst = [], []
    for i in range(0, len(arr) - in_steps - out_steps + 1, step):
        x_lst.append(arr[i:(i + in_steps)])
        y_lst.append(arr[(i + in_steps):(i + in_steps + out_steps)])
    x = np.array(x_lst).reshape(-1, in_steps, 1)
    y = np.array(y_lst).reshape(-1, out_steps, 1)
    return x, y


# 区分训练集与测试集
def split_train_test(x, y, test_split=None, test_size=None):
    chk = [test_split, test_size]
    if all(chk) or not any(chk):
        raise Exception('One of the test_split or test_size must be given. ')
    sz = test_size if test_size else int(test_split * x.shape[0])
    # x_train, y_train, x_test, y_test
    return x[:-sz], y[:-sz], x[-sz:], y[-sz:]


# 计算性能指标
def calc_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mape': mape(y_true, y_pred)
    }


# 测试预测值
def test_y_pred(y_true, y_pred, scaler=None, plot_with_figsize=None, y_steps=1):
    y_true = np.array(y_true).reshape(-1,1)
    y_pred = np.array(y_pred).reshape(-1,1)
    if scaler:
        y_true = y_true * scaler.var_**.5 + scaler.mean_ * y_steps
        y_pred = y_pred * scaler.var_**.5 + scaler.mean_ * y_steps
    res = calc_metrics(y_true, y_pred)
    if plot_with_figsize:
        plt.figure(figsize=plot_with_figsize)
        plt.plot(y_true, label='y_true')
        plt.plot(y_pred, 'r--', label='y_pred')
        plt.title('MAE: %.3f  RMSE: %.3f  MAPE: %.3f' %
                  (res['mae'], res['rmse'], res['mape']))
        plt.legend()
    return res


# 测试训练好的模型
def test_model(model, x_test, y_test, scaler=None, plot_with_figsize=None, y_steps=1):
    y_true = y_test
    y_pred = model.predict(x_test)
    test_y_pred(y_true, y_pred, scaler=scaler,
                plot_with_figsize=plot_with_figsize,
                y_steps=y_steps)


if __name__ == '__main__':
    ts1 = get_flow_series_by_idx(0)
    ts2 = get_flow_series_by_idx(0, only_eight_hours=True)
    print(ts1.shape)
    print(ts2.shape)
