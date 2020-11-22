import sys
import fire
import numpy as np
import pandas as pd
import joblib
import tsfel
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

sc_X = StandardScaler()
sc_y = StandardScaler()


def get_data():
    df = pd.read_csv("bearing-3.xlsx.csv")
    column_target = "Eng Brg 3X Vibration"
    df = df[column_target]
    return df


def extract_feature(window_size):

    data = get_data()
    data = data.to_numpy()
    kurtosis = []
    rms = []
    skewness = []
    variance = []
    means = []

    prev_rol = 0
    window = window_size

    range_window = len(data) // window

    print(range_window)
    for i in range(range_window):
        res = data[prev_rol : prev_rol + window].copy()

        # hitung kurtosis untuk rol data ke prev_rol-window
        kurtosis_rol = tsfel.kurtosis(res)
        kurtosis.append(kurtosis_rol)

        rms_rol = tsfel.rms(res)
        rms.append(rms_rol)

        skewness_rol = tsfel.skewness(res)
        skewness.append(skewness_rol)

        variance_rol = tsfel.calc_var(res)
        variance.append(variance_rol)

        means_rol = tsfel.calc_mean(res)
        means.append(means_rol)

        prev_rol += window
    df = {
        "kurtosis": kurtosis,
        "rms": rms,
        "skewness": skewness,
        "variance": variance,
        "means": means,
    }
    df = pd.DataFrame(df)
    return df


def plot_feature(window_size):
    df = extract_feature(window_size)
    print(len(df["kurtosis"]), len(df["rms"]))

    plt.plot(df.rms, label="rms")
    plt.plot(df.kurtosis, label="kurtosis")
    plt.legend()
    plt.show()


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print("explained_variance: ", round(explained_variance, 4))
    print("mean_squared_log_error: ", round(mean_squared_log_error, 4))
    print("r2: ", round(r2, 4))
    print("MAE: ", round(mean_absolute_error, 4))
    print("MSE: ", round(mse, 4))
    print("RMSE: ", round(np.sqrt(mse), 4))


def dump_dataset():
    x_train, x_test, y_train, y_test = load_data_train()
    joblib.dump(x_train, "x_train.data")
    joblib.dump(y_train, "y_train.data")
    joblib.dump(x_test, "x_test.data")
    joblib.dump(y_test, "y_test.data")


def rms(data, slice_window_size):
    rms = []
    for i in range(len(data) // slice_window_size):
        rms.append(
            np.sqrt(
                np.mean(
                    (data[i * slice_window_size : (i + 1) * slice_window_size]) ** 2
                )
            )
        )
    return rms


def load_data():
    df = pd.read_excel("bearing-3.xlsx")
    column_target = "Eng Brg 3X Vibration"
    df = df[df[column_target] >= 1]
    x = np.array(df.index)
    y = df[column_target].to_numpy()
    return x, y


def load_data_train():
    x, y = load_data()
    y = sc_y.fit_transform(y.reshape(len(y), 1))

    x = x.reshape(x.shape[0], 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, train_size=0.7
    )
    return (x_train, x_test, y_train, y_test)


def load_model():
    model = joblib.load("svr.joblib")
    return model


def model_params():

    x_train, x_test, y_train, y_test = load_data_train()
    model = load_model()
    y_hat = model.predict(x_test)
    r2 = compute_score(y_test, y_hat)
    try:
        params = {
            "C": model.C,
            "gamma": model.gamma,
            "kernel": model.kernel,
            "score": r2,
        }
    except:
        params = model.best_params_
        params["score"] = r2

    print(params)


def plot_predicted():
    import matplotlib.pyplot as plt

    x_train, x_test, y_train, y_test = load_data_train()
    model = load_model()
    out = model.predict(x_train)
    plt.plot(x_train, y_train, label="observasi")
    plt.plot(x_train, out, label="model")
    plt.xlabel("t hour")
    plt.ylabel("vibrasi")
    plt.legend()
    plt.savefig("prediksi-bearing.png", dpi=500)
    plt.show()


def compute_score(y, y_hat):
    score = r2_score(y, y_hat)
    return score


def plot_data():
    import matplotlib.pyplot as plt

    x, y = load_data()
    plt.plot(x, y)
    plt.show()


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


def train_model(grid=""):

    x_train, x_test, y_train, y_test = load_data_train()
    len_train = int(len(x_train))
    x_train = x_train[0:len_train]
    y_train = y_train[0:len_train]

    if grid:
        rmse_score = make_scorer(rmse, greater_is_better=False)
        tscv = TimeSeriesSplit(n_splits=10)
        print("> running grid search...")
        tuned_parameters = [
            {"kernel": ["poly"], "C": [10, 20, 30, 40], "gamma": ["auto"]}
        ]

        clf = GridSearchCV(
            SVR(), tuned_parameters, scoring=rmse_score, verbose=1, n_jobs=4
        )

        clf.fit(x_train, y_train.ravel())

        print("Best parameters set found on development set:")
        print(clf.best_params_)
    else:
        # params = {"C":45, "gamma":7e-6, "epsilon": 0.0001, "degree":4}
        # params = {"C":40, "gamma":4e-4, "degree": 7, "epsilon": 0.3}
        params = {"C": 40, "gamma": 4e-4, "degree": 10, "epsilon": 0.03}
        # params = {"C":400, "gamma":0.000001, "epsilon": 0.001}

        print(params)
        clf = SVR(kernel="rbf", gamma="auto", C=10)
        clf.fit(x_train, y_train.ravel())

    y_pred = clf.predict(x_train)
    y_pred = sc_y.inverse_transform(y_pred)
    regression_results(y_train, y_pred)
    print(f"score: {score}")
    joblib.dump(clf, "svr.joblib")


def plot_extracted():
    from tsmoothie.smoother import DecomposeSmoother, LowessSmoother

    smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
    df = pd.read_csv("bearing-3-extracted.csv")
    smoother.smooth(df)
    plt.plot(df, label="data", c="red")
    plt.plot(smoother.smooth_data[0], label="smooth", c="blue")
    plt.legend()
    plt.xlabel("t hour")
    plt.ylabel("vibrasi")
    plt.show()


def plot_predict(n):
    x = n
    x = np.array([[i] for i in range(x, x + n)])
    model = load_model()
    out = model.predict(x)

    plt.plot(x, out, label="model")
    plt.xlabel("t hour")
    plt.ylabel("vibrasi")
    plt.legend()
    plt.savefig("prediksi.png", dpi=500)
    plt.show()


if __name__ == "__main__":
    fire.Fire()
