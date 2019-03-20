# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    y1 = polynomial(x, w)
    return np.sum(np.square(y - y1)) / np.shape(x)[0]


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    L, _ = np.shape(x_train)
    dM = x_train.copy()
    for i in range(1, M):
        temp = np.array(dM[:, i-1]).reshape((L, 1))
        dM = np.hstack((dM, np.multiply(x_train, temp)))
    onesL = np.ones((L, 1))
    dM = np.hstack((onesL, dM))

    return (onesL if M == 0 else dM)


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    dM = design_matrix(x_train, M)
    w = (np.linalg.inv((dM.T)@dM))@(dM.T)@y_train
    return (w, mean_squared_error(x_train, y_train, w))


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd
    średniokwadratowy dopasowania
    """
    dM = design_matrix(x_train, M)
    I = np.eye(np.shape(dM)[1])
    #I[0, 0] = 0
    I = regularization_lambda * I
    p1 = np.linalg.inv(dM.T @ dM + I)

    w = p1 @ dM.T @y_train
    return (w, mean_squared_error(x_train, y_train, w))


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na
    ciągach treningowym i walidacyjnym
    """
    Mlen = len(M_values)
    arr = []
    for i in range(0, Mlen):
        w, train_err = least_squares(x_train, y_train, M_values[i])
        val_err = mean_squared_error(x_val, y_val, w)
        arr.append((w, train_err, val_err))
    return min(arr, key=lambda x: x[2])


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    Llen = len(lambda_values)
    arr = []
    for i in range(0, Llen):
        w, train_err = regularized_least_squares(x_train, y_train, M, lambda_values[i])
        val_err = mean_squared_error(x_val, y_val, w)
        arr.append((w, train_err, val_err, lambda_values[i]))
    return min(arr, key=lambda x: x[2])
