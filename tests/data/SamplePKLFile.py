import pickle
import os
import numpy as np
import numpy.linalg as linalg
import pandas as pd


def generate_matrices_pkl_from_csv():
    """
    Generates A, B, C matrices from data csv.

    The data come from 'storage_data.csv', a simulation of a battery storage system. The
    state variables are the NPP generation (constant at 50 MW) and the battery state of
    charge. The control variables are the charging and discharging rates for the battery
    found via an optimization. The measurement variable is again the state of charge of
    the battery (so that there is a C matrix to find, although trivial).

    The A, B, C matrices are found and stored in a dictionary that can be used by
    LTIStateSpaceMPCPyomoOptimization.

    """

    file_path = os.path.join(os.path.dirname(__file__), "storage_data.csv")
    # load data
    data_df = pd.read_csv(file_path)

    # find A and B matrices first
    x = data_df[["qNPP", "SOC"]].values.T
    u = data_df[["qC", "qD"]].values.T
    X = x[:, :-1]
    Xp = x[:, 1:]
    U = u[:, :-1]
    Om = np.vstack((X, U))
    u, s, vh = linalg.svd(Om)
    Sinv = np.diag(1.0 / s)
    Sinv = np.hstack((Sinv, np.zeros((Om.shape[0], Om.shape[1] - Sinv.shape[1]))))
    G = Xp.dot(vh.T.dot(Sinv.T).dot(u.T))
    A = G[:, : X.shape[0]]
    B = G[:, X.shape[0] :]

    # now find C
    y = data_df["SOC2"].values.reshape((1, -1))
    C = y.dot(linalg.pinv(x))

    save_dict = {"A": A, "B": B, "C": C}

    with open(os.path.join(os.path.dirname(__file__), "ABC.pkl"), "wb") as f:
        pickle.dump(save_dict, f)


if __name__ == "__main__":
    generate_matrices_pkl_from_csv()
