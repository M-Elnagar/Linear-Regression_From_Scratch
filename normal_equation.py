import numpy as np
from sklearn.metrics import r2_score

def normal_equation(X, y):
    
    W = np.linalg.pinv(X.T @ X) @ X.T @ y
    y_pred = X @ W 

    # computing cost function
    err = (y - y_pred)
    cost = (err.T @ err ) / (2 * len(y))
    r2 = r2_score(y, y_pred)

    return y_pred, W , cost , r2
