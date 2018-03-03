from sklearn.metrics import fbeta_score


def compute_F_score(y_true, predictions, t, beta):
    return(fbeta_score(y_true, (predictions > t).astype(int), beta=beta, average='samples'))


# Alternative implementation, gives exactly the same result
def compute_F_score_np(y_true, predictions, t, beta):
    h = (predictions > t).astype(int)
    syh = np.sum(y_true * h, axis=1)
    sy = np.sum(y_true, axis=1)
    sh = np.sum(h, axis=1)

    FB = np.mean((1 + beta**2) * (syh) / ((beta**2) * sy + sh))
    return(FB)
