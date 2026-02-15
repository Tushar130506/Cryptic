import numpy as np
import config
from model_ic_focused import MacroAwareModel


def test_model_fit_predict():
    model = MacroAwareModel(config)
    X = np.random.randn(200, 5)
    y = np.random.randn(200)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
