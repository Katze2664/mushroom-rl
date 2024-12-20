import numpy as np
from mushroom_rl.core import Serializable


class QRegressor(Serializable):
    """
    This class is used to create a regressor that approximates the Q-function
    using a multi-dimensional output where each output corresponds to the
    Q-value of each action. This is used, for instance, by the ``ConvNet`` used
    in examples/atari_dqn.

    """
    def __init__(self, approximator, **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the
                Q-function;
            **params: parameters dictionary to the regressor.

        """
        self.model = approximator(**params)

        self._add_save_attr(
            model=self._get_serialization_method(approximator)
        )

    def fit(self, state, action, q, **fit_params):
        """
        Fit the model.

        Args:
            state (np.ndarray): states;
            action (np.ndarray): actions;
            q (np.ndarray): target q-values;
            **fit_params: other parameters used by the fit method of the
                regressor.

        """
        self.model.fit(state, action, q, **fit_params)

    def predict(self, state, action=None, **predict_params):
        """
        Predict.

        Args:
            state: state for which q-values should be returned
            action[optional]: if provided, returns q-value corresponding to the provided action
            **predict_params: other parameters used by the predict method
                of each regressor.

        Returns:
            The predictions of the model.

        """

        batch_size = state.shape[0]
        q = self.model.predict(state, **predict_params)
        assert q.shape[0] == batch_size

        if action is not None:
            assert action.shape == (batch_size, 1)
            action = action.ravel()
            if q.ndim == 1:
                print(f"Warning: {q.shape=} only has 1 dimension. Is this the intended behaviour?")
                return q[action]
            elif q.ndim == 2:
                return q[np.arange(batch_size), action]
            elif q.ndim == 3:
                return q[np.arange(batch_size), action, :]  # Multi-objective
            else:
                raise ValueError(f"Invalid shape of q array: {q.shape}. Expected a shape with 1, 2 or 3 dimensions.")
        else:
            assert q.ndim in [1, 2, 3], f"Invalid q-value shape: {q.shape}. Expected a shape with 1, 2 or 3 dimensions."
            if q.ndim == 1:
                print(f"Warning: {q.shape=} only has 1 dimension. Is this the intended behaviour?")
            return q

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            self.model.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a'
                                      ' non-parametric regressor.')

    @property
    def weights_size(self):
        return self.model.weights_size

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, w):
        self.model.set_weights(w)

    def diff(self, state, action=None):
        if action is None:
            return self.model.diff(state)
        else:
            return self.model.diff(state, action).squeeze()

    def __len__(self):
        return len(self.model)
