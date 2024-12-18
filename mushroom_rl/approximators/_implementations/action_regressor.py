import numpy as np

from mushroom_rl.core import Serializable


class ActionRegressor(Serializable):
    """
    This class is used to approximate the Q-function with a different
    approximator of the provided class for each action. It is often used in MDPs
    with discrete actions and should not be used in MDPs with continuous
    actions.

    """
    def __init__(self, approximator, n_actions, **params):
        """
        Constructor.

        Args:
            approximator (class): the model class to approximate the
                Q-function of each action;
            n_actions (int): number of different actions of the problem. It
                determines the number of different regressors in the action
                regressor;
            **params: parameters dictionary to create each regressor.

        """
        self.model = list()
        self._n_actions = n_actions
        try:
            self._output_shape = params["output_shape"]
        except KeyError:
            self._output_shape = (1,)
        # For single objective, self._output_shape == (1,)
        # For multi objective, self._output_shape == (1, n_objectives)
        assert len(self._output_shape) == 1 or len(self._output_shape) == 2
        assert self._output_shape[0] == 1

        for i in range(self._n_actions):
            self.model.append(approximator(**params))

        self._add_save_attr(
            _n_actions='primitive',
            _output_shape='primitive',
            model=self._get_serialization_method(approximator)
        )

    def fit(self, state, action, q, **fit_params):
        """
        Fit the model.

        Args:
            state (np.ndarray): states;
            action (np.ndarray): actions;
            q (np.ndarray): target q-values;
            **fit_params: other parameters used by the fit method
                of each regressor.

        """
        for i in range(len(self.model)):
            idxs = np.argwhere((action == i)[:, 0]).ravel()

            if idxs.size:
                self.model[i].fit(state[idxs, :], q[idxs], **fit_params)

    def predict(self, *z, **predict_params):
        """
        Predict.

        Args:
            *z: a list containing states or states and actions depending
                on whether the call requires to predict all q-values or only
                one q-value corresponding to the provided action;
            **predict_params: other parameters used by the predict method
                of each regressor.

        Returns:
            The predictions of the model.

        """
        # Note: To avoid incorrect broadcasting in the loss function of TorchApproximator.fit(), when using
        # ActionRegressor then TorchApproximator.network(state) must return a tensor whose action dimension has been
        # squeezed. For this reason ActionRegressor.predict() must expect ActionRegressor.model[i].predict(state)
        # to return an array whose action dimension has been squeezed.

        assert len(z) == 1 or len(z) == 2

        state = z[0]
        if len(z) == 2:
            action = z[1]
            if len(self._output_shape) == 1:
                q = np.zeros(state.shape[0])
            else:
                q = np.zeros((state.shape[0], self._output_shape[1]))
            
            for i in range(self._n_actions):
                idxs = np.argwhere((action == i)[:, 0]).ravel()
                if idxs.size:
                    q[idxs] = self.model[i].predict(state[idxs], **predict_params)
        else:
            if len(self._output_shape) == 1:
                q = np.zeros((state.shape[0], self._n_actions))
            else:
                q = np.zeros((state.shape[0], self._n_actions, self._output_shape[1]))
            
            for i in range(self._n_actions):
                q[:, i] = self.model[i].predict(state, **predict_params)

        return q

    def reset(self):
        """
        Reset the model parameters.

        """
        try:
            for m in self.model:
                m.reset()
        except AttributeError:
            raise NotImplementedError('Attempt to reset weights of a'
                                      ' non-parametric regressor.')

    @property
    def weights_size(self):
        return self.model[0].weights_size * len(self.model)

    def get_weights(self):
        w = list()
        for m in self.model:
            w.append(m.get_weights())

        return np.concatenate(w, axis=0)

    def set_weights(self, w):
        size = self.model[0].weights_size
        for i, m in enumerate(self.model):
            start = i * size
            stop = start + size
            m.set_weights(w[start:stop])

    def diff(self, state, action):
        if action is None:
            diff = list()
            for m in self.model:
                diff.append(m.diff(state))

            return diff
        else:
            a = action[0]
            s = self.model[a].weights_size
            diff = np.zeros(s * len(self.model))
            diff[s * a:s * (a + 1)] = self.model[a].diff(state)

            return diff

    def __len__(self):
        return len(self.model[0])
