import numpy as np

from mushroom_rl.algorithms.value.dqn import AbstractDQN


class DQN(AbstractDQN):
    """
    Deep Q-Network algorithm.
    "Human-Level Control Through Deep Reinforcement Learning".
    Mnih V. et al.. 2015.

    """
    def _next_q(self, next_state, absorbing):
        q = self.target_approximator.predict(next_state, **self._predict_params)
        if np.any(absorbing):
            q *= 1 - absorbing.reshape(-1, 1)

        return np.max(q, axis=1)

class MO_DQN(AbstractDQN):

    def __init__(self,
                 mdp_info,
                 policy,
                 approximator,
                 approximator_params,
                 batch_size,
                 target_update_frequency,
                 scalarizer,
                 replay_memory=None,
                 initial_replay_size=500,
                 max_replay_size=5000,
                 fit_params=None,
                 predict_params=None,
                 clip_reward=False):
        
        self.scalarizer = scalarizer
        super().__init__(mdp_info,
                         policy,
                         approximator,
                         approximator_params,
                         batch_size,
                         target_update_frequency,
                         replay_memory=replay_memory,
                         initial_replay_size=initial_replay_size,
                         max_replay_size=max_replay_size,
                         fit_params=fit_params,
                         predict_params=predict_params,
                         clip_reward=clip_reward)

    def _next_q(self, next_state, absorbing):
        next_q_all_actions = self.target_approximator.predict(next_state, **self._predict_params)
        assert len(next_q_all_actions.shape) == 3, f"{next_q_all_actions.shape=}"
        batch_size, n_actions, n_objectives = next_q_all_actions.shape

        next_q_scalarized = np.apply_along_axis(func1d=self.scalarizer, axis=2, arr=next_q_all_actions)
        assert next_q_scalarized.shape == (batch_size, n_actions), f"{next_q_scalarized.shape=} {(batch_size, n_actions)=}"

        max_actions = np.argmax(next_q_scalarized, axis=1)
        assert max_actions.shape == (batch_size,), f"{max_actions.shape=} {(batch_size,)=}"

        next_q = next_q_all_actions[np.arange(batch_size), max_actions, :]
        assert next_q.shape == (batch_size, n_objectives), f"{next_q.shape=} {(batch_size, n_objectives)=}"

        assert absorbing.shape == (batch_size,), f"{absorbing.shape=} {(batch_size,)=}"
        assert absorbing.dtype == bool, f"{absorbing.dtype=}"
        next_q[absorbing, :] = 0
        
        return next_q
