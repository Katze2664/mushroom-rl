from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
import numpy as np
from mushroom_rl.utils.record import VideoRecorder


class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None, record_dictionary=None, agent_info_keys=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of each fit;
            callback_step (Callback): callback to execute after each step;

        """
        self.agent = agent
        self.mdp = mdp
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None
        self.agent_info_keys = agent_info_keys

        self._state = None
        self._agent_info = None

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

        if record_dictionary is None:
            record_dictionary = dict()
        self._record = self._build_recorder_class(**record_dictionary)

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, record=False):
        """
        This function moves the agent in the environment and fits the policy using the collected samples.
        The agent can be moved for a given number of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a given number of episodes.
        The environment is reset at the beginning of the learning process.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True.

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        assert (render and record) or (not record), "To record, the render flag must be set to true"

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition = lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter >= self._n_episodes_per_fit

        self._run(n_steps, n_episodes, fit_condition, render, quiet, record, get_env_info=False)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, record=False, get_env_info=False, seeds=None, get_action_info=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from a set of initial states or random seeds
        for the whole episode. The environment is reset at the beginning of the learning process.

        Args:
            initial_states (np.ndarray, None): the starting states of each episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True;
            get_env_info (bool, False): whether to include the environment info in the dataset info;
            seeds (np.ndarray, None): the seeds to initialise the randomisation of the starting state of each episode;
            get_action_info (bool, False): whether to include the agent's draw action info in the dataset info.

        Returns:
            The collected dataset and, optionally, an extra dataset containing the
            environment info and/or agent's draw action info, collected at each step.
        """
        assert (render and record) or (not record), "To record, the render flag must be set to true"

        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet, record, get_env_info, initial_states, seeds, get_action_info)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet, record, get_env_info, initial_states=None, seeds=None, get_action_info=False):
        assert sum([n_episodes is not None, n_steps is not None, initial_states is not None, seeds is not None]) == 1, (
            "Exactly one of n_episodes, n_steps, initial_states or seeds must be not None.\n"
            f"{n_episodes=}\n{n_steps=}\n{initial_states=}\n{seeds=}"
        )

        if n_episodes is not None:
            self._n_episodes = n_episodes
        elif initial_states is not None:
            self._n_episodes = len(initial_states)
        elif seeds is not None:
            self._n_episodes = len(seeds)
        else:
            self._n_episodes = None

        if n_steps is not None:
            move_condition = lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,  dynamic_ncols=True, disable=quiet, leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes, dynamic_ncols=True, disable=quiet, leave=False)

        dataset, dataset_info = self._run_impl(move_condition, fit_condition, steps_progress_bar, episodes_progress_bar,
                                               render, record, initial_states, seeds, get_action_info)

        if get_env_info or get_action_info:
            return dataset, dataset_info
        else:
            return dataset

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar, episodes_progress_bar, render, record,
                  initial_states, seeds, get_action_info):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        dataset_info = defaultdict(list)

        last = True
        while move_condition():
            if last:
                self.reset(initial_states=initial_states, seeds=seeds)

            sample, step_info = self._step(render, record, get_action_info)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)

            for key, value in step_info.items():
                dataset_info[key].append(value)

            if fit_condition():
                self.agent.fit(dataset, **dataset_info)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()
                dataset_info = defaultdict(list)

            last = sample[-1]

        self.agent.stop()
        self.mdp.stop()

        if record:
            self._record.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset, dataset_info

    def _step(self, render, record, get_action_info):
        """
        Single step.

        Args:
            render (bool): whether to render or not.
            get_action_info (bool): whether to include the agent's draw action info in the step info.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        if get_action_info:
            if self.agent_info_keys is None:
                action, action_info = self.agent.draw_action(self._state, get_action_info=True)
            else:
                action, action_info = self.agent.draw_action(self._state, info=self._agent_info, get_action_info=True)
        else:
            action_info = {}
            if self.agent_info_keys is None:
                action = self.agent.draw_action(self._state)
            else:
                action = self.agent.draw_action(self._state, info=self._agent_info)

        next_state, reward, absorbing, step_info = self.mdp.step(action)

        overlapping_keys = action_info.keys() & step_info.keys()
        assert not overlapping_keys, (
            "action_info and step_info have overlapping keys."
            f"{overlapping_keys=}"
            f"{action_info.keys()=}"
            f"{step_info.keys()=}"
        )
        step_info.update(action_info)

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record)

            if record:
                self._record(frame)

        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state
        if self.agent_info_keys is not None:
            self._agent_info = {key: deepcopy(step_info[key]) for key in self.agent_info_keys}  # deepcopy is just a precaution

        return (state, action, reward, next_state, absorbing, last), step_info

    def reset(self, initial_states=None, seeds=None):
        """
        Reset the state of the agent.

        """
        assert (initial_states is None) or (seeds is None), (
            "At least one of `initial_states` or `seeds` must be None. Providing both is not permitted.\n"
            f"{initial_states=}\n"
            f"{seeds=}"
        )

        if self._total_episodes_counter == self._n_episodes:  # TODO: In what circumstances would this occur?
            # move_condition in _run_impl() ensures self._total_episodes_counter < self._n_episodes, so when could
            # self._total_episodes_counter == self._n_episodes? Is it only when reset is called directly by the user?
            assert (initial_states is None) and (seeds is None), (
                "if self._total_episodes_counter == self._n_episodes, initial_states and seeds must both be None.\n"
                f"{self._total_episodes_counter=}, {self._n_episodes=}, {initial_states=}, {seeds=}"
            )

        if initial_states is None:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        if seeds is None:
            seed = None
        else:
            seed = seeds[self._total_episodes_counter]
            assert isinstance(seed, (int, np.integer)), f"seed must be an integer (Python or NumPy), but got type {type(seed)}"
            seed = int(seed)

        self.agent.episode_start()

        reset_output = self.mdp.reset(state=initial_state, seed=seed)
        if isinstance(reset_output, tuple):
            state, step_info = reset_output  # For Gymnasium environments
        else:
            state = reset_output  # For Gym environments
            step_info = {}
        self._state = self._preprocess(state.copy())
        if self.agent_info_keys is not None:
            self._agent_info = {key: deepcopy(step_info[key]) for key in self.agent_info_keys}  # deepcopy is just a precaution
        self.agent.next_action = None
        self._episode_steps = 0

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self.agent.preprocessors:
            state = p(state)

        return state

    def _build_recorder_class(self, recorder_class=None, fps=None, **kwargs):
        """
        Method to create a video recorder class.

        Args:
            recorder_class (class): the class used to record the video. By default, we use the ``VideoRecorder`` class
                from mushroom. The class must implement the ``__call__`` and ``stop`` methods.

        Returns:
             The recorder object.

        """

        if not recorder_class:
            recorder_class = VideoRecorder

        if not fps:
            fps = int(1 / self.mdp.info.dt)

        return recorder_class(fps=fps, **kwargs)
