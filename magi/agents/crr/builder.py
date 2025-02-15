"""CRR agent builder."""
from typing import Any, Callable, Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import reverb
from reverb import rate_limiters

from magi.agents.crr import config as crr_config
from magi.agents.crr import learning

PolicyNetwork = Any


class CRRBuilder(builders.ActorLearnerBuilder):
    """CRR agent specification"""

    def __init__(
        self,
        config: crr_config.CRRConfig,
        logger_fn: Callable[[], loggers.Logger] = lambda: None,
    ):
        self._config = config
        self._logger_fn = logger_fn

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            # TODO(yl): support prioritized sampling
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=rate_limiters.MinSize(self._config.min_replay_size),
            signature=adders_reverb.NStepTransitionAdder.signature(
                environment_spec=environment_spec
            ),
        )
        return [replay_table]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            transition_adder=True,
        )
        return dataset.as_numpy_iterator()

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the actor/environment.

        Args:
          replay_client: Reverb Client which points to the replay server.
        """
        # TODO(yl): support multi step transitions
        return adders_reverb.NStepTransitionAdder(
            client=replay_client, n_step=1, discount=self._config.discount
        )

    def make_actor(
        self,
        random_key: networks_lib.PRNGKey,
        policy_network: PolicyNetwork,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Actor:
        """Create an actor instance."""
        assert variable_source is not None
        variable_client = variable_utils.VariableClient(variable_source, "policy")
        variable_client.update_and_wait()
        actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy_network)

        return actors.GenericActor(
            actor_core, random_key, variable_client=variable_client, adder=adder
        )

    def make_learner(
        self,
        random_key: networks_lib.PRNGKey,
        networks,
        dataset: Iterator[reverb.ReplaySample],
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
    ):
        del replay_client
        learner = learning.CRRLearner(
            policy_network=networks["policy"],
            critic_network=networks["critic"],
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            random_key=random_key,
            dataset=dataset,
            discount=self._config.discount,
            target_update_period=self._config.target_update_period,
            num_action_samples_td_learning=self._config.num_action_samples_td_learning,
            num_action_samples_policy_weight=self._config.num_action_samples_policy_weight,
            baseline_reduce_function=self._config.baseline_reduce_function,
            policy_improvement_modes=self._config.policy_improvement_modes,
            ratio_upper_bound=self._config.ratio_upper_bound,
            beta=self._config.beta,
            logger=self._logger_fn(),
            counter=counter,
        )
        return learner
