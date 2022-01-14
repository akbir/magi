"""Tests for MPO agent."""
from absl.testing import absltest
import acme
from acme import specs
from acme.testing import fakes
from acme.utils import loggers

from magi.agents.mpo import agent as agent_lib
from magi.agents.mpo import networks
from magi.agents.mpo.config import MPOConfig


class MPOTestCase(absltest.TestCase):
    """Integration tests for MPO agent"""

    def test_mpo_local_agent(self):
        # Create a fake environment to test with.
        environment = fakes.ContinuousEnvironment(
            action_dim=2, observation_dim=3, episode_length=10, bounded=True
        )
        spec = specs.make_environment_spec(environment)

        # Make network purely functional
        agent_networks = networks.make_networks(
            spec,
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(32, 32),
        )

        # Construct the agent.
        agent = agent_lib.MPO(
            environment_spec=spec,
            networks=agent_networks,
            config=MPOConfig(
                min_replay_size=1,
                batch_size=1,
            ),
            seed=0,
        )

        # Try running the environment loop. We have no assertions here because all
        # we care about is that the agent runs without raising any errors.
        loop = acme.EnvironmentLoop(
            environment,
            agent,
            logger=loggers.make_default_logger(label="environment", save_data=False),
        )
        loop.run(num_episodes=10)


if __name__ == "__main__":
    absltest.main()
