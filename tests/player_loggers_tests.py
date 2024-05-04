import pickle
import unittest

import numpy as np

from rlgym_ppo_loggers.player_loggers import PlayerVelocityLogger, PlayerHeightLogger, PlayerBoostLogger, \
    PlayerFlipTimeLogger, PlayerRelDistToBallLogger

with open("resources/states", "rb") as f:
    states = pickle.load(f)


class PlayerVelocityLoggerTest(unittest.TestCase):
    def test_collect_metrics(self):
        game_state = states[0]

        logger = PlayerVelocityLogger()
        metrics = logger.collect_metrics(game_state)

        self.assertEqual(metrics.shape, (4,))
        self.assertAlmostEqual(metrics[1], np.linalg.norm([6.25, 6.25, 0.14]), delta=1e-3)
        self.assertAlmostEqual(metrics[3], np.linalg.norm([-0.06406, 0.036, 0.]), delta=1e-3)


class PlayerHeightLoggerTest(unittest.TestCase):
    def test_collect_metrics(self):
        game_state = states[0]

        logger = PlayerHeightLogger()
        metrics = logger.collect_metrics(game_state)

        self.assertEqual(metrics.shape, (2,))
        self.assertAlmostEqual(float(metrics[1]), 17., delta=1e-3)


class PlayerBoostLoggerTest(unittest.TestCase):
    def test_collect_metrics(self):
        game_state = states[0]

        logger = PlayerBoostLogger()
        metrics = logger.collect_metrics(game_state)

        self.assertEqual(metrics.shape, (2,))
        self.assertAlmostEqual(float(metrics[1]), 0.3272, delta=1e-3)


class PlayerFlipTimeLoggerTest(unittest.TestCase):
    def test_collect_metrics(self):
        logger = PlayerFlipTimeLogger()

        metrics = np.array([])

        for i in range(7):
            game_state = states[i]
            metrics = logger.collect_metrics(game_state)

        self.assertEqual(metrics.shape, (2,))
        self.assertEqual(float(metrics[1]), 6.0)

        game_state = states[7]
        metrics = logger.collect_metrics(game_state)
        self.assertEqual(metrics[1], 0.0)


# TODO: WallTimeLoggerTests
# TODO: WallHeightLoggerTests
# TODO: CeilingHeightLoggerTests

class PlayerRelDistToBallLoggerTest(unittest.TestCase):
    def test_collect_metrics(self):
        game_state = states[0]
        logger = PlayerRelDistToBallLogger()
        metrics = logger.collect_metrics(game_state)

        self.assertEqual(metrics.shape, (2,))
        self.assertAlmostEqual(
            float(metrics[1]),
            np.linalg.norm(np.array([0.0, 0.0, 92.99999]) - np.array([-2047.95, -2559.95, 17.])),
            delta=1e-3
        )


if __name__ == '__main__':
    unittest.main()
