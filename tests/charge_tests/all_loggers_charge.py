import time
from typing import List

import numpy as np
import rlgym_sim
import tqdm
from rlgym_sim.utils.gamestates import GameState

from rlgym_ppo_loggers.ball_loggers import get_all_ball_loggers
from rlgym_ppo_loggers.global_loggers import get_all_global_loggers, WandbMetricsLogger
from rlgym_ppo_loggers.player_loggers import get_all_player_loggers

class TimedLogger(WandbMetricsLogger):
    def __init__(self, logger: WandbMetricsLogger, standalone_mode: bool = False):
        super().__init__(standalone_mode)
        self.logger = logger
        self.timer = 0.

    @property
    def metrics(self) -> List[str]:
        return self.logger.metrics

    def collect_metrics(self, game_state: GameState, car_id: int = -1) -> np.ndarray:
        t1 = time.time()
        metrics = self.logger.collect_metrics(game_state, car_id)
        self.timer += (time.time() - t1)
        return metrics


if __name__ == "__main__":
    loggers = [
        *get_all_ball_loggers(),
        *get_all_global_loggers(),
        *get_all_player_loggers()
    ]
    for i in range(len(loggers)):
        loggers[i] = TimedLogger(loggers[i])


    tick_skip = 8

    env = rlgym_sim.make(
        team_size=3,
        tick_skip=tick_skip,
        spawn_opponents=True
    )

    steps = 1_000 * tick_skip
    n_steps = 0
    env.reset()

    progress = tqdm.tqdm(desc="Steps progression")

    base_time = time.time()

    while n_steps < steps:
        actions = [env.action_space.sample() for _ in range(6)]
        _, _, _, info = env.step(actions)

        n_steps += tick_skip
        progress.update(tick_skip)

    env.close()
    progress.close()

    total_collection_time = time.time() - base_time

    loggers.sort(key=lambda l: l.timer, reverse=True)
    print(f"Collected {n_steps} steps in {total_collection_time:.2f} seconds")
    timestep_collection_time = (total_collection_time / n_steps) * 1_000

    progress = tqdm.tqdm(desc="Steps progression")

    base_time_w_loggers = time.time()

    n_steps = 0

    while n_steps < steps:

        actions = [env.action_space.sample() for _ in range(6)]
        _, _, _, info = env.step(actions)

        for l in loggers:
            l.collect_metrics(info["state"])

        n_steps += tick_skip
        progress.update(tick_skip)

    env.close()
    progress.close()

    total_collection_time_w_loggers = time.time() - base_time_w_loggers
    sps = n_steps / total_collection_time
    sps_w_loggers = n_steps / total_collection_time_w_loggers

    print("SPS without logger:", sps)
    print("SPS with logger:", sps_w_loggers)


    time_diff = total_collection_time_w_loggers - total_collection_time

    loggers.sort(key=lambda l: l.timer, reverse=True)
    print(f"Collected {n_steps} steps in {total_collection_time_w_loggers:.2f} seconds")

    print("SPS loss:", sps - sps_w_loggers)
    for l in loggers:
        print(f"{l.logger.__class__.__name__: <40}", ":", f"{l.timer * 100 / time_diff:.4f}", "% of sps loss")




