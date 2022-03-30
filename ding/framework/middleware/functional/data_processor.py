from typing import TYPE_CHECKING, Callable
from easydict import EasyDict
import logging
import torch
from ding.data import Buffer, Dataset, DataLoader, offline_data_save_type
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context


def data_pusher(cfg: EasyDict, buffer_: Buffer):

    def _push(ctx: "Context"):
        for t in ctx.trajectories:
            buffer_.push(t)
        ctx.trajectories = None

    return _push


def offpolicy_data_fetcher(cfg: EasyDict, buffer_: Buffer) -> Callable:

    def _fetch(ctx: "Context"):
        try:
            buffered_data = buffer_.sample(cfg.policy.learn.batch_size)
            assert buffered_data is not None
        except (ValueError, AssertionError):
            # You can modify data collect config to avoid this warning, e.g. increasing n_sample, n_episode.
            logging.warning(
                "Replay buffer's data is not enough to support training, so skip this trianing for waiting more data. "
            )
            ctx.train_data = None
            return
        ctx.train_data = [d.data for d in buffered_data]
        return
        # yield
        # TODO
        # buffer_.update(ctx.train_output)  # such as priority

    return _fetch


# TODO move ppo training for loop to new middleware
def onpolicy_data_fetcher(cfg: EasyDict, buffer_: Buffer) -> Callable:

    def _fetch(ctx: "Context"):
        ctx.train_data = ctx.trajectories
        ctx.train_data.traj_flag = torch.zeros(len(ctx.train_data))
        ctx.train_data.traj_flag[ctx.trajectory_end_idx] = 1
        yield

    return _fetch


def offline_data_fetcher(cfg: EasyDict, dataset: Dataset) -> Callable:
    # collate_fn is executed in policy now
    dataloader = DataLoader(dataset, batch_size=cfg.policy.learn.batch_size, shuffle=True, collate_fn=lambda x: x)

    def _fetch(ctx: "Context"):
        while True:
            for i, data in enumerate(dataloader):
                ctx.train_data = data
                yield
            ctx.train_epoch += 1
        # TODO apply data update (e.g. priority) in offline setting when necessary

    return _fetch


def offline_data_saver(cfg: EasyDict, data_path: str, data_type: str = 'hdf5') -> Callable:

    def _save(ctx: "Context"):
        data = ctx.trajectories
        offline_data_save_type(data, data_path, data_type)
        ctx.trajectories = None

    return _save


# ################ Algorithm-Specific ###########################


def sqil_data_fetcher(cfg: EasyDict, agent_buffer: Buffer, expert_buffer: Buffer) -> Callable:

    def _fetch(ctx: "Context"):
        try:
            agent_buffered_data = agent_buffer.sample(cfg.policy.learn.batch_size // 2)
            expert_buffered_data = expert_buffer.sample(cfg.policy.learn.batch_size // 2)
        except ValueError:
            logging.warning(
                "Replay buffer's data is not enough to support training." +
                "You can modify data collect config, e.g. increasing n_sample, n_episode."
            )
            # TODO
            return
        agent_data = [d.data for d in agent_buffered_data]
        expert_data = [d.data for d in expert_buffered_data]
        ctx.train_data = agent_data + expert_data

    return _fetch
