from typing import TYPE_CHECKING, Callable, List, Union, Tuple, Dict
from easydict import EasyDict
from collections import deque
import logging
import torch
from ding.data import Buffer, Dataset, DataLoader, offline_data_save_type
from ding.data.buffer.middleware import PriorityExperienceReplay
from ding.framework import task

if TYPE_CHECKING:
    from ding.framework import Context


def data_pusher(cfg: EasyDict, buffer_: Buffer):

    def _push(ctx: "Context"):
        for t in ctx.trajectories:
            buffer_.push(t)
        ctx.trajectories = None

    return _push


def offpolicy_data_fetcher(
        cfg: EasyDict, buffer_: Union[Buffer, List[Tuple[Buffer, float]], Dict[str, Buffer]]
) -> Callable:

    def _fetch(ctx: "Context"):
        try:
            if isinstance(buffer_, Buffer):
                buffered_data = buffer_.sample(cfg.policy.learn.batch_size)
                ctx.train_data = [d.data for d in buffered_data]
            elif isinstance(buffer_, List):  # like sqil, r2d3
                buffered_data = []
                for buffer_elem, p in buffer_:
                    data_elem = buffer_elem.sample(int(cfg.policy.learn.batch_size * p))
                    assert data_elem is not None
                    buffered_data.append(data_elem)
                buffered_data = sum(buffered_data, [])
                ctx.train_data = [d.data for d in buffered_data]
            elif isinstance(buffer_, Dict):  # like ppg_offpolicy
                buffered_data = {k: v.sample(cfg.policy.learn.batch_size) for k, v in buffer_.items()}
                ctx.train_data = {k: [d.data for d in v] for k, v in buffered_data.items()}
            else:
                raise TypeError("not support buffer argument type: {}".format(type(buffer_)))

            assert buffered_data is not None
        except (ValueError, AssertionError):
            # You can modify data collect config to avoid this warning, e.g. increasing n_sample, n_episode.
            logging.warning(
                "Replay buffer's data is not enough to support training, so skip this training for waiting more data."
            )
            ctx.train_data = None
            return

        yield

        if isinstance(buffer_, Buffer):
            if any([isinstance(m, PriorityExperienceReplay) for m in buffer_.middleware]):
                index = [d.index for d in buffered_data]
                meta = [d.meta for d in buffered_data]
                # such as priority
                if isinstance(ctx.train_output, deque):
                    priority = ctx.train_output.pop()['priority']
                else:
                    priority = ctx.train_output['priority']
                for idx, m, p in zip(index, meta, priority):
                    m['priority'] = p
                    buffer_.update(index=idx, data=None, meta=m)

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