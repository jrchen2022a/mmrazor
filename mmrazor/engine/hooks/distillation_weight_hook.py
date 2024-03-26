import math

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


def getOneCycleDescend(epoch, epochs) -> float:
    return 1 - round(1/2*(1-math.cos((math.pi * epoch)/epochs)), 2)


@HOOKS.register_module()
class DistillWeightHook(Hook):

    def __init__(self, start_epoch, stop_epoch: int) -> None:
        self.stop_epoch = stop_epoch
        self.start_epoch = start_epoch
        self.delta = stop_epoch - start_epoch

    def before_train_epoch(self, runner) -> None:
        if runner.epoch > self.start_epoch:
            """Change the weight of distillation loss."""
            model = runner.model
            # TODO: refactor after mmengine using model wrapper
            if is_model_wrapper(model):
                model = model.module

            assert hasattr(model, 'distillation_loss_weight')
            assert hasattr(model, 'distillation_stopped')

            if not model.distillation_stopped:
                updated_weight = getOneCycleDescend(runner.epoch - self.start_epoch, self.delta)
                runner.logger.info(f'Distillation loss weight change to {updated_weight}.')
                model.distillation_loss_weight = model.distillation_loss_weight
