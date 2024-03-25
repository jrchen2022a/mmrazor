# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


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
                updated_weight = round(1.0 - (runner.epoch - self.start_epoch)/self.delta , 2)
                runner.logger.info(f'Distillation loss change to {updated_weight}.')
                model.distillation_loss_weight = model.distillation_loss_weight

