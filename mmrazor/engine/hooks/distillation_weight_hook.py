# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmrazor.registry import HOOKS


@HOOKS.register_module()
class DistillWeightHook(Hook):

    # def __init__(self, stop_epoch: int) -> None:
    #     self.stop_epoch = stop_epoch

    def before_train_epoch(self, runner) -> None:
        """Change the weight of distillation loss."""
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        assert hasattr(model, 'distillation_loss_weight')
        updated_weight = 1.0 - runner.epoch/runner.train_loop.max_epochs
        runner.logger.info(f'Distillation loss change to {updated_weight}.')
        model.distillation_loss_weight = model.distillation_loss_weight

