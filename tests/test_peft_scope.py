import unittest

import torch
from torch import nn


class DummyMoment(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 2)


class PeftScopeTests(unittest.TestCase):
    def test_frozen_mode_freezes_moment_and_trains_adapter_only(self):
        from event_post_training.trainer import configure_trainable_parameters
        from agents.event_adapter import EventResidualAdapter

        model = DummyMoment()
        adapter = EventResidualAdapter(input_dim=2, hidden_dim=4)
        trainable = configure_trainable_parameters(model, adapter, mode="frozen_moment")

        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.parameters()))
        self.assertTrue(all(name.startswith("adapter.") for name in trainable))

    def test_peft_mode_only_unfreezes_forecast_head_and_adapter(self):
        from event_post_training.trainer import configure_trainable_parameters
        from agents.event_adapter import EventResidualAdapter

        model = DummyMoment()
        adapter = EventResidualAdapter(input_dim=2, hidden_dim=4)
        trainable = configure_trainable_parameters(model, adapter, mode="peft_moment")

        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        self.assertTrue(all(p.requires_grad for p in adapter.parameters()))
        self.assertIn("moment.head.weight", trainable)
        self.assertNotIn("moment.backbone.weight", trainable)


if __name__ == "__main__":
    unittest.main()
