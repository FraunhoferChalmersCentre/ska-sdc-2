from abc import ABC

import torch
from torchmetrics import Metric


class IncrementalMetric(Metric, ABC):
    def __init__(self):
        super().__init__()

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class IncrementalDice(IncrementalMetric):
    def __init__(self):
        super().__init__()
        self.add_state("nominator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape
        logits = torch.nn.Sigmoid()(preds)

        self.nominator += torch.sum(logits * target).to('cpu')
        self.denominator += torch.sum(logits).to('cpu') + torch.sum(target).to('cpu')

    def compute(self):
        if self.denominator == torch.tensor(0.0):
            return torch.tensor(0.0)
        return torch.tensor(1.0) - torch.tensor(2.0) * self.nominator / self.denominator


class IncrementalAverageMetric(IncrementalMetric):
    def __init__(self, metric_fct):
        super().__init__()
        self.metric_fct = metric_fct
        self.add_state("cumulative_metric", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("numel", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.cumulative_metric += self.metric_fct(preds, target).to('cpu') * torch.numel(preds)
        self.numel += torch.numel(preds)

    def compute(self):
        return self.cumulative_metric / self.numel


class IncrementalCombo(IncrementalMetric):
    def __init__(self, *metrics: IncrementalMetric):
        super().__init__()
        self.metrics = metrics

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for m in self.metrics:
            m.update(preds, target)

    def compute(self):
        m = torch.tensor([m.compute() for m in self.metrics])
        return torch.sum(m)
