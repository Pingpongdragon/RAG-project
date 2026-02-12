"""
Temperature Scaling

论文:
    Guo et al., "On Calibration of Modern Neural Networks"
    ICML 2017  |  https://arxiv.org/abs/1706.04599
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, model, val_loader, device, max_iter: int = 50) -> float:
        model.eval()
        nll = nn.CrossEntropyLoss()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device)
                ).logits
                all_logits.append(logits)
                all_labels.append(batch['hard_label'].to(device))

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        opt = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = nll(self.forward(all_logits), all_labels)
            loss.backward()
            return loss

        opt.step(closure)
        T = self.temperature.item()
        logger.info(f"✅ 校准完成, T = {T:.3f}")
        return T