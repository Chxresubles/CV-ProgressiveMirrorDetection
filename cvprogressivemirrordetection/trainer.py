from typing import Optional
from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from cvprogressivemirrordetection import metrics
from cvprogressivemirrordetection.constants import EPSILON


class ModelTrainer:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading tensors on {self.device.type}")

    def compute_loss(self, pred_masks: Tensor, gt_masks: Tensor) -> Tensor:
        weit = 1 + 5 * torch.abs(
            F.avg_pool2d(gt_masks, kernel_size=31, stride=1, padding=15) - gt_masks
        )
        wbce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        pred_masks = torch.sigmoid(pred_masks)
        intersection = ((pred_masks * gt_masks) * weit).sum(dim=(2, 3))
        union = ((pred_masks + gt_masks) * weit).sum(dim=(2, 3))
        wiou = 1 - intersection / (union - intersection + EPSILON)
        return (wbce + wiou).mean()

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        n_epochs: Optional[int] = 20,
        lr: Optional[float] = 0.001,
        seed: Optional[int] = 42,
    ) -> dict:
        torch.manual_seed(seed=seed)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True)
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

        self.model.train()
        self.model.to(device=self.device)
        for i in range(n_epochs):
            tot_loss = 0.0
            tot_iou = 0.0
            tot_f_score = 0.0
            tot_mae = 0.0
            n_batches = len(dataloader)
            for images, masks in tqdm(dataloader, leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)
                optim.zero_grad()

                pred_masks = self.model(images)

                # Compute loss
                if isinstance(pred_masks, tuple):
                    loss = 0.0
                    for pred_mask in pred_masks:
                        loss += self.compute_loss(pred_mask, masks)
                else:
                    loss = self.compute_loss(pred_masks, masks)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    tot_loss += loss.item()

                    if isinstance(pred_masks, tuple):
                        pred_masks = pred_masks[-1]

                    pred_masks = torch.sigmoid(pred_masks)

                    # Compute IoU
                    tot_iou += metrics.iou(pred_masks, masks).item()

                    # Compute F-score
                    tot_f_score += metrics.f_score(pred_masks, masks).item()

                    # Compute MAE
                    tot_mae += metrics.mae(pred_masks, masks).item()

            scheduler.step()

            # Average metrics across batches
            tot_loss /= n_batches
            tot_iou /= n_batches
            tot_f_score /= n_batches
            tot_mae /= n_batches

            print(
                f"epoch {i + 1}/{n_epochs}: loss={tot_loss} iou={tot_iou} f_score={tot_f_score} mae={tot_mae}"
            )

        return self.evaluate(dataset)

    def evaluate(self, dataset: torch.utils.data.Dataset) -> dict:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=False)

        tot_iou = 0.0
        tot_f_score = 0.0
        tot_mae = 0.0
        n_batches = len(dataloader)

        self.model.eval()
        self.model.to(device=self.device)
        with torch.no_grad():
            for images, masks in tqdm(dataloader, leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)
                pred_masks = self.model(images)

                # Compute IoU
                tot_iou += metrics.iou(pred_masks, masks).item()

                # Compute F-score
                tot_f_score += metrics.f_score(pred_masks, masks).item()

                # Compute MAE
                tot_mae += metrics.mae(pred_masks, masks).item()

            # Average metrics across batches
            tot_iou /= n_batches
            tot_f_score /= n_batches
            tot_mae /= n_batches

        return {
            "iou": tot_iou,
            "f_score": tot_f_score,
            "mae": tot_mae,
        }
