import torch
import torch.nn as nn
import torch.nn.functional as F

class RecallTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        TP = (pred * target).sum(dim=1)
        FN = ((1 - pred) * target).sum(dim=1)
        FP = (pred * (1 - target)).sum(dim=1)

        tversky = TP / (TP + self.alpha * FN + self.beta * FP + 1e-7)
        loss = 1 - tversky
        return loss.mean()

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, from_logits=True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        """
        pred: (B,1,H,W)  logits 或 probabilities
        target: (B,H,W) 或 (B,1,H,W)   值 ∈ {0,1}
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)  # (B,1,H,W)

        if self.from_logits:
            pred_prob = torch.sigmoid(pred)
        else:
            pred_prob = pred

        pred_flat = pred_prob.contiguous().view(pred.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BinaryBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        pred: (B,1,H,W) logits
        target: (B,H,W) 或 (B,1,H,W)
        """
        if target.dim() == 3:
            target = target.unsqueeze(1)

        return self.loss_fn(pred, target.float())


class StandardSegLoss(nn.Module):
    """
    Final: CT 分支使用二分类的 BCE + Dice
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, from_logits=True):
        super().__init__()
        self.bce = BinaryBCEWithLogitsLoss()
        self.dice = BinaryDiceLoss(from_logits=False)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        """
        pred: (B,1,H,W)  —— logits
        target: (B,H,W) 或 (B,1,H,W) —— int {0,1}
        """
        bce = self.bce(pred, target)
        dice = self.dice(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice