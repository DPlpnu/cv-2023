import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def __call__(self, x, y, label):
        dist = torch.nn.functional.pairwise_distance(x, y)

        loss = (1 - label) * torch.pow(dist, 2) + label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)

        return loss
