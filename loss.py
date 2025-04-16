import torch.nn as nn
import torch
class SupConLoss(nn.Module):
    def __init__(self,temperature=0.07):
        super(SupConLoss,self).__init__()
        self.temperature = temperature

    def forward(self,features,labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()

        exp_logits = torch.exp(logits) * (1 - torch.eye(batch_size, device=device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos.mean()
        return loss