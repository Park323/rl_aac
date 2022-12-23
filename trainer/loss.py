import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def policy_loss(logp, reward):
    reward = torch.tensor(reward, device=logp.device)
    return -1. * ( logp * reward ).mean()

def contrastive_loss(x1, x2, queue=None):
    device = x1.device

    t = torch.tensor(0.).to(device) # temperature
    logits = x1 @ x2.T * torch.exp(t)
    
    labels = torch.arange(logits.size(0)).to(device)
    loss_a = F.cross_entropy(logits, labels, reduction='mean')
    loss_t = F.cross_entropy(logits.T, labels, reduction='mean')
    loss = (loss_a + loss_t) / 2

    # n_samples = 0
    
    # loss = torch.tensor(0.).to(x1.device)
    
    # _, uq_index = torch.unique(queue, dim=0, return_inverse=True) # Find redundant target
    
    # for i, _output in enumerate(x1):
    #     _output = _output.unsqueeze(0)
    #     sims = F.cosine_similarity(_output, x2)
    #     positive_mask = (uq_index == uq_index[i]) #.to(sims.dtype) # detect the items with positive match.
        
    #     n_samples += 1
    #     loss -= torch.log(torch.exp(sims[positive_mask]).sum() / (torch.exp(sims[~positive_mask]).sum() + 1e-8)) 
    #     #loss += F.binary_cross_entropy_with_logits(sims, positive_mask)

    # for i, _output in enumerate(x1):
    #     sims = F.cosine_similarity(_output, x2)

    #     index = torch.arange(x1.size(0))
    #     index_mask = index[uq_index == uq_index[i]] # detect the items with positive match.

    #     numer = torch.exp(sims[index_mask]).sum()
    #     denom = torch.exp(sims[~index_mask]).sum() # Add non-similar objects' similarity
    #     loss -= torch.log(numer/denom)

    return loss