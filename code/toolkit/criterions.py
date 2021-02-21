import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_criterion(name):
    if name == "cross_entropy":
        loss = nn.CrossEntropyLoss().to(device)
    elif name == "contrastive_loss":
        loss = ContrastiveLoss().to(device)
    elif name == "l1":
        loss = nn.L1Loss().to(device)
    return loss


def create_regularizer(args):
    param, function = 0, no_regularization
    if args.regularization == "doubly_stochastic_attention":
        param = args.alpha_c
        function = doubly_stochastic_attention
    return param, function


def no_regularization(arg1, arg2):
    return 0


def doubly_stochastic_attention(alpha_c, alphas):
    return alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()


# ============================================================================ #
class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, max_violation=True):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.max_violation = max_violation

    def forward(self, images_embedded, captions_embedded):
        # compute image-caption score matrix
        scores = cosine_sim(images_embedded, captions_embedded)
        diagonal = scores.diag().view(images_embedded.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        I = Variable(mask).to(device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # Sum up caption retrieval and image retrieval loss
        sum_of_losses = cost_s.sum() + cost_im.sum()

        # Normalize loss by batch size
        normalized_loss = sum_of_losses / images_embedded.size(0)

        return normalized_loss


def cosine_sim(images_embedded, captions_embedded):
    """Cosine similarity between all the image and sentence pairs
    """
    return images_embedded.mm(captions_embedded.t())
