import torch.optim


def create_optimizer(model, lr):
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    return optimizer

