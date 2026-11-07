"""Derive demanded source operands and preserve original native projection stages."""


def derive(text,mod,cosine,sine,support):
    return {'Ptext':text,'R':text[:,support],'txtMod':mod,
            'fullRopeCos':cosine,'fullRopeSin':sine,'ropeCos':cosine[support],'ropeSin':sine[support]}


def project_norm(x,weight,gamma,heads):
    import torch.nn.functional as F
    width=gamma.numel()
    projected=F.linear(x,weight).unflatten(-1,(heads,width))
    normal=F.rms_norm(projected,(width,),gamma,eps=1e-6)
    return projected,normal
