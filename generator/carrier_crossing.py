"""Complete carrier replacement and signed 2x2 accounting."""
from encoder_atoms import np


def replace_geometry(template,donor):
    import torch
    if template.shape!=donor.shape or template.dtype!=donor.dtype or template.device!=donor.device or not template.numel():raise ValueError('Carrier ABI differs')
    shape=tuple(template.shape);stride=tuple(template.stride());offset=template.storage_offset()
    if any(v<0 for v in stride) or torch._debug_has_internal_overlap(template)==1:raise ValueError('Overlapping/negative receiver layout')
    required=offset+1+sum((n-1)*s for n,s in zip(shape,stride));storage=torch.empty(required,dtype=template.dtype,device=template.device)
    value=torch.as_strided(storage,shape,stride,offset);value.copy_(donor)
    if value.untyped_storage().data_ptr() in {template.untyped_storage().data_ptr(),donor.untyped_storage().data_ptr()}:raise ValueError('Carrier aliases receiver/donor storage')
    return value


def factorial(cells):
    a,b,c,d=[np.asarray(cells[k],np.float64) for k in ['R0A','R0B','R1A','R1B']]
    if any(x.shape!=a.shape for x in [b,c,d]) or not all(np.isfinite(x).all() for x in [a,b,c,d]):raise ValueError('Crossing field geometry/finite contract differs')
    d0=b-a;d1=d-c;receiver=c-a;interaction=d1-d0;diagonal=d-a;closure=diagonal-receiver-d0-interaction
    energy=lambda x:float(np.sum(x*x));cross=lambda x,y:float(2*np.sum(x*y))
    e={'receiverEnergy':energy(receiver),'carrierEnergy':energy(d0),'interactionEnergy':energy(interaction),'diagonalEnergy':energy(diagonal),
        'twiceReceiverCarrier':cross(receiver,d0),'twiceReceiverInteraction':cross(receiver,interaction),'twiceCarrierInteraction':cross(d0,interaction)}
    e['energyClosure']=e['diagonalEnergy']-e['receiverEnergy']-e['carrierEnergy']-e['interactionEnergy']-e['twiceReceiverCarrier']-e['twiceReceiverInteraction']-e['twiceCarrierInteraction']
    return {'responseR0':d0,'responseR1':d1,'receiverEffectA':receiver,'interaction':interaction,'diagonalChange':diagonal,'closure':closure},e
