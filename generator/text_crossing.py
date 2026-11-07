"""Ordered signed response decomposition for one fixed first text write."""
from encoder_atoms import np


def decompose(old,new):
    names=['R0A','R0B','R1A','R1B'];a,b,c,d=[np.asarray(old[n],np.float64) for n in names];u,v=[np.asarray(new[n],np.float64) for n in ['A','B']]
    if any(x.shape!=a.shape for x in [b,c,d,u,v]) or not all(np.isfinite(x).all() for x in [a,b,c,d,u,v]):raise ValueError('Response geometry/finite contract differs')
    d0=b-a;d1=d-c;d10=v-u;interaction=d1-d0;remaining=d10-d0;text=d1-d10
    return {'responseR0':d0,'responseR1':d1,'responseR1TextR0':d10,'originalInteraction':interaction,'remainingReceiverDifference':remaining,'firstTextWriteEffect':text,'closure':interaction-remaining-text}


def account(fields):
    d0=fields['responseR0'];d1=fields['responseR1'];d10=fields['responseR1TextR0'];i=fields['originalInteraction'];u=fields['remainingReceiverDifference'];t=fields['firstTextWriteEffect']
    energy=lambda x:float(np.sum(x*x));dot=lambda a,b:float(np.sum(a*b));ei=energy(i);eu=energy(u);et=energy(t);cross=2*dot(u,t)
    rms=lambda x:float(np.sqrt(np.mean(x*x)))
    cosine=lambda a,b:dot(a,b)/(energy(a)*energy(b))**.5 if energy(a)*energy(b) else None
    return {'scalars':int(i.size),'responseR0Rms':rms(d0),'responseR1Rms':rms(d1),'responseR1TextR0Rms':rms(d10),'originalInteractionEnergy':ei,'remainingReceiverEnergy':eu,'firstTextWriteEnergy':et,
        'twiceRemainingText':cross,'energyClosure':ei-eu-et-cross,'remainingOverOriginalInteraction':(eu/ei)**.5 if ei else None,'textOverOriginalInteraction':(et/ei)**.5 if ei else None,
        'responseR0TextR0Cosine':cosine(d0,d10),'responseR1TextR0Cosine':cosine(d1,d10),'remainingTextCosine':cosine(u,t),'remainingOriginalCosine':cosine(u,i),'textOriginalCosine':cosine(t,i),'maxClosureAbs':float(np.max(np.abs(fields['closure'])))}
