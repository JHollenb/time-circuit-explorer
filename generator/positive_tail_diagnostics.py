"""Post-seal exact-tail diagnostics and signed errors; never used by serving."""
from encoder_atoms import np,floats
from context_statistics_study import exact_head,summary
from context_transport import nearest
from sparse_tilt_study import logsum
from centroid_tilt_study import direct_centroid
from tail_factorial import ordered_support
from tail_factorial_study import energy
from positive_tail import positive_read

METHODS=['quadratic','00','11']
TERMS=['mass','value','controlArithmetic']
VECTORS=[m+'__'+k for m in ['quadratic','00'] for k in TERMS]+['gaussianValueRemainder']
SCALARS=['hasTail','quadraticTailWeight','centroidTailWeight','exactTailWeight','meanT','varianceT','minimumT','maximumT','logKappa2','centeredLogKappa',
    'massGaugeLogKappa','centeringGaugeDefect','gaussianLogRemainder','anchorMassBelowMinus1','exactMassBelowMinus1','quadraticMassBelowMinus1',
    'logKappa2MomentClosure','tailValueMomentClosure','quadraticTailLogMassError','quadraticTailValueErrorRelativeRms','gaussianTailValueErrorRelativeRms']


def diagnose_head(p,asset,frozen,budget):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;x=q/d**.25;u=k/d**.25-asset['keyCenter'];routes=nearest(x,asset)[0]
    if not np.array_equal(routes,frozen['nearestIndex']):raise ValueError('Frozen diagnostic route differs')
    banks={m:np.zeros((n,d)) for m in METHODS};banks['11']=exact_head(p)['raw'];fields={k:np.zeros((n,d)) for k in VECTORS};fields.update({k:np.full(n,np.nan) for k in SCALARS});fields['hasTail'][:]=0
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1]);selected=ordered_support(frozen['supportPositions'][i],allowed,budget)
        if not len(allowed):continue
        U=u[allowed];V=v[allowed];ell=asset['landmarks'][routes[i]];s=U@ell;delta=x[i]-ell;raw,z=positive_read(s,U,V,selected,delta)
        banks['quadratic'][i]=raw;banks['00'][i]=direct_centroid(s,U,V,selected,delta)
        if not z['hasTail']:
            for m in ['quadratic','00']:fields[m+'__controlArithmetic'][i]=banks[m][i]-banks['11'][i]
            for key in ['quadraticTailWeight','centroidTailWeight','exactTailWeight']:fields[key][i]=0.
            continue
        anchor=z['anchor'];t=z['t'];vt=z['values'];h0=anchor@vt;h2=z['tailMean'];hs=z['selectedMean'];at=z['currentScores'][z['tail']]
        lze=logsum(at);pe=np.exp(at-lze);he=pe@vt;we=float(np.exp(lze-logsum(np.array([z['logSelectedMass'],lze]))));w0=float(np.exp(z['logCentroidTailMass']-logsum(np.array([z['logSelectedMass'],z['logCentroidTailMass']]))))
        for m,w,h in [('quadratic',z['tailWeight'],h2),('00',w0,h0)]:
            A=(w-we)*(h-hs);B=we*(h-he);fields[m+'__mass'][i]=A;fields[m+'__value'][i]=B;fields[m+'__controlArithmetic'][i]=banks[m][i]-banks['11'][i]-A-B
        mt=float(anchor@t);second=float(anchor@(t*t));var=float(anchor@((t-mt)**2));cov1=(anchor*(t-mt))@vt;cov2=(anchor*(t*t-second))@vt;k2=1+mt+.5*second
        centered=logsum(s[z['tail']]+t)-z['logAnchorTailMass'];masslog=lze-z['logCentroidTailMass'];gauss=he-h0-cov1;fields['gaussianValueRemainder'][i]=gauss
        below=t<-1.;den=float(np.sqrt(np.mean((he-h0)**2)));rms=lambda a:float(np.sqrt(np.mean(a*a)))
        values={'hasTail':1.,'quadraticTailWeight':z['tailWeight'],'centroidTailWeight':w0,'exactTailWeight':we,'meanT':mt,'varianceT':var,'minimumT':float(t.min()),'maximumT':float(t.max()),
            'logKappa2':z['logKappa2'],'centeredLogKappa':centered,'massGaugeLogKappa':masslog,'centeringGaugeDefect':masslog-centered,'gaussianLogRemainder':centered-mt-.5*var,
            'anchorMassBelowMinus1':float(anchor[below].sum()),'exactMassBelowMinus1':float(pe[below].sum()),'quadraticMassBelowMinus1':float(z['tailProbability'][below].sum()),
            'logKappa2MomentClosure':z['logKappa2']-float(np.log(k2)),'tailValueMomentClosure':float(np.max(np.abs(h2-h0-(cov1+.5*cov2)/k2))),
            'quadraticTailLogMassError':z['logQuadraticTailMass']-lze,'quadraticTailValueErrorRelativeRms':rms(h2-he)/den if den else np.nan,'gaussianTailValueErrorRelativeRms':rms(gauss)/den if den else np.nan}
        for key,value in values.items():fields[key][i]=value
    return banks,fields


def head_rows(label,banks,masks,d,heads):
    out=[]
    for m in METHODS:
        for precision in ['raw','rounded']:
            a=banks[m][precision].reshape(-1,heads,d);b=banks['11'][precision].reshape(-1,heads,d)
            for region,use in masks:
                if not use.any():continue
                for h in [None,*range(heads)]:out.append({**label,'method':m,'precision':precision,'region':region,'queryHead':h,'score':summary(a[use] if h is None else a[use,h],b[use] if h is None else b[use,h])})
    return out


def decompositions(label,banks,fields,masks,d,heads):
    out=[]
    for m in ['quadratic','00']:
        raw=banks[m]['raw']-banks['11']['raw'];rounded=banks[m]['rounded']-banks['11']['rounded'];terms={k:fields[m+'__'+k] for k in TERMS};terms['rounding']=rounded-raw
        terms={k:a.reshape(-1,heads,d) for k,a in terms.items()};err=rounded.reshape(-1,heads,d)
        for region,use in masks:
            if not use.any():continue
            for h in [None,*range(heads)]:
                part=lambda a:a[use] if h is None else a[use,h]
                out.append({**label,'method':m,'region':region,'queryHead':h,**energy(part(err),{k:part(a) for k,a in terms.items()})})
    return out
