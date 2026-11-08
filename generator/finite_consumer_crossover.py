"""Finite crossed-state construction and signed response decompositions; no model."""
import math
import numpy as np

STATE_FIELDS = ('deltaN','deltaO','sumNO','sumON','rNO','rON','diagN','diagO')
VECTOR_FIELDS = ('N','O','Cno','Con','E','Uno','Vno','gno','Uon','Von','gon')
DOT_FIELDS = (('N','O'),('N','Cno'),('N','Con'),('Uno','Vno'),('Uon','Von'))
POLICY = 'FP64 subtraction and sum, then FP32 RNE and BF16 RNE. Stored diagonals remain authoritative.'
SCORING = 'Contiguous FP64 products then np.sum(dtype=float64); separately formed signed paths and regrouping residual; math.fsum Gram terms.'


def finite(value):
    value = np.asarray(value, np.float64)
    if not np.isfinite(value).all(): raise ValueError('Nonfinite diagnostic value')
    return value


def admit(value):
    value = finite(value).astype('<f4')
    if not np.isfinite(value).all(): raise ValueError('Nonfinite FP32 admission')
    bits = value.view('<u4'); words = ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) >> 16).astype('<u2')
    if np.any((words & 0x7f80) == 0x7f80): raise ValueError('Nonfinite BF16 admission')
    return (words.astype('<u4') << 16).view('<f4').astype(np.float64)


def construct(a,b,p,q,position):
    a,b,p,q = [finite(x) for x in (a,b,p,q)]
    if a.ndim!=2 or not a.size or any(x.shape!=a.shape for x in (b,p,q)) or type(position) is not int or not 0<=position<len(a): raise ValueError('Crossed state geometry differs')
    dn,do = b[position]-a[position],q[position]-p[position]
    sno,son = a[position]+do,p[position]+dn
    no,on = admit(sno),admit(son);xno,xon = a.copy(),p.copy();xno[position],xon[position] = no,on
    fields = dict(zip(STATE_FIELDS,(dn,do,sno,son,no-sno,on-son,admit(a[position]+dn)-b[position],admit(p[position]+do)-q[position])))
    return {'Xno':xno,'Xon':xon,**fields}


def norm2(value):
    value = finite(value); result = float(np.sum(value*value,dtype=np.float64))
    if not math.isfinite(result): raise ValueError('Nonfinite squared norm')
    return result


def dot(a,b):
    value = float(np.sum(finite(a)*finite(b),dtype=np.float64))
    if not math.isfinite(value): raise ValueError('Nonfinite inner product')
    return value


def scalar_summary(width,norms,dots):
    def relative(n): return math.sqrt(norms[n]/norms['N']) if norms['N'] else None
    def cosine(a,b): return dots[a+':'+b]/math.sqrt(norms[a]*norms[b]) if norms[a]*norms[b] else None
    paths = {}
    for suffix in ('no','on'):
        u,v,g = ('U'+suffix,'V'+suffix,'g'+suffix)
        terms = [norms[u],norms[v],2*dots[u+':'+v],2*dots['sum:'+g],norms[g]]
        paths[suffix] = {'terms':terms,'gramSum':math.fsum(terms),'normSquaredMinusGram':norms['E']-math.fsum(terms),
                         'termCosine':cosine(u,v),'Urelative':relative(u),'Vrelative':relative(v),'gregroupingRMS':math.sqrt(norms[g]/width)}
    return {'rms':{k:math.sqrt(v/width) for k,v in norms.items()},'relativeToNative':{k:relative(k) for k in VECTOR_FIELDS},
            'responseCosines':{k:cosine('N',k) for k in ('O','Cno','Con')},'paths':paths}


def decompose(fa,fb,fp,fq,fxno,fxon):
    arrays = [finite(x) for x in (fa,fb,fp,fq,fxno,fxon)]
    if arrays[0].ndim!=1 or not arrays[0].size or any(x.shape!=arrays[0].shape for x in arrays): raise ValueError('Response row geometry differs')
    fa,fb,fp,fq,fxno,fxon = arrays
    n,o,cno,con = fb-fa,fq-fp,fxno-fa,fxon-fp;e = o-n
    uno,vno,uon,von = o-cno,cno-n,o-con,con-n
    gno,gon = e-(uno+vno),e-(uon+von)
    vectors = dict(zip(VECTOR_FIELDS,(n,o,cno,con,e,uno,vno,gno,uon,von,gon)))
    norms = {k:norm2(x) for k,x in vectors.items()};dots = {a+':'+b:dot(vectors[a],vectors[b]) for a,b in DOT_FIELDS}
    dots.update({'sum:gno':dot(uno+vno,gno),'sum:gon':dot(uon+von,gon)})
    return vectors,{'width':len(n),'normSquared':norms,'innerProducts':dots,**scalar_summary(len(n),norms,dots)}
