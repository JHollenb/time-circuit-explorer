"""Observe native SDPA without changing positional or keyword call semantics."""


def wrap_sdpa(original,phase,records):
    def attention(query,key,value,*args,**kwargs):
        records.append({**phase,'QShape':list(query.shape),'KShape':list(key.shape),'VShape':list(value.shape),
            'nominalQKAndAVMACs':query.numel()//query.shape[-1]*key.shape[-2]*(query.shape[-1]+value.shape[-1])})
        return original(query,key,value,*args,**kwargs)
    return attention
