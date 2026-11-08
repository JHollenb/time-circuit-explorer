"""Frozen successor computation, with repeated factorization diagnostics omitted."""
from own_successor import compute,abis,SOURCE_NAMES,PARAMETERS,PREFIX
from own_successor import work as previous_work


def work():
    value=previous_work()
    return {**value,'diagnostic':{},'diagnosticMACs':0,'assayMACs':value['candidateMACs'],
            'scope':'Identical frozen87 numerical producer on parent70/73 input. '
                    'No repeated factorization diagnostic products. QR/scalar/data work additional; original maps/clocks paid.'}
