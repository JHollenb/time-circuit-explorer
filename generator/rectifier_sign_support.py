"""Exact finite coefficient supports and prospective arithmetic, no learned maps."""
import numpy as np

EXPRESSIONS = ('raw', 'centered', 'combined')
BITORDER = 'little'


def measure(gate, rectifier_mean, activation_mean, selected, split=512, tile_rows=16, dimension=3072):
    gate = np.asarray(gate)
    rectifier_mean = np.asarray(rectifier_mean)
    activation_mean = np.asarray(activation_mean)
    selected = np.asarray(selected)
    rows, channels = gate.shape
    if gate.dtype != np.float64 or rectifier_mean.dtype != np.float64 or activation_mean.dtype != np.float64:
        raise ValueError('Named expressions require stored float64 operands')
    if rectifier_mean.shape != (2, channels) or activation_mean.shape != (2, channels):
        raise ValueError('Two stream means required')
    if not 0 < split < rows or channels % 8 or tile_rows <= 0 or split % tile_rows or (rows-split) % tile_rows:
        raise ValueError('Fixed stream/tile/packing geometry differs')
    if selected.ndim != 1 or selected.dtype != np.int64 or not np.array_equal(selected, np.unique(selected)):
        raise ValueError('Sorted unique integer selected support required')
    if len(selected) and (selected[0] < 0 or selected[-1] >= channels):
        raise ValueError('Selected support outside channels')
    if dimension <= 0 or len(selected) == channels:
        raise ValueError('Positive source dimension and omitted domain required')
    if not all(np.isfinite(x).all() for x in [gate, rectifier_mean, activation_mean]):
        raise ValueError('Nonfinite source operand')
    omitted = np.ones(channels, dtype=bool)
    omitted[selected] = False
    tiles = rows // tile_rows
    fields = {
        'supportBits': np.zeros((3, rows, channels//8), dtype=np.uint8),
        'rowCounts': np.zeros((3, rows), dtype=np.int64),
        'channelCounts': np.zeros((3, 2, channels), dtype=np.int64),
        'tileUnionBits': np.zeros((3, tiles, channels//8), dtype=np.uint8),
        'tileUnionCounts': np.zeros((3, tiles), dtype=np.int64),
        'streamTermBits': np.zeros((2, channels//8), dtype=np.uint8),
    }
    stream_term = np.subtract(activation_mean, rectifier_mean)
    if not np.isfinite(stream_term).all():
        raise ValueError('Nonfinite combined stream term')
    stream_live = (stream_term != 0) & omitted
    fields['streamTermBits'][:] = np.packbits(stream_live, axis=-1, bitorder=BITORDER)
    for stream, lo, hi in [(0, 0, split), (1, split, rows)]:
        for start in range(lo, hi, tile_rows):
            tile = start // tile_rows
            stop = start + tile_rows
            positive = np.maximum(gate[start:stop], np.float64(0))
            coefficients = (positive, np.subtract(positive, rectifier_mean[stream]),
                            np.add(positive, stream_term[stream]))
            for index, coefficient in enumerate(coefficients):
                if not np.isfinite(coefficient).all():
                    raise ValueError('Nonfinite finite-expression coefficient')
                support = (coefficient != 0) & omitted
                union = np.any(support, axis=0)
                fields['supportBits'][index, start:stop] = np.packbits(support, axis=-1, bitorder=BITORDER)
                fields['rowCounts'][index, start:stop] = np.sum(support, axis=1)
                fields['channelCounts'][index, stream] += np.sum(support, axis=0)
                fields['tileUnionBits'][index, tile] = np.packbits(union, bitorder=BITORDER)
                fields['tileUnionCounts'][index, tile] = np.sum(union)
    count = int(np.sum(omitted))
    panels = {}
    for name, lo, hi in [('all', 0, rows), ('text', 0, split), ('image', split, rows)]:
        n = hi-lo
        baseline = 2*dimension*n*count
        panel = {'rows': n, 'omittedChannels': count, 'denseOmittedUpDownMACs': baseline, 'expressions': {}}
        for index, expression in enumerate(EXPRESSIONS):
            per_row = fields['rowCounts'][index, lo:hi]
            per_tile = fields['tileUnionCounts'][index, lo//tile_rows:hi//tile_rows]
            live = int(np.sum(per_row))
            ideal = 2*dimension*live
            tiled = 2*dimension*tile_rows*int(np.sum(per_tile))
            panel['expressions'][expression] = {
                'liveCoefficients': live, 'coefficientDensity': live/(n*count),
                'rowCountMin': int(np.min(per_row)), 'rowCountMedian': float(np.median(per_row)),
                'rowCountMax': int(np.max(per_row)), 'idealRowwiseUpDownMACs': ideal,
                'idealRowwiseSavingsMACs': baseline-ideal, 'idealRowwiseSavingsFraction': (baseline-ideal)/baseline,
                'tileUnionCountMin': int(np.min(per_tile)), 'tileUnionCountMax': int(np.max(per_tile)),
                'fixedTileUpDownMACs': tiled, 'fixedTileSavingsMACs': baseline-tiled,
                'fixedTileSavingsFraction': (baseline-tiled)/baseline,
            }
        panels[name] = panel
    streams = {}
    for index, name, n in [(0, 'text', split), (1, 'image', rows-split)]:
        live = int(np.sum(stream_live[index]))
        streams[name] = {'rows': n, 'liveOmittedChannels': live, 'zeroOmittedChannels': count-live,
                         'hypotheticalCompileMACs': dimension*dimension*live,
                         'hypotheticalApplyMACs': n*dimension*dimension if live else 0,
                         'hypotheticalFP64MapBytes': dimension*dimension*8 if live else 0,
                         'directStreamTermUpDownMACs': 2*dimension*n*live}
    compile_macs = sum(x['hypotheticalCompileMACs'] for x in streams.values())
    apply_macs = sum(x['hypotheticalApplyMACs'] for x in streams.values())
    raw = panels['all']['expressions']['raw']
    baseline = panels['all']['denseOmittedUpDownMACs']
    fold = {'streams': streams, 'hypotheticalCompileMACs': compile_macs,
            'hypotheticalApplyMACs': apply_macs,
            'hypotheticalFP64MapBytes': sum(x['hypotheticalFP64MapBytes'] for x in streams.values()),
            'plusRawRowwiseBranchMACs': compile_macs+apply_macs+raw['idealRowwiseUpDownMACs'],
            'plusRawFixedTileBranchMACs': compile_macs+apply_macs+raw['fixedTileUpDownMACs'],
            'rowwiseSavingsVsOmittedBaselineMACs': baseline-compile_macs-apply_macs-raw['idealRowwiseUpDownMACs'],
            'tileSavingsVsOmittedBaselineMACs': baseline-compile_macs-apply_macs-raw['fixedTileUpDownMACs'],
            'constructed': False,
            'scope': 'Shape-derived arithmetic only. No maps/Upmeans accessed. Bias/mean/gather/launch/storage costs excluded, not zero; no cross-prompt reuse or finite reassociation parity established.'}
    return fields, panels, fold


def abis(rows=1536, channels=9216, tile_rows=16):
    return {'supportBits': ([3, rows, channels//8], 'uint8'),
            'rowCounts': ([3, rows], 'int64'), 'channelCounts': ([3, 2, channels], 'int64'),
            'tileUnionBits': ([3, rows//tile_rows, channels//8], 'uint8'),
            'tileUnionCounts': ([3, rows//tile_rows], 'int64'),
            'streamTermBits': ([2, channels//8], 'uint8')}
