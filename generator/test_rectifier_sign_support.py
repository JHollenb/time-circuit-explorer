"""Independent scalar/bitwise reference, including exact zeros and tile unions."""
import numpy as np
from rectifier_sign_support import measure, EXPRESSIONS, BITORDER


def mechanics():
    gate = np.array([[-2, 3, 0, 1, 2, -0., 4, 1e-300],
                     [1, -1, -2, 0, 3, 1, -4, -1],
                     [-1, 2, 2, 3, 0, -1, 1, 2],
                     [2, 0, 1, -2, 1, 2, 0, -0.],
                     [-3, 1, 0, 2, 3, 1e-300, 2, -1],
                     [0, 2, 1, -1, 1, 2, 3, 1],
                     [1, 3, -1, 1, 2, 0, -2, 2],
                     [2, -1, 2, 0, -2, 3, 1, 3]], dtype=np.float64)
    rectifier = np.array([[1, 2, 0, 1, 2, 0, 3, 0], [1, 2, 0, 2, 1, 0, 2, 1]], dtype=np.float64)
    activation = np.array([[0, 1, 0, 0, 1, 0, 2, 0], [0, 1, 0, 1, 0, 0, 1, 0]], dtype=np.float64)
    selected = np.array([1, 6], dtype=np.int64)
    fields, panels, fold = measure(gate, rectifier, activation, selected, split=4, tile_rows=2, dimension=3)
    expected = [[[False for _ in range(8)] for _ in range(8)] for _ in range(3)]
    for row in range(8):
        stream = row//4
        for channel in range(8):
            positive = max(float(gate[row, channel]), 0.)
            values = [positive, positive-float(rectifier[stream, channel]),
                      positive+(float(activation[stream, channel])-float(rectifier[stream, channel]))]
            for index, value in enumerate(values):
                expected[index][row][channel] = channel not in [1, 6] and value != 0.
    for index, name in enumerate(EXPRESSIONS):
        bits = np.unpackbits(fields['supportBits'][index], axis=-1, bitorder=BITORDER).astype(bool)
        assert np.array_equal(bits, expected[index])
        for row in range(8):
            byte = sum((1 << column) for column, live in enumerate(expected[index][row]) if live)
            assert int(fields['supportBits'][index, row, 0]) == byte
            assert int(fields['rowCounts'][index, row]) == sum(expected[index][row])
        for stream in range(2):
            for channel in range(8):
                assert fields['channelCounts'][index, stream, channel] == sum(expected[index][row][channel] for row in range(4*stream, 4*stream+4))
        unions = []
        for start in range(0, 8, 2):
            union = [any(expected[index][row][channel] for row in [start, start+1]) for channel in range(8)]
            unions.append(sum(union))
            byte = sum(1 << channel for channel, live in enumerate(union) if live)
            assert fields['tileUnionBits'][index, start//2, 0] == byte
        assert fields['tileUnionCounts'][index].tolist() == unions
        account = panels['all']['expressions'][name]
        assert account['idealRowwiseUpDownMACs'] == 6*sum(sum(x) for x in expected[index])
        assert account['fixedTileUpDownMACs'] == 12*sum(unions)
        assert account['idealRowwiseSavingsMACs'] == 288-account['idealRowwiseUpDownMACs']
    c_live = sum(activation[s, j]-rectifier[s, j] != 0 for s in range(2) for j in range(8) if j not in [1, 6])
    assert fold['hypotheticalCompileMACs'] == 9*c_live
    assert fold['hypotheticalApplyMACs'] == 72
    assert fold['plusRawRowwiseBranchMACs'] == 9*c_live+72+panels['all']['expressions']['raw']['idealRowwiseUpDownMACs']
    assert fields['rowCounts'][0, 0] == 3  # Includes 1e-300: exact nonzero, no threshold.
    assert expected[0][0][0] is False and expected[1][0][0] is True
    assert expected[0][0][3] is True and expected[1][0][3] is False
    return {'schema': 'tce-rectifier-sign-support-toy-v1', 'status': 'passed',
            'rows': 8, 'channels': 8, 'scalarCoefficientChecks': 192,
            'scope': 'Independent scalar supports/manual bit packing, all row/channel/tile counts and prospective MACs. Exact zeros/signed zeros/tiny nonzeros and centering-induced liveness retained.'}


def test_scalar_reference():
    assert mechanics()['status'] == 'passed'
