"""Independent tiny full-query versus selected-query ancestry checks."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from selected_source_v2 import PREFIX, AFFINES, NORMS, form, splice, load_pages


def fixture():
    torch.manual_seed(419)
    width = 8
    weights = {n: torch.randn((24 if n.endswith('linear_in.weight') else width),
                              (12 if n.endswith('linear_out.weight') else width), dtype=torch.float64) / 4 for n in AFFINES}
    weights.update({n: torch.rand(4, dtype=torch.float64) + .5 for n in NORMS})
    def attention(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    native = SimpleNamespace(apply_rotary_emb=lambda x, rope, **kw: x,
                             dispatch_attention_fn=attention,
                             Flux2SwiGLU=lambda: lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
    values = [torch.randn(1, n, width, dtype=torch.float64) for n in [5, 3]]
    mods = [torch.randn(1, 6 * width, dtype=torch.float64) / 5 for _ in range(2)]
    return weights, native, values, mods


def test_selected_rows_equal_independent_full_query_program():
    weights, native, (image, text), (im, tm) = fixture()
    s = [0, 2]
    rope = (torch.ones(8, 4), torch.zeros(8, 4))
    got = form(image, text, im, tm, rope, s, weights, native)
    mi, mt = im[:, None].chunk(6, -1), tm[:, None].chunk(6, -1)
    xi = (1 + mi[1]) * F.layer_norm(image, (8,), eps=1e-6) + mi[0]
    xt = (1 + mt[1]) * F.layer_norm(text, (8,), eps=1e-6) + mt[0]
    lin = lambda n, x: F.linear(x, weights[PREFIX + n + '.weight'])
    rms = lambda n, x: F.rms_norm(x.unflatten(-1, (2, 4)), (4,), weights[PREFIX + 'attn.' + n + '.weight'], eps=1e-6)
    q = rms('norm_added_q', lin('attn.add_q_proj', xt))
    k = torch.cat([rms('norm_added_k', lin('attn.add_k_proj', xt)), rms('norm_k', lin('attn.to_k', xi))], 1)
    v = torch.cat([lin('attn.add_v_proj', xt), lin('attn.to_v', xi)], 1).unflatten(-1, (2, 4))
    h = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).flatten(2, 3)
    a = lin('attn.to_add_out', h)
    u = text + mt[2] * a
    x = F.layer_norm(u, (8,), eps=1e-6) * (1 + mt[4]) + mt[3]
    gate, up = lin('ff_context.linear_in', x).chunk(2, -1)
    m = lin('ff_context.linear_out', F.silu(gate) * up)
    for n, expected in dict(A=a, U=u, X=x, M=m, T=u + mt[5] * m).items():
        torch.testing.assert_close(got[n], expected[:, s], atol=1e-12, rtol=1e-12)
    assert got['Q'].shape[1] == len(s) and got['K'].shape[1] == 8


def test_splice_keeps_p_mlp_sequential_ancestry():
    weights, native, (image, text), (im, tm) = fixture()
    rope = (torch.ones(8, 4), torch.zeros(8, 4))
    p = form(image, text, im, tm, rope, [0, 2], weights, native)
    direction = torch.linspace(-.3, .2, 8, dtype=torch.float64)
    q = form(image + direction, text - 2 * direction, im, tm, rope, [0, 2], weights, native)
    got = splice(p, q)
    torch.testing.assert_close(got['PQP'], (p['R'] + q['gateA'] * q['A']) + q['gateM'] * p['M'])
    wrong = (p['R'] + q['gateA'] * q['A']) + q['gateM'] * q['M']
    assert not torch.allclose(got['PQP'], wrong)


def page_fixture(tmp_path, replace_page=False):
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from encoder_atoms import read, save, sha, identity
    store = LocalStore(tmp_path / 'store')
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    page, _, _, _ = store._save_slot(value + 1 if replace_page else value)
    metadata = read(store.root / 'pages' / (page + '.json'))
    manifest = {'payload': describe({'image': value})}
    cut = {'manifest': manifest, 'slots': {'image': page}}
    cut_file = tmp_path / 'cut.json'
    save(cut_file, cut)
    files = {str(store.root / 'pages' / (page + '.json')): page,
             str(store.root / 'blobs' / (metadata['tensor_blob'] + '.safetensors')): metadata['tensor_blob']}
    return value, {'cutID': identity(manifest), 'cut': cut,
                   'cutFile': {'path': str(cut_file), 'sha256': sha(cut_file)},
                   'pages': {'image': {'pageID': page, 'metadata': metadata, 'store': str(store.root),
                                       'files': files, 'declaredDescriptor': describe(value)}}}


def test_source_page_restores_authenticated_values_and_stride(tmp_path):
    value, record = page_fixture(tmp_path)
    got = load_pages(record, 'cpu')['image']
    assert got.stride() == value.stride()
    torch.testing.assert_close(got, value, rtol=0, atol=0)


def test_independently_valid_foreign_page_cannot_replace_cut_slot(tmp_path):
    import pytest
    _, record = page_fixture(tmp_path, replace_page=True)
    with pytest.raises(ValueError, match='declared cut slot'):
        load_pages(record, 'cpu')
