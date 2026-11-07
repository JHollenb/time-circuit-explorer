"""Tiny independent dense-equation oracle for routed original computations."""
from pathlib import Path
import copy
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from rotating_writes import image_rows, joint, single


class RotatingWrites(unittest.TestCase):
    def test_fixed_rotation_covers_every_image_row(self):
        counts = np.zeros(1024, dtype=int)
        for b in range(25):
            idx = image_rows(b)
            self.assertEqual(len(idx), 256)
            counts[idx] += 1
        self.assertEqual(int((counts == 7).sum()), 256)
        self.assertEqual(int((counts == 6).sum()), 768)
        for b, side in [(-1, 32), (0, 3)]:
            with self.assertRaises(ValueError): image_rows(b, side)

    def fixture(self):
        import torch
        torch.manual_seed(913)
        nn = torch.nn
        linear = lambda a, b: nn.Linear(a, b, bias=False, dtype=torch.float64)
        norm = lambda: nn.LayerNorm(8, elementwise_affine=False, eps=1e-6, dtype=torch.float64)
        class Swi(nn.Module):
            def forward(self, x):
                a, b = x.chunk(2, dim=-1)
                return torch.nn.functional.silu(a) * b
        def ff(): return nn.Sequential(linear(8, 48), Swi(), linear(24, 8))
        a = NS(heads=2, processor=NS(), norm_q=nn.Identity(), norm_k=nn.Identity(),
            norm_added_q=nn.Identity(), norm_added_k=nn.Identity(),
            to_q=linear(8, 8), to_k=linear(8, 8), to_v=linear(8, 8),
            add_q_proj=linear(8, 8), add_k_proj=linear(8, 8), add_v_proj=linear(8, 8),
            to_out=[linear(8, 8), nn.Identity()], to_add_out=linear(8, 8))
        j = NS(attn=a, norm1=norm(), norm1_context=norm(), norm2=norm(), norm2_context=norm(), ff=ff(), ff_context=ff())
        s = NS(norm=norm(), attn=NS(inner_dim=8, heads=2, processor=NS(), to_qkv_mlp_proj=linear(8, 72),
            norm_q=nn.Identity(), norm_k=nn.Identity(), mlp_act_fn=Swi(), to_out=linear(32, 8)))
        def split(x, n):return tuple(tuple(x[:, None, :].chunk(3*n, dim=-1)[3*i:3*i+3]) for i in range(n))
        def rope(x, emb, sequence_dim):
            c, t = emb
            return x * c[None, :, None, :] + x.flip(-1) * t[None, :, None, :]
        calls = []
        def attention(q, k, v, **kw):
            calls.append((list(q.shape), list(k.shape)))
            logits = torch.einsum('bqhd,bkhd->bhqk', q, k) / q.shape[-1]**.5
            return torch.einsum('bhqk,bkhd->bqhd', logits.softmax(-1), v)
        native = NS(Flux2Modulation=NS(split=split), apply_rotary_emb=rope, dispatch_attention_fn=attention)
        return torch, j, s, native, calls

    @staticmethod
    def oracle(block, image, text, mi, mt, rope, selected, kind):
        def arr(x):return x.detach().numpy()
        def mat(x, module):return x @ arr(module.weight).T
        def norm(x):return (x-x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True)+1e-6)
        def swi(x):
            a, b = np.split(x, 2, axis=-1)
            return a / (1+np.exp(-a)) * b
        def ff(x, m):return mat(swi(mat(x, m[0])), m[2])
        def attention(q, k, v):
            q, k, v = [x.reshape(1, x.shape[1], 2, 4) for x in [q, k, v]]
            c, t = [arr(x) for x in rope]
            q = q*c[None, :, None, :] + q[..., ::-1]*t[None, :, None, :]
            k = k*c[None, :, None, :] + k[..., ::-1]*t[None, :, None, :]
            logits = np.einsum('bqhd,bkhd->bhqk', q, k)/2
            prob = np.exp(logits-logits.max(-1, keepdims=True));prob /= prob.sum(-1, keepdims=True)
            return np.einsum('bhqk,bkhd->bqhd', prob, v).reshape(1, q.shape[1], 8)
        im, tx = arr(image), arr(text);a = block.attn
        if kind == 'joint':
            ai, bi, gi, af, bf, gf = np.split(arr(mi)[:, None, :], 6, axis=-1)
            at, bt, gt, aft, bft, gft = np.split(arr(mt)[:, None, :], 6, axis=-1)
            xi, xt = (1+bi)*norm(im)+ai, (1+bt)*norm(tx)+at
            q, k, v = [np.concatenate([mat(xt, getattr(a, t)), mat(xi, getattr(a, i))], axis=1)
                for t, i in [('add_q_proj','to_q'), ('add_k_proj','to_k'), ('add_v_proj','to_v')]]
            y = attention(q, k, v);yt, yi = y[:, :tx.shape[1]], y[:, tx.shape[1]:]
            zi = im + gi*mat(yi, a.to_out[0]);zt = tx + gt*mat(yt, a.to_add_out)
            zi = zi + gf*ff(norm(zi)*(1+bf)+af, block.ff)
            zt = zt + gft*ff(norm(zt)*(1+bft)+aft, block.ff_context)
            out = im.copy();out[:, selected] = zi[:, selected]
            return zt, out
        state = np.concatenate([tx, im], axis=1)
        shift, scale, gate = np.split(arr(mi)[:, None, :], 3, axis=-1)
        x = (1+scale)*norm(state)+shift
        projected = mat(x, a.to_qkv_mlp_proj)
        q, k, v = np.split(projected[..., :24], 3, axis=-1)
        y = attention(q, k, v)
        out = state.copy();dense = state+gate*mat(np.concatenate([y, swi(projected[..., 24:])], -1), a.to_out)
        out[:, selected] = dense[:, selected]
        return out

    def test_joint_sequential_mlp_and_full_readable_state(self):
        torch, j, _, native, calls = self.fixture()
        x = torch.randn(1, 16, 8, dtype=torch.float64);e = torch.randn(1, 3, 8, dtype=torch.float64)
        mi, mt = [torch.randn(1, 48, dtype=torch.float64)*.2 for _ in range(2)]
        rope = (torch.cos(torch.arange(76, dtype=torch.float64).reshape(19, 4)*.01), torch.sin(torch.arange(76, dtype=torch.float64).reshape(19, 4)*.01))
        selected = torch.tensor(image_rows(0, 4));untouched = np.setdiff1d(np.arange(16), selected)
        with torch.inference_mode():
            actual = joint(j, x, e, mi, mt, rope, selected, native)
            expected = self.oracle(j, x, e, mi, mt, rope, selected.numpy(), 'joint')
            for a, b in zip(actual, expected): np.testing.assert_allclose(a.numpy(), b, rtol=1e-11, atol=1e-11)
            self.assertTrue(torch.equal(actual[1][:, untouched], x[:, untouched]))
            changed = x.clone();changed[:, 15, 0] += 2
            other = joint(j, changed, e, mi, mt, rope, selected, native)
            self.assertGreater(float((other[1][:, selected]-actual[1][:, selected]).abs().max()), 1e-7)
            self.assertTrue(torch.equal(other[1][:, untouched], changed[:, untouched]))
        self.assertEqual(calls[0], ([1, 7, 2, 4], [1, 19, 2, 4]))

    def test_single_parallel_inputs_and_physical_sliced_outputs(self):
        torch, _, s, native, calls = self.fixture()
        x = torch.randn(1, 16, 8, dtype=torch.float64);e = torch.randn(1, 3, 8, dtype=torch.float64)
        state = torch.cat([e, x], 1);mod = torch.randn(1, 24, dtype=torch.float64)*.2
        rope = (torch.cos(torch.arange(76, dtype=torch.float64).reshape(19, 4)*.01), torch.sin(torch.arange(76, dtype=torch.float64).reshape(19, 4)*.01))
        selected = torch.cat([torch.arange(3), 3+torch.tensor(image_rows(1, 4))]);ops=[]
        original = torch.nn.functional.linear
        def linear(a, w, b=None):
            ops.append((list(a.shape), list(w.shape), w.storage_offset()))
            return original(a, w, b)
        with torch.inference_mode(), patch('torch.nn.functional.linear', side_effect=linear):
            actual = single(s, state, mod, rope, selected, native)
        expected = self.oracle(s, x, e, mod, None, rope, selected.numpy(), 'single')
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-11, atol=1e-11)
        keep = np.setdiff1d(np.arange(19), selected)
        self.assertTrue(torch.equal(actual[:, keep], state[:, keep]))
        self.assertEqual(ops, [([1,7,8],[8,8],0), ([1,19,8],[16,8],64), ([1,7,8],[48,8],192), ([1,7,32],[8,32],0)])
        self.assertEqual(calls, [([1,7,2,4],[1,19,2,4])])

    def test_worker_imports_are_lazy(self):
        code = 'import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');import rotating_writes,rotating_write_images,rotating_write_study;assert not {"torch","diffusers","transformers"}.intersection(sys.modules)'
        subprocess.run([sys.executable,'-I','-c',code], check=True, capture_output=True, timeout=20)

    def test_both_images_must_seal_before_targets_release(self):
        import rotating_write_study as study
        from encoder_atoms import save
        for failed in [False, True]:
            with tempfile.TemporaryDirectory() as temp:
                events=[];save(Path(temp)/'request.json',{})
                def child(*args, **kwargs):
                    events.append('images')
                    if failed:raise RuntimeError('candidate child failed')
                def sealed(root):
                    events.append('seal');(Path(root)/'candidate-seal.json').write_text('{}')
                with patch.object(study,'verify'),patch.object(study.subprocess,'run',side_effect=child),patch.object(study,'seal',side_effect=sealed),patch.object(study,'score',side_effect=lambda *args:events.append('targets')):
                    if failed:
                        with self.assertRaises(RuntimeError):study.execute(temp)
                        self.assertEqual(events,['images'])
                    else:
                        study.execute(temp);self.assertEqual(events,['images','seal','targets'])

    def test_release_rejects_missing_ports_phase_drift_and_broken_handoffs(self):
        import rotating_write_study as study
        item=lambda digest:{'payloadSHA256':digest}
        fs={p:item(p) for p in study.PORTS};fs.update(C3=item('C3'),latent3=item('latent3'))
        source={'prefix':{c:{'latent3':item('latent3'),'E':item('E')} for c in ['00','10']},'ids':item('ids')}
        image={'requestSHA256':'request','fields':{c:copy.deepcopy(fs) for c in ['00','10']},'images':{c:{'path':c+'.png','sha256':'seal'} for c in ['00','10']},'routes':[],'actual':[]}
        for c in ['00','10']:
            image['actual'] += [{'case':c,'kind':'routed-input','hidden_statesPayloadSHA256':'latent3','encoder_hidden_statesPayloadSHA256':'E','img_idsPayloadSHA256':'ids'},
                {'case':c,'kind':'final-commit','incomingPayloadSHA256':'latent3','velocityPayloadSHA256':'velocity3','outgoingPayloadSHA256':'latent4','finalStepIndex':4}]
            for i in range(25):image['routes'].append({'case':c,'blockIndex':i,'blockKind':'joint' if i<5 else 'single','selectedImageRows':image_rows(i).tolist(),
                'unchangedInputPayloadSHA256':'carry','unchangedOutputPayloadSHA256':'carry','fullInputStatePayloadSHA256':str(i),'fullOutputStatePayloadSHA256':str(i+1)})
        mutations=[None,lambda v:v['fields']['00'].pop('decodedRaw'),lambda v:v['images'].pop('10'),
            lambda v:v['routes'][5].update(blockIndex=0),lambda v:v['routes'][6].update(fullInputStatePayloadSHA256='other'),
            lambda v:v['routes'][4].update(selectedImageRows=image_rows(1).tolist())]
        for mutate in mutations:
            changed=copy.deepcopy(image)
            if mutate:mutate(changed)
            def read(path):return {'request.json':{'requestSHA256':'request'},'inputs.json':source,'images.json':changed}[Path(path).name]
            with patch.object(study,'read',side_effect=read),patch.object(study,'sha',return_value='seal'),patch.object(study,'save') as saved,patch('paired_fields.load'):
                if mutate:
                    with self.assertRaises(ValueError):study.seal('unused')
                    saved.assert_not_called()
                else:
                    study.seal('unused');saved.assert_called_once()
