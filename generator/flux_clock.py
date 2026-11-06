"""Finite clock operands and selected parameter loading; no model imports at load."""
from pathlib import Path
import hashlib
import json
import math
import struct
import sys
import time
from encoder_atoms import read, sha
from paired_fields import load, payload

CLOCK_PREFIXES = ('time_guidance_embed.', 'double_stream_modulation_img.',
    'double_stream_modulation_txt.', 'single_stream_modulation.', 'norm_out.linear.')
CLOCK_WEIGHTS = ('time_guidance_embed.timestep_embedder.linear_1.weight',
    'time_guidance_embed.timestep_embedder.linear_2.weight',
    'double_stream_modulation_img.linear.weight', 'double_stream_modulation_txt.linear.weight',
    'single_stream_modulation.linear.weight', 'norm_out.linear.weight')
PORTS = ('temb', 'double_img', 'double_txt', 'single', 'output_affine')


def header(path):
    with Path(path).open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < size <= 16 << 20: raise ValueError('Invalid tensor header')
        data = stream.read(size)
    entries = {k: v for k, v in json.loads(data).items() if k != '__metadata__'}
    return entries, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': size}


def partition(entries):
    removed = {n for n in entries if n.startswith(CLOCK_PREFIXES)}
    if removed != set(CLOCK_WEIGHTS): raise ValueError('Unexpected clock parameter closure')
    for n, x in entries.items():
        if x['dtype'] != 'BF16' or x['data_offsets'][1] - x['data_offsets'][0] != math.prod(x['shape']) * 2:
            raise ValueError('Unexpected original tensor format: ' + n)
    return sorted(removed), sorted(set(entries) - removed)


def selected_load(module, path, names, entries, device):
    import torch
    from safetensors import safe_open
    parameters = dict(module.named_parameters())
    if set(parameters) != set(names): raise ValueError('Selected model/parameter names differ')
    rows = []
    with safe_open(str(path), framework='pt', device='cpu') as stream, torch.inference_mode(False):
        for name in names:
            old = parameters[name]; entry = entries[name]
            if old.device.type != 'meta' or list(old.shape) != entry['shape']: raise ValueError('Expected matching meta-only parameter')
            value = stream.get_tensor(name)
            if value.dtype != torch.bfloat16 or list(value.shape) != entry['shape']: raise ValueError('Loaded original tensor format differs')
            parent, _, leaf = name.rpartition('.')
            setattr(module.get_submodule(parent) if parent else module, leaf,
                torch.nn.Parameter(value.to(device), requires_grad=False))
            rows.append({'name': name, 'shape': entry['shape'], 'bytes': value.numel() * value.element_size(),
                'dataOffsets': entry['data_offsets'], 'payloadSHA256': payload(value)})
    if any(x.device.type == 'meta' for x in [*module.parameters(), *module.buffers()]): raise ValueError('Unmaterialized selected model state')
    return rows


def snapshot(torch):
    import os
    return {'cublasWorkspaceConfig': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(),
        'threads': torch.get_num_threads(), 'cudaMatmulTF32': torch.backends.cuda.matmul.allow_tf32,
        'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'bf16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'cudnnTF32': torch.backends.cudnn.allow_tf32, 'cudnnBenchmark': torch.backends.cudnn.benchmark, 'cudnnDeterministic': torch.backends.cudnn.deterministic}


def check_mode(kwargs, numerical, contract):
    if numerical != contract['numericalProfile']: raise ValueError('Unsupported numerical profile')
    if kwargs.get('guidance') is not None or kwargs.get('kv_cache') is not None or kwargs.get('kv_cache_mode') is not None or kwargs.get('num_ref_tokens', 0) != 0:
        raise ValueError('Unsupported guidance/reference/cache mode')
    if kwargs.get('joint_attention_kwargs') not in (None, {}): raise ValueError('Unsupported attention overrides')
    x = kwargs['hidden_states']; e = kwargs['encoder_hidden_states']
    if list(x.shape) != contract['latentShape'] or list(e.shape) != contract['EShape'] or str(x.dtype) != contract['dtype'] or str(e.dtype) != contract['dtype']:
        raise ValueError('Unsupported clock consumer geometry/dtype')


def clock_path(t, torch):
    # Follow the pinned pipeline and transformer boundaries, without cancelling /1000 *1000.
    expanded = t.expand(1).to(torch.bfloat16)
    argument = expanded / 1000
    internal = argument.to(torch.bfloat16) * 1000
    return {'scheduler': t.reshape(1), 'pipeline_cast': expanded, 'transformer_argument': argument, 'internal': internal}


def deny_targets(paths):
    denied = {str(Path(p).resolve()) for p in paths}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)) and str(Path(args[0]).resolve()) in denied:
            raise PermissionError('Evaluation target unavailable before candidate seal')
    sys.addaudithook(audit)


def install_tables(model, table, device, phase, calls):
    import torch
    rows = table['rows']; contract = table['contract']; dtype = contract['dtype']; numerical = contract['numericalProfile']
    clock_keys = [r['clockOperands']['internal']['payloadSHA256'] for r in rows]
    temb_keys = [r['fields']['temb']['payloadSHA256'] for r in rows]
    if len(set(clock_keys)) != len(rows) or len(set(temb_keys)) != len(rows): raise ValueError('Ambiguous clock table')
    class Gather(torch.nn.Module):
        def __init__(self, port, keys):
            super().__init__(); self.port = port; self.keys = keys
            for i, r in enumerate(rows): self.register_buffer('row' + str(i), load(r['fields'][port], device))
        def select(self, x):
            started = time.perf_counter()
            if snapshot(torch) != numerical: raise ValueError('Unsupported clock numerical profile')
            expected_shape = [1] if self.port == 'temb' else rows[0]['fields']['temb']['shape']
            if list(x.shape) != expected_shape or str(x.dtype) != dtype or x.device.type != device:
                raise ValueError('Unsupported clock operand geometry/dtype/device')
            key = payload(x)
            if key not in self.keys: raise ValueError('Unsupported clock operand payload')
            i = self.keys.index(key); y = getattr(self, 'row' + str(i))
            calls.append({**phase, 'port': self.port, 'clock': i, 'inputPayloadSHA256': key,
                'outputPayloadSHA256': rows[i]['fields'][self.port]['payloadSHA256'], 'logicalOutputBytes': y.numel() * y.element_size(),
                'wallSecondsWithInputHash': time.perf_counter() - started})
            return y
        def forward(self, x): return self.select(x)
    class Time(Gather):
        def forward(self, timestep, guidance):
            if guidance is not None: raise ValueError('Guidance unavailable in compiled clock')
            return self.select(timestep)
    class Norm(Gather):
        def __init__(self, norm):
            super().__init__('output_affine', temb_keys); self.norm = norm
        def forward(self, x, conditioning_embedding):
            if str(x.dtype) != dtype or x.shape[0] != 1: raise ValueError('Unsupported normalization consumer')
            emb = self.select(conditioning_embedding); scale, shift = torch.chunk(emb, 2, dim=1)
            return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    model.time_guidance_embed = Time('temb', clock_keys)
    model.double_stream_modulation_img = Gather('double_img', temb_keys)
    model.double_stream_modulation_txt = Gather('double_txt', temb_keys)
    model.single_stream_modulation = Gather('single', temb_keys)
    model.norm_out = Norm(model.norm_out.norm)
    if any(n.startswith(CLOCK_PREFIXES) for n, _ in model.named_parameters()): raise ValueError('Clock parameters survived replacement')
    return model
