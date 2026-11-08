"""Copy a compiled causal template prefix; never fabricate the dynamic remainder."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


class TokenInput:
    """The existing Prefix.tokenize policy, without AtomStore or numerical imports."""
    def __init__(self, root, manifest):
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.m = manifest
        for key in ('tokenizer', 'template'):
            if sha(Path(root)/manifest[key]['path']) != manifest[key]['sha256']:
                raise ValueError('Tokenizer/template identity changed')
        self.tokenizer = Tokenizer.from_file(str(Path(root)/manifest['tokenizer']['path']))
        p = manifest['tokenPolicy']
        self.tokenizer.enable_padding(length=512, direction=p['paddingSide'], pad_id=p['padID'], pad_token=p['padToken'])
        self.tokenizer.enable_truncation(max_length=512, direction=p['truncationSide'], strategy='longest_first')
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        def refuse(message):
            raise ValueError(message)
        env.globals['raise_exception'] = refuse
        self.template = env.from_string((Path(root)/manifest['template']['path']).read_text())

    def tokenize(self, prompt):
        if not isinstance(prompt, str):
            raise TypeError('Raw text required')
        p = self.m['tokenPolicy']
        text = self.template.render(messages=[{'role': 'user', 'content': prompt}], tools=None, documents=None,
                                    add_generation_prompt=True, enable_thinking=False, bos_token=None,
                                    eos_token=p['eosToken'], pad_token=p['padToken'])
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        return {'ids': encoded.ids, 'mask': encoded.attention_mask, 'positions': list(range(512))}, text


class PrefixState:
    """Own input packet + numeric fixed rows + explicitly unknown dynamic rows."""
    def __init__(self, root):
        self.root = Path(root)
        self.program = read(self.root/'program.json')
        self.bank = (self.root/'prefix.bf16').read_bytes()
        p = self.program
        if hashlib.sha256(self.bank).hexdigest() != p['bankSHA256'] or len(self.bank) != p['bankBytes']:
            raise ValueError('Compiled prefix changed')
        if (p['length'], p['width'], p['layers'], p['prefixIDs']) != (512, 2560, 27, [151644, 872]):
            raise ValueError('Compiled ABI differs')
        if len(self.bank) != 28*2*2560*2:
            raise ValueError('Compiled prefix byte geometry differs')
        self.packet = None

    def advance(self, packet):
        self.supported = False
        ids, mask, positions = [packet[k] for k in ('ids', 'mask', 'positions')]
        if any(len(x) != 512 for x in (ids, mask, positions)):
            raise ValueError('Complete 512-row packet required')
        if any(type(x) is not int or not 0 <= x < 151936 for x in ids):
            raise ValueError('Invalid full-vocabulary token')
        if any(type(x) is not int or x not in (0, 1) for x in mask) or any(type(x) is not int for x in positions) or positions != list(range(512)):
            raise ValueError('Mask/position ABI differs')
        if any(a < b for a, b in zip(mask, mask[1:])):
            raise ValueError('Right-padded causal packet required')
        previous = self.packet
        self.packet = {k: list(packet[k]) for k in ('ids', 'mask', 'positions')}
        self.supported = ids[:2] == self.program['prefixIDs'] and mask[:2] == [1, 1]
        changed = [] if previous is None else [i for i in range(512) if any(previous[k][i] != packet[k][i] for k in self.packet)]
        return {'status': 'partial' if self.supported else 'unsupported-prefix', 'changedInputPositions': changed,
                'compiledPositions': [0, 1] if self.supported else [],
                'unresolvedPositions': [2, 512] if self.supported else [0, 512],
                'unresolvedOperations': ['dynamic contextual attention', 'dynamic residual/RMS/SwiGLU', 'layer1 QKV', 'denoising', 'pixels'],
                'completeH1Producer': False, 'imageGenerator': False}

    def state(self, layer):
        if self.packet is None or not self.supported:
            raise ValueError('No supported owned input')
        if type(layer) is not int or not 0 <= layer <= 27:
            raise ValueError('Unknown compiled state layer')
        stride = 2*2560*2
        return self.bank[layer*stride:(layer+1)*stride]
