"""Complete source/cache/output custody and compact lossless publication."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha, identity
from publish_source_native import checked
from publish_sparse_text_kv import NPY_BYTES
from projected_complement import fields, WO, ACTOR_PARAMETERS
from selected_source_v2 import PARAMETERS

TEMPLATE = Path(__file__).with_name('publish_sparse_text_kv.py')


def items(p):
    s = read(p / 'inputs.json')
    values = [{'path': path, 'sha256': digest} for path, digest in s['sourceMetadataFiles'].items()]
    values += [s['parentCompiler'], s['parentCompilerSeal']]
    values += [x for name in ['producer.json', 'compiler.json', 'primary.json', 'targets.json'] for x in fields(read(p / name))]
    values += [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root)
    p = root / 'result'
    s = read(p / 'inputs.json')
    api, allowed_roots = Api(), [job['result']['diagnostics']['cwd'], s['modelRoot']]
    for job_key, custody_key in [('parentJobID', 'parentCustody'), ('sourceJobID', 'sourceCustody')]:
        parent = api.json('GET', '/api/jobs/' + s[job_key])
        if parent['state'] != 'succeeded' or parent['payload_custody'] != s[custody_key]:
            raise ValueError('Parent/source scheduler custody changed')
        allowed_roots.append(parent['result']['diagnostics']['cwd'])
    allowed_roots += ['/mnt/big/mdb/research/job-a2048028d7bf/flux-binding-family-result',
                      '/mnt/hdd8tb/mrun-work/job-cd781e9a2a10']
    base = job['result']['diagnostics']['cwd']
    unique = {}
    for item in items(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting complete byte identity')
        unique[item['path']] = item
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast',
                             'python3 -c ' + shlex.quote(NPY_BYTES)],
                            input=json.dumps({'allowedRoots': allowed_roots, 'items': list(unique.values())}),
                            text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout)
    receipt.update(schema='tce-selected-source-bytes-v1', jobID=job['job_id'],
                   producerSealSHA256=sha(p / 'producer-seal.json'), primarySealSHA256=sha(p / 'primary-seal.json'),
                   verifierSourceSHA256=sha(__file__), byteTemplateSourceSHA256=sha(TEMPLATE),
                   scope='Full authenticated source cuts/pages/blobs, original assets, reused cache and operands, own source/splice/reader outputs and archived targets. SSH byte I/O only; no model/array calculation.')
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses'])}


def publish(root, destination):
    root, collection, job, request = checked(root)
    p = root / 'result'
    receipt = read(root / 'retained-array-verification.json')
    if (receipt['producerSealSHA256'] != sha(p / 'producer-seal.json')
            or receipt['primarySealSHA256'] != sha(p / 'primary-seal.json')
            or receipt['verifierSourceSHA256'] != sha(__file__) or receipt['byteTemplateSourceSHA256'] != sha(TEMPLATE)):
        raise ValueError('Byte verification source/release differs')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    streams = []
    for file in p.iterdir():
        if file.suffix == '.jsonl':
            raw = file.read_bytes()
            target = out / (file.name + '.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless query transport differs')
            streams.append({'source': file.name, 'sourceSHA256': sha(file), 'sourceBytes': len(raw),
                            'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(file, out / file.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-selected-source-publication-v1', 'jobID': job['job_id'],
         'requestSHA256': request['requestSHA256'], 'payloadCustody': collection['custody'],
         'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'],
         'losslessStreams': streams, 'scope': 'Demanded source rows into unchanged conditional reader, retaining original maps and historical prefixes/query/cache. No new images or arbitrary-prompt zero-neural claim.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p = Path(source)
    r, s, prod, c, a, se, pe, report, receipt = [read(p / name) for name in
        ['request.json', 'inputs.json', 'producer.json', 'compiler.json', 'primary.json', 'producer-execution.json',
         'primary-execution.json', 'report.json', 'retained-array-verification.json']]
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for name, digest in read(p / 'files.json').items():
        if sha(p / name) != digest:
            raise ValueError('Published member differs: ' + name)
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]:
            raise ValueError('Executed owned source changed')
    for name in ['producer', 'compiler', 'primary']:
        seal = read(p / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(p / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete phase JSON/request/seal join differs')
    if c['sourceProducerSealSHA256'] != sha(p / 'producer-seal.json') or a['compilerSealSHA256'] != sha(p / 'compiler-seal.json'):
        raise ValueError('Source-to-reader release link differs')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['byteTemplateSourceSHA256'] != sha(TEMPLATE):
        raise ValueError('Complete byte verifier source differs')
    for item in items(p):
        if receipt['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Complete source/output byte custody absent')
        if item.get('payloadSHA256') and receipt['npyPayloadWitnesses'][item['path']]['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Independent NPY payload witness absent')
    if set(x['name'] for x in se['loadedParameters']) != set(PARAMETERS) or set(x['name'] for x in pe['loadedParameters']) != set(ACTOR_PARAMETERS):
        raise ValueError('Original selected parameter closure differs')
    if len(se['linears']) != 16 or len(se['attentions']) != 2 or any(x['queryShape'] != [1, 4, 24, 128] or x['keyShape'] != [1, 1536, 24, 128] for x in se['attentions']):
        raise ValueError('Physical selected-query source geometry differs')
    if sum(x['nominalMACs'] for x in se['linears']) + sum(x['nominalQKAVMACs'] for x in se['attentions']) != se['nominalMatrixMACs']:
        raise ValueError('Source physical cost sum differs')
    if sum(x['nominalMACs'] for x in pe['linears'] + pe['contractions']) != pe['nominalMatrixMACs'] or se['nominalMatrixMACs'] + pe['nominalMatrixMACs'] != report['totalIncrementalMatrixMACs']:
        raise ValueError('Total physical cost sum differs')
    head_blocks = [x for x in pe['linears'] if x['parameter'] == WO]
    if len(pe['linears']) != 52 or [x['head'] for x in head_blocks] != list(range(24)) * 2 or any(x['weightShape'] != [3072, 128] or x['inputShape'] != [4, 128] for x in head_blocks):
        raise ValueError('Unchanged physical reader head blocks differ')
    for e in [se, pe]:
        guard = e['fileQuarantine']
        if not set(guard['successfulFileOpenCounts']).issubset(guard['allowedFiles']) or e['parameterInputRNGClosure'] is not True:
            raise ValueError('Actual input/parameter closure differs')
    source_weights = {x['name']: x['payloadSHA256'] for x in se['loadedParameters']}
    for role, source_record in s['sourceCuts'].items():
        cut = source_record['cut']
        if identity(cut['manifest']) != source_record['cutID']:
            raise ValueError('Authenticated source cut differs')
        declared = {x['key']['value']: x['value'] for x in cut['manifest']['payload']['items']}
        for slot, page in source_record['pages'].items():
            if cut['slots'][slot] != page['pageID'] or page['declaredDescriptor'] != declared[slot] or page['metadata']['value'] != declared[slot]:
                raise ValueError('Cut-slot-page descriptor membership differs')
        context = {x['key']['value']: x['value'] for x in declared['context']['items']}
        expected = {'image': declared['image']['sha256'], 'text': declared['text']['sha256'],
                    'img_mod': context['img_mod']['sha256'], 'txt_mod': context['txt_mod']['sha256'],
                    'cos': context['rope']['items'][0]['sha256'], 'sin': context['rope']['items'][1]['sha256'], **source_weights}
        actual = prod['sources'][role]
        check_frame(actual, expected, r['requestSHA256'])
        for name, digest in actual['evidence']['metrics']['capturedPayloads'].items():
            if actual['fields'][name]['payloadSHA256'] != digest or actual['outputFrame']['slots'][name]['fingerprint'] != digest:
                raise ValueError('Source typed output capture differs')
    expected = {role + '.' + name: prod['sources'][role]['fields'][name]['payloadSHA256']
                for role in ['P', 'Q'] for name in ['R', 'A', 'U', 'M', 'T', 'gateA', 'gateM']}
    check_frame(prod['splice'], expected, r['requestSHA256'])
    for arm, actor in a['actors'].items():
        if c['actorInputs'][arm + '.rows'] != prod['splice']['fields'][arm]:
            raise ValueError('Reader did not receive own emitted rows')
        if prod['splice']['outputFrame']['slots'][arm]['fingerprint'] != prod['splice']['fields'][arm]['payloadSHA256']:
            raise ValueError('Splice typed output differs')
        weights = {x['name']: x['payloadSHA256'] for x in pe['loadedParameters']}
        expected = {n: x['payloadSHA256'] for n, x in c['actorInputs'].items() if not n.endswith('.rows')}
        expected.update(textRows=prod['splice']['fields'][arm]['payloadSHA256'],
                        Kweight=weights['transformer_blocks.1.attn.add_k_proj.weight'],
                        Vweight=weights['transformer_blocks.1.attn.add_v_proj.weight'],
                        KnormWeight=weights['transformer_blocks.1.attn.norm_added_k.weight'], WoFP64=pe['WoFP64']['payloadSHA256'])
        check_frame(actor, expected, r['requestSHA256'])
        if actor['evidence']['metrics']['capturedPayloadSHA256'] != actor['fields']['R_attention']['payloadSHA256']:
            raise ValueError('Reader typed output capture differs')
    raw = gzip.decompress((p / 'queries.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != read(p / 'provenance.json')['losslessStreams'][0]['sourceSHA256']:
        raise ValueError('Lossless query stream differs')
    rows = [json.loads(line) for line in raw.splitlines()]
    if [x['imageQueryIndex'] for x in rows] != list(range(1024)):
        raise ValueError('Actual dependent query coverage differs')
    return {'schema': 'tce-selected-source-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']),
            'typedSourceActs': 2, 'typedSpliceActs': 1, 'typedReaderActs': 2, 'sourcePageMembershipJoins': 6,
            'sourceAffines': len(se['linears']), 'sourceRectangularSDPA': len(se['attentions']),
            'readerPhysicalHeadBlocks': len(head_blocks), 'queryRecords': len(rows),
            'fullByteFiles': len(receipt['files']), 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
            'scope': 'Source/cut/page/seal/complete-byte/typed-dependency/physical-work/quarantine mechanics only; no scientific effect threshold or image/general-generator certification.'}


def check_frame(actual, expected, program):
    src, dst, ev = actual['sourceFrame'], actual['outputFrame'], actual['evidence']
    if ({n: x['fingerprint'] for n, x in src['slots'].items()} != expected or src['frame_id'] != ev['input_frame_id']
            or dst['parent_frame_id'] != src['frame_id'] or dst['frame_id'] != ev['output_frame_id'] or dst['program_fingerprint'] != program):
        raise ValueError('Actual typed source/output lineage differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'publish', 'audit'])
    parser.add_argument('source')
    parser.add_argument('destination', nargs='?')
    args = parser.parse_args()
    result = publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source)
    print(json.dumps(result, indent=2))
