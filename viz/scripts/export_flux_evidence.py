"""Export existing FLUX evidence into a compact static teaching pack. No model calls.

Uses the public, corrected tensor archive reader for native BF16/int64 arrays.
Forecast bundles are streamed and each selected framed leaf is hash-verified;
the multi-gigabyte bundle never needs to be held in memory.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import numpy as np

AUDIT_SHA = "314fe2d2c63aedf1392b87f0c599a417bfc241e54fac8a9f127d868c834f3115"
ROWS = [0, 31, 127, 255, 511, 767, 1023]
FEATURES = [0, 1, 2, 3, 31, 63, 127, 1023, 3071]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(node):
    kind = node["kind"]
    if kind == "scalar":
        return node["value"]
    if kind == "array":
        return node
    if kind == "mapping":
        return {k: plain(v) for k, v in node["items"].items()}
    if kind in ("list", "tuple"):
        return [plain(v) for v in node["items"]]
    raise ValueError("Unknown forecast descriptor: " + kind)


def main(workspace, output):
    workspace, output = workspace.resolve(), output.resolve()
    source = workspace / "artifacts/mdb-flux-initial-noise-20261105-corrected"
    data = source / "run-01/extracted/flux-initial-noise-result"
    reader_path = workspace / "saturn-pub/packages/mdb/src/mdb/tensor_archive.py"
    spec = importlib.util.spec_from_file_location("flux_export_archive", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    assert sha(source / "audit.json") == AUDIT_SHA, "Accepted audit identity changed"
    sources, bundle_checks, leaves = {}, set(), {}

    def record(path, digest=None):
        relative = str(path.relative_to(workspace))
        sources[relative] = digest or sha(path)
        return relative

    def read(path):
        record(path)
        return json.loads(path.read_text())

    def native(ref):
        assert ref.get("encoding") in ("bf16-bits-v1", "native-npy-v1")
        value = reader.load_tensor_archive(data, ref)
        record(data / ref["path"], ref["sha256"])
        return value

    def forecast(ref, root):
        path = root / ref["path"]
        assert path.resolve().is_relative_to(root.resolve())
        assert ref["storage"] == "bundle-v1"
        if path not in bundle_checks:
            assert sha(path) == ref["bundle_sha256"], "Forecast bundle changed"
            record(path, ref["bundle_sha256"])
            bundle_checks.add(path)
        with path.open("rb") as stream:
            stream.seek(ref["offset"])
            raw = stream.read(ref["nbytes"])
        assert len(raw) == ref["nbytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
        stream = io.BytesIO(raw)
        value = np.load(stream, allow_pickle=False)
        assert stream.tell() == len(raw) and list(value.shape) == ref["shape"] and str(value.dtype) == ref["dtype"]
        assert hashlib.sha256(value.tobytes(order="C")).hexdigest() == ref["content_sha256"]
        assert np.isfinite(value).all()
        leaves[ref["sha256"]] = {k: ref[k] for k in ("path", "offset", "nbytes", "content_sha256", "shape", "dtype")}
        return value

    def write(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))

    def source_ref(ref, root=data, row_slice=None):
        value = {"path":str((root/ref["path"]).relative_to(workspace)), "sha256":ref["sha256"]}
        if "descriptor" in ref:
            value.update(dtype=ref["descriptor"]["dtype"], encoding=ref["encoding"])
        else:
            value.update(dtype=ref["dtype"], offset=ref["offset"], nbytes=ref["nbytes"], contentSHA256=ref["content_sha256"])
        if row_slice:
            value["rowSlice"] = row_slice
        return value

    def sampled(array, identity, metrics=None, source_cursor=None, source=None):
        values = array.reshape(-1, array.shape[-1])
        rows = [r for r in ROWS if r < values.shape[0]]
        features = [f for f in FEATURES if f < values.shape[1]]
        return {"shape": list(array.shape), "rows": rows, "features": features,
                "values": values[np.ix_(rows, features)].tolist(), "sha256": identity,
                "metrics": metrics, "measurement": "coordinate-sample", "sourceCursor": source_cursor, "source":source}

    audit = read(source / "audit.json")
    binding = read(data / "binding.json")
    summary = read(data / "summary.json")
    assert binding["logical_steps"] == [0, 1, 2, 3] and len(binding["final_blocks"]) == 20
    prefix, segment = read(data / "native-prefix.json"), read(data / "native-segment.json")
    boundaries = segment["boundaries"]
    assert len(boundaries) == 112 and [b["index"] for b in boundaries] == list(range(112))
    assert [b["cut"] for b in prefix["boundaries"]] == [b["cut"] for b in boundaries[:56]]
    origin = binding["origins"]["full"]["inputs"]
    sigmas, timesteps, ids = native(origin["sigmas"]), native(origin["timesteps"]), native(origin["img_ids"])
    initial = native(origin["latent"])
    native_start = [initial] + [native(boundaries[s*28-1]["closure"]["latent"]) for s in range(1, 4)]
    initial_rms = float(np.sqrt(np.mean(initial.astype(np.float64)**2)))
    histories, tile_scale, euler = {}, 0, {"histories": {}, "hostile": read(data / "cuda-scalar-qualification.json")["cases"]}
    # Native PNGs are byte copies, not replacement or generated illustrations.
    for name in ("full.predicted", "matched.predicted", "native", "sham"):
        path = data / (name + ".png")
        record(path)
        (output / "images").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output / "images" / path.name)

    for mode in ("full", "matched"):
        root = data / "transaction" / mode
        encoded = read(root / "forecasts/passes.json")
        seal = read(root / "seal.json")
        assert "passes" in seal["forecasts"]
        program = plain(encoded["predictions"])
        start = program["origin_logical_step"]
        current = initial if start == 0 else native_start[2]
        current_source = source_ref(origin["latent"] if start == 0 else boundaries[55]["closure"]["latent"])
        current_metric = {"rms": 0., "relative_rms": 0., "reference_rms": float(np.sqrt(np.mean(current.astype(np.float64)**2))), "actual_rms": float(np.sqrt(np.mean(current.astype(np.float64)**2))), "max_abs": 0., "signed_mean": 0., "sample_count": int(current.size)}
        records, fixtures = [], []
        for pass_value, pass_audit in zip(program["passes"], audit["histories"][mode]["passes"]):
            step = pass_value["logical_step"]
            assert pass_audit["logical_step"] == step
            print(mode, "pass", step, flush=True)
            incoming = current
            incoming_source = current_source
            context = pass_value["project"]["context"]
            for local in range(28):
                cursor = step*28 + local
                nb, ac = boundaries[cursor], pass_audit["transitions"][local]
                assert ac["native_transition_index"] == cursor and ac["native_cut"] == nb["cut"]
                fields, npred = {}, {}
                kind = ac["kind"]
                if local == 0:
                    refs = {k: pass_value["project"][k] for k in ("image", "text")}
                elif local <= 5:
                    refs = {k: pass_value["joints"][str(local-1)][k] for k in ("image", "text")}
                elif local <= 25:
                    refs = {"both": pass_value["single_steps"][local-6]["prediction"]["output"]}
                elif local == 26:
                    refs = {"noise": pass_value["readout"]["noise"]}
                else:
                    refs = {"latent": pass_value["commit"]["latent"]}
                for field, ref in refs.items():
                    predicted = forecast(ref, root)
                    split = {"image": predicted[:, binding["text_rows"]:], "text": predicted[:, :binding["text_rows"]]} if field == "both" else {field: predicted}
                    for name, value in split.items():
                        key = "noise_prediction" if name == "noise" else name
                        actual_ref = nb["closure"][key]
                        actual = native(actual_ref)
                        metric = ac["streams"][name] if field == "both" else ac["fields"][name]["predicted_minus_native"]
                        assert list(actual.shape) == metric["shape"] and value.shape == actual.shape
                        fields[name] = {"native": sampled(actual, actual_ref["descriptor"]["sha256"], source_cursor=cursor, source=source_ref(actual_ref)),
                                        "predicted": sampled(value, ref["content_sha256"], source_cursor=cursor, source=source_ref(ref, root, ("512:" if name=="image" else ":512") if field=="both" else None)), "error": metric}
                        if name == "latent":
                            current, current_metric = value, metric
                            current_source = source_ref(ref, root)
                        if local == 26:
                            npred[name] = value
                    del predicted
                if local < 27:
                    nr = nb["closure"]["latent"]
                    fields["latent"] = {"native": sampled(native_start[step], nr["descriptor"]["sha256"], source_cursor=step*28-1, source=source_ref(nr)),
                                        "predicted": sampled(incoming, pass_value["input_latent_descriptor"].get("bf16_le_sha256", ""), source_cursor=step*28-1, source=incoming_source), "error": current_metric}
                if local == 26:
                    # Readout reads the image register without mutating either stream.
                    for stream in ("image", "text"):
                        fields[stream] = records[-1]["fields"][stream]
                # The context is pass-local; retain its real shape, sampled modulation, and identity.
                if local < 27:
                    ctx = nb["closure"]["context"]
                    fields["context"] = {"shape": ctx["temb"]["descriptor"]["shape"], "modulation": ctx["single_mod"]["descriptor"]["preview"], "sha256":ctx["temb"]["descriptor"]["sha256"]}
                record_value = {"cursor": cursor, "pass": step, "local": local, "kind": kind, "block": ac["index"], "pointAfter":nb["point"], "nativeCut": nb["cut"], "fields":fields}
                records.append(record_value)
                if local == 26:
                    noise_pred, noise_native = npred["noise"], native(nb["closure"]["noise_prediction"])
                if local == 27:
                    latent_native = native(nb["closure"]["latent"])
                    cells = []
                    for row in (0,31,511,1023):
                        for feature in (0,1,63,127):
                            for lane, z, v, result in (("native",native_start[step],noise_native,latent_native),(mode,incoming,noise_pred,current)):
                                cells.append({"lane":lane,"row":row,"feature":feature,"z":float(z[0,row,feature]),"v":float(v[0,row,feature]),"expected":float(result[0,row,feature])})
                    fixtures.append({"pass":step,"sigma":float(sigmas[step]),"sigmaNext":float(sigmas[step+1]),"dtBF16":pass_value["commit"]["dt_bf16"],"coordinates":cells})
                    tiles={}
                    for label,value in (("native-in",native_start[step]),("predicted-in",incoming),("native-out",latent_native),("predicted-out",current)):
                        tile=value[0,:,:4].T
                        tile_scale=max(tile_scale,float(np.max(np.abs(tile))))
                        tiles[label]=tile.tolist()
                    write(f"tiles/{mode}-{step}.json",{"pass":step,"history":mode,"channels":[0,1,2,3],"tiles":tiles,"scope":"Four complete latent channels; display view only, not RGB or full-field error reduction."})
            # Exact own-state joins are checked by the accepted audit and retained descriptor sequence.
            assert np.array_equal(current, forecast(pass_value["commit"]["latent"], root))
        assert len(records) == (4-start)*28
        histories[mode] = {"origin":start*28,"records":records,"seal":sha(root/"seal.json")}
        euler["histories"][mode]=fixtures

    endpoints=read(source/"endpoint-error-data.json")
    worst=read(source/"endpoint-worst-coordinates.json")
    independent_metrics = []
    def collect_independent(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "sealed_minus_independent":
                    independent_metrics.append(value)
                else:
                    collect_independent(value)
        elif isinstance(node, list):
            for value in node:
                collect_independent(value)
    collect_independent(audit["histories"])
    write("endpoints.json",{"comparisons":audit["endpoint_comparisons"],"distributions":endpoints["distributions"],
        "worst":{k:v["maximum_pixels"] for k,v in worst["branches"].items()},"controls":audit["control_observations"],
        "independent":{"fields":len(independent_metrics),"scalarValues":sum(x["sample_count"] for x in independent_metrics),
                       "mismatches":sum(x["bitwise_mismatch_count"] for x in independent_metrics)},"scope":audit["scope"]})
    write("timeline.json",{"histories":histories,"sampleRows":ROWS,"sampleFeatures":FEATURES})
    write("euler.json",euler)
    write("manifest.json",{"schema":1,"id":"flux2-klein-4b","prompt":audit["context"]["prompt"],"seed":113,"passes":4,"transitionsPerPass":28,
        "geometry":{"latent":[1,1024,128],"conditioning":[1,512,7680],"image":[1,1024,3072],"text":[1,512,3072],"heads":binding["final_blocks"]["0"]["heads"],"headDim":128,"mlpWidth":9216,"ropeAxes":binding["project"]["axes_dims_rope"]},
        "sigmas":sigmas.tolist(),"timesteps":timesteps.tolist(),"modelIdentity":summary["model_identity"],"profile":"CUDA BF16 · native conditioning and VAE · 512×512 · batch 1",
        "noiseHash":origin["latent"]["descriptor"]["sha256"],"conditioningHash":origin["conditioning"]["descriptor"]["sha256"],
        "idGrid":ids[0,:,1:3].astype(int).tolist(),"latentScale":tile_scale,"auditSHA256":AUDIT_SHA,
        "initial":{"shape":list(initial.shape),"rms":initial_rms,"sample":sampled(initial,origin["latent"]["descriptor"]["sha256"], source=source_ref(origin["latent"]))},
        "scope":"One recorded seed113 history. Full forecast starts at actual Z0; matched forecast starts at observed native Z2. No native model runs occur in this website."})
    record(reader_path)
    write("provenance.json",{"schema":1,"auditSHA256":AUDIT_SHA,"modelIdentity":summary["model_identity"],"newModelForwards":0,
        "exporter":"viz/scripts/export_flux_evidence.py","sourceFiles":sources,"forecastLeaves":leaves,
        "reader":"Public corrected tensor_archive.load_tensor_archive for native BF16/int64. Forecast bundle and selected NPY leaf/content digests verified separately with streaming reads.",
        "artifactHashes":{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='provenance.json'}})
    print("Exported",len(sources),"source identities;",len(leaves),"forecast leaves;",sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),"bytes")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"public/data/flux-klein")
    args=parser.parse_args()
    main(args.workspace,args.output)
