"""Offline, model-free export of existing ISA evidence. Run with common exec python.

No forwards, training, or mutation of research artifacts. Torch only decodes the
existing weights-only parameter archive. Architectural integers become strings.
"""
import hashlib
import json
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "public/data"
SCAN = ROOT / "isa/runs/native-forward-symbols-20261029T072605.019798Z/extracted/compiled-scan-result"
STREAM = ROOT / "isa/runs/native-forward-symbols-20261029T090450.332745Z/extracted/stream-program-result"
LOOP = ROOT / "isa/runs/native-forward-symbols-20261029T100021.116951Z/extracted/symbolic-loop-result"
SOURCES = {}


def record(path):
    b = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(b).hexdigest()
    return b


def load(path):
    return json.loads(record(path))


def safe(value, key=""):
    if isinstance(value, dict):
        return {k: safe(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [safe(v, key) for v in value]
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        return str(value)
    return value


def write(name, value):
    (OUT / name).write_text(json.dumps(safe(value), separators=(",", ":"), allow_nan=False))


def values(d):
    return d["values"]


def main(workspace=None):
    global ROOT, SCAN, STREAM, LOOP
    if workspace is not None:
        ROOT = workspace.resolve()
        SCAN = ROOT / "isa/runs/native-forward-symbols-20261029T072605.019798Z/extracted/compiled-scan-result"
        STREAM = ROOT / "isa/runs/native-forward-symbols-20261029T090450.332745Z/extracted/stream-program-result"
        LOOP = ROOT / "isa/runs/native-forward-symbols-20261029T100021.116951Z/extracted/symbolic-loop-result"
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    report = load(SCAN / "report.json")
    assert report["checkpoint_sha256"] == "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
    case = report["cases"][0]
    native = load(SCAN / "case-carry-depth-0/native-scan-factors.json")["selected_native_factors"]
    mlp = load(SCAN / "case-carry-depth-0/mlp-observation.json")["selected_values"]
    archive = STREAM / "stream-program.pt"
    record(archive)
    parameters = torch.load(archive, map_location="cpu", weights_only=True)["parameters"]
    keys = ["post.2.0.weight", "post.2.0.bias", "post.2.1.weight", "post.2.1.bias", "post.2.3.weight", "post.2.3.bias", "reg_out.weight", "reg_out.bias"]
    weights = {k: parameters[k].tolist() for k in keys}
    f = lambda name: values(native["scan.2." + name])
    write("scan.json", {
        "caseId": "carry-depth-0", "layer": 2, "inputWord": str(case["target_register_word"]),
        "a": f("A_derived_from_a_log"), "d": values(native["parameter.mixers.2.d"]),
        "u": f("u")[0], "dt": f("dt")[0], "b": f("B")[0], "c": f("C")[0], "g": f("G")[0],
        "h": f("H")[0], "residual": f("residual")[0], "mixer": f("hmix")[0],
        "wOut": values(native["parameter.mixers.2.out_proj.weight"]), "outBias": values(native["parameter.mixers.2.out_proj.bias"]),
        "post": {k.removeprefix("post.2."): values(v)[0] for k, v in mlp.items() if k.startswith("post.2.")},
        "weights": weights, "nativeRegs": values(case["native_step0_output_heads"]["regs"])[0],
        "nativeHeads": {k: {"shape":v["shape"],"values":v["values"]} for k,v in case["native_step0_output_heads"].items()},
        "source": str((SCAN / "case-carry-depth-0/native-scan-factors.json").relative_to(ROOT)),
        "traceFingerprint": load(SCAN / "case-carry-depth-0/native-scan-factors.json")["trace_fingerprint"],
    })
    initial = [str(x) for x in case["register_words_before"]]
    heads = case["native_step0_output_heads"]["regs"]["values"][0]
    native_words = [str(sum((1 << i) for i, x in enumerate(reg) if x > 0)) for reg in heads]
    write("carry.json", {"before": initial, "after": native_words, "flagsBefore":case["flags_before"],
        "flagsAfter":[int(x > 0) for x in case["native_step0_output_heads"]["flags"]["values"][0]],
        "route":7, "programHex":case["program_hex"], "rawWindowHex":case["program_hex"][:30].ljust(30,"0"),
        "registerLogits":heads, "length":4, "carryDepth":case["carry_depth_low_bits"],
        "source":str((SCAN/"report.json").relative_to(ROOT))})
    inputs = load(LOOP / "pure-inputs.json")
    comparisons = load(LOOP / "comparisons.json")
    predictions = load(LOOP / "predictions-pre-native.json")
    cases=[]
    for i, inp in enumerate(inputs):
        case_id = predictions["case_ids"][i]
        cases.append({"id":case_id, "initial": {**inp,"register_words_before":[str(x) for x in inp["register_words_before"]]},
            "steps":[r for r in comparisons if r["case_id"] == case_id],
            "predictions":predictions["predictions"][i]["steps"]})
    write("loop.json", {"cases":cases, "source":str((LOOP/"comparisons.json").relative_to(ROOT)), "summary":load(LOOP/"report.json")["summary"]})
    quotient = load(ROOT/"isa/outputs/response-quotient-r3-mamba/findings.json")
    frontier = load(ROOT/"isa/outputs/stream-frontier-r3-mamba/findings.json")
    write("frontier.json", {"cells":[{"h":int(b["cell"][0]), "conv":int(b["cell"][1]), "logit":b["target"]["32"]["logit"], "bit":b["target"]["32"]["bit"]} for b in frontier["branches"] if b["case_id"] == "frontier-zero"],
        "exceptions":quotient["heldout_metrics"]["target_bit_exceptions"], "exactWords":93,"totalWords":96,
        "source":str((ROOT/"isa/outputs/stream-frontier-r3-mamba/findings.json").relative_to(ROOT)),
        "boundary":"Source-qualified streamed equations, calibrated against native baselines. Full H plus convolution swaps reproduce donor future register-logit suffixes in the two tested contexts; individual port necessity is unresolved."})
    readings = [
        ("state-math", "State, context, and the mathematics of time circuits", "2026-11-05-153154-state-context-and-the-mathematics-of-time-circuits.md"),
        ("mlp", "How the same MLP changes its response", "2026-10-29-142000-how-the-same-mlp-changes-its-response.md"),
        ("carry", "Tracing a carry computation through state and MLP laws", "2026-10-29-172053-tracing-a-carry-computation-through-state-and-mlp-laws.md"),
        ("symbols", "A dynamic symbol table read off the model and frozen to predict", "2026-10-30-140215-a-dynamic-symbol-table-read-off-the-model-and-frozen-to-predict.md"),
    ]
    reading_index=[]
    for slug,title,name in readings:
        path=ROOT/"obsidian/blog"/name
        content=record(path).decode()
        (OUT / f"reading-{slug}.txt").write_text(content)
        reading_index.append({"id":slug,"title":title,"path":f"./data/reading-{slug}.txt","source":str(path.relative_to(ROOT))})
    write("readings.json",reading_index)
    write("provenance.json", {"schema":"time-circuit-evidence-v1", "checkpoint":report["checkpoint_sha256"],
        "export":"viz/scripts/export_evidence.py", "newModelForwards":0, "sources":SOURCES,
        "scope":"Recorded CPU FP32 observations; JavaScript local calculations use double precision. Compact response law has retained native exceptions. No general semantic ISA claim."})
    print("Exported",len(SOURCES),"hashed sources;",sum(p.stat().st_size for p in OUT.iterdir()),"bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT, help="Research workspace containing isa/ and obsidian/; defaults to this repository's parent")
    main(parser.parse_args().workspace)
