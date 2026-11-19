"""Read native action logits on the audited live trajectories of the BEGIN study."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import torch
from torch.func import functional_call

from experiments.coverage_exit.fast_window import WindowBlock
from experiments.undirected.model import PortGraphModel


def digest(data):
    return hashlib.sha256(data).hexdigest()


def describe(parts, first, stop):
    if not parts:
        return {"positions": 0, "balanced_content_accuracy": None, "median_group_margin": None}
    logits, labels = (torch.cat(group) for group in zip(*parts))
    best, content = logits[:, first:stop].max(-1)
    outside = torch.cat((logits[:, :first], logits[:, stop:]), dim=-1).amax(-1)
    counts = torch.bincount(labels-first, minlength=stop-first)
    correct = torch.bincount(labels[(content+first).eq(labels)]-first, minlength=stop-first)
    selected = logits.argmax(-1)
    return dict(positions=labels.numel(), label_counts=counts.tolist(), correct_within_group=correct.tolist(),
        balanced_content_accuracy=float((correct.double()/counts).mean()) if counts.min()>0 else None,
        median_group_margin=float(torch.quantile(best-outside,.5)), chosen_group=int(((selected>=first)&(selected<stop)).sum()))


@torch.inference_mode()
def analyze(protocol):
    raw = protocol.read_bytes()
    meta = json.loads(raw)
    root = Path(meta["output"])
    verification = root / "verification.json"
    validated = json.loads(verification.read_bytes())
    if validated["protocol_sha256"] != digest(raw) or not validated["external_streams_exact_between_arms"]:
        raise ValueError("need the matching verified coverage study")
    wrapper = WindowBlock(PortGraphModel().double())
    rows = []
    for seed in meta["seeds"]:
        for arm in meta["arms"]:
            files = sorted((root/"training"/f"seed_{seed:03d}"/arm/"audit").glob("*.pt.gz"))
            if len(files) != meta["updates"]:
                raise ValueError("incomplete trajectory")
            groups = {(half,name):[] for half in (0,1) for name in ("BEGIN","FINISH")}
            for index, file in enumerate(files):
                data = file.read_bytes()
                if digest(data) != validated["files"][str(file.resolve())]:
                    raise ValueError("changed live record")
                audit = torch.load(io.BytesIO(gzip.decompress(data)), map_location="cpu",weights_only=False)
                record = audit["record"]
                params = {"model."+k:v for k,v in record["parameters"].items()}
                buffers = {"model."+k:v for k,v in record["buffers"].items()}
                logits, end = functional_call(wrapper,(params,buffers),
                    (record["observations"],record["start_h"],record["elapsed"],True,True),strict=True)
                if not torch.equal(end,record["end_h"]) or not torch.equal(logits.argmax(-1),audit["actions"]["model_actions"]):
                    raise ValueError("reconstructed state or pre-assistance actions differ")
                labels = record["targets"]
                half = int(index >= len(files)//2)
                for name, lo, hi in (("BEGIN",1,3),("FINISH",3,11)):
                    mask = (labels>=lo)&(labels<hi)
                    if mask.any():
                        groups[(half,name)].append((logits[mask],labels[mask]))
            for (half,name), parts in groups.items():
                lo, hi = (1,3) if name=="BEGIN" else (3,11)
                rows.append(dict(seed=seed, arm=arm, half=half, teacher_action_group=name, **describe(parts,lo,hi)))
    target = root/"native-action-margins.json"
    output = dict(protocol_sha256=digest(raw), verified_inputs_sha256=digest(verification.read_bytes()),
        reader_sha256=digest(Path(__file__).read_bytes()), aggregation="Live label positions, including repeated positions per request. First and last 64 updates separately. Balanced accuracy averages per-class recall among the existing logits in the teacher group. Margin is best group logit minus best outside logit. No fitted decoder.", rows=rows)
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return target


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    print(analyze(args.protocol))
