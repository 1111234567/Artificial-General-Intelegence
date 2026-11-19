"""Verify the morning report, selected evidence and downloadable CPU artifact."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile


ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/pulse-game"
OUT=DATA/"morning-20261119"
EVIDENCE=DATA/"research-2026-11-19"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))


def main():
    docs=[ROOT/"docs"/f for f in ("pulse-game-morning-2026-11-19.md","pulse-game-evidence-2026-11-19.md",
          "pulse-game-transfer-boundaries-2026-11-19.md","pulse-game-night-2026-11-19.md")]
    docs+=[ROOT/"experiments/pulse_game/README.md",OUT/"README.md"]
    link_counts={}
    for file in docs:
        found=re.findall(r"!?\[[^]\n]*\]\(([^)\n]+)\)",file.read_text(encoding="utf-8-sig"))
        local=[]
        for raw in found:
            if raw.startswith(("https://","http://","#")):continue
            target=(file.parent/unquote(raw.split("#")[0])).resolve()
            if not target.exists():raise FileNotFoundError((file,raw))
            local.append(str(target.relative_to(ROOT)))
        link_counts[str(file.relative_to(ROOT))]=len(local)
    checks=read(OUT/"portable-verification.json");restart=read(OUT/"fresh-process-restart.json")
    archive=OUT/"continuous-learning-cpu.zip"
    h=sha(archive)
    assert h==checks["archive_sha256"]==restart["archive_sha"]
    assert len(checks["examples"])==5 and all(r["exact_native_aggregate"] for r in checks["examples"])
    assert len(restart["checks"])==3 and all(c["exact"] for c in restart["checks"].values())
    assert sha(OUT/"reference-agent.npz")==checks["weights_sha256"]
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read("manifest.json"))
        for name,expect in manifest["sha256"].items():assert hashlib.sha256(z.read(name)).hexdigest()==expect,name
        assert set(z.namelist())==set(manifest["sha256"])|{"manifest.json"}
    names=("legacy_grounded-paired.json","legacy_grounded-report.md","legacy_grounded-replay.json","legacy_grounded-release.pt",
           "lifetime-review-statistics.json","lifetime-review-report.md","active-symmetry-final.json","symmetry-spike-summary.json",
           "attractor-dendrite-summary.json","attractor-motif-mechanism.md","precision-cache-verified.json",
           "physical-throughput/summary.json","untyped_revisit/summary.json","long-life-sealed/complete.json",
           "unified-physical-sealed/complete.json","self-organized-twelve-births/complete.json","matched-sensory-topology/complete.json",
           "asynchronous-credit-sealed/complete.json","object-content-sealed/complete.json","repeated-game-switches/complete.json")
    evidence=[EVIDENCE/n for n in names]
    evidence+=[DATA/k for k in ("context-active-20261119/manifest.json","context-local-20261119/manifest.json",
       "twostep-actual-reward-20261119/manifest.json","twostep-adaptive-20261119/manifest.json","twostep-20261119/manifest.json",
       "independent-lifelong-births-v1/registered.json","independent-lifelong-births-v1/whole_recipe_verified.json",
       "native-long-life-v1/complete.json","native-long-life-local-v1/complete.json","native-long-life-selective-v1/complete.json")]
    evidence+=[OUT/n for n in ("independent-birth-statistics.json","learning-and-retention.json","credit-and-transfer.json",
             "topology-comparison.json","learning-results.png","learning-results.pdf","single-life-trace.png","single-life-trace.json",
             "portable-verification.json","fresh-process-restart.json","reference-selection.json","reference-agent.npz",
             "readout-options.json","all-tests.stdout.log")]
    tests=(OUT/"all-tests.stdout.log").read_text(encoding="utf-8-sig")
    m=re.search(r"(\d+) passed",tests)
    if not m or re.search(r"[1-9]\d* failed",tests):raise ValueError("Suite did not pass")
    head=read(OUT/"learning-and-retention.json")
    assert head["local_addresses"]["seed_intersections"]=={}
    for mode in ("native_stream","native_local_stream","native_selective_stream"):
        assert head[mode]["complete"] and head[mode]["requests_per_life"]==96000
    results=dict(verified_at=datetime.now(timezone.utc).astimezone().isoformat(),
       files={p.relative_to(ROOT).as_posix():sha(p) for p in docs+evidence},
       test_passes=int(m.group(1)),links_checked=link_counts,portable_archive_sha256=h,
       reference=checks["identity"],portable_cpu_cases=len(checks["examples"]),fresh_process_modes=len(restart["checks"]),
       independent_developmental_starts=len(head["local_addresses"]["independent_births"]),
       life_requests={m:head[m]["requests_per_life"] for m in ("native_stream","native_local_stream","native_selective_stream")})
    (OUT/"release-verification.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    print("verified",len(results["files"]),"files,",sum(link_counts.values()),"local references,",m.group(1),"tests, 5 portable runs, 3 process restarts")


if __name__=="__main__":main()
