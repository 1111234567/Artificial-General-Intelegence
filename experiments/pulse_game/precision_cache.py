"""Measure real 16-bit public-cache storage with unchanged FP32 CUDA dynamics."""

from argparse import ArgumentParser
from itertools import permutations
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.undirected._cuda_nvrtc import compile_kernels
from . import longrun_plastic, longrun_study, longrun_world, mixture_flash, unified_physical, unified_world
from . import research_audit as audit
from .attractor_search import OUT


FORMATS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ROOT = OUT / "unified-physical-evolution-v1" / "rls-learned-s1373--full-e19781"
GPU_FILE = Path(unified_physical.__file__).with_name("unified_gpu.cu")


def single(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Original CUDA source changed at " + old)
    return text.replace(old, new)


def cache_source(source, precision):
    # Use actual 16-bit device arrays. Inline PTX rounds only the store and
    # reconstructs a float before the original matrix/dot/RLS expression.
    if precision == "fp32":
        return source
    head = """
typedef unsigned short cache16;
__device__ __forceinline__ cache16 store_cache(float x) {
  cache16 y; asm("cvt.rn.BITS.f32 %0, %1;" : "=h"(y) : "f"(x)); return y;
}
__device__ __forceinline__ float read_cache(cache16 x) {
  float y; asm("cvt.f32.BITS %0, %1;" : "=f"(y) : "h"(x)); return y;
}
""".replace("BITS", "f16" if precision == "fp16" else "bf16")
    edits = [
        ("float* keys,float* codes,int trials", "cache16* keys,cache16* codes,int trials"),
        ("play(const float* keys,const float* codes,const float* dist", "play(const cache16* keys,const cache16* codes,const float* dist"),
        ("float* out=keys+", "cache16* out=keys+"),
        ("out[k]=read[s][k]/fmaxf(sqrtf(r),1.e-12f);out[4+k]=write[s][k]/fmaxf(sqrtf(w),1.e-12f);",
         "out[k]=store_cache(read[s][k]/fmaxf(sqrtf(r),1.e-12f));out[4+k]=store_cache(write[s][k]/fmaxf(sqrtf(w),1.e-12f));"),
        ("codes[ev*S+i]=loc[i]/fmaxf(sqrtf(norm),1.e-7f);", "codes[ev*S+i]=store_cache(loc[i]/fmaxf(sqrtf(norm),1.e-7f));"),
        ("codes[(e-1)*S+i]*saved_r", "read_cache(codes[(e-1)*S+i])*saved_r"),
        ("=keys[((e-1)*2+choice)*8+4+i];", "=read_cache(keys[((e-1)*2+choice)*8+4+i]);"),
        ("master[k]*codes[e*S+k]", "master[k]*read_cache(codes[e*S+k])"),
        ("keys[(e*2+side)*8+k]*bank[i][k]", "read_cache(keys[(e*2+side)*8+k])*bank[i][k]"),
    ]
    for old, new in edits:
        source = single(source, old, new)
    return head + source


def add_pulse_trace(source):
    source = single(source, "float* gainlog,\n int* summary", "float* gainlog,\n unsigned char* pulse_trace,int* summary")
    anchor = "if(active){vel=.65f*vel+.15f*(motor[1]-motor[0])+wind[ix*15+t];"
    line = "pulse_trace[out*16+t]=(active?4:0)+int(motor[0])+2*int(motor[1]);\n     "
    return single(source, anchor, line + anchor)


def compiled(shape, precision, long, *, trace=False):
    if long:
        source = longrun_plastic.source(*shape)
    else:
        n, h, s, cap = shape
        source = f"#define N {n}\n#define H {h}\n#define S {s}\n#define CAP {cap}\n#define G 19\n" + GPU_FILE.read_text()
    source = cache_source(source, precision)
    if trace:
        source = add_pulse_trace(source)
    return compile_kernels(source, ("prepare","play"), device=torch.device("cuda:0"), options=("--fmad=false",))


class TrialBatch:
    def __init__(self, e, front, route, genes, long=False):
        self.long = long
        self.e = e
        arena, body = e.arena, e.arena.menu.body
        self.n, self.tr = body.ctx.shape
        self.shape = (front["matrix"].shape[0], route["first"][1].size, route["slots"], front["capacity"])
        ot, ct = e.sensory_timing if hasattr(e, "sensory_timing") else mixture_flash.timing(body.seeds,self.tr)
        self.prepare_inputs = [self.gpu(a) for a in (2*arena.menu.chemicals-1,arena.appearance,ot.astype(np.uint8),
                                                    ct.astype(np.uint8),front["matrix"],front["sensors"],
                                                    *front["ground"],front["transform"],*route["first"],*route["second"])]
        self.address_mode={"learned":0,"random":0,"single":1,"public":2,"shared_public":3}[route["routing"]]
        self.share = float(route.get("shared",0.))
        blocks = (self.tr*16+127)//128
        noise=np.stack([np.random.default_rng(np.random.SeedSequence([s,745930])).standard_normal((blocks*128,2),dtype=np.float32)[:self.tr*16] for s in body.seeds])
        self.world_inputs = [self.gpu(v) for v in (body.distances,body.wind,arena.menu.realized,arena.probe.astype(np.uint8),noise)]
        self.checks = e.headline_probes if long else (
            e.checks["B_after_0"],e.checks[f"B_after_{e.steps[-1]}"],e.checks["return_A"])
        self.rule = int(route["updater"] == "rls")
        self.genes = self.gpu(genes)
        self.pop = len(genes)
        self.stats = torch.empty((self.pop,self.n,6),device="cuda",dtype=torch.int32)
        self.none8 = torch.empty(1,device="cuda",dtype=torch.int8)
        self.none32 = torch.empty(1,device="cuda",dtype=torch.float32)
        self.format = {}
        for tag, dtype in FORMATS.items():
            normal = compiled(self.shape, tag, long)
            capture = compiled(self.shape, tag, long, trace=True)
            keys = torch.empty((self.n,self.tr,16,2,8),device="cuda",dtype=dtype)
            addr = torch.empty((self.n,self.tr,16,route["slots"]),device="cuda",dtype=dtype)
            self.format[tag] = (normal,capture,keys,addr)

    @staticmethod
    def gpu(x):
        return torch.as_tensor(np.ascontiguousarray(x),device="cuda")

    def prepare(self, tag):
        code, _, keys, addr = self.format[tag]
        code["prepare"]((self.n,),(64,),*self.prepare_inputs,keys,addr,self.tr,self.address_mode,self.share)

    def play(self, tag, *, detail=False, trace=False):
        normal, capture, keys, addr = self.format[tag]
        code = capture if trace else normal
        shape = (self.pop,self.n,self.tr)
        if detail:
            outputs = [torch.empty(shape,device="cuda",dtype=torch.int8) for _ in range(3)]
            gain = torch.empty(shape,device="cuda",dtype=torch.float32)
        else:
            outputs, gain = [self.none8]*3, self.none32
        taps = [torch.empty(shape+(16,),device="cuda",dtype=torch.uint8)] if trace else []
        code["play"]((self.n,self.pop),(32,),keys,addr,*self.world_inputs,self.genes,*outputs,gain,
                     *taps,self.stats,self.n,self.tr,*self.checks,self.rule,int(detail),0)
        if detail:
            return tuple(a.cpu().numpy() for a in (*outputs,*taps,self.stats))

    def sizes(self):
        return {tag:sum(k.numel()*k.element_size() for k in item[2:]) for tag,item in self.format.items()}


def wall_ms(action):
    torch.cuda.synchronize()
    start=time.perf_counter()
    action()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000


def timing(batch, repeats):
    for _ in range(3):
        for tag in FORMATS:
            batch.prepare(tag)
            batch.play(tag)
    torch.cuda.synchronize()
    warmed={}
    for task in ("resident_play","prepare_then_play","prepare_only"):
        samples=[]
        for order in list(permutations(FORMATS))*repeats:
            row={}
            for tag in order:
                def measure():
                    if task!="resident_play":
                        batch.prepare(tag)
                    if task!="prepare_only":
                        batch.play(tag)
                row[tag]=wall_ms(measure)
            samples.append(row)
        medians={tag:float(np.median([row[tag] for row in samples])) for tag in FORMATS}
        paired={tag:[row["fp32"]/row[tag] for row in samples] for tag in ("fp16","bf16")}
        warmed[task]={"milliseconds_by_paired_round":samples, "median_ms":medians,
                      "median_paired_fp32_over_variant":{k:float(np.median(v)) for k,v in paired.items()},
                      "paired_speedup_iqr":{k:np.quantile(v,[.25,.75]).tolist() for k,v in paired.items()},
                      "individual_paired_speedups":paired,
                      "ordered_variants":[list(order) for order in list(permutations(FORMATS))*repeats]}
    return warmed


def accuracy(batch, native):
    observed={}
    for tag in FORMATS:
        batch.prepare(tag)
        observed[tag]=batch.play(tag,detail=True,trace=True)
    original=batch.play("fp32",detail=True)
    correct=observed["fp32"]
    if any(not np.array_equal(x,y) for x,y in zip(original,(*correct[:3],correct[-1]),strict=True)):
        raise ValueError("Instrumented 32-bit cache changes the reference rollout")
    for i, outcome in enumerate(native):
        view=(outcome.contact_side,outcome.contact_tick,outcome.reward)
        if any(not np.array_equal(x[i],y) for x,y in zip(correct[:3],view,strict=True)):
            raise ValueError("Benchmark inputs or reference do not match the native FP32 cohort")
    truth = batch.e.arena.truths[None]
    probes = batch.e.arena.probe[None]
    correct_ref = (correct[0]>=0)&(correct[0]==truth)
    ref_valid=np.broadcast_to(probes,correct[0].shape)
    reference_keys, reference_addr = batch.format["fp32"][2:]
    summary={}
    for tag,current in observed.items():
        a,b,c,pulse,counts=current
        active_applied=np.where(pulse&4,pulse&3,0)
        base_applied=np.where(correct[3]&4,correct[3]&3,0)
        hit=(a>=0)&(a==truth)
        assert np.array_equal(hit[ref_valid],(c>0)[ref_valid])
        stored_keys, stored_addr = batch.format[tag][2:]
        rounding = (torch.equal(stored_keys,reference_keys.to(FORMATS[tag]))
                    and torch.equal(stored_addr,reference_addr.to(FORMATS[tag])))
        summary[tag]={"native_rounded_store_equals_torch_cast":rounding,
                      "contact_sides_changed":int((a!=correct[0]).sum()),"contact_ticks_changed":int((b!=correct[1]).sum()),
                      "rewards_changed":int((c!=correct[2]).sum()),
                      "raw_motor_ticks_changed":int(((pulse&3)!=(correct[3]&3)).sum()),
                      "physically_applied_motor_ticks_changed":int((active_applied!=base_applied).sum()),
                      "scored_probe_correct":int(hit[ref_valid].sum()),
                      "correct_probe_labels_changed":int((hit[ref_valid]!=correct_ref[ref_valid]).sum()),
                      "first_b_or_first_probe_accuracy":float(hit[...,batch.checks[0]].mean()),
                      "second_headline_probe_accuracy":float(hit[...,batch.checks[1]].mean()),
                      "third_headline_probe_accuracy":float(hit[...,batch.checks[2]].mean()),
                      "actual_all_probes_accuracy":float(hit[ref_valid].mean()),
                      "kernel_public_outcome_counts_equal":bool(np.array_equal(counts,correct[-1]))}
    return {"physical_requests":int(correct[0].size),"motor_tick_opportunities":int(correct[3].size),
            "scored_physical_probes":int(ref_valid.sum()),"native_fp32_entries_exact":len(native),
            "variants":summary}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--repeats",type=int,default=3,help="sets of all six paired orders")
    parser.add_argument("--case",choices=("all","unified","long2304","long9216"),default="all")
    args=parser.parse_args()
    if args.repeats<1 or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device and balanced timing repeats are required")
    torch.set_num_threads(1)
    front,route,motion,body,_,fit=longrun_study.origin(ROOT)
    selected=json.loads((OUT/"long-life-plasticity-v1"/(ROOT.name+".json")).read_text())
    rigid=np.zeros(5,np.float32)
    learned=np.asarray(longrun_plastic.choices()[selected["global_validation_winner"]],np.float32)
    rng=np.random.default_rng(73164009)
    keys={"unified":rng.integers(550_000_000,620_000_000,size=64).tolist(),
          "long2304":rng.integers(620_000_000,670_000_000,size=16).tolist(),
          "long9216":rng.integers(670_000_000,720_000_000,size=8).tolist()}
    result={"device":torch.cuda.get_device_name(0),"torch":torch.__version__,"checkpoint":str(ROOT/"frozen.json"),
            "checkpoint_sha256":hashlib.sha256((ROOT/"frozen.json").read_bytes()).hexdigest(),
            "unmodified_fp32_kernel_sha256":hashlib.sha256(GPU_FILE.read_bytes()).hexdigest(),
            "precision_scope":"only retained public keys and addresses; all float math and mutable state still FP32",
            "time_scope":"synchronized per-call wall time with resident inputs and preallocated buffers; no tracing/compilation",
            "cases":{}}
    suffix = "" if args.case == "all" else "-" + args.case
    out=audit.OUT/f"precision-cache-verified{suffix}.json"
    for tag,seeds in keys.items():
        if args.case not in ("all",tag):
            continue
        if tag=="unified":
            episode=unified_world.make(seeds)
            # Repeated selected and native phenotypes approximate small resident population scoring.
            native=unified_physical.params(front,route,motion)
            genes=np.tile(np.stack((body,native)),(32,1))
            native_cohort=unified_physical.Cohort(episode,front,route,motion)
            g=np.asarray(fit["best"]["genome"])
            native_records=[x for x,_ in native_cohort.run(np.stack((g,np.zeros_like(g))),fit["settings"]["mode"],detail=True)]
        else:
            blocks=192 if tag=="long2304" else 768
            episode=longrun_world.make(seeds,blocks=blocks,mode="reverse_return",noise=.05 if blocks==192 else .07)
            genes=np.column_stack((np.repeat(body[None,:],4,axis=0),np.stack((learned,rigid,learned,rigid))))
            native_cohort=longrun_plastic.Cohort(episode,front,route,motion)
            native_records=native_cohort.adjustments(body,np.stack((learned,rigid)),detail=True)
        del native_cohort
        sample=TrialBatch(episode,front,route,genes,long=tag!="unified")
        effect=accuracy(sample,native_records)
        speed=timing(sample,args.repeats)
        result["cases"][tag]={"seeds":seeds,"requests_per_life":sample.tr,"phenotype_entries":len(genes),
                              "distinct_genotypes":len(np.unique(genes,axis=0)),
                              "physical_cache_bytes":sample.sizes(),"accuracy":effect,"timing":speed}
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        print(tag, sample.sizes(), effect, {k:v["median_paired_fp32_over_variant"] for k,v in speed.items()},flush=True)
        del sample
        torch.cuda.synchronize()
    print(out,flush=True)


if __name__=="__main__":
    main()
