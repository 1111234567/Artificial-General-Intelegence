"""Synthetic count/provenance rejection tests; no learner is advanced."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.credit.report import (
    CONTRASTS, MODES, PROTOCOL_SHA256, _semantic, aggregate, expected_work, load_cycles,
    load_cycle_measurement_revision, load_protocol, normalize, write_artifacts,
)
from experiments.credit.run import MODES as FLAGS
from experiments.integration.test_report import _record as old_record


def digest(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def fixture(*, seeds=(0, 1, 2), modes=MODES):
    files = {f"F:/synthetic/experiments/{folder}/{name}": digest((folder, name))
             for folder, names in (("credit", ("run.py", "local.py", "test_run.py", "test_local.py", "evaluate.py", "cycle.py")),
                                   ("continuity", ("run.py", "world.py", "model.py", "test_run.py")),
                                   ("integration", ("run.py", "diagnose.py"))) for name in names}
    hashes = lambda folder, names: {name: files[f"F:/synthetic/experiments/{folder}/{name}"] for name in names}
    selected = [mode for mode in ("readout_only", "bp64", "bp_replay", "diag_sym", "diag_sym_replay") if mode in modes]
    protocol = dict(format="credit-study-protocol-v1", seeds=list(seeds), config=dict(mode="bp64", chunk_ticks=64,
        duration_range=[4., 8.], gap_range=[4., 12.], regime_name="credit"), device="cuda",
        modes={mode: asdict(FLAGS[mode]) for mode in modes}, train_ticks=128, dt=.25, batch_size=64, n_jobs=4, width=128,
        updated_parameters=dict(readout_only=5676, frozen=0, other=34348), learning_rate=.001, gradient_clip=1.,
        replay=dict(warmup_ticks=64, loss_ticks=64), file_hashes=files,
        origins=[dict(seed=seed, checkpoint=f"F:/synthetic/A/seed_{seed:03d}/checkpoint.pt", sha256=digest((seed, "Afile"))) for seed in seeds],
        evaluation=dict(conditions=["iid", "long_wait", "longer", "sparse", "long_sparse"], steps=2048,
            intervention_modes=selected, intervention_conditions=["longer", "long_sparse"],
            extra_interventions=["finish_content_oracle", "timing_oracle"]),
        cycles=dict(modes=selected, seeds=list(seeds), start_stage_ticks=128, sequence=["A", "B"], segment_ticks=128,
            milestones=[64,128], regimes={"A": [[1.,4.],[.5,2.]], "B": [[4.,8.],[4.,12.]]},
            eval_conditions=["iid", "long_sparse"], eval_steps=512))
    reports, evaluations = [], []
    for seed in seeds:
        origin = {name: digest((seed, name)) for name in ("semantic_hash", "model_hash", "optimizer_hash", "hidden_hash", "world_hash", "replay_hash", "rng_hash")}
        origin.update(checkpoint=f"F:/synthetic/A/seed_{seed:03d}/checkpoint.pt", checkpoint_sha256=digest((seed,"Afile")),
                      tick=131072, update=2048, inherited_source_hashes=hashes("continuity", ("run.py", "world.py", "model.py", "test_run.py")))
        for mode in modes:
            flags = protocol["modes"][mode]
            trace, norms = [], {name:0. for name in ("recurrent_delta_sum", "encoder_delta_sum", "head_delta_sum", "all_delta_sum", "gradient_norm_sum", "clip_scale_sum")}
            action, data = digest((seed, "Aaction")), digest((seed, "Adata"))
            for index in range(2):
                update = None
                if flags["method"] != "frozen":
                    update = dict(recurrent_delta=0. if flags["method"] == "readout" else .01,
                                  encoder_delta=0. if flags["method"] == "readout" else .02,
                                  head_delta=.03, all_delta=.04, gradient_norm=.3, clip_scale=1., raw_parameter_gradient_norms={})
                    for name in update:
                        if name != "raw_parameter_gradient_norms":
                            norms[name+"_sum"] += update[name]
                action_chunk, data_chunk = digest((seed, mode,index,"ac")), digest((seed,mode,index,"dc"))
                action = hashlib.sha256(bytes.fromhex(action)+bytes.fromhex(action_chunk)).hexdigest()
                data = hashlib.sha256(bytes.fromhex(data)+bytes.fromhex(data_chunk)).hexdigest()
                trace.append(dict(stage="credit", stage_ticks=(index+1)*64, tick=131072+(index+1)*64,
                    world_tick=131071+(index+1)*64, now=(131071+(index+1)*64)*.25, loss=.2,
                    optimizer_steps_this_chunk=int(flags["method"] != "frozen"),
                    work=expected_work(mode, protocol, (index+1)*64), parameter_update=update,
                    old=dict(block_index=index, start_tick=index*128, weighted_loss=.3) if flags["replay"] else None,
                    live_hidden_hash=digest((seed,mode,index,"h")), action_chunk_hash=action_chunk,data_chunk_hash=data_chunk,
                    action_hash=action,data_hash=data,event_hash=digest((seed,"train_events"))))
            report = dict(format="credit-trainer-v1", config=dict(seed=seed,batch_size=64,n_jobs=4,device="cuda",train_dts=[.25],
                lr=.001,clip=1.,tbptt=64,replay_block_ticks=128,replay_burnin=64,replay_capacity=32),
                credit_config={**protocol["config"],"mode":mode}, mode=flags,
                variant=dict(name="online128",width=128,kind="learned",policy="online",reset_every=0),
                model=dict(kind="learned",state_scalars=128,input_size=96,n_jobs=4,n_actions=11,total_parameters=34348),
                stage_ticks=128,stage_updates=2,stage_gradient_updates=0 if mode=="frozen" else 2,
                ticks=131200,updates=2050,world_resets=0,hidden_resets=0,effective_updated_parameters=protocol["updated_parameters"].get(mode,34348),
                origin=deepcopy(origin),source_hashes=origin["inherited_source_hashes"],
                credit_source_hashes=hashes("credit",("run.py","local.py","test_run.py","test_local.py")),
                dependency_hashes={"integration/run.py":files["F:/synthetic/experiments/integration/run.py"]},
                hidden_hash=digest((seed,mode,"hidden_report")),world_hash=digest((seed,mode,"world")),
                event_hash=digest((seed,"train_events")),final_parameter_hash=origin["model_hash"] if mode=="frozen" else digest((seed,mode,"W")),
                action_hash=action,data_hash=data,work=expected_work(mode,protocol,128),update_norms=norms,trace=trace,
                replay=dict(stored_blocks=32,capacity_blocks=32,block_ticks=128,stored_tick_ranges=[[i*128,(i+1)*128] for i in range(32)]),
                regime_changes=[dict(tick=131072,stage_tick=0,regime=dict(duration_range=[4.,8.],gap_range=[4.,12.]))],
                live_credit=None,old_credit=None)
            if flags["method"] == "diag":
                for namespace, observed, loss, resets in (("live_credit",128,128,2 if flags["reset"]=="block" else 128 if flags["reset"]=="tick" else 0),
                    ("old_credit",256,128,2)):
                    if namespace=="old_credit" and not flags["replay"]:
                        continue
                    report[namespace]=dict(observed_ticks=observed,loss_ticks=loss,pending_loss_ticks=0,eligibility_resets=resets,
                        reconstructed_flow_ticks=observed,reconstructed_flow_matvecs=2*observed,eligibility_bytes=7340032,
                        E_rms=.1,F_rms=.2,E_absmax=.5,F_absmax=.8,
                        diagonal_homogeneous_gain=dict(last_min=.7,last_max=.9,geometric_mean_abs_gain_over_64_observations=2e-7))
            report["_provenance"]=dict(path=f"F:/synthetic/main/seed_{seed:03d}/{mode}/report.json",sha256=digest((seed,mode,"report")))
            reports.append(report)
            pairs=[(name,"none") for name in protocol["evaluation"]["conditions"]]
            if mode in selected:
                pairs += [(name,intervention) for name in ("longer","long_sparse") for intervention in ("finish_content_oracle","timing_oracle")]
            records=[record(report,condition,intervention,modes=modes) for condition,intervention in pairs]
            evaluations.append(dict(format="credit-evaluation-v1",seed=seed,mode=mode,stage_ticks=128,training_ticks=131200,
                credit_config=report["credit_config"],model=report["model"],records=records,
                checkpoint=f"F:/synthetic/main/seed_{seed:03d}/{mode}/checkpoint.pt",checkpoint_sha256=digest((seed,mode,"cp")),
                checkpoint_state_hash=digest((seed,mode,"state")),source_sha256=files["F:/synthetic/experiments/credit/evaluate.py"],
                observer_sha256=files["F:/synthetic/experiments/integration/diagnose.py"],
                state_snapshot=dict(activity=dict(rms=.2),recurrent=dict(frobenius=14.)),
                _provenance=dict(path=f"F:/synthetic/eval/{seed}_{mode}.json",sha256=digest((seed,mode,"eval")))))
    return protocol,reports,evaluations


def record(report, condition, intervention="none", *, modes=MODES, steps=2048):
    seed,mode=report["config"]["seed"],report["credit_config"]["mode"]
    proxy={**report,"integration_config":{"mode":"bp64"}}
    row=old_record(proxy,condition,intervention)
    clean=32+seed+list(modes).index(mode)+(1 if intervention=="finish_content_oracle" else 2 if intervention=="timing_oracle" else 0)
    completed=clean+5
    row["primary"].update(completed=completed,clean_completed=clean,completion_rate=completed/64,clean_completion_rate=clean/64,
        per_actor=dict(matured=[1]*64,completed=[1]*completed+[0]*(64-completed),clean_completed=[1]*clean+[0]*(64-clean)))
    row["request_diagnosis"]["primary"].update(completed=completed,clean_completed=clean,uncompleted_despite_legal_opportunity=64-completed-3)
    row["post_fork_counters"]["completed_all"]=completed
    row["post_fork_finish_precision"]=completed/100
    row["start_hidden_hash"]=digest((seed,mode,"forkh"))
    row["source_continuation_hash"]=digest((seed,mode,"source"))
    row["action_hash"]=digest((seed,mode,condition,intervention,"actions"))
    row["model_action_hash"]=row["action_hash"] if intervention=="none" else digest((seed,mode,"model"))
    if intervention!="none":
        row["paired_baseline_action_hash"]=digest((seed,mode,condition,"none","actions"))
    if steps!=2048:
        old_total=row["opportunities"]["opportunities"]
        total=steps*64*4
        difference=old_total-total
        row["steps"]=steps;row["end_tick"]=row["fork_tick"]+steps;row["end_time"]=row["fork_time"]+steps*.25
        row["opportunities"]["opportunities"]=total
        row["opportunities"]["inactive_opportunities"]-=difference
        row["inactive_history"][0]["opportunities"]-=difference
        idle=row["opportunities"]["inactive_opportunities"];actions=row["opportunities"]["inactive_actions"]
        row["inactive_actions_per_idle_opportunity"]=actions/idle
        row["inactive_actions_per_1000_idle_opportunities"]=1000*actions/idle
        row["inactive_actions_per_1000_total_opportunities"]=1000*actions/total
    return row


def cycle_fixture(protocol, reports, evaluations, folder):
    """Small JSON-only familiar A/B revisitation cohort with a known trend."""
    completions=[]
    protocol["cycles"]["sequence"]=["A","B"]*2
    spec=protocol["cycles"]
    for original in reports:
        seed,mode=original["config"]["seed"],original["credit_config"]["mode"]
        if mode not in spec["modes"]:
            continue
        original_eval=next(x for x in evaluations if x["seed"]==seed and x["mode"]==mode)
        completed=[]
        for segment,name in enumerate(spec["sequence"]):
            for offset in spec["milestones"]:
                target=spec["start_stage_ticks"]+segment*spec["segment_ticks"]+offset
                current=deepcopy(original)
                current.update(stage_ticks=target,stage_updates=target//64,stage_gradient_updates=target//64,
                               ticks=original["origin"]["tick"]+target,updates=2048+target//64)
                template=deepcopy(original["trace"][-1])
                action,data=current["action_hash"],current["data_hash"]
                for index in range(2,target//64):
                    row=deepcopy(template)
                    at=(index+1)*64
                    ach,dch=digest((seed,mode,index,"ac")),digest((seed,mode,index,"dc"))
                    action=hashlib.sha256(bytes.fromhex(action)+bytes.fromhex(ach)).hexdigest()
                    data=hashlib.sha256(bytes.fromhex(data)+bytes.fromhex(dch)).hexdigest()
                    row.update(stage_ticks=at,tick=131072+at,world_tick=131071+at,now=(131071+at)*.25,
                        action_chunk_hash=ach,data_chunk_hash=dch,action_hash=action,data_hash=data,
                        work=expected_work(mode,protocol,at))
                    if row["old"] is not None:
                        row["old"].update(block_index=index%32,start_tick=(index%32)*128)
                    current["trace"].append(row)
                current.update(action_hash=action,data_hash=data,work=expected_work(mode,protocol,target))
                current["update_norms"]={key:value*(target//64)/2 for key,value in original["update_norms"].items()}
                if current["live_credit"]:
                    current["live_credit"].update(observed_ticks=target,loss_ticks=target,reconstructed_flow_ticks=target,
                        reconstructed_flow_matvecs=2*target)
                if current["old_credit"]:
                    current["old_credit"].update(observed_ticks=(target//64)*128,loss_ticks=target,
                        reconstructed_flow_ticks=(target//64)*128,reconstructed_flow_matvecs=(target//64)*256,
                        eligibility_resets=target//64)
                current["regime_changes"]=deepcopy(original["regime_changes"])
                for index in range(segment+1):
                    kind=spec["sequence"][index]
                    current["regime_changes"].append(dict(stage_tick=128+index*128,
                        regime=dict(duration_range=spec["regimes"][kind][0],gap_range=spec["regimes"][kind][1])))
                destination=Path(folder)/f"{seed}_{mode}"/f"{segment}_{offset}"
                destination.mkdir(parents=True)
                milestone=dict(segment=segment,regime=name,offset=offset,stage_ticks=target,
                    checkpoint=str(destination/"checkpoint.pt"),checkpoint_sha256=digest((seed,mode,target,"file")),
                    state_hash=digest((seed,mode,target,"state")),evaluation=str(destination/"evaluation.json"))
                value=dict(format="credit-evaluation-v1",seed=seed,mode=mode,stage_ticks=target,training_ticks=current["ticks"],
                    checkpoint_state_hash=milestone["state_hash"],credit_config=current["credit_config"],
                    source_sha256=original_eval["source_sha256"],observer_sha256=original_eval["observer_sha256"],
                    cycle=milestone,origin=original_eval["checkpoint"],origin_sha256=original_eval["checkpoint_sha256"],
                    protocol_sha256=digest("synthetic protocol"),records=[])
                for condition in spec["eval_conditions"]:
                    row=record(current,condition,modes=tuple(protocol["modes"]),steps=512)
                    extra=segment+offset//64
                    row["primary"]["clean_completed"]+=extra
                    row["primary"]["completed"]+=extra
                    n,clean,done=64,row["primary"]["clean_completed"],row["primary"]["completed"]
                    row["primary"].update(clean_completion_rate=clean/n,completion_rate=done/n,
                        per_actor=dict(matured=[1]*64,completed=[1]*done+[0]*(64-done),clean_completed=[1]*clean+[0]*(64-clean)))
                    row["request_diagnosis"]["primary"].update(completed=done,clean_completed=clean,
                        uncompleted_despite_legal_opportunity=n-done-3)
                    row["post_fork_counters"]["completed_all"]=done
                    row["post_fork_finish_precision"]=done/100
                    value["records"].append(row)
                Path(milestone["evaluation"]).write_text(json.dumps(value),encoding="utf-8")
                completed.append(milestone)
        completions.append(dict(origin=original_eval["checkpoint"],origin_sha256=original_eval["checkpoint_sha256"],
            protocol_sha256=digest("synthetic protocol"),completed=completed,complete=True,
            final_stage_ticks=current["stage_ticks"],report=current,
            source_sha256=next(digest_ for path,digest_ in protocol["file_hashes"].items() if path.endswith("credit/cycle.py"))))
    return completions


def cpu_cycle_revision(protocol, completions, folder):
    folder=Path(folder)
    adapter=digest("CPU measurement adapter")
    source=lambda suffix:next(value for path,value in protocol["file_hashes"].items() if path.endswith(suffix))
    body=dict(format="credit-cycle-measurement-revision-v2",evaluation_device="cpu",training_device="cuda",
        base_protocol_sha256=PROTOCOL_SHA256,eval_conditions=protocol["cycles"]["eval_conditions"],
        eval_steps=protocol["cycles"]["eval_steps"],dt=protocol["dt"],adapter_source_sha256=adapter,
        frozen_evaluator_sha256=source("credit/evaluate.py"),frozen_cycle_sha256=source("credit/cycle.py"),
        reuse_root=str((folder/"immutable-v1").resolve()))
    path=folder/"measurement-revision-v2.json"
    path.write_text(json.dumps(body),encoding="utf-8")
    revision=load_cycle_measurement_revision(path)
    common=dict(measurement_backend="cpu",measurement_revision=str(path.resolve()),
        measurement_revision_sha256=revision["_provenance"]["sha256"],measurement_adapter_sha256=adapter)
    for index,completion in enumerate(completions):
        branch=folder/f"v2branch_{index}"
        branch.mkdir()
        for milestone in completion["completed"]:
            value=json.loads(Path(milestone["evaluation"]).read_text())
            value.update(**common,training_device="cuda",training_state_hash_before=milestone["state_hash"],
                         training_state_hash_after=milestone["state_hash"])
            Path(milestone["evaluation"]).write_text(json.dumps(value),encoding="utf-8")
        complete_path=branch/"complete.json"
        complete_path.write_text(json.dumps(completion),encoding="utf-8")
        completion["_provenance"]=dict(path=str(complete_path.resolve()),sha256=hashlib.sha256(complete_path.read_bytes()).hexdigest())
        final=completion["completed"][-1]
        sidecar=dict(format="credit-cycle-cpu-measurement-v2",**common,
            source_complete_sha256=completion["_provenance"]["sha256"],final_training_state_hash=final["state_hash"],
            final_checkpoint=final["checkpoint"],final_checkpoint_sha256=final["checkpoint_sha256"],
            final_stage_ticks=completion["final_stage_ticks"],completed_milestones=len(completion["completed"]),
            measurement_forks=len(completion["completed"])*len(protocol["cycles"]["eval_conditions"]),
            retained_v1_milestones=0,bootstrap=str(branch/"bootstrap-complete.json"),complete=True)
        sidecar_path=branch/"measurement-complete.json"
        sidecar_path.write_text(json.dumps(sidecar),encoding="utf-8")
        completion["_measurement"]={**sidecar,"_provenance":dict(path=str(sidecar_path.resolve()),sha256=hashlib.sha256(sidecar_path.read_bytes()).hexdigest())}
    return revision


class CreditReportTests(unittest.TestCase):
    def test_full_synthetic_roster_mean_sd_and_preselected_paired_contrasts(self):
        protocol,reports,evaluations=fixture()
        result=aggregate(reports,evaluations,protocol=protocol)
        self.assertEqual(result["audit"]["status"],"passed",result["audit"]["issues"][:4])
        self.assertEqual(result["counts"],dict(training_runs=33,baseline_forks=165,additional_diagnostic_forks=60))
        row=next(x for x in result["main_summary"] if x["mode"]=="bp64" and x["condition"]=="iid" and x["metric"]=="strict_completion")
        self.assertEqual(row["n"],3);self.assertAlmostEqual(row["sample_sd"],1/64)
        self.assertEqual({(x["after"],x["before"]) for x in result["paired_contrasts"]},set(CONTRASTS))
        delta=next(x for x in result["paired_contrasts"] if x["after"]=="diag_sym" and x["before"]=="readout_only" and x["condition"]=="iid" and x["metric"]=="strict_completion")
        self.assertEqual(delta["n"],3);self.assertAlmostEqual(delta["mean"],3/64);self.assertEqual(delta["sample_sd"],0.)

    def test_missing_seed_is_error_or_explicitly_incomplete_never_actor_replication(self):
        protocol,reports,evaluations=fixture()
        reports=[x for x in reports if x["config"]["seed"]!=2]
        evaluations=[x for x in evaluations if x["seed"]!=2]
        failed=aggregate(reports,evaluations,protocol=protocol)
        self.assertEqual(failed["audit"]["status"],"failed")
        partial=aggregate(reports,evaluations,protocol=protocol,allow_incomplete=True)
        self.assertEqual(partial["audit"]["error_count"],0)
        self.assertGreater(partial["audit"]["warning_count"],0)
        row=next(x for x in partial["main_summary"] if x["mode"]=="bp64")
        self.assertEqual((row["n"],row["n_total"],row["n_undefined"]),(2,3,1))

    def test_bad_counts_denominators_hashes_and_budgets_are_not_relaxed(self):
        mutations=(lambda p,r,e:e[0]["records"][0]["primary"].update(clean_completed=65),
            lambda p,r,e:e[0]["records"][0]["primary"].update(clean_completion_rate=.01),
            lambda p,r,e:e[0]["records"][0].update(start_world_state_hash="not-a-hash"),
            lambda p,r,e:r[0]["credit_source_hashes"].update({"run.py":"0"*64}),
            lambda p,r,e:r[3]["work"].update(old_warmup_ticks=0),
            lambda p,r,e:r[1]["trace"][0]["parameter_update"].update(recurrent_delta=.2),
            lambda p,r,e:r[2]["update_norms"].update(all_delta_sum=100),
            lambda p,r,e:e[0]["records"].pop())
        for index,mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                protocol,reports,evaluations=fixture(seeds=(0,))
                mutate(protocol,reports,evaluations)
                result=aggregate(reports,evaluations,protocol=protocol,allow_incomplete=True)
                self.assertGreater(result["audit"]["error_count"],0)

    def test_intervention_pairing_and_original_action_links_are_checked(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        evaluations[0]["records"][-1]["start_scheduler_hash"]=digest("changed schedule")
        result=aggregate(reports,evaluations,protocol=protocol)
        self.assertTrue(any(x["code"]=="PAIRED_DIAGNOSTIC" for x in result["audit"]["issues"]))

    def test_private_old_exposures_must_match_across_replay_rules(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp_replay","diag_sym_replay","diag_random_replay"))
        reports[-1]["trace"][0]["old"]["block_index"]=7
        result=aggregate(reports,evaluations,protocol=protocol)
        self.assertTrue(any(x["code"]=="PAIRED_OLD_EXPOSURES" for x in result["audit"]["issues"]))

    def test_cpu_semantic_normalization_changes_only_execution_device(self):
        state=dict(format="credit-trainer-v1",continuation=dict(config=dict(device="cuda",seed=0),hidden=torch.tensor([[.5]]),
            model={"w":torch.tensor([1.])},runtime="excluded",seconds=4),live_credit={"E":torch.zeros(1)},
            source_hashes={"run.py":"x"},sources={"run.py":"source"},dependencies={},dependency_hashes={})
        original=deepcopy(state)
        value=_semantic(state,"cpu")
        self.assertEqual(value["continuation"]["config"]["device"],"cpu")
        self.assertEqual(state["continuation"]["config"]["device"],"cuda")
        self.assertNotIn("runtime",value["continuation"])
        self.assertTrue(torch.equal(original["live_credit"]["E"],value["live_credit"]["E"]))
        altered=deepcopy(state);altered["continuation"]["hidden"].add_(1)
        from experiments.continuity.run import state_digest
        self.assertNotEqual(state_digest(_semantic(altered,"cpu")),state_digest(value))

    def test_global_protocol_hash_rejects_modified_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"protocol.json"
            path.write_text('{"format":"synthetic"}',encoding="utf-8")
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(load_protocol(path,expected_sha256=expected)["format"],"synthetic")
            path.write_text('{"format":"modified"}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"global frozen protocol hash"):
                load_protocol(path,expected_sha256=expected)

    def test_cycle_aggregates_paired_familiar_regime_revisitation(self):
        protocol,reports,evaluations=fixture(modes=("bp64","diag_sym"))
        with tempfile.TemporaryDirectory() as folder:
            cycles=cycle_fixture(protocol,reports,evaluations,folder)
            result=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles)
        self.assertEqual(result["audit"]["status"],"passed",result["audit"]["issues"][:3])
        self.assertEqual(result["cycles"]["status"],"complete")
        self.assertEqual(result["cycles"]["evaluation_device"],"cuda")
        self.assertEqual(result["cycles"]["completed_runs"],6)
        changes=result["cycles"]["paired_changes"]
        within=next(x for x in changes if x["kind"]=="within_segment_4096_minus_1024" and x["metric"]=="strict_completion")
        self.assertEqual(within["n"],3);self.assertAlmostEqual(within["mean"],1/64)
        revisit=next(x for x in changes if x["kind"]=="same_regime_revisit" and x["metric"]=="strict_completion")
        self.assertEqual(revisit["n"],3);self.assertAlmostEqual(revisit["mean"],2/64)
        self.assertIn("not new-task",result["notes"][-1])

    def test_cycle_bad_fullstate_link_and_missing_run_are_detected(self):
        protocol,reports,evaluations=fixture(modes=("bp64",))
        with tempfile.TemporaryDirectory() as folder:
            cycles=cycle_fixture(protocol,reports,evaluations,folder)
            path=Path(cycles[0]["completed"][0]["evaluation"])
            data=json.loads(path.read_text());data["checkpoint_state_hash"]=digest("wrong state")
            path.write_text(json.dumps(data),encoding="utf-8")
            result=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles,allow_incomplete=True)
            self.assertTrue(any(x["code"]=="CYCLE_EVAL_FULL_STATE" for x in result["audit"]["issues"]))
            path.write_text(json.dumps({**data,"checkpoint_state_hash":cycles[0]["completed"][0]["state_hash"]}),encoding="utf-8")
            partial=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles[:-1],allow_incomplete=True)
            self.assertEqual(partial["audit"]["error_count"],0)
            self.assertGreater(partial["audit"]["warning_count"],0)
            self.assertEqual(partial["cycles"]["status"],"incomplete")

    def test_cpu_cycle_revision_preserves_raw_cuda_semantic_links_and_loads_sidecar(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        with tempfile.TemporaryDirectory() as folder:
            cycles=cycle_fixture(protocol,reports,evaluations,folder)
            revision=cpu_cycle_revision(protocol,cycles,folder)
            result=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles,
                             cycle_measurement_revision=revision)
            self.assertEqual(result["audit"]["status"],"passed",result["audit"]["issues"][:3])
            self.assertEqual(result["cycles"]["evaluation_device"],"cpu")
            self.assertIn("not matched cases",result["cycles"]["interpretation"])
            self.assertTrue(all(row["checkpoint_state_hash"]==row["training_state_hash_before"]==row["training_state_hash_after"]
                for row in (json.loads(Path(point["evaluation"]).read_text()) for point in cycles[0]["completed"])))
            loaded=load_cycles([Path(folder)/"v2branch_0"])
            self.assertEqual(loaded[0]["_measurement"]["source_complete_sha256"],loaded[0]["_provenance"]["sha256"])
            old_default=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles)
            self.assertGreater(old_default["audit"]["error_count"],0)

    def test_cpu_cycle_wrong_backend_revision_adapter_or_callback_state_is_rejected(self):
        for field,bad_value in (("measurement_backend","cuda"),("measurement_revision_sha256","0"*64),
                                ("measurement_adapter_sha256","1"*64),("training_state_hash_after","2"*64)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
                cycles=cycle_fixture(protocol,reports,evaluations,folder)
                revision=cpu_cycle_revision(protocol,cycles,folder)
                path=Path(cycles[0]["completed"][0]["evaluation"])
                value=json.loads(path.read_text());value[field]=bad_value
                path.write_text(json.dumps(value),encoding="utf-8")
                result=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles,
                                 cycle_measurement_revision=revision,allow_incomplete=True)
                self.assertGreater(result["audit"]["error_count"],0)

    def test_cpu_cycle_revision_must_bind_original_protocol_and_frozen_completion(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        with tempfile.TemporaryDirectory() as folder:
            cycles=cycle_fixture(protocol,reports,evaluations,folder)
            revision=cpu_cycle_revision(protocol,cycles,folder)
            revision["base_protocol_sha256"]="0"*64
            cycles[0]["_measurement"]["source_complete_sha256"]="1"*64
            result=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles,
                             cycle_measurement_revision=revision)
            codes={row["code"] for row in result["audit"]["issues"]}
            self.assertIn("CYCLE_MEASUREMENT_REVISION",codes)
            self.assertIn("CPU_SIDECAR_BINDS_FROZEN_COMPLETION",codes)

    def test_missing_cycle_file_is_integrity_error_even_when_incomplete_is_allowed(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        with tempfile.TemporaryDirectory() as folder:
            cycles=cycle_fixture(protocol,reports,evaluations,folder)
            Path(cycles[0]["completed"][0]["evaluation"]).unlink()
            result=aggregate(reports,evaluations,protocol=protocol,cycle_completions=cycles,allow_incomplete=True)
            self.assertGreater(result["audit"]["error_count"],0)
            self.assertTrue(any(x["code"]=="CYCLE_SCHEMA_OR_FILE" for x in result["audit"]["issues"]))

    def test_artifact_output_cannot_overwrite_a_recorded_raw_input(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        result=aggregate(reports,evaluations,protocol=protocol)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"summary.json"
            path.write_text("immutable raw source",encoding="utf-8")
            result["inputs"][str(path.resolve())]=digest("raw")
            with self.assertRaisesRegex(ValueError,"immutable input"):
                write_artifacts(result,folder)
            self.assertEqual(path.read_text(),"immutable raw source")

    def test_transient_windows_artifact_replace_error_is_retried(self):
        from experiments.credit.report import atomic_json as original_atomic_json
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        result=aggregate(reports,evaluations,protocol=protocol)
        attempts=[]
        def transient(path,value):
            attempts.append(str(path))
            if len(attempts)<=2:
                raise PermissionError(5,"simulated transient Windows sharing violation")
            return original_atomic_json(path,value)
        with tempfile.TemporaryDirectory() as folder:
            with patch("experiments.credit.report.atomic_json",side_effect=transient), \
                 patch("experiments.credit.report.time.sleep") as sleeping:
                write_artifacts(result,folder)
            self.assertEqual(sleeping.call_count,2)
            self.assertEqual(json.loads((Path(folder)/"audit.json").read_text())["status"],"passed")
            self.assertEqual(json.loads((Path(folder)/"summary.json").read_text())["counts"],result["counts"])

    def test_json_normalization_and_artifact_writes_do_not_mutate_inputs(self):
        protocol,reports,evaluations=fixture(seeds=(0,),modes=("bp64",))
        evaluations[0]["records"][0]["regime"]["duration_range"]=(1.,4.)
        before=deepcopy((protocol,reports,evaluations))
        result=aggregate(reports,evaluations,protocol=protocol)
        self.assertEqual(result["audit"]["status"],"passed",result["audit"]["issues"][:3])
        self.assertEqual((protocol,reports,evaluations),before)
        with tempfile.TemporaryDirectory() as folder:
            write_artifacts(result,folder)
            self.assertTrue((Path(folder)/"summary.json").is_file())
            self.assertTrue((Path(folder)/"paired_contrasts.csv").is_file())
            self.assertEqual(json.loads((Path(folder)/"audit.json").read_text())["status"],"passed")


if __name__=="__main__":
    unittest.main()
