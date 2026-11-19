"use strict";

globalThis.adaptLocalSnapshot = (() => {
  const defaultArms = ["control", "local", "modulated"];
  const selectionRule = "local_rule_revisit";

  function hasChampion(row) {
    return Number.isInteger(row?.generation) && row.generation > 0
      && Number.isInteger(row.candidates) && row.candidates > 0
      && typeof row.champion?.id === "string" && row.champion.id.length > 0;
  }

  function candidateFields(row) {
    if (!row) return {};
    const { champion, mean } = row;
    return {
      ...champion,
      best_candidate: champion.id,
      best_fitness: champion.strict,
      average_edges: champion.mean_edges,
      active_edges: champion.final_edges,
      mean_fitness: mean?.strict,
      mean_selection_score: mean?.selection_score,
      mean_symbol_late: mean?.symbol_late_mean,
      mean_pair_strict_rate: mean?.pair_strict_rate,
      selection_rule: selectionRule,
    };
  }

  function latestHoldout(rows, arm) {
    let latest = null;
    for (const row of rows) {
      const metrics = row?.arms?.[arm];
      if (!Number.isInteger(row?.generation) || row.generation <= 0
          || !metrics || !Object.keys(metrics).length || row.generation <= (latest?.generation ?? 0)) continue;
      const { generation, seed_count, actors, visits, final } = row;
      latest = { ...metrics, generation, seed_count, actors, visits, final };
    }
    return latest;
  }

  return function adaptLocalSnapshot(snapshot) {
    const rawProgress = snapshot?.progress ?? {};
    const protocol = snapshot?.local_rule;
    const arms = protocol?.arms ?? defaultArms;
    const currentArm = rawProgress.current_arm ?? null;
    const latest = rawProgress.latest;
    const current = currentArm && latest?.arm === currentArm && latest.generation === rawProgress.generation
      && latest.candidates === rawProgress.generation_completed && hasChampion(latest)
      && typeof latest.sealed === "boolean" ? latest : null;
    const histories = new Map(arms.map((arm) => [arm, new Map()]));

    function addSealed(row) {
      if (row?.sealed !== true || !hasChampion(row)) return;
      const history = histories.get(row.arm);
      if (history && !history.has(row.generation)) history.set(row.generation, row);
    }

    for (const row of snapshot?.history ?? []) addSealed(row);
    for (const arm of arms) {
      const row = rawProgress.arms_progress?.[arm]?.latest;
      if (row?.arm === arm) addSealed(row);
    }
    addSealed(current);

    const audits = [...(snapshot?.holdout_history ?? []), snapshot?.holdout];
    let lastSealed = null;
    let armSealed = null;
    const branches = arms.map((arm) => {
      const rows = [...histories.get(arm).values()].sort((a, b) => a.generation - b.generation);
      const sealed = rows.at(-1);
      // Each generation runs in protocol arm order, so ties go to the later arm.
      if (sealed && sealed.generation >= (lastSealed?.generation ?? 0)) lastSealed = sealed;
      if (arm === currentArm) armSealed = sealed;
      const history = rows.map((row) => ({
        ...candidateFields(row),
        generation: row.generation,
        candidates: row.candidates,
        sealed: row.sealed,
        arm,
      }));
      const counters = rawProgress.arms_progress?.[arm];
      return {
        id: arm,
        arm,
        history,
        latest: history.at(-1) ?? null,
        holdout: latestHoldout(audits, arm),
        sealed_generations: counters?.sealed_generations ?? sealed?.generation,
        candidates: counters?.candidates,
      };
    });

    const display = current ?? armSealed ?? lastSealed;
    const displaySummary = display ? {
      arm: display.arm,
      generation: display.generation,
      candidates: display.candidates,
      sealed: display.sealed,
      source: current ? "current" : armSealed ? "arm_sealed" : "latest_sealed",
    } : null;
    const checkpointSource = armSealed ?? lastSealed;
    const sealedCheckpoint = checkpointSource
      ? `arms/${checkpointSource.arm}/generations/${String(checkpointSource.generation).padStart(4, "0")}/elites.pt`
      : undefined;
    const device = rawProgress.device ?? snapshot?.process?.device;
    const executionPrecision = rawProgress.execution_precision
      ?? (device === "cuda" ? "FP16 循环计算 / FP32 累加" : undefined);

    return {
      ...snapshot,
      progress: {
        ...rawProgress,
        ...candidateFields(display),
        display_summary: displaySummary,
        checkpoint: rawProgress.checkpoint ?? sealedCheckpoint,
        execution_precision: executionPrecision,
      },
      private_symbol: {
        current: currentArm,
        branches,
        visits: 16,
        requests_per_visit: 4,
        actors: protocol?.actors ?? 8,
        context_sampling: "iid_per_request",
        selection_rule: selectionRule,
        local_rule: true,
      },
    };
  };
})();
