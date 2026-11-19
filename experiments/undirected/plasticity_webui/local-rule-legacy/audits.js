"use strict";

(() => {
  const armLabels = { control: "原规则", local: "逐步局部规则", modulated: "接收节点调制" };
  const armIds = Object.keys(armLabels);
  const views = [["holdout", "留出成绩"], ["receipt", "回执干预"]];
  const effects = [
    ["same_truthful_correct_gain", "原题"],
    ["other_context_truthful_correct_gain", "换情境"],
    ["other_raw_truthful_correct_gain", "换符号"],
  ];
  const points = new Intl.NumberFormat("zh-CN", {
    minimumFractionDigits: 2, maximumFractionDigits: 2, signDisplay: "exceptZero",
  });
  const selectedGeneration = { holdout: "latest", receipt: "latest" };
  let selectedDelay = "8";
  let view = "holdout";
  let lastSnapshot;
  let tableSignature;

  function updateGenerations(select, records) {
    const generations = [...records.keys()].sort((a, b) => b - a);
    const latest = generations[0];
    const choices = [["latest", latest ? `最新 · 第 ${latest} 代` : "等待评估"]];
    for (const generation of generations) choices.push([String(generation), `第 ${generation} 代`]);
    const selected = selectedGeneration[view];
    if (selected !== "latest" && !records.has(Number(selected))) {
      choices.push([selected, `第 ${selected} 代（暂无数据）`]);
    }
    syncSelect(select, choices, selected);
    select.disabled = generations.length === 0;
    return selected === "latest" ? latest : Number(selected);
  }

  function updateTable(entries, columns, metadata = null) {
    const label = view === "receipt" ? "真实回执相对颠倒回执的首答正确率差" : "各组留出评估成绩";
    byId("holdout-data").setAttribute("aria-label", label);
    byId("holdout-rows").closest("table").setAttribute("aria-label", label);
    const signature = JSON.stringify([view, entries, metadata]);
    if (tableSignature === signature) return false;
    renderHoldoutRows(entries, columns, armLabels);
    byId("holdout-columns").firstElementChild.textContent = "训练组";
    tableSignature = signature;
    return true;
  }

  function renderHoldout(record, phase) {
    const columns = [
      ["strict", percent, "严格完成率"],
      ["completion", percent, "总完成率"],
      ["first_finish_correct", percent, "首次作答正确率", "就绪窗口内第一次作答，未答计错"],
      ["pair_strict_rate", percent, "双回访成功率", "同一符号在 A、B 两个情境中的配对回访均严格完成"],
      ["selection_score", fitness, "选拔得分"],
    ];
    byId("holdout-data").hidden = !record;
    if (!record) {
      put("holdout-detail", phase === "holdout" ? "正在评估留出世界。" : "等待留出评估结果。");
      return;
    }
    const details = [`第 ${record.generation} 代`, "各岛冠军均值"];
    if (numeric(record.seed_count)) details.push(`${format(record.seed_count)} 个未见世界`);
    if (numeric(record.actors)) details.push(`每个世界 ${format(record.actors)} 个独立生命`);
    put("holdout-detail", details.join(" · "));
    updateTable(armIds.map((arm) => [arm, record.arms?.[arm] ?? {}]), columns);
  }

  function delayLabel(delay) {
    return delay === "0" ? "立即回访" : `隔 ${delay} 道干扰题回访`;
  }

  function renderReceipt(record, phase, delaySelect) {
    const delays = [...new Set(Object.values(record?.arms ?? {}).flatMap((arm) => Object.keys(arm.delays ?? {})))];
    delays.sort((a, b) => Number(a) - Number(b));
    const delay = delays.includes(selectedDelay) ? selectedDelay : delays[0];
    syncSelect(delaySelect, delays.map((value) => [value, delayLabel(value)]), delay ?? "");
    delaySelect.disabled = delays.length === 0;
    byId("holdout-data").hidden = !delay;
    if (!delay) {
      put("holdout-detail", phase === "receipt_audit" ? "正在评估回执干预。" : "等待回执干预结果。");
      return;
    }

    put("holdout-detail", `第 ${record.generation} 代 · ${delayLabel(delay)} · 真实回执减颠倒回执的首答正确率，单位为百分点。括号内为按世界配对的 95% 区间。`);
    const records = armIds.map((arm) => record.arms?.[arm]);
    const results = records.map((arm) => arm?.delays?.[delay]);
    const columns = [["events", number, "有效教学 / 机会"],
      ...effects.map(([key, label]) => [key, points, label])];
    const entries = armIds.map((arm, index) => [arm, Object.fromEntries(effects.map(([key]) => (
      [key, numeric(results[index]?.[key]?.mean) ? results[index][key].mean * 100 : undefined]
    )))]);
    if (!updateTable(entries, columns, [delay, records])) return;

    const rows = byId("holdout-rows").rows;
    for (const [index, row] of Array.from(rows).entries()) {
      const arm = records[index];
      if (numeric(arm?.events) && numeric(arm.opportunities)) {
        row.cells[1].textContent = `${format(arm.events)} / ${format(arm.opportunities)}`;
      }
      for (const [effectIndex, [key]] of effects.entries()) {
        const interval = results[index]?.[key]?.ci95;
        if (!Array.isArray(interval) || interval.length !== 2 || !interval.every(numeric)) continue;
        const hint = document.createElement("div");
        hint.className = "muted ci-range";
        hint.textContent = `(${interval.map((value) => format(value * 100, points)).join(", ")})`;
        row.cells[effectIndex + 2].append(hint);
      }
    }
  }

  globalThis.renderLocalAudits = function renderLocalAudits(snapshot) {
    const modeSelect = byId("private-holdout-branch");
    const generationSelect = byId("private-holdout-clock");
    const delaySelect = byId("receipt-delay");
    if (!lastSnapshot) {
      modeSelect.setAttribute("aria-label", "评估类型");
      generationSelect.setAttribute("aria-label", "评估代数");
      delaySelect.setAttribute("aria-label", "干扰题数");
      modeSelect.addEventListener("change", () => {
        view = modeSelect.value;
        renderLocalAudits(lastSnapshot);
      });
      generationSelect.addEventListener("change", () => {
        selectedGeneration[view] = generationSelect.value;
        renderLocalAudits(lastSnapshot);
      });
      delaySelect.addEventListener("change", () => {
        selectedDelay = delaySelect.value;
        renderLocalAudits(lastSnapshot);
      });
    }
    lastSnapshot = snapshot;

    byId("private-holdout-controls").hidden = false;
    modeSelect.hidden = false;
    generationSelect.hidden = false;
    delaySelect.hidden = view !== "receipt";
    syncSelect(modeSelect, views, view);
    put("holdout-title", view === "receipt" ? "回执干预" : "留出评估");

    const history = view === "receipt" ? snapshot.receipt_history : snapshot.holdout_history;
    const records = new Map();
    if (view === "holdout" && Number.isInteger(snapshot.holdout?.generation)) {
      records.set(snapshot.holdout.generation, snapshot.holdout);
    }
    for (const row of history ?? []) {
      if (Number.isInteger(row.generation)) records.set(row.generation, row);
    }
    const generation = updateGenerations(generationSelect, records);
    const record = records.get(generation);
    const phase = snapshot.progress?.phase;
    if (view === "receipt") renderReceipt(record, phase, delaySelect);
    else renderHoldout(record, phase);
  };
})();
