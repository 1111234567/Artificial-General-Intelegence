"use strict";

const byId = (id) => document.getElementById(id);
const missing = "—";
const numeric = (value) => typeof value === "number" && Number.isFinite(value);
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
const fitness = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 });
const percent = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 });
const localTime = new Intl.DateTimeFormat("zh-CN", {
  month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
});

const format = (value, formatter = number) => numeric(value) ? formatter.format(value) : missing;
const fraction = (completed, total) => numeric(completed) && numeric(total) && total > 0 ? completed / total : null;
const counter = (completed, total) => numeric(total) ? `${format(completed)} / ${format(total)}` : format(completed);

function put(id, value) {
  const element = byId(id);
  if (element.textContent !== value) element.textContent = value;
}

function putTime(id, timestamp) {
  const parsed = typeof timestamp === "string" ? Date.parse(timestamp) : NaN;
  const element = byId(id);
  if (!Number.isFinite(parsed)) {
    element.removeAttribute("datetime");
    element.removeAttribute("title");
    put(id, missing);
    return;
  }
  element.dateTime = new Date(parsed).toISOString();
  element.title = element.dateTime;
  put(id, localTime.format(parsed));
}

function candidateName(candidate) {
  const identity = candidate && typeof candidate === "object" ? candidate.id ?? candidate.candidate_id : candidate;
  return typeof identity === "string" || numeric(identity) ? String(identity) : "";
}

const stageLabels = { timing: "完成时机", version: "版本辨识", symbol: "符号映射", conflict: "干扰与冲突" };
const armNames = ["full", "weights"];
const modeLabels = { full: "完整可塑性", weights: "仅权重更新", frozen: "冻结", shuffled: "打乱对照" };
const uncappedModeLabels = { full: "完整可塑性", no_reward_write: "关闭回执写权", frozen: "冻结可塑性",
  fixed_guess: "公开窗口与版本 · 固定猜符号" };
const pairedModeLabels = {
  full: "完整组冠军 · 完整可塑性",
  weights: "权重组冠军 · 仅权重更新",
  full_weights: "完整组冠军 · 仅权重更新",
  full_frozen: "完整组冠军 · 冻结",
  weights_frozen: "权重组冠军 · 冻结",
  fixed_guess: "公开窗口与版本 · 固定猜符号",
};
const privateArmLabels = { control: "原规则", local: "逐步局部规则", modulated: "接收节点调制" };
const privateArms = Object.keys(privateArmLabels);
const privateHoldoutModeLabels = { full: "完整可塑性", no_reward_write: "关闭回执写权", frozen: "冻结可塑性" };
const privateClockLabels = { regular_native: "常规时序", regular_padded: "常规时序（填充）", long: "长时序" };
const pairName = (pair) => numeric(pair) ? `种子对 ${String(pair).padStart(2, "0")}` : missing;
const branchName = (branch) => branch ? (privateArmLabels[branch.arm] ?? branch.arm) : missing;
const privateLateRounds = (visits = 16) => `第 ${visits * 2 + 1}–${visits * 4} 题`;
const tableColumns = [
  ["completion", percent, "完成率"], ["strict_a", percent, "严格 A"], ["strict_b", percent, "严格 B"],
  ["strict_return", percent, "严格返回"], ["weight_updates", number, "权重更新"], ["births", number, "新增边"],
  ["deaths", number, "退出边"], ["active_edges", number, "活跃边"],
];
const pairedMetrics = [
  ["best_fitness", fitness, "最优适应度"], ["completion", percent, "完成率"], ["ready_coverage", percent, "就绪覆盖率"],
  ["timing_clean", percent, "时机合规率", "在有效窗口内尝试完成，且从未违规尝试的请求占比"],
  ["version_coverage", percent, "版本正确覆盖率", "首次窗口内尝试版本正确的请求占全部请求的比例"],
  ["version_accuracy", percent, "版本准确率", "在有效窗口内尝试完成的请求中首次尝试版本正确的比例"],
  ["symbol_accuracy", percent, "符号准确率", "在有效窗口内尝试完成的请求中首次尝试符号正确的比例"],
  ["symbol_first", percent, "首轮映射准确率", "首轮各符号请求中，按有效窗口内尝试完成的请求计算"],
  ["symbol_late", percent, "末轮映射准确率", "末轮各符号请求中，按有效窗口内尝试完成的请求计算"],
  ["strict_return", percent, "严格返回"], ["idle", percent, "空闲动作率"], ["legal_attempt", percent, "窗口内尝试率"],
  ["births", number, "新增边"], ["deaths", number, "退出边"], ["weight_updates", number, "权重更新"],
];
const pairedHoldoutMetrics = [
  ...pairedMetrics.slice(1),
  ["active_edges", number, "活跃边"],
];
const uncappedMetrics = [
  ["ready_coverage", percent, "就绪覆盖率", "至少在有效窗口内尝试完成一次的请求占比"],
  ["version_accuracy", percent, "版本准确率", "在有效窗口内尝试完成的请求中，首次尝试版本正确的比例"],
  ["symbol_accuracy", percent, "符号准确率", "在有效窗口内尝试完成的请求中，首次尝试符号正确的比例"],
  ["strict", percent, "严格完成率", "成功完成且之前没有无效动作的请求占比"],
];
const uncappedHoldoutMetrics = [
  tableColumns[0], ...uncappedMetrics,
  ["strict_return", percent, "严格返回", "两种情境的首次访问结束后，后续访问中的严格完成率"],
  ["average_edges", number, "时均活跃边", "跨时间步和 actor 的平均活跃边数"],
  ["active_edges", number, "最终活跃边", "每个 actor 的最终活跃边数均值"],
  ["sparsity_bonus", percent, "稀疏分数"],
  ...tableColumns.slice(4, 7),
];
const privateHoldoutMetrics = (rounds) => [
  ["strict", percent, "严格完成率", "在就绪窗口内答对，且此前无被拒绝的主端口作答"],
  ["recovered_completion_rate", percent, "补救完成率", "有过被拒绝的主端口作答，随后在就绪窗口内答对，占全部请求"],
  ["completion", percent, "总完成率", "严格完成与补救完成之和"],
  ["premature_request_rate", percent, "抢答请求率", "窗口打开前提交过答案的请求占比"],
  ["ready_coverage", percent, "窗口内作答率", "至少在就绪窗口内提交过一次答案的请求占比"],
  ["symbol_late_mean", percent, `${rounds}首答 · A/B 均值`, "就绪窗口内第一次作答。未答计错，A、B 等权。"],
  ["symbol_late_a", percent, `${rounds}首答 · 情境 A`],
  ["symbol_late_b", percent, `${rounds}首答 · 情境 B`],
  ["mean_edges", number, "全程平均连接数"], ["final_edges", number, "结束时连接数"],
  ...tableColumns.slice(4, 7),
];

function privateMetric(metrics, key) {
  if (key !== "recovered_completion_rate") return metrics?.[key];
  const { strict, completion } = metrics ?? {};
  return numeric(strict) && numeric(completion) && completion >= strict ? completion - strict : undefined;
}

function renderPrivateReadout(progress, candidate, data) {
  byId("private-readout").hidden = !data || byId("candidate-details").getAttribute("aria-expanded") !== "true";
  if (!data) return;
  const { visits = 8, requests_per_visit: requests = 4, actors } = data;
  const runSize = `${numeric(actors) ? `每个候选评估 ${actors} 次独立生命。` : ""}每个生命共 ${visits * requests} 个请求。`;
  const sampling = data.context_sampling === "iid_per_request" ? "每个请求独立抽取情境，A、B 各有 50% 概率。" : "A、B 按轮交替。";
  put("private-metric-title", data.branches?.length === 1 ? "本代领先候选" : "当前分支领先候选");
  put("private-answers-title", `${privateLateRounds(visits)}首答正确率`);
  put("private-answers-note", `${runSize}${sampling}同一符号在两套规则下答案不同。首答取就绪窗口内第一次作答，未答计错。`);
  put("private-candidate", candidate || (progress.phase === "search" ? "等待本代候选结果" : ""));
  for (const element of byId("private-readout").querySelectorAll("[data-private-value]")) {
    const value = format(privateMetric(progress, element.dataset.privateValue), percent);
    if (element.textContent !== value) element.textContent = value;
  }
}

function renderCurriculum(progress, paired) {
  byId("curriculum").hidden = !paired;
  byId("paired").hidden = !paired;
  byId("legacy-metrics").hidden = paired;
  byId("legacy-graph").hidden = paired;
  byId("analysis").classList.toggle("paired-analysis", paired);
  if (!paired) return;

  const stage = progress.stage;
  const stageName = progress.stage_title || stageLabels[stage];
  const stageDone = progress.phase === "complete";
  const generation = progress.generation;
  const bounds = progress.stage_bounds;
  const stageStatus = stageDone ? "最后一阶段" : progress.phase === "holdout" ? "正在留出评估" : "当前阶段";
  put("stage-detail", stageName ? `${stageStatus} · ${stageName}` : "");
  const stageOrder = Object.keys(stageLabels);
  const stageIndex = stageOrder.indexOf(stage);
  for (const item of byId("stage-list").children) {
    const key = item.dataset.stage;
    const range = bounds?.[key];
    const [start, end] = Array.isArray(range) ? range : [];
    const state = stageOrder.indexOf(key) < stageIndex || (stageDone && key === stage) ? "done" : key === stage ? "current" : "pending";
    item.dataset.state = state;
    if (state === "current") item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
    item.querySelector(".stage-range").textContent = numeric(start) && numeric(end) ? `第 ${format(start)}–${format(end)} 代` : "";
  }

  const arms = progress.arms ?? {};
  for (const arm of armNames) {
    const done = progress.arm_counts?.[arm];
    byId(`${arm}-count`).hidden = !numeric(done);
    put(`${arm}-count`, numeric(done) ? `已评估 ${counter(done, progress.population)}` : "");
  }
  const identities = armNames
    .map((name) => [modeLabels[name], candidateName(arms[name]?.best_candidate)])
    .filter(([, candidate]) => candidate);
  const detail = [stageName, numeric(generation) ? `第 ${format(generation)} 代` : "", "各组适应度领先候选"]
    .filter(Boolean)
    .join(" · ");
  put("paired-detail", identities.length ? `${detail} · ${identities.map(([label, candidate]) => `${label} ${candidate}`).join(" · ")}` : detail);
  const rows = pairedMetrics.map(([key, formatter, label, description]) => {
    const row = document.createElement("tr");
    const header = document.createElement("th");
    header.scope = "row";
    header.textContent = label;
    if (description) header.title = description;
    row.append(header);
    for (const arm of armNames) {
      const cell = document.createElement("td");
      cell.textContent = format(arms[arm]?.[key], formatter);
      row.append(cell);
    }
    return row;
  });
  byId("paired-rows").replaceChildren(...rows);
}

function renderUncapped(progress, uncapped, privateMode) {
  const advanced = uncapped || privateMode;
  byId("candidate-metrics").classList.toggle("uncapped-metrics", uncapped);
  document.querySelectorAll("[data-legacy-metric]").forEach((element) => { element.hidden = advanced; });
  document.querySelectorAll("[data-uncapped-metric]").forEach((element) => { element.hidden = !advanced; });
  byId("version-accuracy").parentElement.hidden = !uncapped;
  byId("strict-return-stat").hidden = privateMode;
  byId("average-edges-stat").hidden = !advanced;
  byId("initial-edges-stat").hidden = !privateMode || !numeric(progress.initial_edges);
  const hasSparsity = uncapped && numeric(progress.sparsity_bonus);
  byId("sparsity-stat").hidden = !hasSparsity;
  put("active-edges-label", uncapped ? "最终活跃边" : "活跃边");
  put("average-edges-label", "时均活跃边");
  put("weight-updates-label", privateMode ? "权重改动次数" : "权重更新");
  put("births-label", privateMode ? "新增连接次数" : "新增边");
  put("deaths-label", privateMode ? "移除连接次数" : "退出边");
  byId("strict-return-label").title = uncapped ? "两种情境的首次访问结束后，后续访问中的严格完成率" : "只统计每个符号的首次返回探查";
  if (!advanced) {
    put("graph-detail", "适应度领先候选的实测图变化");
    return;
  }

  uncappedMetrics.forEach(([key, formatter]) => put(key.replaceAll("_", "-"), format(progress[key], formatter)));
  put("average-edges", format(progress.average_edges));
  put("sparsity", format(progress.sparsity_bonus, percent));
  if (privateMode) {
    put("active-edges-label", "结束时连接数");
    put("average-edges-label", "全程平均连接数");
    put("initial-edges-label", "出生时连接数");
    put("active-edges", format(progress.final_edges ?? progress.active_edges));
    put("average-edges", format(progress.mean_edges ?? progress.average_edges));
    put("initial-edges", format(progress.initial_edges));
    put("graph-detail", "每个开启的传信方向算一条连接，反向另计。连接数按独立生命平均，改权、新增和退出为所有生命的累计次数。");
    return;
  }
  put("graph-detail", `边数按 actor 平均，新增与退出为所有 actor 的累计次数。${hasSparsity ? "稀疏分数在严格完成率、严格返回和完成率相同时用于排序。" : "两种情境都出现严格完成时才显示稀疏分数。"}`);
}

function renderHoldout(holdout, phase, paired, uncapped) {
  const modes = holdout?.modes;
  const entries = modes && Object.entries(modes).filter(([, metrics]) => metrics && typeof metrics === "object");
  const region = byId("holdout-data");
  region.hidden = !entries?.length;
  if (region.hidden) {
    const details = {
      holdout: "正在评估最优候选及对照模式。",
      complete: "本次运行目录尚无可读取的留出评估结果。",
      failed: "留出评估结果暂不可用。",
    };
    const waiting = paired ? "每个阶段结束后评估两组冠军与各对照模式。"
      : uncapped ? "训练期间定期评估领先候选与固定猜符号对照。" : "搜索结束后评估最优候选及各对照模式。";
    const evaluating = paired && phase === "holdout" ? "正在运行本阶段的留出评估。" : "";
    put("holdout-detail", evaluating || details[phase] || waiting);
    return;
  }

  const candidate = candidateName(holdout.candidate);
  const description = [];
  if (paired) {
    const stage = stageLabels[holdout.stage];
    if (stage) description.push(`最近完成 · ${stage}`);
    if (numeric(holdout.generation)) description.push(`第 ${format(holdout.generation)} 代`);
    for (const [arm, label] of [["full", "完整组冠军"], ["weights", "权重组冠军"]]) {
      const identity = candidateName(holdout.candidates?.[arm]);
      if (identity) description.push(`${label} ${identity}`);
    }
  }
  if (uncapped && numeric(holdout.generation)) description.push(`最近完成 · 第 ${format(holdout.generation)} 代`);
  if (candidate) description.push(`候选 ${candidate}`);
  if (Array.isArray(holdout.seeds)) description.push(`${format(holdout.seeds.length)} 个种子`);
  if (numeric(holdout.actors)) description.push(`${format(holdout.actors)} 个 actor`);
  description.push(uncapped ? "各指标为留出种子均值" : "图统计为各留出种子测量值的均值");
  put("holdout-detail", description.join(" · "));

  const available = ([key]) => entries.some(([, metrics]) => numeric(metrics[key]));
  const returnColumns = [
    ["first_finish_correct_a_first_return", percent, "A 首次回访首答"],
    ["first_finish_correct_b_first_return", percent, "B 首次回访首答"],
  ];
  const columns = paired ? pairedHoldoutMetrics.filter(available)
    : uncapped ? [...returnColumns, ...uncappedHoldoutMetrics].filter(available) : tableColumns;
  const labels = paired ? pairedModeLabels : uncapped ? uncappedModeLabels : modeLabels;
  renderHoldoutRows(entries, columns, labels);
}

function renderHoldoutRows(entries, columns, labels) {
  const headers = [["模式"], ...columns.map(([, , label, description]) => [label, description])].map(([label, description]) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    if (description) cell.title = description;
    return cell;
  });
  byId("holdout-columns").replaceChildren(...headers);
  const rows = entries.map(([name, metrics]) => {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = Object.hasOwn(labels, name) ? labels[name] : name;
    row.append(label);
    for (const [key, formatter] of columns) {
      const cell = document.createElement("td");
      cell.textContent = format(metrics[key], formatter);
      row.append(cell);
    }
    return row;
  });
  byId("holdout-rows").replaceChildren(...rows);
}

function syncSelect(select, choices, selected) {
  if (select.options.length !== choices.length || choices.some(([value, label], index) => (
    select.options[index].value !== value || select.options[index].textContent !== label
  ))) {
    select.replaceChildren(...choices.map(([value, label]) => new Option(label, value)));
  }
  if (select.value !== selected) select.value = selected;
}

let privateHoldoutBranch = "";
let privateHoldoutClock = "";

function renderPrivateHoldout(data, phase) {
  const branches = data.branches ?? [];
  const single = branches.length === 1;
  const branchSelect = byId("private-holdout-branch");
  branchSelect.hidden = branches.length < 2;
  if (!branches.some(({ id }) => id === privateHoldoutBranch)) privateHoldoutBranch = "";
  syncSelect(branchSelect, branchSelect.hidden ? [] : [
    ["", "自动选择分支"], ...branches.map((branch) => [branch.id, branchName(branch)]),
  ], privateHoldoutBranch);
  const current = branches.find(({ id }) => id === data.current) ?? (single ? branches[0] : null);
  const mostRecent = branches.reduce((latest, branch) => {
    if (!branch.holdout) return latest;
    return !latest || branch.holdout.generation >= latest.holdout.generation ? branch : latest;
  }, null);
  const selected = branches.find(({ id }) => id === privateHoldoutBranch) ?? (current?.holdout ? current : mostRecent ?? current);
  const holdout = selected?.holdout;
  const conditions = holdout?.conditions ?? {};
  const clockKeys = Object.keys(conditions);
  const clock = Object.hasOwn(conditions, privateHoldoutClock) ? privateHoldoutClock : clockKeys[0];
  const clockSelect = byId("private-holdout-clock");
  clockSelect.hidden = clockKeys.length < 2;
  syncSelect(clockSelect, clockKeys.map((key) => [key, privateClockLabels[key] ?? key]), clock ?? "");
  const modes = conditions[clock]?.modes ?? {};
  const entries = Object.keys(privateHoldoutModeLabels).filter((key) => modes[key]).map((key) => [key, modes[key]]);
  byId("holdout-data").hidden = entries.length === 0;
  if (!entries.length) {
    const evaluating = phase === "holdout" && (!selected || selected.id === data.current);
    const waiting = evaluating ? "正在评估" : "等待首次留出评估";
    put("holdout-detail", selected ? `${branchName(selected, single)} · ${waiting}` : waiting);
    return;
  }
  const description = [branchName(selected, single), privateClockLabels[clock] ?? clock];
  if (numeric(holdout.generation)) description.push(`第 ${format(holdout.generation)} 代`);
  if (Array.isArray(holdout.seeds)) description.push(`${format(holdout.seeds.length)} 个留出种子`);
  if (numeric(holdout.actors)) description.push(`每种子 ${format(holdout.actors)} 次独立生命`);
  description.push("各岛领先候选的留出均值");
  put("holdout-detail", description.join(" · "));
  const withRecovery = entries.map(([name, metrics]) => [name, {
    ...metrics, recovered_completion_rate: privateMetric(metrics, "recovered_completion_rate"),
  }]);
  const columns = privateHoldoutMetrics(privateLateRounds(data.visits))
    .filter(([key]) => withRecovery.some(([, metrics]) => numeric(metrics[key])));
  renderHoldoutRows(withRecovery, columns, privateHoldoutModeLabels);
}

function statusFor(progress, processActive, paired, uncapped, privateMode) {
  const phase = progress?.phase;
  const state = progress?.state;
  if (phase === "failed" || state === "failed") return ["failed", "运行失败", "查看标准错误或运行记录。"];
  if (phase === "complete" || state === "complete") return ["complete", "已完成", "候选搜索和留出评估已结束。"];
  if (state === "paused") return ["paused", "已暂停", "已写入当前进度与检查点。"];
  if (processActive === false && state === "running") {
    return ["paused", "进程已退出", "显示退出前最后一次写入的进度。"];
  }
  if (phase === "receipt_audit") return ["running", "回执干预中", "正在比较同一教学动作收到真实或颠倒回执后的作答。"];
  if (phase === "final_comparison") return ["running", "终代复评中", "正在汇总相同留出世界上的首代与终代成绩。"];
  if (phase === "holdout") {
    const mode = (privateMode ? privateHoldoutModeLabels : uncapped ? uncappedModeLabels : pairedModeLabels)[progress.holdout_mode];
    const detail = paired
      ? `当前阶段候选评估已结束，正在使用留出种子${mode ? `评估 ${mode}` : "评估两组冠军"}。`
      : uncapped || privateMode ? `正在评估第 ${format(progress.generation)} 代的留出结果${mode ? ` · ${mode}` : ""}。`
      : "候选搜索已结束，正在运行留出评估。";
    return ["running", "留出评估中", detail];
  }
  if (phase === "search") return ["running", "候选搜索中", "计数按训练器写入的候选批次更新。"];
  if (processActive === true) return ["running", "正在启动", "训练进程已启动，正在等待首份进度。"];
  return ["waiting", "等待训练", "正在等待训练写入进度。"];
}

let lastSnapshot = null;

function remainingTime(progress, tone) {
  if (tone === "complete") return "已完成";
  if (tone !== "running") return missing;

  const { completed_candidates: done, total_candidates: total, candidates_per_minute: rate } = progress;
  if (!numeric(done) || done < 0 || !numeric(total) || total <= 0) return missing;
  if (done >= total) return progress.phase === "holdout" ? "留出评估中" : "收尾中";
  if (!numeric(rate) || rate <= 0) return missing;

  const estimate = (total - done) / rate;
  if (!numeric(estimate)) return missing;
  if (estimate < 1) return "不到 1 分钟";
  const minutes = Math.ceil(estimate);
  return [
    [Math.floor(minutes / 1440), "天"],
    [Math.floor(minutes / 60) % 24, "小时"],
    [minutes % 60, "分钟"],
  ].filter(([value]) => value > 0).map(([value, unit]) => `${value} ${unit}`).join(" ");
}

function renderSnapshot(snapshot) {
  const progress = snapshot.progress ?? {};
  const process = snapshot.process ?? {};
  const privateData = snapshot.private_symbol;
  const privateMode = Boolean(privateData);
  const singlePrivateBranch = privateData?.branches?.length === 1;
  const uncapped = !privateMode && [progress, process, snapshot.holdout].some((data) => data?.experiment_kind === "uncapped_continuous");
  const paired = !privateMode && !uncapped && (Object.hasOwn(stageLabels, progress.stage) || Boolean(progress.arms || progress.stage_bounds || snapshot.holdout?.candidates));
  const topologies = progress.topology_groups;
  const continuing = numeric(progress.continuation_start_generation);
  const [tone, label, detail] = statusFor(progress, snapshot.process_active, paired, uncapped, privateMode);
  byId("status").dataset.tone = tone;
  put("status-text", label);
  put("phase-detail", detail);
  const title = privateData?.local_rule ? "无边数上限 · 连续可塑性训练" : privateMode ? (singlePrivateBranch ? "私有符号训练" : "私有符号配对训练") : continuing ? "四岛有向图 · 回执学习" : topologies ? "有向与无向图 · 回执学习" : paired ? "可塑性课程训练"
    : uncapped ? "无边数上限 · 连续可塑性训练" : "可塑性进化训练";
  put("title", title);
  put("experiment-label", privateMode ? "有向图实验" : "无向图实验");
  document.title = `${title} · AGI`;
  put("run-name", snapshot.run_name);
  const precision = progress.execution_precision?.toUpperCase();
  put("run-detail", `设备 ${progress.device ?? process.device ?? missing}${precision ? ` · ${precision}` : ""} · PID ${process.pid ?? missing}`);

  const failure = typeof progress.failure === "string" ? progress.failure : "";
  const unreadable = Object.keys(snapshot.read_errors ?? {}).map((name) => `${name}.json`);
  const warnings = [...(failure ? [failure] : []), ...(unreadable.length ? [`读取失败 · ${unreadable.join("、")}`] : [])];
  showNotice(warnings.join(" · "));

  const overall = fraction(progress.completed_candidates, progress.total_candidates);
  const generationTotal = progress.generation_total ?? progress.population;
  byId("overall-bar").value = overall ?? 0;
  byId("generation-bar").value = fraction(progress.generation_completed, generationTotal) ?? 0;
  put("overall-label", "全部候选（三组合计）");
  put("generation-label", "本代候选（当前组）");
  byId("private-branch-stat").hidden = !privateMode;
  const current = privateData?.branches?.find(({ id }) => id === privateData.current)
    ?? privateData?.branches?.find(({ id }) => id === progress.display_summary?.arm);
  put("private-branch-label", privateData?.current ? "当前训练组" : "最后封存组");
  put("private-branch", branchName(current, singlePrivateBranch));
  put("overall-percent", format(overall, percent));
  put("overall-count", counter(progress.completed_candidates, progress.total_candidates));
  put("gen-count", counter(progress.generation_completed, generationTotal));
  put("generation", counter(progress.generation, progress.generations));
  put("rate", numeric(progress.candidates_per_minute) ? `${format(progress.candidates_per_minute)} / 分钟` : missing);
  put("remaining-time", remainingTime(progress, tone));
  putTime("updated", progress.updated_utc);
  putTime("started", progress.started_utc ?? process.started_utc);
  put("checkpoint", typeof progress.checkpoint === "string" && progress.checkpoint ? progress.checkpoint : missing);
  const journal = typeof progress.candidate_journal === "string" ? progress.candidate_journal : "";
  byId("candidate-journal-stat").hidden = !privateMode || !journal;
  put("candidate-journal", journal);

  put("best-fitness", format(progress.best_fitness, fitness));
  for (const [key, formatter] of tableColumns) {
    const id = key.replaceAll("_", "-");
    if (byId(id)) put(id, format(progress[key], formatter));
  }
  const candidate = candidateName(progress.best_candidate);
  put("candidate", candidate);
  const behavior = [
    ["空闲动作率", "idle"], ["端口步错误率", "error"], ["窗口内尝试率", "legal_attempt"],
    ["过早完成尝试率", "early"], ["抢答请求率", "premature_request_rate"], ["超时完成尝试率", "late"],
    ["A 首次回访首答", "first_finish_correct_a_first_return"],
    ["B 首次回访首答", "first_finish_correct_b_first_return"],
  ];
  const behaviorText = behavior.filter(([, key]) => numeric(progress[key]))
    .map(([label, key]) => `${label} ${format(progress[key], percent)}`).join(" · ");
  put("candidate-errors", behaviorText);
  byId("candidate-errors").hidden = !behaviorText;
  renderCurriculum(progress, paired);
  byId("legacy-metrics").hidden = false;
  renderPrivateReadout(progress, candidate, privateData);
  renderLocalCandidate(progress);
  put("graph-title", "图结构与权重");
  if (topologies || continuing) {
    put("active-edges-label", "最终活跃方向数");
    put("graph-detail", continuing ? "边数按传信方向计。新增和退出为 16 条生命的累计次数。"
      : "边数按传信方向计。一条无向边计为两个方向。新增和退出为 16 条生命的累计次数。");
  }
  byId("private-holdout-controls").hidden = !privateMode;
  byId("private-chart-scope").hidden = !privateMode || singlePrivateBranch;
  if (privateMode) {
    renderLocalAudits(snapshot);
    renderPrivateHistory(privateData, snapshot.run_name);
    return;
  }
  const legacyHoldout = continuing && snapshot.holdout?.generation < progress.continuation_start_generation;
  renderHoldout(legacyHoldout ? null : snapshot.holdout, progress.phase, paired, uncapped);
  setHistory(snapshot.history, continuing ? "directed" : topologies ? "topology" : paired ? "paired" : uncapped ? "uncapped" : "legacy");
}

function renderLocalCandidate(progress) {
  byId("candidate-metrics").classList.add("uncapped-metrics");
  document.querySelectorAll("[data-legacy-metric]").forEach(el => { el.hidden = true; });
  document.querySelectorAll("[data-uncapped-metric]").forEach(el => { el.hidden = false; });
  for (const [id, key, formatter] of [
    ["strict", "strict", percent], ["completion", "completion", percent],
    ["ready-coverage", "ready_coverage", percent], ["first-finish-correct", "first_finish_correct", percent],
    ["pair-strict-rate", "pair_strict_rate", percent], ["selection-score", "selection_score", fitness],
    ["average-edges", "mean_edges", number], ["active-edges", "final_edges", number],
    ["initial-edges", "initial_edges", number], ["bias-updates", "bias_updates", number]
  ]) put(id, format(progress[key], formatter));
  byId("average-edges-stat").hidden = false;
  byId("initial-edges-stat").hidden = false;
  put("active-edges-label", "最终活跃边");
  put("graph-detail", "活跃边数按独立生命平均。权重、偏置和连接改动次数为 8 个生命的累计值。");
  const row = progress.display_summary;
  const caption = row ? [privateArmLabels[row.arm], `第 ${format(row.generation)} 代`, candidateName(progress.best_candidate)].filter(Boolean).join(" · ") : "等待本组首批候选";
  put("candidate", caption);
  const notes = [["空闲动作率", "idle"], ["抢答请求率", "premature_request_rate"], ["补救完成率", "recovery_rate"]]
    .filter(([, key]) => numeric(progress[key])).map(([label,key]) => `${label} ${format(progress[key], percent)}`);
  put("candidate-errors", notes.join(" · "));
  byId("candidate-errors").hidden = !notes.length;
}

byId("candidate-details").addEventListener("click", (event) => {
  const button = event.currentTarget;
  const expanded = button.getAttribute("aria-expanded") !== "true";
  button.setAttribute("aria-expanded", String(expanded));
  byId("private-readout").hidden = !expanded;
});

function showNotice(message) {
  put("notice", message);
  byId("notice").hidden = !message;
}

async function fetchData(path) {
  const response = await fetch(path, { cache: "no-store", signal: AbortSignal.timeout(6000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

const foregroundRefreshMs = 1000;

async function pollSnapshot() {
  const started = performance.now();
  try {
    const snapshot = adaptLocalSnapshot(await fetchData("/api/snapshot"));
    for (const key of Object.keys(snapshot.read_errors ?? {})) {
      if (lastSnapshot?.[key]) snapshot[key] = lastSnapshot[key];
    }
    renderSnapshot(snapshot);
    lastSnapshot = snapshot;
  } catch {
    showNotice(lastSnapshot ? "暂时无法连接本地控制台，当前显示上次收到的数据。" : "暂时无法连接本地控制台，稍后将自动重试。");
  } finally {
    const terminal = ["complete", "failed"].includes(lastSnapshot?.progress?.phase);
    const interval = document.hidden || terminal ? 10000 : foregroundRefreshMs;
    setTimeout(pollSnapshot, Math.max(0, interval - (performance.now() - started)));
  }
}

let stream = "stdout";
let logs = {};

function renderLog() {
  const data = logs[stream];
  const metadata = [stream === "stdout" ? "标准输出" : "标准错误"];
  if (data?.truncated) metadata.push("显示最新 32 KiB 范围内的日志");
  put("log-detail", metadata.join(" · "));
  const target = byId("log-text");
  const follow = target.scrollHeight - target.clientHeight - target.scrollTop < 15;
  const previous = target.scrollTop;
  put("log-text", data?.text || (data?.error ? "日志暂时不可读取" : "等待日志写入"));
  target.scrollTop = follow ? target.scrollHeight : previous;
}

document.querySelectorAll("[data-stream]").forEach((button) => {
  button.addEventListener("click", () => {
    stream = button.dataset.stream;
    document.querySelectorAll("[data-stream]").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
    byId("log-text").scrollTop = byId("log-text").scrollHeight;
    renderLog();
  });
});

async function pollLogs() {
  const started = performance.now();
  try {
    logs = await fetchData("/api/logs");
    renderLog();
  } catch {
    put("log-detail", "日志连接暂时中断，稍后将自动重试。");
  } finally {
    const interval = document.hidden ? 12000 : foregroundRefreshMs;
    setTimeout(pollLogs, Math.max(0, interval - (performance.now() - started)));
  }
}

const legacySeries = {
  fitness: [
    ["best_fitness", "本代最优", "#d4f983"], ["mean_fitness", "本代均值", "#72e4b3"],
  ],
  behavior: [
    ["completion", "完成率", "#d4f983"], ["strict_a", "严格 A", "#72e4b3"],
    ["strict_b", "严格 B", "#83bdf6"], ["strict_return", "严格返回", "#f3c684"],
  ],
};
const pairedSeries = {
  fitness: [
    ["best_fitness", "完整组最优", "#d4f983", "full"], ["best_fitness", "权重组最优", "#83bdf6", "weights"],
  ],
  behavior: [
    ["completion", "完整组完成率", "#d4f983", "full"], ["completion", "权重组完成率", "#83bdf6", "weights"],
    ["strict_return", "完整组严格返回", "#72e4b3", "full"], ["strict_return", "权重组严格返回", "#cba5f4", "weights"],
  ],
};
const uncappedSeries = {
  fitness: [
    ["best_fitness", "本代领先候选", "#d4f983"], ["mean_fitness", "本代均值", "#72e4b3"],
  ],
  behavior: [
    ["completion", "完成率", "#d4f983"], ["ready_coverage", "就绪覆盖率", "#72e4b3"],
    ["version_accuracy", "版本准确率", "#83bdf6"], ["symbol_accuracy", "符号准确率", "#cba5f4"],
    ["strict", "严格完成率", "#f4a58f"], ["strict_return", "严格返回", "#f3c684"],
  ],
  edges: [
    ["average_edges", "时均活跃边", "#83bdf6"], ["active_edges", "最终活跃边", "#72e4b3"],
  ],
};
const topologySeries = {
  ...uncappedSeries,
  fitness: [
    ["first_return_mean", "有向领先候选", "#d4f983", "directed"],
    ["first_return_mean", "无向领先候选", "#83bdf6", "reciprocal"],
    ["mean_first_return", "有向种群均值", "#72e4b3", "directed"],
    ["mean_first_return", "无向种群均值", "#cba5f4", "reciprocal"],
  ],
};
const directedSeries = {
  ...uncappedSeries,
  fitness: [
    ["first_return_mean", "领先候选回访首答", "#d4f983"],
    ["mean_first_return", "全种群回访首答", "#72e4b3"],
    ["strict", "领先候选严格完成率", "#83bdf6"],
  ],
};
const privateChartColors = [
  ["#d4f983", "#72e4b3"],
  ["#83bdf6", "#cba5f4"],
  ["#f3c684", "#f4a58f"],
];
function privateSeriesFor(metrics) {
  return privateArms.flatMap((arm, index) => metrics.map(([key, label], metricIndex) => [
    key, label, privateChartColors[index][metricIndex], arm,
  ]));
}

const privateSeries = {
  fitness: privateSeriesFor([["strict", "领先候选"], ["mean_fitness", "本代均值"]]),
  score: privateSeriesFor([["selection_score", "领先候选选拔分"], ["mean_selection_score", "本代平均选拔分"]]),
  symbols: privateSeriesFor([["symbol_late_mean", "领先候选 · A/B 均值"], ["mean_symbol_late", "全种群 · A/B 均值"]]),
  behavior: privateSeriesFor([["premature_request_rate", "抢答请求率"], ["ready_coverage", "窗口内作答率"]]),
  edges: privateSeriesFor([["mean_edges", "时均活跃边"], ["final_edges", "最终活跃边"]]),
};
const privateAnswerSeries = {
  overall: privateSeries.symbols,
  contexts: privateSeriesFor([["symbol_late_a", "情境 A 首答"], ["symbol_late_b", "情境 B 首答"]]),
  revisits: privateSeriesFor([["pair_strict_rate", "领先候选双回访成功率"], ["mean_pair_strict_rate", "本代平均双回访成功率"]]),
};
const seriesByMode = { legacy: legacySeries, paired: pairedSeries, uncapped: uncappedSeries,
  topology: topologySeries, directed: directedSeries, private: privateSeries };
const selectionScoreKeys = new Set(["selection_score", "mean_selection_score"]);
const scoreRuleLabels = {
  local_rule_revisit: "严格完成每题加 1 分，补救完成每题加 0.1 分，双回访均严格完成每对加 1 分，答对后每次多余动作扣 0.1 分，合计除以 512 道请求",
};

function privateHistory(branches, selectionRule) {
  const byGeneration = new Map();
  for (const branch of branches) {
    for (const summary of branch.history ?? []) {
      if (!numeric(summary?.generation)) continue;
      const { generation } = summary;
      if (!byGeneration.has(generation)) byGeneration.set(generation, { generation, arms: {} });
      byGeneration.get(generation).arms[branch.arm] = summary.selection_rule == null && selectionRule
        ? { ...summary, selection_rule: selectionRule } : summary;
    }
  }
  return [...byGeneration.values()];
}

let privateChartData = null;
let privateChartSelection = "";
let privateAnswerView = "overall";

function renderPrivateHistory(data) {
  privateChartData = data;
  const branches = data.branches ?? [];
  const groupLabel = (branch) => privateArmLabels[branch.arm] ?? branch.arm;
  const current = branches.find(({ id }) => id === data.current)
    ?? branches.reduce((last, branch) => (branch.latest?.generation ?? -1) >= (last?.latest?.generation ?? -1) ? branch : last, null);
  const choices = branches.length < 2 ? [] : [
    ["", "当前训练组"], ["all", "全部训练组"],
    ...branches.map((branch) => [branch.id, groupLabel(branch)]),
  ];
  if (!choices.some(([value]) => value === privateChartSelection)) privateChartSelection = "";
  syncSelect(byId("private-chart-scope"), choices, privateChartSelection);
  const selected = branches.find(({ id }) => id === privateChartSelection) ?? current;
  const visible = privateChartSelection === "all" ? branches : selected ? [selected] : [];
  const scope = privateChartSelection === "all" ? "全部训练组"
    : selected ? (privateChartSelection ? groupLabel(selected) : `当前训练组 · ${groupLabel(selected)}`) : "当前训练组";
  const requestCount = (data.visits ?? 16) * (data.requests_per_visit ?? 4);
  const chartConfig = {
    arms: visible.map(({ arm }) => arm),
    lateRequests: `第 ${Math.floor(requestCount / 2) + 1}–${requestCount} 题`,
    selectionRule: data.selection_rule,
  };
  setHistory(privateHistory(visible, data.selection_rule), "private", scope, chartConfig);
}

const canvas = byId("chart");
const plot = byId("plot");
const context = canvas.getContext("2d");
const fitButton = byId("chart-fit-button");
const fitPreference = "agi-plasticity-show-fit";
const fitCache = new Map();
let chartType = "fitness";
let chartMode = "legacy";
let chartScopeDetail = null;
let privateChartConfig = null;
let history = [];
let hoverX = null;
let showFit = false;
try {
  showFit = localStorage.getItem(fitPreference) === "true";
} catch {
  // The toggle still works if browser storage is disabled.
}

function activeChartSeries() {
  const series = seriesByMode[chartMode][chartType];
  if (chartMode !== "private" || !series) return series;
  const selected = chartType === "symbols" ? privateAnswerSeries[privateAnswerView] : series;
  const { arms } = privateChartConfig;
  const visible = selected.filter(([, , , arm]) => arms.includes(arm));
  return visible.map(([key, label, color, arm], index) => arms.length === 1
    ? [key, label, privateChartColors[0][index], arm]
    : [key, `${privateArmLabels[arm]} · ${label}`, color, arm]);
}
const chartValue = (row, key, arm) => {
  const metrics = arm ? (row.topology_groups ?? row.arms)?.[arm] : row;
  return chartMode === "private" ? privateMetric(metrics, key) : metrics?.[key];
};
const scoreRule = (row, arm) => row.arms?.[arm]?.selection_rule ?? null;
const sameSeriesSegment = (previous, row, key, arm) => (!arm || previous.stage === row.stage)
  && (!selectionScoreKeys.has(key) || scoreRule(previous, arm) === scoreRule(row, arm));
const hasEdgeChart = () => Boolean(seriesByMode[chartMode].edges);
const percentageChart = () => chartType === "behavior" || chartType === "symbols" || (chartType === "fitness" && hasEdgeChart());

function selectChart(type) {
  chartType = type;
  document.querySelectorAll("[data-chart]").forEach((item) => item.setAttribute("aria-pressed", String(item.dataset.chart === type)));
}

function setHistory(rows, mode, scopeDetail = null, config = null) {
  if (chartMode !== mode || chartScopeDetail !== scopeDetail || config?.arms.join() !== privateChartConfig?.arms.join()
      || config?.lateRequests !== privateChartConfig?.lateRequests || config?.selectionRule !== privateChartConfig?.selectionRule) {
    chartMode = mode;
    chartScopeDetail = scopeDetail;
    privateChartConfig = config;
    if (!activeChartSeries()) selectChart("fitness");
    renderLegend();
  }
  history = (rows ?? []).filter((row) => row && numeric(row.generation)).sort((left, right) => left.generation - right.generation);
  fitCache.clear();
  drawChart();
}

function renderLegend() {
  const items = activeChartSeries().map(([, label, color]) => {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.color = color;
    item.append(swatch, document.createTextNode(label));
    return item;
  });
  byId("chart-legend").replaceChildren(...items);
  const uncapped = hasEdgeChart();
  const topology = chartMode === "topology" || chartMode === "directed";
  const privateMode = chartMode === "private";
  const lateRequests = privateChartConfig?.lateRequests;
  const rule = privateChartConfig?.selectionRule;
  const revisits = privateAnswerView === "revisits";
  byId("edge-chart-button").hidden = !uncapped;
  byId("score-chart-button").hidden = !privateMode;
  byId("symbol-chart-button").hidden = !privateMode;
  byId("private-answer-view").hidden = !privateMode || chartType !== "symbols";
  put("fitness-chart-button", topology ? "回访首答" : uncapped ? "严格完成率" : "适应度");
  if (privateMode) put("symbol-chart-button", "符号与回访");
  put("behavior-chart-button", privateMode ? "作答时机" : "行为率");
  put("plot-empty", chartType === "edges" ? "等待活跃边记录" : "等待首代评估完成");
  const titles = {
    fitness: topology ? "历代首次回访的首答正确率" : uncapped ? "历代严格完成率" : "历代适应度",
    score: "历代选拔得分",
    symbols: revisits ? "历代双回访成功率" : "历代后半生首答正确率",
    behavior: privateMode ? "历代作答时机" : "历代领先候选行为率",
    edges: "历代领先候选活跃边数",
  };
  const details = {
    fitness: topology ? "A、B 等权平均，未答计错" : uncapped ? "各代领先候选和全代平均的严格完成率"
      : "每个点对应一代已完成的评估",
    behavior: uncapped ? "各代领先候选的行为率" : "行为率取各代适应度领先候选",
    edges: "各代领先候选的时均和最终活跃边数，按 actor 平均",
  };
  fitButton.setAttribute("aria-pressed", String(showFit));
  put("trend-title", privateMode ? titles[chartType] : "历代趋势");
  canvas.setAttribute("aria-label", showFit ? `${titles[chartType]}，含拟合线` : titles[chartType]);
  const privateDetails = {
    fitness: "在窗口内答对，且此前主端口没有被拒动作，计为严格完成。每组分别显示该代选拔得分最高的候选和全体候选的平均严格完成率。",
    score: rule ? `各代领先候选和种群平均的原始选拔分。${scoreRuleLabels[rule] ?? rule}。`
      : "各代领先候选和种群平均的原始选拔分。计分规则改变时分段绘制和拟合。",
    symbols: revisits ? "同一生命里，某个符号在 A、B 都出现过后，后续回访按次序配对。两题均严格完成才计为成功。按合格配对数计算成功率，分别显示领先候选和全代均值。"
      : `每个生命的${lateRequests}，按窗口内首次回答统计正确率，漏答计错。${privateAnswerView === "contexts"
        ? "每组分别显示领先候选在情境 A 和 B 的成绩。" : "先对 A、B 两种情境等权平均，再分别显示领先候选和全代均值。"}`,
    behavior: "各代领先候选。窗口打开前提交过答案计抢答，窗口内提交过答案计窗口内作答。两项均按请求统计。",
    edges: "各代领先候选。每个开启的传信方向算一条，全程按时间步和独立生命平均，结束时只对独立生命平均。",
  };
  const detail = privateMode ? `${chartScopeDetail} · ${privateDetails[chartType]}`
    : chartMode === "paired" ? "各组逐代领先候选 · 按课程阶段分段" : details[chartType];
  put("trend-detail", showFit ? `${detail} 亮线为局部线性拟合，淡线为逐代实测值。` : detail);
}

function fitSegment(points) {
  const span = Math.min(points.length, 61, Math.max(5, Math.ceil(points.length / 9)));
  const half = Math.floor(span / 2);
  return points.map(([generation], i) => {
    const start = Math.max(0, Math.min(points.length - span, i - half));
    const end = start + span - 1;
    const radius = 1 + Math.max(generation - points[start][0], points[end][0] - generation);
    let sum = 0;
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumXY = 0;
    for (let j = start; j <= end; j += 1) {
      const [x, y] = points[j];
      const dx = x - generation;
      const weight = (1 - (Math.abs(dx) / radius) ** 3) ** 3;
      sum += weight;
      sumX += weight * dx;
      sumY += weight * y;
      sumXX += weight * dx * dx;
      sumXY += weight * dx * y;
    }
    const determinant = sum * sumXX - sumX * sumX;
    const fitted = determinant > 0 ? (sumY * sumXX - sumX * sumXY) / determinant : sumY / sum;
    return [generation, fitted];
  });
}

function fittedSegments(key, arm) {
  const cacheKey = `${key}:${arm ?? ""}`;
  if (!fitCache.has(cacheKey)) {
    const fitted = [];
    let points = [];
    let previous = null;
    const endSegment = () => {
      if (points.length >= 3) fitted.push(fitSegment(points));
      points = [];
    };
    for (const row of history) {
      const value = chartValue(row, key, arm);
      if (!numeric(value) || (previous && (row.generation !== previous.generation + 1
          || !sameSeriesSegment(previous, row, key, arm)))) endSegment();
      if (numeric(value)) points.push([row.generation, value]);
      previous = row;
    }
    endSegment();
    fitCache.set(cacheKey, fitted);
  }
  return fitCache.get(cacheKey);
}

function fittedValueAt(generation, key, arm) {
  for (const segment of fittedSegments(key, arm)) {
    const point = segment[generation - segment[0][0]];
    if (point?.[0] === generation) return point[1];
  }
  return undefined;
}

function chartDomain(visibleSeries) {
  if (chartType === "behavior") return [0, 1];
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const row of history) {
    for (const [key, , , arm] of visibleSeries) {
      const value = chartValue(row, key, arm);
      if (!numeric(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  if (minimum === Infinity) return [0, 1];
  const margin = minimum === maximum ? Math.max(Math.abs(minimum) * 0.06, chartType === "edges" ? 1 : 0.01) : (maximum - minimum) * 0.09;
  const min = chartType === "edges" || percentageChart() ? Math.max(0, minimum - margin) : minimum - margin;
  return [min, percentageChart() ? Math.min(1, maximum + margin) : maximum + margin];
}

function scoreTransitions() {
  const last = new Map();
  const boundaries = new Map();
  for (const row of history) {
    for (const arm of privateChartConfig.arms) {
      if (!numeric(chartValue(row, "selection_score", arm))) continue;
      const previous = last.get(arm);
      if (previous && scoreRule(previous, arm) !== scoreRule(row, arm)) {
        boundaries.set(row.generation, (previous.generation + row.generation) / 2);
      }
      last.set(arm, row);
    }
  }
  return boundaries;
}

function drawChart() {
  const revisions = chartType === "score" ? scoreTransitions() : new Map();
  byId("score-phase-note").hidden = revisions.size === 0;
  if (revisions.size) put("score-phase-note", `改分起点在第 ${[...revisions.keys()].join("、")} 代。`);
  if (!context) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.scale(pixelRatio, pixelRatio);

  const activeSeries = activeChartSeries();
  const ready = history.some((row) => activeSeries.some(([key, , , arm]) => numeric(chartValue(row, key, arm))));
  byId("plot-empty").hidden = ready;
  byId("chart-tooltip").hidden = true;
  const [minY, maxY] = chartDomain(activeSeries);
  const minX = history[0]?.generation ?? 1;
  const maxX = history.at(-1)?.generation ?? minX;
  const formatter = percentageChart() ? percent : chartType === "edges" ? number : fitness;
  context.font = "11px Consolas, 'Microsoft YaHei UI', monospace";
  const labelWidth = Math.max(context.measureText(format(minY, formatter)).width, context.measureText(format(maxY, formatter)).width);
  const finalTickPadding = Math.ceil(context.measureText(`第 ${maxX} 代`).width / 2) + 3;
  const bounds = { left: Math.max(40, labelWidth + 11), right: width - finalTickPadding, top: 15, bottom: height - 27 };
  if (bounds.left >= bounds.right) return;
  const x = (generation) => bounds.left + (bounds.right - bounds.left) * (maxX === minX ? 0.5 : (generation - minX) / (maxX - minX));
  const y = (value) => bounds.bottom - (bounds.bottom - bounds.top) * (value - minY) / (maxY - minY);

  context.lineWidth = 1;
  context.strokeStyle = "#2d3932";
  context.fillStyle = "#a0b1a7";
  context.textBaseline = "middle";
  context.textAlign = "right";
  for (let i = 0; i <= 4; i += 1) {
    const value = minY + (maxY - minY) * i / 4;
    const rowY = y(value);
    context.beginPath();
    context.moveTo(bounds.left, rowY);
    context.lineTo(bounds.right, rowY);
    context.stroke();
    if (ready) context.fillText(format(value, formatter), bounds.left - 8, rowY);
  }
  if (!ready) return;

  context.textAlign = "center";
  const ticks = Math.max(1, Math.min(width < 390 ? 3 : 5, maxX - minX));
  for (const generation of new Set(Array.from({ length: ticks + 1 }, (_, i) => Math.round(minX + (maxX - minX) * i / ticks)))) {
    context.fillText(`第 ${generation} 代`, x(generation), bounds.bottom + 15);
  }

  if (chartMode === "paired") {
    context.strokeStyle = "#607167";
    context.beginPath();
    for (let index = 1; index < history.length; index += 1) {
      const previous = history[index - 1];
      const row = history[index];
      if (previous.stage === row.stage) continue;
      const boundary = x((previous.generation + row.generation) / 2);
      context.moveTo(boundary, bounds.top);
      context.lineTo(boundary, bounds.bottom);
    }
    context.stroke();
  }
  if (revisions.size) {
    context.strokeStyle = "#657267";
    context.beginPath();
    for (const boundary of revisions.values()) {
      context.moveTo(x(boundary), bounds.top);
      context.lineTo(x(boundary), bounds.bottom);
    }
    context.stroke();
  }

  for (const [key, , color, arm] of activeSeries) {
    const fitted = showFit ? fittedSegments(key, arm) : [];
    context.globalAlpha = fitted.length ? 0.32 : 1;
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    let previous = null;
    for (const row of history) {
      const value = chartValue(row, key, arm);
      if (!numeric(value)) {
        previous = null;
        continue;
      }
      const contiguous = chartMode !== "private" || previous?.generation + 1 === row.generation;
      if (previous && contiguous && sameSeriesSegment(previous, row, key, arm)) context.lineTo(x(row.generation), y(value));
      else context.moveTo(x(row.generation), y(value));
      previous = row;
    }
    context.stroke();
    if (history.length <= 30) {
      for (const row of history) {
        const value = chartValue(row, key, arm);
        if (!numeric(value)) continue;
        context.beginPath();
        context.arc(x(row.generation), y(value), 2.7, 0, 2 * Math.PI);
        context.fill();
      }
    }
  }
  context.globalAlpha = 1;

  if (showFit) {
    context.save();
    context.beginPath();
    context.rect(bounds.left - 2, bounds.top - 2, bounds.right - bounds.left + 4, bounds.bottom - bounds.top + 4);
    context.clip();
    context.lineWidth = 2.7;
    context.lineJoin = "round";
    context.lineCap = "round";
    for (const [key, , color, arm] of activeSeries) {
      context.strokeStyle = color;
      context.beginPath();
      for (const segment of fittedSegments(key, arm)) {
        const [generation, value] = segment[0];
        context.moveTo(x(generation), y(value));
        for (const [nextGeneration, nextValue] of segment.slice(1)) {
          context.lineTo(x(nextGeneration), y(nextValue));
        }
      }
      context.stroke();
    }
    context.restore();
  }

  if (hoverX !== null && hoverX >= bounds.left - 12 && hoverX <= bounds.right + 12) {
    const nearest = history.reduce((previous, row) => Math.abs(x(row.generation) - hoverX) < Math.abs(x(previous.generation) - hoverX) ? row : previous);
    const selected = x(nearest.generation);
    context.beginPath();
    context.strokeStyle = "#7a8f80";
    context.lineWidth = 1;
    context.moveTo(selected, bounds.top);
    context.lineTo(selected, bounds.bottom);
    context.stroke();
    showTooltip(nearest, activeSeries, formatter, selected, width);
  }
}

function showTooltip(row, activeSeries, formatter, x, width) {
  const tooltip = byId("chart-tooltip");
  const title = [`第 ${row.generation} 代`, stageLabels[row.stage], showFit ? "拟合值" : null]
    .filter(Boolean).join(" · ");
  const lines = [title];
  lines.push(...activeSeries.map(([key, label, , arm]) => {
    const value = showFit ? fittedValueAt(row.generation, key, arm) : chartValue(row, key, arm);
    return `${label} ${format(value, formatter)}`;
  }));
  tooltip.replaceChildren(...lines.map((line) => {
    const element = document.createElement("div");
    element.textContent = line;
    return element;
  }));
  tooltip.hidden = false;
  tooltip.style.left = `${Math.max(0, Math.min(x + 12, width - tooltip.offsetWidth))}px`;
}

document.querySelectorAll("[data-chart]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!seriesByMode[chartMode][button.dataset.chart]) return;
    selectChart(button.dataset.chart);
    renderLegend();
    drawChart();
  });
});
const privateScopeSelect = byId("private-chart-scope");
privateScopeSelect.setAttribute("aria-label", "趋势图训练组");
privateScopeSelect.addEventListener("change", (event) => {
  if (!privateChartData) return;
  privateChartSelection = event.target.value;
  renderPrivateHistory(privateChartData);
});
const privateAnswerSelect = byId("private-answer-view");
privateAnswerSelect.setAttribute("aria-label", "符号与回访指标");
privateAnswerSelect.addEventListener("change", (event) => {
  privateAnswerView = event.target.value;
  renderLegend();
  drawChart();
});
fitButton.addEventListener("click", () => {
  showFit = !showFit;
  try {
    localStorage.setItem(fitPreference, String(showFit));
  } catch {
    // Keep the current page preference even without storage.
  }
  renderLegend();
  drawChart();
});
plot.addEventListener("pointermove", (event) => {
  hoverX = event.clientX - plot.getBoundingClientRect().left;
  drawChart();
});
plot.addEventListener("pointerleave", () => {
  hoverX = null;
  drawChart();
});

new ResizeObserver(drawChart).observe(plot);
renderLegend();
pollSnapshot();
pollLogs();
