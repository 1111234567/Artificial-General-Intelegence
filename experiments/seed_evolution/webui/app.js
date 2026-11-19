"use strict";

const refreshInterval = 5000;
const feedMaxAge = 3 * refreshInterval;
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 });
const score = new Intl.NumberFormat("zh-CN", {
    style: "percent", minimumFractionDigits: 2, maximumFractionDigits: 2,
});
const clock = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
});
const chartClock = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const dateAndTime = new Intl.DateTimeFormat("zh-CN", {
    month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const statusLabels = {
    waiting: "等待启动", running: "运行中", stale: "进度未更新",
    stopped: "已停止", failed: "运行失败", complete: "已完成",
};
const phaseLabels = {
    search: "候选搜索", finals: "终训", test: "持出测试",
    complete: "已完成", failed: "运行失败",
};
const armLabels = { shared: "共享起始池", deep: "进化（多代）", medium: "进化（中等）", wide: "进化（宽种群）", random: "随机重启" };
const byId = (id) => document.getElementById(id);
const finite = Number.isFinite;
const display = (value, unit = "") => finite(value) ? number.format(value) + unit : "—";
const timestamp = (value) => typeof value === "string" ? Date.parse(value) : NaN;
const fraction = (value, total) => finite(value) && finite(total) && total > 0 ? 100 * value / total : null;
const clampPercent = (value) => finite(value) ? Math.min(100, Math.max(0, value)) : 0;
const setText = (id, value) => { byId(id).textContent = value; };
const countPair = (value, total) => `${display(value)} / ${display(total)}`;

let snapshot = null;
let refreshing = false;
let pollingPaused = false;
let refreshError = "";
let selectedLog = "stdout";
let copyReset;

function duration(milliseconds) {
    if (!finite(milliseconds) || milliseconds < 0) return "—";
    const seconds = Math.floor(milliseconds / 1000);
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor(seconds / 3600) % 24;
    const minutes = Math.floor(seconds / 60) % 60;
    if (days) return `${days} 天 ${hours} 小时`;
    if (hours) return `${hours} 小时 ${minutes} 分`;
    return minutes ? `${minutes} 分 ${seconds % 60} 秒` : `${seconds} 秒`;
}

function setButtonLabel(id, label) {
    const button = byId(id);
    const span = button.querySelector("[data-label]");
    if (span) {
        span.textContent = label;
    } else {
        const node = [...button.childNodes].find((child) => child.nodeType === Node.TEXT_NODE
            && child.textContent.trim());
        if (node) node.textContent = ` ${label}`;
        else button.append(document.createTextNode(label));
    }
}

function meter(id, value) {
    const bar = byId(id);
    bar.style.width = `${clampPercent(value)}%`;
    const track = bar.closest('[role="meter"], [role="progressbar"]');
    if (!track) return;
    if (finite(value)) track.setAttribute("aria-valuenow", String(clampPercent(value)));
    else track.removeAttribute("aria-valuenow");
}

function renderConnection() {
    const updatedAt = timestamp(snapshot?.updated_at);
    const stale = finite(updatedAt) && Date.now() - updatedAt > feedMaxAge && !pollingPaused;
    const runStatus = snapshot?.run?.status;
    const label = refreshError ? "刷新失败" : stale ? "数据未更新" : statusLabels[runStatus] ?? "等待数据";
    const state = refreshError ? "failed" : stale ? "stale" : runStatus ?? "waiting";
    const badge = byId("statusBadge");
    badge.dataset.state = state;
    badge.title = refreshError || "";
    setText("statusText", label);
    const updated = finite(updatedAt) ? `更新于 ${clock.format(updatedAt)}` : "尚未更新";
    setText("lastUpdated", pollingPaused ? `${updated} · 已暂停` : updated);
    if (!snapshot) {
        byId("runError").hidden = !refreshError;
        setText("runError", refreshError);
    }
}

function renderRun(run, throughput, updatedAt) {
    setText("runId", run.id || "—");
    byId("runError").hidden = !run.error;
    setText("runError", run.error || "");
    setText("completedCount", display(run.completed));
    setText("totalCount", display(run.total));
    const progress = fraction(run.completed, run.total);
    setText("runPercent", display(progress, "%"));
    meter("progressFill", progress);
    const phase = phaseLabels[run.phase] ?? run.phase ?? "等待阶段";
    setText("phaseTag", (run.phase === "finals" || run.phase === "test") && finite(run.final_total)
        ? `${phase} · ${countPair(run.final_completed, run.final_total)}` : phase);
    setText("activeRound", run.round === 0 ? "起始池" : countPair(run.round, run.max_round));
    setText("activeRep", finite(run.active_rep) ? display(run.active_rep + 1) : "—");
    setText("activeBatch", finite(run.active_batch) ? display(run.active_batch + 1) : "—");
    const pending = finite(run.trained) && finite(run.completed)
        ? Math.max(0, run.trained - run.completed) : run.pending_scores;
    setText("pendingScores", display(pending));
    setText("trainedCount", display(run.trained));
    if (byId("finalProgress")) {
        setText("finalProgress", countPair(run.final_completed, run.final_total));
    }
    setText("throughputRate", display(throughput.per_minute));
    setText("throughputWindow", finite(throughput.window_minutes)
        ? `近 ${display(throughput.window_minutes)} 分钟`
        : "—");
    const eta = timestamp(throughput.eta_at);
    setText("etaValue", finite(eta) ? dateAndTime.format(eta) : "—");
    byId("etaValue").title = finite(eta) ? new Date(eta).toLocaleString("zh-CN") : "";
    const finishedAt = timestamp(run.finished_at);
    setText("elapsedValue", duration((finite(finishedAt) ? finishedAt : timestamp(updatedAt))
        - timestamp(run.started_at)));
}

function renderArms(arms) {
    const rows = arms.map((arm) => {
        const row = document.createElement("tr");
        const name = document.createElement("th");
        name.scope = "row";
        name.title = arm.id;
        const dot = document.createElement("span");
        dot.className = "arm-dot";
        dot.dataset.arm = arm.id;
        dot.setAttribute("aria-hidden", "true");
        name.append(dot, document.createTextNode(armLabels[arm.id] ?? arm.id));
        row.append(name);

        row.insertCell().textContent = display(arm.width);
        row.insertCell().textContent = countPair(arm.rounds_completed, arm.rounds_total);
        const progress = row.insertCell();
        progress.className = "arm-progress";
        const count = document.createElement("span");
        count.textContent = countPair(arm.completed, arm.total);
        const track = document.createElement("span");
        track.className = "arm-progress-track";
        track.setAttribute("aria-hidden", "true");
        const fill = document.createElement("span");
        fill.className = "arm-progress-fill";
        fill.style.width = `${clampPercent(fraction(arm.completed, arm.total))}%`;
        track.append(fill);
        progress.append(count, track);
        row.insertCell().textContent = finite(arm.best) ? score.format(arm.best) : "—";
        return row;
    });
    if (!rows.length) {
        const row = document.createElement("tr");
        const cell = row.insertCell();
        cell.colSpan = 5;
        cell.textContent = "暂无分组数据";
        rows.push(row);
    }
    byId("armsBody").replaceChildren(...rows);
}

function renderHardware(hardware, run) {
    setText("cpuValue", display(hardware.cpu_percent, "%"));
    meter("cpuMeter", hardware.cpu_percent);
    setText("cpuDetails", finite(run.cpu_workers) ? `${display(run.cpu_workers)} 个评分进程` : "—");
    const gpu = hardware.gpu ?? {};
    setText("gpuName", gpu.name ?? "GPU");
    setText("gpuValue", display(gpu.utilization, "%"));
    meter("gpuMeter", gpu.utilization);
    setText("gpuMemory", finite(gpu.used_mib) && finite(gpu.total_mib)
        ? `${countPair(gpu.used_mib, gpu.total_mib)} MiB`
        : "—");
    meter("gpuMemMeter", fraction(gpu.used_mib, gpu.total_mib));
    const gpuDetails = [];
    if (finite(gpu.temp_c)) gpuDetails.push(display(gpu.temp_c, " °C"));
    if (finite(gpu.power_w)) gpuDetails.push(display(gpu.power_w, " W"));
    if (finite(run.gpu_batch)) gpuDetails.push(`批大小 ${display(run.gpu_batch)}`);
    setText("gpuDetails", gpuDetails.join(" · ") || "—");
    const ram = fraction(hardware.ram_used_gib, hardware.ram_total_gib);
    setText("ramValue", display(ram, "%"));
    meter("ramMeter", ram);
    setText("ramDetails", finite(hardware.ram_used_gib) && finite(hardware.ram_total_gib)
        ? `${countPair(hardware.ram_used_gib, hardware.ram_total_gib)} GiB`
        : "—");
}

function svgElement(name, attributes, label) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
    if (label !== undefined) element.textContent = label;
    return element;
}

const chart = byId("throughputChart");
let drawnChart;
let chartTooltip;
let chartSignature = "";

function renderChart(series) {
    const points = series.map((point) => ({ at: timestamp(point.at), rate: point.per_minute }))
        .filter(({ at, rate }) => finite(at) && finite(rate) && rate >= 0)
        .sort((left, right) => left.at - right.at);
    const width = Math.max(240, chart.clientWidth || 620);
    const height = Math.max(150, chart.clientHeight || 208);
    const signature = JSON.stringify([width, height, points]);
    if (signature === chartSignature) return;
    chartSignature = signature;
    const focusedAt = document.activeElement?.getAttribute("data-sample-at");
    drawnChart?.remove();
    chartTooltip?.remove();
    byId("chartEmpty").hidden = points.length > 0;
    if (!points.length) return;

    const left = 42;
    const right = width - 24;
    const top = 16;
    const bottom = height - 29;
    const peak = Math.max(1, ...points.map(({ rate }) => rate));
    const magnitude = 10 ** Math.floor(Math.log10(peak / 4));
    const step = Math.ceil(peak / (4 * magnitude)) * magnitude;
    const first = points[0].at;
    const last = points[points.length - 1].at;
    const x = (at) => first === last ? (left + right) / 2 : left + (right - left) * (at - first) / (last - first);
    const y = (rate) => bottom - (bottom - top) * rate / (4 * step);
    const stroke = "var(--chart-line, #56d8ba)";
    drawnChart = svgElement("svg", {
        viewBox: `0 0 ${width} ${height}`, width: "100%", height: "100%",
        role: "group", "aria-label": "候选评估速率，单位为候选每分钟",
    });
    drawnChart.style.display = "block";

    const defs = svgElement("defs", {});
    const gradient = svgElement("linearGradient", { id: "throughputGradient", x2: "0%", y2: "100%" });
    gradient.append(
        svgElement("stop", { offset: "0%", "stop-color": stroke, "stop-opacity": 0.28 }),
        svgElement("stop", { offset: "100%", "stop-color": stroke, "stop-opacity": 0.02 }),
    );
    defs.append(gradient);
    drawnChart.append(defs);
    const axes = svgElement("g", { fill: "var(--chart-label, #9ab3af)", "font-size": 11, "aria-hidden": true });
    for (let index = 0; index <= 4; index++) {
        const rate = index * step;
        axes.append(
            svgElement("line", {
                x1: left, x2: right, y1: y(rate), y2: y(rate), stroke: "var(--chart-grid, #253e3c)",
                "stroke-dasharray": index ? "4 4" : "none",
            }),
            svgElement("text", { x: left - 9, y: y(rate) + 4, "text-anchor": "end" }, display(rate)),
        );
    }
    const tickCount = first === last ? 1 : width < 380 ? 3 : 5;
    for (let index = 0; index < tickCount; index++) {
        const at = tickCount === 1 ? first : first + (last - first) * index / (tickCount - 1);
        axes.append(svgElement("text", {
            x: x(at), y: height - 8, "text-anchor": "middle",
        }, chartClock.format(at)));
    }
    drawnChart.append(axes);

    if (points.length > 1) {
        const coordinates = points.map(({ at, rate }) => `${x(at)},${y(rate)}`).join(" ");
        drawnChart.append(
            svgElement("polygon", {
                points: `${left},${bottom} ${coordinates} ${right},${bottom}`,
                fill: "url(#throughputGradient)", "aria-hidden": true,
            }),
            svgElement("polyline", {
                points: coordinates, fill: "none", stroke, "stroke-width": 2.5,
                "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": true,
            }),
        );
    }
    const guide = svgElement("line", {
        y1: top, y2: bottom, stroke, "stroke-dasharray": "3 5", opacity: 0, "aria-hidden": true,
    });
    drawnChart.append(guide);
    chartTooltip = document.createElement("div");
    chartTooltip.className = "chart-tooltip";
    chartTooltip.hidden = true;
    chartTooltip.setAttribute("aria-hidden", "true");
    Object.assign(chartTooltip.style, {
        position: "absolute", transform: "translate(-50%, -100%)", pointerEvents: "none",
        padding: "5px 8px", whiteSpace: "nowrap", fontSize: "12px", zIndex: 1,
        color: "var(--chart-tooltip-text, #e3f8f2)", background: "var(--chart-tooltip-bg, #203a37)",
        borderRadius: "5px",
    });

    let restoreFocus;
    for (const { at, rate } of points) {
        const cx = x(at);
        const cy = y(rate);
        const label = `${clock.format(at)} · ${display(rate)} 候选/分钟`;
        const point = svgElement("g", {
            tabindex: 0, role: "img", "aria-label": label, "data-sample-at": at,
        });
        const marker = svgElement("circle", {
            cx, cy, r: points.length === 1 ? 4.5 : 2.6, fill: stroke,
            stroke: "var(--chart-point-border, #101e1b)", "stroke-width": 1,
        });
        point.append(svgElement("circle", { cx, cy, r: 9, fill: "transparent", cursor: "crosshair" }), marker);
        const show = () => {
            marker.setAttribute("r", "5");
            guide.setAttribute("x1", cx);
            guide.setAttribute("x2", cx);
            guide.setAttribute("opacity", 0.5);
            chartTooltip.textContent = label;
            chartTooltip.style.left = `${Math.max(83, Math.min(width - 83, cx)) / width * 100}%`;
            chartTooltip.style.top = `${Math.max(30, cy - 9) / height * 100}%`;
            chartTooltip.hidden = false;
        };
        const hide = () => {
            marker.setAttribute("r", points.length === 1 ? "4.5" : "2.6");
            guide.setAttribute("opacity", 0);
            chartTooltip.hidden = true;
        };
        point.addEventListener("pointerenter", show);
        point.addEventListener("pointerleave", () => {
            if (document.activeElement !== point) hide();
        });
        point.addEventListener("focus", show);
        point.addEventListener("blur", hide);
        point.addEventListener("keydown", (event) => {
            if (event.key === "Escape") hide();
        });
        drawnChart.append(point);
        if (String(at) === focusedAt) restoreFocus = point;
    }
    chart.append(drawnChart, chartTooltip);
    restoreFocus?.focus({ preventScroll: true });
}

const logControls = { stdout: byId("logStdout"), stderr: byId("logStderr") };
const logBox = byId("logOutput");
const currentLogs = () => snapshot?.logs?.[selectedLog] ?? [];

function renderLogs(follow = false) {
    for (const [log, button] of Object.entries(logControls)) {
        if (button.getAttribute("role") === "tab") {
            button.setAttribute("aria-selected", String(log === selectedLog));
            button.tabIndex = log === selectedLog ? 0 : -1;
        } else {
            button.setAttribute("aria-pressed", String(log === selectedLog));
        }
    }
    const lines = currentLogs();
    byId("copyLogs").disabled = lines.length === 0;
    const content = lines.length ? lines.join("\n") : selectedLog === "stdout" ? "暂无标准输出" : "暂无错误输出";
    if (logBox.textContent === content && !follow) return;
    const scrollTop = logBox.scrollTop;
    const atEnd = logBox.scrollHeight - logBox.clientHeight - scrollTop <= 6;
    logBox.textContent = content;
    logBox.scrollTop = follow || atEnd ? logBox.scrollHeight : scrollTop;
}

function selectLog(log) {
    if (selectedLog === log) return;
    selectedLog = log;
    window.clearTimeout(copyReset);
    setButtonLabel("copyLogs", "复制");
    renderLogs(true);
}

for (const [log, button] of Object.entries(logControls)) {
    button.addEventListener("click", () => selectLog(log));
    button.addEventListener("keydown", (event) => {
        if (button.getAttribute("role") !== "tab") return;
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === "Home" ? "stdout" : event.key === "End" ? "stderr"
            : log === "stdout" ? "stderr" : "stdout";
        selectLog(next);
        logControls[next].focus();
    });
}

async function copyLogs() {
    const content = currentLogs().join("\n");
    if (!content) return;
    const copiedLog = selectedLog;
    let copied = false;
    try {
        await navigator.clipboard.writeText(content);
        copied = true;
    } catch {
        const selection = document.createElement("textarea");
        selection.value = content;
        selection.readOnly = true;
        Object.assign(selection.style, { position: "fixed", opacity: 0 });
        document.body.append(selection);
        const focused = document.activeElement;
        try {
            selection.select();
            copied = document.execCommand("copy");
        } catch {
            copied = false;
        } finally {
            selection.remove();
            focused?.focus({ preventScroll: true });
        }
    }
    if (selectedLog !== copiedLog) return;
    setButtonLabel("copyLogs", copied ? "已复制" : "复制失败");
    window.clearTimeout(copyReset);
    copyReset = window.setTimeout(() => setButtonLabel("copyLogs", "复制"), 1600);
}

async function refresh() {
    if (refreshing) return;
    refreshing = true;
    byId("refreshNow").disabled = true;
    const controller = new AbortController();
    const deadline = window.setTimeout(() => controller.abort(), 8000);
    try {
        const response = await fetch("/api/status", { cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(`状态请求返回 HTTP ${response.status}`);
        const data = await response.json();
        if (!data?.run || !data.throughput || !data.hardware || !Array.isArray(data.arms)) {
            throw new Error("状态接口返回的数据结构不完整");
        }
        snapshot = data;
        renderRun(data.run, data.throughput, data.updated_at);
        renderArms(data.arms);
        renderHardware(data.hardware, data.run);
        renderChart(data.throughput.series ?? []);
        renderLogs();
        refreshError = "";
    } catch (error) {
        refreshError = error.name === "AbortError" ? "状态请求超时" : error.message;
    } finally {
        window.clearTimeout(deadline);
        refreshing = false;
        byId("refreshNow").disabled = false;
        renderConnection();
    }
}

byId("pauseRefresh").addEventListener("click", () => {
    pollingPaused = !pollingPaused;
    byId("pauseRefresh").setAttribute("aria-pressed", String(pollingPaused));
    setButtonLabel("pauseRefresh", pollingPaused ? "恢复刷新" : "暂停刷新");
    renderConnection();
    if (!pollingPaused) void refresh();
});
byId("refreshNow").addEventListener("click", refresh);
byId("copyLogs").addEventListener("click", copyLogs);
if ("ResizeObserver" in window) {
    const chartResize = new ResizeObserver(() => renderChart(snapshot?.throughput.series ?? []));
    chartResize.observe(chart);
}
renderLogs();
void refresh();
window.setInterval(() => {
    if (!pollingPaused) void refresh();
}, refreshInterval);
