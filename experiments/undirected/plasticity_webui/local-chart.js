const arms = [
  ["control", "原规则", "#d4f983"],
  ["local", "逐步局部规则", "#69d6c3"],
  ["modulated", "接收节点调制", "#91afff"],
];
const charts = new WeakMap();
const font = '12px Consolas, "Microsoft YaHei UI", sans-serif';
const fullNumber = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 });
const percentage = new Intl.NumberFormat("zh-CN", { style: "percent", maximumFractionDigits: 2 });

// Match the legacy chart's tricube-weighted local linear regression.
function fitSegment(points) {
  const span = Math.min(points.length, 61, Math.max(5, Math.ceil(points.length / 9)));
  const half = Math.floor(span / 2);
  return points.map(([generation], index) => {
    const start = Math.max(0, Math.min(points.length - span, index - half));
    const end = start + span - 1;
    const radius = 1 + Math.max(generation - points[start][0], points[end][0] - generation);
    let sum = 0;
    let sumX = 0;
    let sumY = 0;
    let sumXX = 0;
    let sumXY = 0;
    for (let i = start; i <= end; i += 1) {
      const [x, y] = points[i];
      const dx = x - generation;
      const weight = (1 - (Math.abs(dx) / radius) ** 3) ** 3;
      sum += weight;
      sumX += weight * dx;
      sumY += weight * y;
      sumXX += weight * dx * dx;
      sumXY += weight * dx * y;
    }
    const determinant = sum * sumXX - sumX * sumX;
    return [generation, determinant > 0 ? (sumY * sumXX - sumX * sumXY) / determinant : sumY / sum];
  });
}

function model(history, metric, scope, fit) {
  const specs = metric.series ?? arms.map(([arm, label, color]) => ({ arm, label, color, key: metric.key }));
  const rows = new Map();
  for (const row of history) {
    const { arm, generation } = row ?? {};
    if (!Number.isSafeInteger(generation) || generation < 1) continue;
    if (!rows.has(arm)) rows.set(arm, new Map());
    rows.get(arm).set(generation, row);
  }
  const generations = new Set();
  let min = Infinity;
  let max = -Infinity;
  const series = specs.map((spec) => {
    const raw = new Map();
    const fitted = new Map();
    const segments = [];
    const armRows = [...(rows.get(spec.arm) ?? [])].sort(([left], [right]) => left - right);
    let segment = [];
    let previous = null;
    for (const [generation, row] of armRows) {
      const value = row[spec.scope ?? scope]?.[spec.key];
      if (!Number.isFinite(value)) continue;
      if (generation !== previous + 1 || !segment.length) {
        segment = [];
        segments.push(segment);
      }
      segment.push([generation, value]);
      previous = generation;
      raw.set(generation, value);
      generations.add(generation);
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    const curves = segments.map((points) => {
      const smooth = fit && points.length >= 3 ? fitSegment(points) : [];
      for (const [generation, value] of smooth) {
        fitted.set(generation, value);
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
      return { points, smooth };
    });
    return { ...spec, raw, fitted, curves, latest: previous };
  });
  return { series, generations: [...generations].sort((a, b) => a - b), min, max };
}

function valueLabel(value, unit) {
  if (!Number.isFinite(value)) return "—";
  if (unit === "percent") return percentage.format(value);
  if (value && Math.abs(value) < 0.000001) return value.toExponential(2);
  return fullNumber.format(value);
}

function tickLabel(value, step, unit) {
  const epsilon = Math.abs(step) * 1e-10;
  if (Math.abs(value) < epsilon) value = 0;
  const magnitude = Math.abs(value);
  let scale = unit === "percent" ? 0.01 : 1;
  let suffix = unit === "percent" ? "%" : "";
  if (unit !== "percent") {
    if (magnitude >= 1e9) [scale, suffix] = [1e9, "B"];
    else if (magnitude >= 1e6) [scale, suffix] = [1e6, "M"];
    else if (magnitude >= 1e3) [scale, suffix] = [1e3, "k"];
  }
  const scaled = value / scale;
  const digits = Math.min(7, Math.max(0, Math.ceil(-Math.log10(step / scale)) + 1));
  if (scaled && Math.abs(scaled) < 1e-7) return scaled.toExponential(1) + suffix;
  return Number(scaled.toFixed(digits)).toString() + suffix;
}

function nearestIndex(generations, target) {
  let low = 0;
  let high = generations.length - 1;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (generations[middle] >= target) high = middle;
    else low = middle + 1;
  }
  return low && target - generations[low - 1] <= generations[low] - target ? low - 1 : low;
}

function plotPath(ctx, points, x, y, showDots) {
  ctx.beginPath();
  points.forEach(([generation, value], index) => {
    if (index === 0) ctx.moveTo(x(generation), y(value));
    else ctx.lineTo(x(generation), y(value));
  });
  ctx.stroke();
  if (showDots || points.length === 1) {
    for (const [generation, value] of points) {
      ctx.beginPath();
      ctx.arc(x(generation), y(value), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function renderLegend(state) {
  const { legend, metric, data, fit } = state;
  const doc = legend.ownerDocument;
  const tag = ["UL", "OL"].includes(legend.tagName) ? "li" : "span";
  const nodes = data.series.map(({ label, color, raw, latest }) => {
    const item = doc.createElement(tag);
    item.className = "local-chart-legend-item";
    const swatch = doc.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.color = color;
    swatch.setAttribute("aria-hidden", "true");
    item.append(swatch, latest === null ? `${label}　—`
      : `${label}　${valueLabel(raw.get(latest), metric.unit)}（第 ${fullNumber.format(latest)} 代）`);
    return item;
  });
  if (fit && data.generations.length) {
    const key = doc.createElement(tag);
    key.className = "local-chart-fit-key";
    key.textContent = data.series.some(({ fitted }) => fitted.size)
      ? "亮线为拟合，淡线为实测。少于 3 个连续实测点的区段仅显示实测。"
      : "线性拟合至少需要 3 个连续实测点。";
    nodes.push(key);
  }
  legend.replaceChildren(...nodes);
}

function renderTooltip(state, selected, x, width) {
  const { tooltip, metric, data, fit } = state;
  if (!tooltip) return;
  const doc = tooltip.ownerDocument;
  const title = doc.createElement("div");
  title.textContent = `第 ${fullNumber.format(selected)} 代`;
  const rows = data.series.map(({ label, color, raw, fitted }) => {
    const row = doc.createElement("div");
    const name = doc.createElement("span");
    name.style.color = color;
    name.textContent = label + "　";
    const actual = valueLabel(raw.get(selected), metric.unit);
    const fitValue = fitted.has(selected) ? valueLabel(fitted.get(selected), metric.unit)
      : raw.has(selected) ? "样本不足" : "—";
    row.append(name, fit ? `实测 ${actual}　拟合 ${fitValue}` : actual);
    return row;
  });
  tooltip.replaceChildren(title, ...rows);
  tooltip.style.boxSizing = "border-box";
  tooltip.style.maxWidth = `${Math.max(0, width - 8)}px`;
  tooltip.hidden = false;
  const tooltipWidth = tooltip.offsetWidth;
  const left = x + 12 + tooltipWidth <= width ? x + 12 : x - 12 - tooltipWidth;
  tooltip.style.left = `${Math.max(4, Math.min(left, width - tooltipWidth - 4))}px`;
}

function draw(state) {
  const { canvas, ctx, data, metric, tooltip } = state;
  const { series, generations, min, max } = data;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  state.axis = null;
  if (tooltip) tooltip.hidden = true;
  if (!ctx || !width || !height) return;
  const dpr = canvas.ownerDocument.defaultView?.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(width * dpr));
  const pixelHeight = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  ctx.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = font;
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#a0b1a7";
  ctx.textAlign = "center";
  if (!generations.length) {
    ctx.fillText("暂无已封存的指标数据", width / 2, height / 2);
    return;
  }

  const minX = generations[0];
  const maxX = generations.at(-1);
  const percent = metric.unit === "percent";
  const margin = min === max ? Math.max(Math.abs(min) * 0.04, percent ? 0.005 : 0.01)
    : (max - min) * 0.08;
  const minY = min >= 0 ? Math.max(0, min - margin) : min - margin;
  const maxY = percent && max <= 1 ? Math.min(1, max + margin) : max + margin;
  const intervals = height < 180 ? 3 : 4;
  const step = (maxY - minY) / intervals;
  const yTicks = Array.from({ length: intervals + 1 }, (_, i) => minY + i * step);
  const yLabels = yTicks.map((value) => tickLabel(value, step, metric.unit));
  const generationLabel = (generation) => `第 ${fullNumber.format(generation)} 代`;
  const tickWidth = ctx.measureText(generationLabel(maxX)).width;
  const labelWidth = Math.max(...yLabels.map(label => ctx.measureText(label).width));
  const left = Math.max(labelWidth + 13, Math.ceil(tickWidth / 2) + 5);
  const right = width - Math.ceil(tickWidth / 2) - 5;
  const top = 13;
  const bottom = height - 26;
  if (left >= right || top >= bottom) return;
  const x = generation => minX === maxX ? (left + right) / 2
    : left + (generation - minX) / (maxX - minX) * (right - left);
  const y = value => bottom - (value - minY) / (maxY - minY) * (bottom - top);
  state.axis = { left, right, minX, maxX };

  ctx.lineWidth = 1;
  ctx.strokeStyle = "#35433b";
  ctx.textAlign = "right";
  ctx.beginPath();
  yTicks.forEach((value, index) => {
    const rowY = y(value);
    ctx.moveTo(left, rowY);
    ctx.lineTo(right, rowY);
    ctx.fillText(yLabels[index], left - 8, rowY);
  });
  ctx.stroke();
  ctx.textAlign = "center";
  const tickCount = Math.max(1, Math.min(6, Math.floor((right - left) / (tickWidth + 15))));
  const xTicks = new Set(Array.from({ length: tickCount + 1 },
    (_, index) => Math.round(minX + (maxX - minX) * index / tickCount)));
  for (const generation of xTicks) ctx.fillText(generationLabel(generation), x(generation), bottom + 15);

  const selected = state.hover ?? state.pinned;
  if (selected !== null) {
    ctx.strokeStyle = "#83958a";
    ctx.beginPath();
    ctx.moveTo(x(selected), top);
    ctx.lineTo(x(selected), bottom);
    ctx.stroke();
  }
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const { color, curves } of series) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.8;
    for (const { points, smooth } of curves) {
      ctx.globalAlpha = smooth.length ? 0.4 : 1;
      plotPath(ctx, points, x, y, generations.length <= 24);
    }
  }
  ctx.globalAlpha = 1;
  ctx.lineWidth = 2.6;
  for (const { color, curves } of series) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    for (const { smooth } of curves) {
      if (smooth.length) plotPath(ctx, smooth, x, y, false);
    }
  }
  if (selected !== null) {
    for (const { color, raw } of series) {
      if (!raw.has(selected)) continue;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(x(selected), y(raw.get(selected)), 4, 0, 2 * Math.PI);
      ctx.stroke();
    }
    renderTooltip(state, selected, x(selected), width);
  }
}

function selectByPointer(state, event, pin) {
  if (event.pointerType === "touch" && !pin) return;
  const { axis, data, canvas } = state;
  if (!axis) return;
  const { left, right, minX, maxX } = axis;
  const rect = canvas.getBoundingClientRect();
  const pointer = (event.clientX - rect.left) * canvas.clientWidth / rect.width;
  const target = minX + (pointer - left) / (right - left) * (maxX - minX);
  const generation = pointer < left - 12 || pointer > right + 12 ? null
    : data.generations[nearestIndex(data.generations, target)];
  if (pin) {
    state.pinned = state.pinned === generation ? null : generation;
    state.hover = null;
  } else {
    if (state.hover === generation) return;
    state.hover = generation;
  }
  draw(state);
}

function selectByKeyboard(state, event) {
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  const { generations } = state.data;
  if (!generations.length) return;
  const current = state.hover ?? state.pinned;
  let index = current === null ? generations.length - 1 : nearestIndex(generations, current);
  switch (event.key) {
    case "Escape": index = -1; break;
    case "Home": index = 0; break;
    case "End": index = generations.length - 1; break;
    case "ArrowLeft": if (current !== null) index = Math.max(0, index - 1); break;
    case "ArrowRight": if (current !== null) index = Math.min(generations.length - 1, index + 1); break;
    default: return;
  }
  event.preventDefault();
  state.hover = null;
  state.pinned = generations[index] ?? null;
  draw(state);
}

export function renderChart({ canvas, tooltip, legend, history = [], metric, scope = "champion", fit = false }) {
  let state = charts.get(canvas);
  if (!state) {
    state = { canvas, ctx: canvas.getContext("2d"), hover: null, pinned: null };
    charts.set(canvas, state);
    if (canvas.tabIndex < 0) canvas.tabIndex = 0;
    canvas.addEventListener("pointermove", (event) => selectByPointer(state, event, false));
    canvas.addEventListener("pointerleave", () => {
      if (state.hover === null) return;
      state.hover = null;
      draw(state);
    });
    canvas.addEventListener("click", (event) => selectByPointer(state, event, true));
    canvas.addEventListener("keydown", (event) => selectByKeyboard(state, event));
    const ResizeObserver = canvas.ownerDocument.defaultView?.ResizeObserver;
    if (ResizeObserver) {
      const observer = new ResizeObserver(() => draw(state));
      observer.observe(canvas);
    }
  }
  Object.assign(state, { tooltip, legend, metric, fit, data: model(history ?? [], metric, scope, fit) });
  if (!state.data.generations.includes(state.hover)) state.hover = null;
  if (!state.data.generations.includes(state.pinned)) state.pinned = null;
  canvas.setAttribute("aria-label", `${metric.label}。按左右方向键查看已封存代，Esc 清除选择。`);
  if (legend) renderLegend(state);
  draw(state);
}
