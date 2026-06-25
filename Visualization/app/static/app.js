const CATEGORY_STYLES = {
  car: { color: "#2563eb", shape: "rect", weight: 2 },
  van: { color: "#0891b2", shape: "rect", weight: 2 },
  truck: { color: "#b91c1c", shape: "heavyRect", weight: 4 },
  bus: { color: "#7c3aed", shape: "heavyRect", weight: 4 },
  freight_car: { color: "#92400e", shape: "heavyRect", weight: 4 },
  bicycle: { color: "#059669", shape: "diamond", weight: 2 },
  tricycle: { color: "#84cc16", shape: "diamond", weight: 2 },
  "awning-tricycle": { color: "#c026d3", shape: "diamond", weight: 2 },
  motor: { color: "#f97316", shape: "diamond", weight: 2 },
  pedestrian: { color: "#dc2626", shape: "circle", weight: 2 },
  people: { color: "#111827", shape: "circle", weight: 2 },
};

const DEFAULT_STYLE = { color: "#475467", shape: "rect", weight: 2 };
const LANE_PALETTE = [
  "#2dd4bf",
  "#60a5fa",
  "#fbbf24",
  "#a78bfa",
  "#34d399",
  "#fb7185",
  "#38bdf8",
  "#f472b6",
  "#a3e635",
  "#f97316",
  "#818cf8",
  "#14b8a6",
  "#eab308",
  "#22c55e",
  "#06b6d4",
  "#c084fc",
];
const ROAD_MASK_COLOR = "rgba(0, 0, 0, 0.88)";
const DEFAULT_TRAIL_LENGTH = 50;
const STATIC_ARROW_SUPPRESSED_CLASSES = new Set(["car", "truck", "bus", "freight_car", "van", "motor", "tricycle", "awning-tricycle"]);
const HEADING_CONFIG = {
  use_legacy_visual_heading_refinement: false,
  heading_smooth_window: 8,
  min_motion_threshold: 2.0,
  arrow_length_scale: 1.0,
  arrow_min_length: 6,
  arrow_max_length: 160,
};

const state = {
  datasets: [],
  datasetId: null,
  version: null,
  metadata: null,
  tracks: [],
  objects: [],
  objectInfoMap: new Map(),
  frames: [],
  laneGeometry: { available: false, shapes: [] },
  frameMap: new Map(),
  objectMap: new Map(),
  headingCache: new Map(),
  frameIds: [],
  classNames: [],
  selectedClasses: new Set(),
  currentFrame: 0,
  minFrame: 0,
  maxFrame: 0,
  playing: false,
  speed: 1,
  trailLength: DEFAULT_TRAIL_LENGTH,
  objectFilter: "",
  selectedObject: null,
  hoveredLaneKey: null,
  background: null,
  lastTick: 0,
  frameAccumulator: 0,
  canvasBox: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0, width: 0, height: 0 },
};

const els = {
  datasetSummary: document.getElementById("datasetSummary"),
  scanButton: document.getElementById("scanButton"),
  forceScan: document.getElementById("forceScan"),
  datasetSearch: document.getElementById("datasetSearch"),
  datasetList: document.getElementById("datasetList"),
  objectInput: document.getElementById("objectInput"),
  frameInput: document.getElementById("frameInput"),
  jumpButton: document.getElementById("jumpButton"),
  selectAllClasses: document.getElementById("selectAllClasses"),
  clearAllClasses: document.getElementById("clearAllClasses"),
  classSelectState: document.getElementById("classSelectState"),
  classFilters: document.getElementById("classFilters"),
  classStatsTotal: document.getElementById("classStatsTotal"),
  classStatsList: document.getElementById("classStatsList"),
  dimensionSummary: document.getElementById("dimensionSummary"),
  lengthDistributionLayers: document.getElementById("lengthDistributionLayers"),
  dimensionTableBody: document.getElementById("dimensionTableBody"),
  bboxToggle: document.getElementById("bboxToggle"),
  labelToggle: document.getElementById("labelToggle"),
  laneToggle: document.getElementById("laneToggle"),
  keepUnrelatedToggle: document.getElementById("keepUnrelatedToggle"),
  trailRange: document.getElementById("trailRange"),
  trailValue: document.getElementById("trailValue"),
  frameMetric: document.getElementById("frameMetric"),
  objectMetric: document.getElementById("objectMetric"),
  statusText: document.getElementById("statusText"),
  prevButton: document.getElementById("prevButton"),
  playButton: document.getElementById("playButton"),
  nextButton: document.getElementById("nextButton"),
  speedSelect: document.getElementById("speedSelect"),
  timeline: document.getElementById("timeline"),
  canvas: document.getElementById("trackCanvas"),
  canvasWrap: document.getElementById("canvasWrap"),
  tooltip: document.getElementById("tooltip"),
  emptyState: document.getElementById("emptyState"),
  legendList: document.getElementById("legendList"),
  detailContent: document.getElementById("detailContent"),
  scanResult: document.getElementById("scanResult"),
};

const ctx = els.canvas.getContext("2d");

function asNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function styleFor(className) {
  return CATEGORY_STYLES[className] || DEFAULT_STYLE;
}

function api(path, options) {
  return fetch(path, options).then(async (res) => {
    const contentType = res.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await res.json() : await res.text();
    if (!res.ok) throw new Error(payload.error || payload || `HTTP ${res.status}`);
    return payload;
  });
}

async function loadDatasets() {
  const payload = await api("/api/datasets");
  state.datasets = payload.converted || [];
  renderDatasets();
  const firstAvailable = state.datasets.find((item) => item.is_available !== false);
  if (!state.datasets.length) {
    state.objects = [];
    els.datasetSummary.textContent = "没有已转换数据集";
    els.emptyState.classList.remove("hidden");
    renderClassStats();
    renderDimensionAnalysis();
  } else if (!firstAvailable) {
    state.objects = [];
    els.datasetSummary.textContent = "Final Data 缺少必要 CSV";
    els.emptyState.classList.remove("hidden");
    renderClassStats();
    renderDimensionAnalysis();
  } else if (!state.datasetId) {
    state.objects = [];
    els.datasetSummary.textContent = "请选择数据集";
    els.emptyState.classList.remove("hidden");
    renderClassStats();
    renderDimensionAnalysis();
  }
}

function renderDatasets() {
  const keyword = els.datasetSearch.value.trim().toLowerCase();
  const items = state.datasets.filter((item) => {
    const text = `${item.dataset_id} ${item.version} ${item.display_name || ""}`.toLowerCase();
    return !keyword || text.includes(keyword);
  });

  els.datasetList.innerHTML = "";
  if (!items.length) {
    const div = document.createElement("div");
    div.className = "dataset-meta";
    div.textContent = state.datasets.length ? "没有匹配的数据集" : "Final Data 为空";
    els.datasetList.appendChild(div);
    return;
  }

  for (const item of items) {
    const btn = document.createElement("button");
    const active = item.dataset_id === state.datasetId && item.version === state.version;
    btn.className = `dataset-item ${active ? "active" : ""}`;
    btn.disabled = item.is_available === false;
    const metaText = item.is_available === false ? "不可用" : "可用";
    btn.innerHTML = `
      <div class="dataset-name">${escapeHtml(item.dataset_id)}</div>
      <div class="dataset-version">${escapeHtml(item.version)}</div>
      <div class="dataset-meta">${metaText}</div>
      ${item.is_available === false ? `<div class="dataset-meta">缺少 ${escapeHtml((item.missing_files || []).join(", "))}</div>` : ""}
    `;
    btn.addEventListener("click", () => {
      if (item.is_available !== false) loadDataset(item.dataset_id, item.version);
    });
    els.datasetList.appendChild(btn);
  }
}

async function loadDataset(datasetId, version) {
  setStatus("读取数据中...");
  state.datasetId = datasetId;
  state.version = version;
  state.objects = [];
  renderClassStats();
  renderDimensionAnalysis();
  state.metadata = await api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/metadata`);
  const [tracks, frames, objects, lanes] = await Promise.all([
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/tracks`),
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/frames`),
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/objects`),
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/lanes`).catch((err) => ({
      available: false,
      reason: String(err.message || err),
      shapes: [],
    })),
  ]);

  state.tracks = tracks.map(normalizeTrack).filter(Boolean);
  state.frames = frames.map((row) => ({
    ...row,
    frame_id: asNumber(row.frame_id),
    timestamp: asNumber(row.timestamp),
    width: asNumber(row.width),
    height: asNumber(row.height),
    num_objects: asNumber(row.num_objects) || 0,
  }));
  state.objects = objects.map(normalizeObject);
  state.laneGeometry = normalizeLaneGeometry(lanes);
  buildIndexes();
  renderDatasets();
  buildClassFilters();
  renderLegend();
  renderClassStats();
  renderDimensionAnalysis();
  await loadBackground(datasetId, version);

  state.currentFrame = state.minFrame;
  state.objectFilter = "";
  state.selectedObject = null;
  state.hoveredLaneKey = null;
  state.headingCache = new Map();
  els.objectInput.value = "";
  els.frameInput.value = "";
  els.timeline.min = String(state.minFrame);
  els.timeline.max = String(state.maxFrame);
  els.timeline.value = String(state.currentFrame);
  els.emptyState.classList.add("hidden");
  els.datasetSummary.textContent = `${datasetId} · ${state.frameIds.length} 帧 · ${state.objects.length} 目标`;
  setStatus("");
  resizeCanvas();
  draw();
}

function normalizeTrack(row) {
  const objectId = asNumber(row.object_id);
  const frameId = asNumber(row.frame_id);
  if (objectId === null || frameId === null) return null;
  const out = {
    ...row,
    frame_id: frameId,
    object_id: objectId,
    raw_object_id: asNumber(row.raw_object_id),
    width: asNumber(row.width),
    length: asNumber(row.length),
    total_frames: asNumber(row.total_frames),
    confidence: asNumber(row.confidence),
    x1: asNumber(row.x1),
    y1: asNumber(row.y1),
    x2: asNumber(row.x2),
    y2: asNumber(row.y2),
    cx: asNumber(row.cx),
    cy: asNumber(row.cy),
    width: asNumber(row.width),
    height: asNumber(row.height),
    raw_mean_width: asNumber(row.raw_mean_width),
    raw_mean_height: asNumber(row.raw_mean_height),
    corrected_width: asNumber(row.corrected_width),
    corrected_height: asNumber(row.corrected_height),
    missing_ratio: asNumber(row.missing_ratio),
    is_interpolated: row.is_interpolated === true || row.is_interpolated === "true",
    angle_deg: asNumber(row.angle_deg),
    heading_screen_rad: asNumber(row.heading_screen_rad),
    category_id: asNumber(row.category_id),
    lane_id: asNumber(row.lane_id),
  };
  for (let i = 1; i <= 4; i += 1) {
    out[`q${i}_x`] = asNumber(row[`q${i}_x`]);
    out[`q${i}_y`] = asNumber(row[`q${i}_y`]);
  }
  return out;
}

function normalizeObject(row) {
  return {
    ...row,
    object_id: asNumber(row.object_id),
    raw_object_id: asNumber(row.raw_object_id),
    start_frame: asNumber(row.start_frame),
    end_frame: asNumber(row.end_frame),
    displacement: asNumber(row.displacement),
    path_length: asNumber(row.path_length),
    stationary_extent: asNumber(row.stationary_extent),
    mean_speed: asNumber(row.mean_speed),
    max_speed: asNumber(row.max_speed),
    static_ratio: asNumber(row.static_ratio),
    lane_id: asNumber(row.lane_id),
    startLaneId: asNumber(row.startLaneId),
    endLaneId: asNumber(row.endLaneId),
    is_static: String(row.is_static).toLowerCase() === "true",
  };
}

function normalizeLaneGeometry(lanes) {
  const geometry = lanes || { available: false, shapes: [] };
  const shapes = Array.isArray(geometry.shapes) ? geometry.shapes : [];
  return {
    ...geometry,
    shapes: shapes.map((shape, index) => ({
      ...shape,
      _key: `${shape.role || "shape"}:${shape.label || ""}:${index}`,
      points: Array.isArray(shape.points)
        ? shape.points
            .map((point) => ({ x: asNumber(point.x), y: asNumber(point.y) }))
            .filter((point) => point.x !== null && point.y !== null)
        : [],
    })),
  };
}

function buildIndexes() {
  state.frameMap = new Map();
  state.objectMap = new Map();
  state.objectInfoMap = new Map();
  for (const obj of state.objects) {
    state.objectInfoMap.set(obj.object_id, obj);
  }
  for (const row of state.tracks) {
    if (!state.frameMap.has(row.frame_id)) state.frameMap.set(row.frame_id, []);
    state.frameMap.get(row.frame_id).push(row);
    if (!state.objectMap.has(row.object_id)) state.objectMap.set(row.object_id, []);
    state.objectMap.get(row.object_id).push(row);
  }
  for (const rows of state.objectMap.values()) rows.sort((a, b) => a.frame_id - b.frame_id);
  state.frameIds = Array.from(new Set([...state.frames.map((f) => f.frame_id), ...state.frameMap.keys()]))
    .filter((v) => Number.isFinite(v))
    .sort((a, b) => a - b);
  state.minFrame = state.frameIds[0] || 0;
  state.maxFrame = state.frameIds[state.frameIds.length - 1] || 0;
  state.classNames = Array.from(new Set(state.tracks.map((r) => r.class_name || "unknown"))).sort();
  state.selectedClasses = new Set(state.classNames);
  updateClassSelectState();
}

async function loadBackground(datasetId, version) {
  state.background = null;
  try {
    const img = new Image();
    img.src = `/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/background?ts=${Date.now()}`;
    await img.decode();
    state.background = img;
  } catch {
    state.background = null;
  }
}

function buildClassFilters() {
  els.classFilters.innerHTML = "";
  for (const className of state.classNames) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = state.selectedClasses.has(className);
    input.dataset.className = className;
    input.addEventListener("change", () => {
      if (input.checked) state.selectedClasses.add(className);
      else state.selectedClasses.delete(className);
      updateClassSelectState();
      draw();
    });
    const span = document.createElement("span");
    span.textContent = className;
    label.appendChild(input);
    label.appendChild(span);
    els.classFilters.appendChild(label);
  }
  updateClassSelectState();
}

function updateClassSelectState() {
  const total = state.classNames.length;
  const selected = state.selectedClasses.size;
  if (!total) els.classSelectState.textContent = "未加载";
  else if (selected === total) els.classSelectState.textContent = "全选";
  else if (selected === 0) els.classSelectState.textContent = "全部隐藏";
  else els.classSelectState.textContent = `部分选中 ${selected}/${total}`;

  els.classFilters.querySelectorAll("input[type='checkbox']").forEach((input) => {
    input.checked = state.selectedClasses.has(input.dataset.className);
  });
}

function setAllClasses(selected) {
  state.selectedClasses = selected ? new Set(state.classNames) : new Set();
  updateClassSelectState();
  draw();
}

function computeClassStats() {
  const counts = new Map();
  for (const obj of state.objects) {
    const className = obj.class_name || "unknown";
    counts.set(className, (counts.get(className) || 0) + 1);
  }
  const total = state.objects.length;
  return Array.from(counts.entries())
    .map(([className, count]) => ({
      className,
      count,
      percent: total ? (count / total) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count || a.className.localeCompare(b.className));
}

function renderClassStats() {
  const total = state.objects.length;
  els.classStatsList.innerHTML = "";
  if (!total) {
    els.classStatsTotal.textContent = state.datasetId ? "当前数据集没有目标" : "未加载数据集";
    return;
  }

  els.classStatsTotal.textContent = `总目标 ${total}`;
  for (const item of computeClassStats()) {
    const style = styleFor(item.className);
    const row = document.createElement("div");
    row.className = "class-stat-row";
    row.innerHTML = `
      <div class="class-stat-head">
        <span class="class-stat-name">
          <span class="class-stat-dot" style="background:${style.color}"></span>
          <span>${escapeHtml(item.className)}</span>
        </span>
        <span class="class-stat-value">${item.count} · ${item.percent.toFixed(1)}%</span>
      </div>
      <div class="class-stat-track">
        <div class="class-stat-fill" style="width:${Math.max(0, Math.min(100, item.percent))}%; background:${style.color}"></div>
      </div>
    `;
    els.classStatsList.appendChild(row);
  }
}

function renderDimensionAnalysis() {
  if (!els.dimensionSummary || !els.dimensionTableBody || !els.lengthDistributionLayers) return;
  const dimensions = state.objects
    .filter((obj) => Number.isFinite(obj.length) || Number.isFinite(obj.width))
    .sort((a, b) => (a.object_id ?? Infinity) - (b.object_id ?? Infinity));
  const analysis = window.DimensionAnalysis.buildVehicleLengthDistribution(state.objects);
  if (!state.datasetId) {
    els.dimensionSummary.textContent = "未加载数据集";
  } else if (!analysis.total_count) {
    els.dimensionSummary.textContent = "当前数据集没有可统计的车辆长度";
  } else if (analysis.car_count) {
    els.dimensionSummary.textContent = `车辆 ${analysis.total_count} 辆 · car ${analysis.car_reference_min.toFixed(1)}–${analysis.car_reference_max.toFixed(1)} m：${analysis.car_reference_count}/${analysis.car_count} 辆（${fmt(analysis.car_reference_ratio * 100, 1)}%） · ${analysis.bin_width.toFixed(1)} m分箱`;
  } else {
    els.dimensionSummary.textContent = `车辆 ${analysis.total_count} 辆 · 当前无 car · ${analysis.bin_width.toFixed(1)} m分箱`;
  }

  els.lengthDistributionLayers.innerHTML = "";
  for (const group of analysis.groups) {
    const layer = document.createElement("div");
    layer.className = "length-distribution-layer";
    const header = document.createElement("div");
    header.className = "length-distribution-header";
    const overflowText = group.overflow_count ? ` · >${analysis.axis_max} m：${group.overflow_count} 辆` : "";
    header.innerHTML = `<strong>${escapeHtml(group.class_name)}</strong><span>${group.count} 辆 · 峰值 ${group.peak_label}：${group.peak_count} 辆${overflowText}</span>`;
    const canvas = document.createElement("canvas");
    canvas.className = "class-length-histogram";
    canvas.setAttribute("aria-label", `${group.class_name}车辆长度分布，共${group.count}辆`);
    layer.append(header, canvas);
    els.lengthDistributionLayers.appendChild(layer);
    drawClassLengthHistogram(canvas, group, analysis);
  }

  els.dimensionTableBody.innerHTML = "";
  for (const obj of dimensions) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${obj.object_id === null ? "" : obj.object_id}</td>
      <td>${escapeHtml(obj.class_name || "unknown")}</td>
      <td>${fmt(obj.length, 3)}</td>
      <td>${fmt(obj.width, 3)}</td>
    `;
    els.dimensionTableBody.appendChild(row);
  }
}

function drawClassLengthHistogram(canvas, group, analysis) {
  const cssWidth = Math.max(canvas.clientWidth || 300, 220);
  const cssHeight = 116;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  const chart = canvas.getContext("2d");
  chart.setTransform(ratio, 0, 0, ratio, 0, 0);
  chart.clearRect(0, 0, cssWidth, cssHeight);
  chart.fillStyle = "#ffffff";
  chart.fillRect(0, 0, cssWidth, cssHeight);
  const margin = { left: 24, right: 8, top: 14, bottom: 26 };
  const plotWidth = cssWidth - margin.left - margin.right;
  const plotHeight = cssHeight - margin.top - margin.bottom;
  const overflowWidth = 28;
  const regularPlotWidth = plotWidth - overflowWidth;
  const slot = regularPlotWidth / group.counts.length;
  const maxCount = Math.max(...group.counts, group.overflow_count, 1);
  let carReferenceX = null;

  if (group.class_name === "car") {
    const referenceStart = margin.left + (analysis.car_reference_min / analysis.axis_max) * regularPlotWidth;
    const referenceEnd = margin.left + (analysis.car_reference_max / analysis.axis_max) * regularPlotWidth;
    carReferenceX = margin.left + (analysis.car_reference_value / analysis.axis_max) * regularPlotWidth;
    chart.fillStyle = "rgba(34, 197, 94, 0.16)";
    chart.fillRect(referenceStart, margin.top, Math.max(2, referenceEnd - referenceStart), plotHeight);
    chart.strokeStyle = "#16a34a";
    chart.lineWidth = 1.5;
    chart.beginPath();
    chart.moveTo(carReferenceX, margin.top);
    chart.lineTo(carReferenceX, margin.top + plotHeight);
    chart.stroke();
  }

  chart.strokeStyle = "#d0d5dd";
  chart.lineWidth = 1;
  chart.beginPath();
  chart.moveTo(margin.left, margin.top);
  chart.lineTo(margin.left, margin.top + plotHeight);
  chart.lineTo(margin.left + plotWidth, margin.top + plotHeight);
  chart.stroke();

  const color = styleFor(group.class_name).color;
  group.counts.forEach((count, index) => {
    const height = (count / maxCount) * (plotHeight - 4);
    const x = margin.left + index * slot + 1;
    const y = margin.top + plotHeight - height;
    chart.fillStyle = color;
    chart.fillRect(x, y, Math.max(1, slot - 2), height);
    if (count > 0) {
      chart.fillStyle = "#344054";
      chart.font = "8px Segoe UI, Arial";
      chart.textAlign = "center";
      chart.fillText(String(count), x + Math.max(1, slot - 2) / 2, Math.max(margin.top + 8, y - 2));
    }
  });

  const overflowX = margin.left + regularPlotWidth + 4;
  const overflowBarWidth = Math.max(4, overflowWidth - 8);
  const overflowHeight = (group.overflow_count / maxCount) * (plotHeight - 4);
  if (group.overflow_count) {
    chart.fillStyle = "#dc2626";
    chart.fillRect(overflowX, margin.top + plotHeight - overflowHeight, overflowBarWidth, overflowHeight);
    chart.fillStyle = "#991b1b";
    chart.font = "8px Segoe UI, Arial";
    chart.textAlign = "center";
    chart.fillText(String(group.overflow_count), overflowX + overflowBarWidth / 2, Math.max(margin.top + 8, margin.top + plotHeight - overflowHeight - 2));
  }

  chart.fillStyle = "#667085";
  chart.font = "9px Segoe UI, Arial";
  for (let value = 0; value <= analysis.axis_max; value += 3) {
    const x = margin.left + (value / analysis.axis_max) * regularPlotWidth;
    chart.textAlign = value === analysis.axis_max ? "right" : value === 0 ? "left" : "center";
    chart.fillText(String(value), x, cssHeight - 9);
  }
  chart.textAlign = "center";
  chart.fillStyle = group.overflow_count ? "#991b1b" : "#667085";
  chart.fillText(`>${analysis.axis_max}`, overflowX + overflowBarWidth / 2, cssHeight - 9);
  if (carReferenceX !== null) {
    chart.fillStyle = "#15803d";
    chart.font = "9px Segoe UI, Arial";
    chart.textAlign = "center";
    chart.fillText("4.3", carReferenceX, margin.top + plotHeight + 10);
  }
}

function renderLegend() {
  els.legendList.innerHTML = "";
  for (const className of state.classNames) {
    const style = styleFor(className);
    const item = document.createElement("div");
    item.className = "legend-item";
    const symbol = document.createElement("span");
    symbol.className = `legend-symbol ${style.shape === "circle" ? "circle" : ""} ${style.shape === "diamond" ? "diamond" : ""} ${style.weight > 2 ? "heavy" : ""}`;
    symbol.style.color = style.color;
    item.appendChild(symbol);
    item.appendChild(document.createTextNode(className));
    els.legendList.appendChild(item);
  }
}

function resizeCanvas() {
  const rect = els.canvasWrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  els.canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function imageSize() {
  return {
    width: state.metadata?.image_width || state.background?.naturalWidth || 1920,
    height: state.metadata?.image_height || state.background?.naturalHeight || 1080,
  };
}

function computeCanvasBox() {
  const rect = els.canvas.getBoundingClientRect();
  const { width: iw, height: ih } = imageSize();
  const scale = Math.min(rect.width / iw, rect.height / ih);
  const width = iw * scale;
  const height = ih * scale;
  state.canvasBox = {
    scaleX: scale,
    scaleY: scale,
    offsetX: (rect.width - width) / 2,
    offsetY: (rect.height - height) / 2,
    width,
    height,
  };
  return state.canvasBox;
}

function worldToScreen(x, y) {
  const box = state.canvasBox;
  return { x: box.offsetX + x * box.scaleX, y: box.offsetY + y * box.scaleY };
}

function screenToWorld(x, y) {
  const box = state.canvasBox;
  return { x: (x - box.offsetX) / box.scaleX, y: (y - box.offsetY) / box.scaleY };
}

function playbackFraction() {
  return state.playing ? Math.max(0, Math.min(1, state.frameAccumulator || 0)) : 0;
}

function interpolateNumber(a, b, t) {
  if (a === null || b === null) return a;
  return a + (b - a) * t;
}

function interpolateAngleRad(a, b, t) {
  if (a === null || b === null) return a;
  const diff = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + diff * t;
}

function interpolateDisplayRow(row, t) {
  if (t <= 0) return row;
  const rows = state.objectMap.get(row.object_id) || [];
  const next = rows.find((item) => item.frame_id === row.frame_id + 1);
  if (!next) return row;
  const out = { ...row, _interpolated_display: true };
  ["x1", "y1", "x2", "y2", "cx", "cy", "width", "height"].forEach((key) => {
    out[key] = interpolateNumber(row[key], next[key], t);
  });
  for (let i = 1; i <= 4; i += 1) {
    out[`q${i}_x`] = interpolateNumber(row[`q${i}_x`], next[`q${i}_x`], t);
    out[`q${i}_y`] = interpolateNumber(row[`q${i}_y`], next[`q${i}_y`], t);
  }
  out.heading_screen_rad = interpolateAngleRad(row.heading_screen_rad, next.heading_screen_rad, t);
  out.angle_deg = interpolateNumber(row.angle_deg, next.angle_deg, t);
  return out;
}

function visibleCurrentRows() {
  let rows = state.frameMap.get(state.currentFrame) || [];
  rows = rows.filter((row) => state.selectedClasses.has(row.class_name || "unknown"));
  if (state.objectFilter) {
    const wanted = Number(state.objectFilter);
    if (Number.isFinite(wanted)) rows = rows.filter((row) => row.object_id === wanted);
  }
  const t = playbackFraction();
  return t > 0 ? rows.map((row) => interpolateDisplayRow(row, t)) : rows;
}

function draw() {
  if (!ctx) return;
  const rect = els.canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const box = computeCanvasBox();
  ctx.fillStyle = "#eef2f6";
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (state.background) {
    ctx.drawImage(state.background, box.offsetX, box.offsetY, box.width, box.height);
  } else {
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(box.offsetX, box.offsetY, box.width, box.height);
    ctx.strokeStyle = "#cbd5e1";
    ctx.strokeRect(box.offsetX, box.offsetY, box.width, box.height);
  }

  if (els.keepUnrelatedToggle && !els.keepUnrelatedToggle.checked) drawRoadMask();
  if (els.laneToggle.checked) drawLaneGeometry();
  const rows = visibleCurrentRows();
  drawTrails(rows);
  for (const row of rows) drawTarget(row);
  updateMetrics(rows.length);
}

function drawTrails(currentRows) {
  for (const row of currentRows) {
    const history = trajectoryFor(row.object_id, state.currentFrame, state.trailLength);
    if (row._interpolated_display) history.push(row);
    if (history.length < 2) continue;
    const style = styleFor(row.class_name);
    drawSmoothTrail(history, style.color);
  }
}

function drawSmoothTrail(history, color) {
  const points = history.map((item) => worldToScreen(item.cx, item.cy));
  for (let i = 1; i < points.length; i += 1) {
    const prev = points[i - 1];
    const current = points[i];
    const next = points[i + 1];
    const start = i === 1 ? prev : { x: (prev.x + current.x) / 2, y: (prev.y + current.y) / 2 };
    const end = next ? { x: (current.x + next.x) / 2, y: (current.y + next.y) / 2 } : current;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    if (next) ctx.quadraticCurveTo(current.x, current.y, end.x, end.y);
    else ctx.lineTo(end.x, end.y);
    ctx.strokeStyle = hexToRgba(color, Math.max(0.1, (i / points.length) * 0.75));
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

function trajectoryFor(objectId, frameId, trailLength) {
  const rows = state.objectMap.get(objectId) || [];
  const start = frameId - trailLength;
  return rows.filter((row) => row.frame_id <= frameId && row.frame_id >= start);
}

function drawTarget(row) {
  const style = styleFor(row.class_name);
  const center = worldToScreen(row.cx, row.cy);
  ctx.save();
  if (els.bboxToggle.checked) drawBBox(row, style);
  else drawCenterPoint(center, style.color);
  if (shouldDrawDirection(row)) drawDirection(row, style.color);
  if (els.labelToggle.checked) drawLabel(row, center, style.color);
  ctx.restore();
}

function shouldDrawDirection(row) {
  if (state.version !== "full") return true;
  const obj = state.objectInfoMap.get(row.object_id);
  return !(obj && obj.is_static && STATIC_ARROW_SUPPRESSED_CLASSES.has(row.class_name));
}

function drawBBox(row, style) {
  ctx.strokeStyle = style.color;
  ctx.fillStyle = "transparent";
  ctx.lineWidth = style.weight || 2;
  if (hasQuad(row)) {
    drawPolygon(quadPoints(row).map((p) => worldToScreen(p.x, p.y)), false);
  } else if (row.x1 !== null && row.y1 !== null) {
    const p = worldToScreen(row.x1, row.y1);
    const w = (row.x2 - row.x1) * state.canvasBox.scaleX;
    const h = (row.y2 - row.y1) * state.canvasBox.scaleY;
    ctx.strokeRect(p.x, p.y, w, h);
  } else {
    drawCenterPoint(worldToScreen(row.cx, row.cy), style.color);
  }
}

function drawCenterPoint(center, color) {
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(center.x, center.y, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function hasQuad(row) {
  return [1, 2, 3, 4].every((i) => row[`q${i}_x`] !== null && row[`q${i}_y`] !== null);
}

function quadPoints(row) {
  if (hasQuad(row)) {
    return [1, 2, 3, 4].map((i) => ({ x: row[`q${i}_x`], y: row[`q${i}_y`] }));
  }
  return [
    { x: row.x1, y: row.y1 },
    { x: row.x2, y: row.y1 },
    { x: row.x2, y: row.y2 },
    { x: row.x1, y: row.y2 },
  ];
}

function drawPolygon(points, fill) {
  ctx.beginPath();
  points.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.closePath();
  if (fill) ctx.fill();
  ctx.stroke();
}

function headingFor(row) {
  const rows = state.objectMap.get(row.object_id) || [];
  const currentIndex = rows.findIndex((item) => item.frame_id === row.frame_id);
  if (currentIndex < 0) return state.headingCache.get(row.object_id) || null;

  const current = rows[currentIndex];
  let candidate = null;
  for (let i = currentIndex - 1; i >= 0; i -= 1) {
    const item = rows[i];
    if (current.frame_id - item.frame_id > HEADING_CONFIG.heading_smooth_window) break;
    if (item.cx !== null && item.cy !== null) candidate = item;
  }

  if (candidate) {
    const dx = current.cx - candidate.cx;
    const dy = current.cy - candidate.cy;
    if (Math.hypot(dx, dy) >= HEADING_CONFIG.min_motion_threshold) {
      const heading = Math.atan2(dy, dx);
      state.headingCache.set(row.object_id, heading);
      return heading;
    }
  }
  return state.headingCache.get(row.object_id) || null;
}

function headingFromCsv(row) {
  if (row.heading_screen_rad !== null) return row.heading_screen_rad;
  if (row.angle_deg !== null) return (row.angle_deg * Math.PI) / 180 - Math.PI / 2;
  return null;
}

function legacyVisualHeadingFor(row) {
  const longSide = boxLongSideInfo(row);
  const motionRad = headingFor(row);
  let rad = longSide ? longSide.angle : motionRad;
  if (rad === null) return null;

  if (longSide && motionRad !== null) {
    const dot = Math.cos(rad) * Math.cos(motionRad) + Math.sin(rad) * Math.sin(motionRad);
    if (dot < 0) rad += Math.PI;
  }
  return rad;
}

function drawDirection(row, color) {
  const rad = headingFromCsv(row) ?? (HEADING_CONFIG.use_legacy_visual_heading_refinement ? legacyVisualHeadingFor(row) : null);
  if (rad === null) return;
  const longSide = boxLongSideInfo(row);
  const center = worldToScreen(row.cx, row.cy);
  const baseLen = (longSide ? longSide.length : targetLongSideScreenLength(row)) * HEADING_CONFIG.arrow_length_scale;
  const len = Math.max(HEADING_CONFIG.arrow_min_length, Math.min(HEADING_CONFIG.arrow_max_length, baseLen));
  const end = { x: center.x + Math.cos(rad) * len, y: center.y + Math.sin(rad) * len };
  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(end.x, end.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(end.x, end.y);
  ctx.lineTo(end.x - Math.cos(rad - 0.45) * 7, end.y - Math.sin(rad - 0.45) * 7);
  ctx.lineTo(end.x - Math.cos(rad + 0.45) * 7, end.y - Math.sin(rad + 0.45) * 7);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function boxLongSideInfo(row) {
  if (!hasQuad(row)) return null;
  const points = quadPoints(row).map((point) => worldToScreen(point.x, point.y));
  let best = null;
  points.forEach((point, idx) => {
    const next = points[(idx + 1) % points.length];
    const dx = next.x - point.x;
    const dy = next.y - point.y;
    const length = Math.hypot(dx, dy);
    if (!best || length > best.length) best = { angle: Math.atan2(dy, dx), length };
  });
  return best && best.length > 0 ? best : null;
}

function targetLongSideScreenLength(row) {
  const longSide = boxLongSideInfo(row);
  if (longSide) return longSide.length;
  if (row.x1 !== null && row.y1 !== null && row.x2 !== null && row.y2 !== null) {
    const a = worldToScreen(row.x1, row.y1);
    const b = worldToScreen(row.x2, row.y2);
    return Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y));
  }
  return Math.max(row.width || 0, row.height || 0) * state.canvasBox.scaleX;
}

function drawLabel(row, center, color) {
  const text = `${row.object_id} ${row.class_name || ""}`;
  ctx.font = "12px Segoe UI, Arial";
  const width = ctx.measureText(text).width + 10;
  const x = center.x + 7;
  const y = center.y - 20;
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.fillRect(x, y, width, 18);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, width, 18);
  ctx.fillStyle = "#18202c";
  ctx.fillText(text, x + 5, y + 13);
}

function drawLaneGeometry() {
  const shapes = state.laneGeometry && state.laneGeometry.available ? state.laneGeometry.shapes || [] : [];
  if (!shapes.length) return;

  ctx.save();
  for (const shape of shapes.filter((item) => item.role === "road")) {
    const points = screenPointsForShape(shape);
    if (points.length < 2) continue;
    tracePoints(points, true);
    ctx.fillStyle = "rgba(255, 255, 255, 0.03)";
    ctx.strokeStyle = "rgba(15, 23, 42, 0.5)";
    ctx.lineWidth = 1.4;
    ctx.fill();
    ctx.stroke();
  }

  for (const shape of shapes.filter((item) => item.role === "lane")) {
    const points = screenPointsForShape(shape);
    if (points.length < 3) continue;
    const color = laneColor(shape);
    const isHovered = shape._key === state.hoveredLaneKey;
    tracePoints(points, true);
    ctx.fillStyle = hexToRgba(color, isHovered ? 0.34 : 0.17);
    ctx.strokeStyle = hexToRgba(color, isHovered ? 0.98 : 0.78);
    ctx.lineWidth = isHovered ? 3 : 1.6;
    ctx.shadowColor = isHovered ? hexToRgba(color, 0.4) : "transparent";
    ctx.shadowBlur = isHovered ? 10 : 0;
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
    drawLaneGeometryLabel(shape, points, color, isHovered);
  }

  for (const shape of shapes.filter((item) => item.role !== "road" && item.role !== "lane")) {
    const points = screenPointsForShape(shape);
    if (points.length < 2) continue;
    tracePoints(points, false);
    ctx.strokeStyle = shape.role === "drivingline" ? "rgba(124, 58, 237, 0.88)" : "rgba(217, 119, 6, 0.86)";
    ctx.lineWidth = shape.role === "drivingline" ? 2.2 : 1.7;
    ctx.setLineDash(shape.role === "drivingline" ? [8, 6] : []);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

function drawRoadMask() {
  const shapes = state.laneGeometry && state.laneGeometry.available ? state.laneGeometry.shapes || [] : [];
  const roadShapes = shapes.filter((shape) => shape.role === "road" && (shape.points || []).length >= 3);
  if (!roadShapes.length) return;

  const box = state.canvasBox;
  const path = new Path2D();
  path.rect(box.offsetX, box.offsetY, box.width, box.height);
  for (const shape of roadShapes) {
    const points = screenPointsForShape(shape);
    if (points.length < 3) continue;
    points.forEach((point, index) => {
      if (index === 0) path.moveTo(point.x, point.y);
      else path.lineTo(point.x, point.y);
    });
    path.closePath();
  }

  ctx.save();
  ctx.fillStyle = ROAD_MASK_COLOR;
  ctx.fill(path, "evenodd");
  ctx.restore();
}

function screenPointsForShape(shape) {
  return (shape.points || []).map((point) => worldToScreen(point.x, point.y));
}

function tracePoints(points, closePath) {
  ctx.beginPath();
  points.forEach((point, idx) => {
    if (idx === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  if (closePath) ctx.closePath();
}

function laneColor(shape) {
  const laneId = asNumber(shape.lane_id);
  if (laneId !== null) return LANE_PALETTE[Math.abs(laneId) % LANE_PALETTE.length];
  let hash = 0;
  const source = String(shape.label || shape._key || "");
  for (let i = 0; i < source.length; i += 1) hash = (hash * 31 + source.charCodeAt(i)) | 0;
  return LANE_PALETTE[Math.abs(hash) % LANE_PALETTE.length];
}

function drawLaneGeometryLabel(shape, points, color, isHovered = false) {
  if (!shape.lane_id || !points.length) return;
  const cx = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const cy = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  const text = `lane_${shape.lane_id}`;
  ctx.font = "11px Segoe UI, Arial";
  const width = ctx.measureText(text).width + 10;
  ctx.fillStyle = isHovered ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.78)";
  ctx.fillRect(cx - width / 2, cy - 9, width, 18);
  ctx.strokeStyle = hexToRgba(color, isHovered ? 0.98 : 0.68);
  ctx.strokeRect(cx - width / 2, cy - 9, width, 18);
  ctx.fillStyle = "#172033";
  ctx.fillText(text, cx - width / 2 + 5, cy + 4);
}

function updateMetrics(currentCount) {
  els.frameMetric.textContent = `帧 ${state.currentFrame} / ${state.maxFrame}`;
  els.objectMetric.textContent = `目标 ${currentCount}`;
  els.timeline.value = String(state.currentFrame);
}

function setStatus(text) {
  els.statusText.textContent = text;
}

function stepFrame(delta) {
  if (!state.frameIds.length) return;
  state.currentFrame = clampFrame(state.currentFrame + delta);
  draw();
}

function tick(ts) {
  if (!state.lastTick) state.lastTick = ts;
  const elapsed = (ts - state.lastTick) / 1000;
  state.lastTick = ts;
  if (state.playing && state.metadata) {
    const fps = Number(state.metadata.fps || 10);
    state.frameAccumulator += elapsed * fps * state.speed;
    const steps = Math.floor(state.frameAccumulator);
    if (steps >= 1) {
      state.frameAccumulator -= steps;
      state.currentFrame += steps;
      if (state.currentFrame > state.maxFrame) state.currentFrame = state.minFrame;
    }
    draw();
  }
  requestAnimationFrame(tick);
}

function setPlayback(playing) {
  state.playing = playing;
  els.playButton.textContent = playing ? "暂停" : "播放";
  state.lastTick = 0;
  state.frameAccumulator = 0;
}

function clampFrame(frame) {
  return Math.max(state.minFrame, Math.min(state.maxFrame, Math.round(frame)));
}

function applyInputs() {
  const objectText = els.objectInput.value.trim();
  const frameText = els.frameInput.value.trim();
  const objectId = asNumber(objectText);
  const frameId = asNumber(frameText);

  if (!objectText && !frameText) {
    state.objectFilter = "";
    state.selectedObject = null;
    setStatus("已恢复全目标显示");
    draw();
    return;
  }

  if (objectText) {
    if (objectId === null) {
      setStatus("object_id 需要输入数字");
      return;
    }
    if (!state.objectMap.has(objectId)) {
      setStatus(`未找到 object_id ${objectId}`);
      return;
    }
    state.objectFilter = String(objectId);
    state.selectedObject = objectId;
    if (!frameText) {
      const rows = state.objectMap.get(objectId) || [];
      if (rows.length) state.currentFrame = clampFrame(rows[0].frame_id);
      state.lastTick = 0;
      state.frameAccumulator = 0;
    }
  }

  if (frameText) {
    if (frameId === null) {
      setStatus("frame_id 需要输入数字");
      return;
    }
    if (frameId < state.minFrame || frameId > state.maxFrame) {
      setStatus(`frame_id 超出范围：${state.minFrame} - ${state.maxFrame}`);
      return;
    }
    state.currentFrame = clampFrame(frameId);
  }

  if (objectText && frameText) setStatus(`显示 object_id ${state.objectFilter}，跳转到第 ${state.currentFrame} 帧`);
  else if (objectText) setStatus(`显示 object_id ${state.objectFilter}，跳转到首次出现帧 ${state.currentFrame}`);
  else if (frameText) setStatus(`跳转到第 ${state.currentFrame} 帧`);
  draw();
}

function syncTrailLength(value, shouldDraw = true) {
  const min = Number(els.trailRange.min || 0);
  const max = Number(els.trailRange.max || 160);
  const next = Math.max(min, Math.min(max, Math.round(Number(value) || DEFAULT_TRAIL_LENGTH)));
  state.trailLength = next;
  els.trailRange.value = String(next);
  els.trailValue.textContent = String(next);
  if (shouldDraw) draw();
}

function findHitObject(clientX, clientY) {
  const rect = els.canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const world = screenToWorld(x, y);
  const rows = visibleCurrentRows();
  let best = null;
  let bestDist = Infinity;
  for (const row of rows) {
    const pad = 8 / state.canvasBox.scaleX;
    const inBox = row.x1 !== null && world.x >= row.x1 - pad && world.x <= row.x2 + pad && world.y >= row.y1 - pad && world.y <= row.y2 + pad;
    const dist = Math.hypot(world.x - row.cx, world.y - row.cy);
    if ((inBox || dist * state.canvasBox.scaleX < 10) && dist < bestDist) {
      bestDist = dist;
      best = row;
    }
  }
  return best;
}

function findHitLane(clientX, clientY) {
  if (!els.laneToggle.checked || !(state.laneGeometry && state.laneGeometry.available)) return null;
  const rect = els.canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const world = screenToWorld(x, y);
  const lanes = (state.laneGeometry.shapes || []).filter((shape) => shape.role === "lane" && (shape.points || []).length >= 3);
  for (let i = lanes.length - 1; i >= 0; i -= 1) {
    if (pointInPolygon(world, lanes[i].points)) return lanes[i];
  }
  return null;
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi || 1e-9) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function setHoveredLane(shape) {
  const nextKey = shape ? shape._key : null;
  if (state.hoveredLaneKey === nextKey) return;
  state.hoveredLaneKey = nextKey;
  draw();
}

function showObjectDetail(row) {
  if (!row) {
    els.detailContent.textContent = "悬停或点击目标";
    return;
  }
  const obj = state.objectInfoMap.get(row.object_id) || {};
  els.detailContent.innerHTML = `
    <div class="detail-grid">
      <span>object_id</span><strong>${row.object_id}</strong>
      <span>raw_object_id</span><strong>${row.raw_object_id === null ? "" : row.raw_object_id}</strong>
      <span>class</span><strong>${escapeHtml(row.class_name || "")}</strong>
      <span>length</span><strong>${fmt(obj.length, 3)} m</strong>
      <span>width</span><strong>${fmt(obj.width, 3)} m</strong>
      <span>frame</span><strong>${row.frame_id}</strong>
      <span>lane_id</span><strong>${row.lane_id === null ? "" : row.lane_id}</strong>
      <span>center</span><strong>${fmt(row.cx)}, ${fmt(row.cy)}</strong>
      <span>bbox</span><strong>${fmt(row.x1)}, ${fmt(row.y1)} - ${fmt(row.x2)}, ${fmt(row.y2)}</strong>
      <span>confidence</span><strong>${row.confidence === null ? "" : fmt(row.confidence, 3)}</strong>
      <span>displacement</span><strong>${fmt(obj.displacement, 1)}</strong>
      <span>mean_speed</span><strong>${fmt(obj.mean_speed, 3)}</strong>
      <span>static_ratio</span><strong>${fmt(obj.static_ratio, 3)}</strong>
    </div>
  `;
}

function showTooltip(row, clientX, clientY) {
  if (!row) {
    els.tooltip.classList.add("hidden");
    els.tooltip.classList.remove("lane-tooltip");
    return;
  }
  els.tooltip.classList.remove("lane-tooltip");
  const obj = state.objectInfoMap.get(row.object_id) || {};
  els.tooltip.innerHTML = `
    <strong>${row.object_id}</strong> ${escapeHtml(row.class_name || "")}<br>
    length × width: ${fmt(obj.length, 2)} × ${fmt(obj.width, 2)} m<br>
    frame: ${row.frame_id}<br>
    lane_id: ${row.lane_id === null ? "" : row.lane_id}<br>
    center: ${fmt(row.cx)}, ${fmt(row.cy)}<br>
    confidence: ${row.confidence === null ? "" : fmt(row.confidence, 3)}
  `;
  const wrap = els.canvasWrap.getBoundingClientRect();
  els.tooltip.style.left = `${Math.min(clientX - wrap.left + 14, wrap.width - 230)}px`;
  els.tooltip.style.top = `${Math.max(10, clientY - wrap.top + 14)}px`;
  els.tooltip.classList.remove("hidden");
}

function showLaneTooltip(shape, clientX, clientY) {
  if (!shape) {
    els.tooltip.classList.add("hidden");
    els.tooltip.classList.remove("lane-tooltip");
    return;
  }
  const color = laneColor(shape);
  const pointCount = (shape.points || []).length;
  els.tooltip.classList.add("lane-tooltip");
  els.tooltip.innerHTML = `
    <div class="tooltip-title"><span class="tooltip-swatch" style="background:${color}"></span><span>${escapeHtml(shape.label || `lane_${shape.lane_id || ""}`)}</span></div>
    lane_id: ${shape.lane_id || ""}<br>
    type: ${escapeHtml(shape.role || "lane")}<br>
    points: ${pointCount}
  `;
  const wrap = els.canvasWrap.getBoundingClientRect();
  els.tooltip.style.left = `${Math.min(clientX - wrap.left + 14, wrap.width - 210)}px`;
  els.tooltip.style.top = `${Math.max(10, clientY - wrap.top + 14)}px`;
  els.tooltip.classList.remove("hidden");
}

function fmt(value, digits = 1) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "";
  return Number(value).toFixed(digits);
}

function hexToRgba(hex, alpha) {
  const value = hex.replace("#", "");
  const bigint = parseInt(value, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
}

els.scanButton.addEventListener("click", async () => {
  setStatus("转换中...");
  els.scanResult.textContent = "";
  try {
    const result = await api(`/api/scan?force=${els.forceScan.checked ? "true" : "false"}`, { method: "POST" });
    els.scanResult.textContent = JSON.stringify(result, null, 2);
    await loadDatasets();
    setStatus("扫描完成");
  } catch (err) {
    els.scanResult.textContent = String(err.message || err);
    setStatus("转换失败");
  }
});

els.datasetSearch.addEventListener("input", renderDatasets);
els.jumpButton.addEventListener("click", applyInputs);
els.selectAllClasses.addEventListener("click", () => setAllClasses(true));
els.clearAllClasses.addEventListener("click", () => setAllClasses(false));
els.playButton.addEventListener("click", () => setPlayback(!state.playing));
els.prevButton.addEventListener("click", () => stepFrame(-1));
els.nextButton.addEventListener("click", () => stepFrame(1));
els.speedSelect.addEventListener("change", () => {
  state.speed = Number(els.speedSelect.value) || 1;
});
els.trailRange.addEventListener("input", () => syncTrailLength(els.trailRange.value));
els.timeline.addEventListener("input", () => {
  state.currentFrame = Number(els.timeline.value);
  draw();
});
els.bboxToggle.addEventListener("change", draw);
els.labelToggle.addEventListener("change", draw);
els.laneToggle.addEventListener("change", () => {
  if (els.laneToggle.checked && !(state.laneGeometry && state.laneGeometry.available)) {
    setStatus((state.laneGeometry && state.laneGeometry.reason) || "当前数据集没有可绘制的车道几何");
  }
  if (!els.laneToggle.checked) setHoveredLane(null);
  draw();
});
els.keepUnrelatedToggle.addEventListener("change", draw);

els.canvas.addEventListener("mousemove", (event) => {
  const laneHit = findHitLane(event.clientX, event.clientY);
  setHoveredLane(laneHit);
  const hit = findHitObject(event.clientX, event.clientY);
  if (hit) {
    showTooltip(hit, event.clientX, event.clientY);
    showObjectDetail(hit);
  } else if (laneHit) {
    showLaneTooltip(laneHit, event.clientX, event.clientY);
  } else {
    showTooltip(null, event.clientX, event.clientY);
  }
});
els.canvas.addEventListener("mouseleave", () => {
  setHoveredLane(null);
  els.tooltip.classList.add("hidden");
  els.tooltip.classList.remove("lane-tooltip");
});
els.canvas.addEventListener("click", (event) => {
  const hit = findHitObject(event.clientX, event.clientY);
  state.selectedObject = hit ? hit.object_id : null;
  showObjectDetail(hit);
});

window.addEventListener("resize", () => {
  resizeCanvas();
  renderDimensionAnalysis();
});

syncTrailLength(DEFAULT_TRAIL_LENGTH, false);
loadDatasets().catch((err) => {
  setStatus("读取失败");
  renderClassStats();
  els.scanResult.textContent = String(err.message || err);
});
requestAnimationFrame(tick);
