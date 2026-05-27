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
const DEFAULT_TRAIL_LENGTH = 50;
const HEADING_CONFIG = {
  heading_smooth_window: 8,
  min_motion_threshold: 2.0,
  arrow_length_scale: 0.8,
  arrow_min_length: 8,
  arrow_max_length: 40,
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
  bboxToggle: document.getElementById("bboxToggle"),
  labelToggle: document.getElementById("labelToggle"),
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
  if (!state.datasetId && state.datasets.length) {
    await loadDataset(state.datasets[0].dataset_id, state.datasets[0].version);
  } else if (!state.datasets.length) {
    els.datasetSummary.textContent = "没有已转换数据集";
    els.emptyState.classList.remove("hidden");
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
    div.textContent = state.datasets.length ? "没有匹配的数据集" : "Adjusted results 为空";
    els.datasetList.appendChild(div);
    return;
  }

  for (const item of items) {
    const btn = document.createElement("button");
    const active = item.dataset_id === state.datasetId && item.version === state.version;
    btn.className = `dataset-item ${active ? "active" : ""}`;
    btn.innerHTML = `
      <div class="dataset-name">${escapeHtml(item.dataset_id)}</div>
      <div class="dataset-version">${escapeHtml(item.version)}</div>
      <div class="dataset-meta">${item.total_frames || 0} 帧 · ${item.object_count || 0} 目标 · ${item.row_count || 0} 行</div>
      ${item.version === "moving_filtered" ? `<div class="dataset-meta">过滤 ${item.filtered_object_count || 0} 个静止目标</div>` : ""}
    `;
    btn.addEventListener("click", () => loadDataset(item.dataset_id, item.version));
    els.datasetList.appendChild(btn);
  }
}

async function loadDataset(datasetId, version) {
  setStatus("读取数据中...");
  state.datasetId = datasetId;
  state.version = version;
  state.metadata = await api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/metadata`);
  const [tracks, frames, objects] = await Promise.all([
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/tracks`),
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/frames`),
    api(`/api/datasets/${encodeURIComponent(datasetId)}/${encodeURIComponent(version)}/objects`),
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
  buildIndexes();
  renderDatasets();
  buildClassFilters();
  renderLegend();
  await loadBackground(datasetId, version);

  state.currentFrame = state.minFrame;
  state.objectFilter = "";
  state.selectedObject = null;
  state.headingCache = new Map();
  els.objectInput.value = "";
  els.frameInput.value = "";
  els.timeline.min = String(state.minFrame);
  els.timeline.max = String(state.maxFrame);
  els.timeline.value = String(state.currentFrame);
  els.emptyState.classList.add("hidden");
  els.datasetSummary.textContent = `${datasetId} / ${version} · ${state.frameIds.length} 帧 · ${state.objects.length} 目标`;
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
    confidence: asNumber(row.confidence),
    x1: asNumber(row.x1),
    y1: asNumber(row.y1),
    x2: asNumber(row.x2),
    y2: asNumber(row.y2),
    cx: asNumber(row.cx),
    cy: asNumber(row.cy),
    width: asNumber(row.width),
    height: asNumber(row.height),
    angle_deg: asNumber(row.angle_deg),
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
    start_frame: asNumber(row.start_frame),
    end_frame: asNumber(row.end_frame),
    displacement: asNumber(row.displacement),
    path_length: asNumber(row.path_length),
    mean_speed: asNumber(row.mean_speed),
    max_speed: asNumber(row.max_speed),
    static_ratio: asNumber(row.static_ratio),
    is_static: String(row.is_static).toLowerCase() === "true",
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

function visibleCurrentRows() {
  let rows = state.frameMap.get(state.currentFrame) || [];
  rows = rows.filter((row) => state.selectedClasses.has(row.class_name || "unknown"));
  if (state.objectFilter) {
    const wanted = Number(state.objectFilter);
    if (Number.isFinite(wanted)) rows = rows.filter((row) => row.object_id === wanted);
  }
  return rows;
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

  const rows = visibleCurrentRows();
  drawTrails(rows);
  for (const row of rows) drawTarget(row);
  updateMetrics(rows.length);
}

function drawTrails(currentRows) {
  for (const row of currentRows) {
    const history = trajectoryFor(row.object_id, state.currentFrame, state.trailLength);
    if (history.length < 2) continue;
    const style = styleFor(row.class_name);
    for (let i = 1; i < history.length; i += 1) {
      const a = history[i - 1];
      const b = history[i];
      const p1 = worldToScreen(a.cx, a.cy);
      const p2 = worldToScreen(b.cx, b.cy);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = hexToRgba(style.color, Math.max(0.1, (i / history.length) * 0.75));
      ctx.lineWidth = 2;
      ctx.stroke();
    }
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
  drawDirection(row, style.color);
  if (els.labelToggle.checked) drawLabel(row, center, style.color);
  ctx.restore();
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

function drawDirection(row, color) {
  const rad = headingFor(row);
  if (rad === null) return;
  const center = worldToScreen(row.cx, row.cy);
  const baseLen = Math.max(row.width || 0, row.height || 0) * state.canvasBox.scaleX * HEADING_CONFIG.arrow_length_scale;
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
      draw();
    }
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
  else if (objectText) setStatus(`显示 object_id ${state.objectFilter}`);
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

function showObjectDetail(row) {
  if (!row) {
    els.detailContent.textContent = "悬停或点击目标";
    return;
  }
  const obj = state.objectInfoMap.get(row.object_id) || {};
  els.detailContent.innerHTML = `
    <div class="detail-grid">
      <span>object_id</span><strong>${row.object_id}</strong>
      <span>class</span><strong>${escapeHtml(row.class_name || "")}</strong>
      <span>frame</span><strong>${row.frame_id}</strong>
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
    return;
  }
  els.tooltip.innerHTML = `
    <strong>${row.object_id}</strong> ${escapeHtml(row.class_name || "")}<br>
    frame: ${row.frame_id}<br>
    center: ${fmt(row.cx)}, ${fmt(row.cy)}<br>
    confidence: ${row.confidence === null ? "" : fmt(row.confidence, 3)}
  `;
  const wrap = els.canvasWrap.getBoundingClientRect();
  els.tooltip.style.left = `${Math.min(clientX - wrap.left + 14, wrap.width - 230)}px`;
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

els.canvas.addEventListener("mousemove", (event) => {
  const hit = findHitObject(event.clientX, event.clientY);
  showTooltip(hit, event.clientX, event.clientY);
  if (hit) showObjectDetail(hit);
});
els.canvas.addEventListener("mouseleave", () => {
  els.tooltip.classList.add("hidden");
});
els.canvas.addEventListener("click", (event) => {
  const hit = findHitObject(event.clientX, event.clientY);
  state.selectedObject = hit ? hit.object_id : null;
  showObjectDetail(hit);
});

window.addEventListener("resize", resizeCanvas);

syncTrailLength(DEFAULT_TRAIL_LENGTH, false);
loadDatasets().catch((err) => {
  setStatus("读取失败");
  els.scanResult.textContent = String(err.message || err);
});
requestAnimationFrame(tick);
