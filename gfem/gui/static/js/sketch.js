// 2D sketcher: polylines (with arcs), rectangles, circles, slots; grid snap; numeric entry; DXF import.
// Produces a sketch feature {plane, offset, loops, distance|angle, op} for extrude / revolve.
import { fmt } from "./charts.js";

export function openSketcher(app, existing) {
  const dlg = document.getElementById("sketch-dlg");
  const cv = dlg.querySelector("canvas");
  const st = {
    loops: existing ? JSON.parse(JSON.stringify(existing.loops || [])) : [],
    tool: "polyline", cur: null, grid: 5, snap: true, view: { cx: 0, cy: 0, s: 4 }, mouse: [0, 0], arc: 0,
  };
  const $ = s => dlg.querySelector(s);
  $("#sk-plane").value = typeof existing?.plane === "string" ? existing.plane : "XY";
  $("#sk-offset").value = existing?.offset ?? 0;
  $("#sk-op").value = existing?.type === "revolve" ? "revolve" : existing?.type === "surface" ? "surface" : "extrude";
  $("#sk-dist").value = existing?.distance ?? 20;
  $("#sk-angle").value = existing?.angle ?? 360;
  $("#sk-sym").checked = !!existing?.symmetric;
  $("#sk-bool").value = "new";
  const bodySel = $("#sk-target");
  bodySel.innerHTML = app.liveFeatures().map(f => `<option value="${f.id}">${f.name || f.id}</option>`).join("");
  dlg.classList.add("open");
  const W = () => cv.clientWidth, H = () => cv.clientHeight;
  const toS = ([x, y]) => [W() / 2 + (x - st.view.cx) * st.view.s, H() / 2 - (y - st.view.cy) * st.view.s];
  const toW = (px, py) => [(px - W() / 2) / st.view.s + st.view.cx, -(py - H() / 2) / st.view.s + st.view.cy];
  const snap = p => st.snap ? p.map(v => Math.round(v / st.grid) * st.grid) : p;
  function fit() {
    const pts = []; for (const L of st.loops) pts.push(...loopPts(L));
    if (!pts.length) { st.view = { cx: 50, cy: 25, s: Math.min(W(), H()) / 140 }; return; }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    st.view.cx = (x0 + x1) / 2; st.view.cy = (y0 + y1) / 2;
    st.view.s = Math.min(W() / Math.max(x1 - x0, 1) , H() / Math.max(y1 - y0, 1)) * 0.75;
  }
  function loopPts(L) {
    if (L.circle) { const [cx, cy, r] = L.circle; return [[cx - r, cy - r], [cx + r, cy + r]]; }
    if (L.rect) return [[L.rect[0], L.rect[1]], [L.rect[2], L.rect[3]]];
    if (L.slot) { const [a, b, c, d, r] = L.slot; return [[Math.min(a, c) - r, Math.min(b, d) - r], [Math.max(a, c) + r, Math.max(b, d) + r]]; }
    return L.points;
  }
  function arcPts(p0, p1, b) {
    if (!b) return [p0, p1];
    const th = 4 * Math.atan(b), dx = p1[0] - p0[0], dy = p1[1] - p0[1], c = Math.hypot(dx, dy);
    const r = c / (2 * Math.sin(Math.abs(th) / 2)), mx = (p0[0] + p1[0]) / 2, my = (p0[1] + p1[1]) / 2;
    const nx = -dy / c, ny = dx / c, h = Math.sqrt(Math.max(r * r - c * c / 4, 0));
    const sg = (b > 0) === (Math.abs(th) < Math.PI) ? 1 : -1;
    const cx = mx + sg * h * nx, cy = my + sg * h * ny;
    const a0 = Math.atan2(p0[1] - cy, p0[0] - cx);
    const out = []; for (let i = 0; i <= 24; i++) { const a = a0 + th * i / 24; out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
    return out;
  }
  function drawLoop(g, L, color) {
    g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
    if (L.circle) { const [cx, cy, r] = L.circle; const [sx, sy] = toS([cx, cy]); g.arc(sx, sy, r * st.view.s, 0, 7); }
    else if (L.rect) { const [x0, y0, x1, y1] = L.rect; const a = toS([x0, y0]), b = toS([x1, y1]); g.rect(a[0], a[1], b[0] - a[0], b[1] - a[1]); }
    else if (L.slot) { const [x0, y0, x1, y1, r] = L.slot; const d = [x1 - x0, y1 - y0], Ld = Math.hypot(...d), t = [d[0] / Ld, d[1] / Ld], n = [-t[1], t[0]];
      const pts = [[x0 + r * n[0], y0 + r * n[1]], [x1 + r * n[0], y1 + r * n[1]], [x1 - r * n[0], y1 - r * n[1]], [x0 - r * n[0], y0 - r * n[1]]];
      drawLoop(g, { points: pts, bulges: [0, -1, 0, -1] }, color); return; }
    else {
      const P = L.points, B = L.bulges || [];
      for (let i = 0; i < P.length; i++) { const seg = arcPts(P[i], P[(i + 1) % P.length], B[i] || 0); seg.forEach((p, k) => { const s = toS(p); (i === 0 && k === 0) ? g.moveTo(...s) : g.lineTo(...s); }); }
      g.closePath();
    }
    g.stroke();
  }
  function draw() {
    const dpr = window.devicePixelRatio || 1;
    cv.width = W() * dpr; cv.height = H() * dpr;
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#fbfcfe"; g.fillRect(0, 0, W(), H());
    // grid
    const step = st.grid * st.view.s;
    if (step > 6) {
      g.strokeStyle = "#e3e8ef"; g.lineWidth = 1; g.beginPath();
      const [wx0, wy1] = toW(0, 0), [wx1, wy0] = toW(W(), H());
      for (let x = Math.floor(wx0 / st.grid) * st.grid; x <= wx1; x += st.grid) { const s = toS([x, 0])[0]; g.moveTo(s, 0); g.lineTo(s, H()); }
      for (let y = Math.floor(wy0 / st.grid) * st.grid; y <= wy1; y += st.grid) { const s = toS([0, y])[1]; g.moveTo(0, s); g.lineTo(W(), s); }
      g.stroke();
    }
    const o = toS([0, 0]); g.strokeStyle = "#c33"; g.beginPath(); g.moveTo(o[0], 0); g.lineTo(o[0], H()); g.stroke();
    g.strokeStyle = "#3a3"; g.beginPath(); g.moveTo(0, o[1]); g.lineTo(W(), o[1]); g.stroke();
    st.loops.forEach((L, i) => drawLoop(g, L, i === 0 ? "#1d4f91" : "#c0392b"));
    // in-progress
    const m = st.mouse;
    g.strokeStyle = "#e67e22"; g.lineWidth = 1.5; g.setLineDash([5, 4]); g.beginPath();
    if (st.cur) {
      if (st.tool === "polyline") {
        const P = [...st.cur.points, m], B = [...st.cur.bulges, st.arc];
        for (let i = 0; i < P.length - 1; i++) arcPts(P[i], P[i + 1], B[i] || 0).forEach((p, k) => { const s = toS(p); (i === 0 && k === 0) ? g.moveTo(...s) : g.lineTo(...s); });
      } else if (st.tool === "rect") { const a = toS(st.cur.p0), b = toS(m); g.rect(a[0], a[1], b[0] - a[0], b[1] - a[1]); }
      else if (st.tool === "circle") { const a = toS(st.cur.p0); g.arc(a[0], a[1], Math.hypot(m[0] - st.cur.p0[0], m[1] - st.cur.p0[1]) * st.view.s, 0, 7); }
      else if (st.tool === "slot") { const a = toS(st.cur.p0), b = toS(m); g.moveTo(...a); g.lineTo(...b); }
    }
    g.stroke(); g.setLineDash([]);
    const ms = toS(m); g.fillStyle = "#e67e22"; g.fillRect(ms[0] - 3, ms[1] - 3, 6, 6);
    $("#sk-coord").textContent = `x=${fmt(m[0])}  y=${fmt(m[1])}  ${st.cur && st.tool === "polyline" ? "(click first point or press Enter to close, Esc to cancel)" : ""}`;
    $("#sk-list").innerHTML = st.loops.map((L, i) => `<div class="sk-item">${i === 0 ? "Outer" : "Inner"} ${L.circle ? "circle" : L.rect ? "rectangle" : L.slot ? "slot" : "profile (" + L.points.length + " pts)"} <button data-del="${i}">✕</button></div>`).join("") || "<i>no profiles</i>";
    $("#sk-list").querySelectorAll("button[data-del]").forEach(b => b.onclick = () => { st.loops.splice(+b.dataset.del, 1); draw(); });
  }
  function closePoly() {
    if (st.cur && st.cur.points.length >= 3) { st.loops.push({ points: st.cur.points, bulges: st.cur.bulges.slice(0, st.cur.points.length) }); }
    st.cur = null; draw();
  }
  cv.onmousemove = e => { const r = cv.getBoundingClientRect(); st.mouse = snap(toW(e.clientX - r.left, e.clientY - r.top)); draw(); };
  cv.onmousedown = e => {
    if (e.button === 1 || e.button === 2) { const r0 = [e.clientX, e.clientY], v0 = { ...st.view };
      const mv = ev => { st.view.cx = v0.cx - (ev.clientX - r0[0]) / st.view.s; st.view.cy = v0.cy + (ev.clientY - r0[1]) / st.view.s; draw(); };
      const up = () => { window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up); return; }
    const p = st.mouse;
    if (st.tool === "polyline") {
      if (!st.cur) st.cur = { points: [p], bulges: [] };
      else {
        const f = st.cur.points[0];
        if (Math.hypot(p[0] - f[0], p[1] - f[1]) * st.view.s < 8 && st.cur.points.length >= 3) { st.cur.bulges.push(st.arc); closePoly(); return; }
        st.cur.bulges.push(st.arc); st.cur.points.push(p);
      }
    } else if (st.tool === "rect") {
      if (!st.cur) st.cur = { p0: p }; else { const a = st.cur.p0; st.loops.push({ rect: [Math.min(a[0], p[0]), Math.min(a[1], p[1]), Math.max(a[0], p[0]), Math.max(a[1], p[1])] }); st.cur = null; }
    } else if (st.tool === "circle") {
      if (!st.cur) st.cur = { p0: p }; else { const a = st.cur.p0, r = Math.hypot(p[0] - a[0], p[1] - a[1]); if (r > 0) st.loops.push({ circle: [a[0], a[1], r] }); st.cur = null; }
    } else if (st.tool === "slot") {
      if (!st.cur) st.cur = { p0: p }; else { const r = +$("#sk-slot-r").value || 2; st.loops.push({ slot: [st.cur.p0[0], st.cur.p0[1], p[0], p[1], r] }); st.cur = null; }
    }
    draw();
  };
  cv.onwheel = e => { e.preventDefault(); const f = e.deltaY < 0 ? 1.15 : 1 / 1.15; st.view.s *= f; draw(); };
  cv.oncontextmenu = e => e.preventDefault();
  const key = e => {
    if (!dlg.classList.contains("open")) return;
    if (e.key === "Escape") { st.cur = null; draw(); }
    if (e.key === "Enter" && st.tool === "polyline") closePoly();
  };
  window.addEventListener("keydown", key);
  dlg.querySelectorAll("[data-tool]").forEach(b => b.onclick = () => { st.tool = b.dataset.tool; st.cur = null;
    dlg.querySelectorAll("[data-tool]").forEach(x => x.classList.toggle("on", x === b)); draw(); });
  $("#sk-arc").onchange = e => { st.arc = +e.target.value; };
  $("#sk-grid").onchange = e => { st.grid = +e.target.value || 1; draw(); };
  $("#sk-snap").onchange = e => { st.snap = e.target.checked; };
  $("#sk-fit").onclick = () => { fit(); draw(); };
  $("#sk-add-pt").onclick = () => {
    const x = +$("#sk-px").value, y = +$("#sk-py").value;
    st.tool = "polyline"; if (!st.cur) st.cur = { points: [[x, y]], bulges: [] }; else { st.cur.bulges.push(st.arc); st.cur.points.push([x, y]); } draw(); };
  $("#sk-close").onclick = closePoly;
  $("#sk-add-circle").onclick = () => { st.loops.push({ circle: [+$("#sk-px").value, +$("#sk-py").value, +$("#sk-r").value || 5] }); draw(); };
  $("#sk-dxf").onchange = async e => {
    const file = e.target.files[0]; if (!file) return;
    const up = await app.api.upload(file);
    const r = await app.api.post("/api/dxf", { path: up.path });
    for (const L of r.loops) {
      const s = r.scale || 1;
      if (L.circle) st.loops.push({ circle: L.circle.map(v => v * s) });
      else st.loops.push({ points: L.points.map(p => [p[0] * s, p[1] * s]), bulges: L.bulges });
    }
    fit(); draw(); app.log(`DXF: ${r.loops.length} closed profiles imported`);
  };
  const finish = ok => {
    window.removeEventListener("keydown", key);
    dlg.classList.remove("open");
    if (!ok) return;
    if (st.cur && st.tool === "polyline") closePoly();
    if (!st.loops.length) { app.log("sketch is empty"); return; }
    const op = $("#sk-op").value;
    const f = { type: op, plane: $("#sk-plane").value, offset: +$("#sk-offset").value || 0, loops: st.loops };
    if (op === "extrude") { f.distance = +$("#sk-dist").value || 10; f.symmetric = $("#sk-sym").checked; }
    else if (op === "surface") { /* planar sheet body for a shell mesh */ }
    else { f.angle = +$("#sk-angle").value || 360; f.axis = $("#sk-axis").value === "x" ? [[0, 0], [1, 0]] : [[0, 0], [0, 1]]; }
    app.addSketchFeature(f, existing, $("#sk-bool").value, bodySel.value);
  };
  $("#sk-ok").onclick = () => finish(true);
  $("#sk-cancel").onclick = () => finish(false);
  requestAnimationFrame(() => { fit(); draw(); });
}
