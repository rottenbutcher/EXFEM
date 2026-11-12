// Minimal canvas line/bar charts (convergence, time histories, histograms).
export class Chart {
  constructor(canvas, opts = {}) {
    this.c = canvas; this.opts = Object.assign({ logy: false, title: "", xlabel: "", ylabel: "", bars: false }, opts);
    this.series = [];
  }
  set(series) { this.series = series; this.draw(); }
  draw() {
    const c = this.c, dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth || 300, H = c.clientHeight || 150;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--fg").trim() || "#222", grid = css.getPropertyValue("--grid").trim() || "#ddd";
    g.clearRect(0, 0, W, H);
    const L = 58, R = 12, T = this.opts.title ? 22 : 10, B = 30;
    const pw = W - L - R, ph = H - T - B;
    let xs = [], ys = [];
    for (const s of this.series) for (const [x, y] of s.data) { if (isFinite(x) && isFinite(y) && (!this.opts.logy || y > 0)) { xs.push(x); ys.push(y); } }
    g.font = "11px system-ui, sans-serif"; g.fillStyle = fg;
    if (this.opts.title) { g.font = "bold 12px system-ui, sans-serif"; g.fillText(this.opts.title, L, 14); g.font = "11px system-ui, sans-serif"; }
    if (!xs.length) { g.fillStyle = "#888"; g.fillText("no data", L + pw / 2 - 20, T + ph / 2); return; }
    let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x1 === x0) x1 = x0 + 1;
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (this.opts.ymin !== undefined) y0 = Math.min(y0, this.opts.ymin);
    if (this.opts.logy) { y0 = Math.floor(Math.log10(y0)); y1 = Math.ceil(Math.log10(y1)); if (y1 === y0) y1 = y0 + 1; }
    else { if (y1 === y0) { y1 = y0 + 1; } const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad; if (this.opts.bars) y0 = Math.min(0, y0); }
    const X = x => L + (x - x0) / (x1 - x0) * pw;
    const Y = y => T + ph - ((this.opts.logy ? Math.log10(y) : y) - y0) / (y1 - y0) * ph;
    // grid
    g.strokeStyle = grid; g.lineWidth = 1; g.beginPath();
    const ny = this.opts.logy ? (y1 - y0) : 5;
    for (let i = 0; i <= ny; i++) {
      const v = y0 + (y1 - y0) * i / ny; const yy = T + ph - (v - y0) / (y1 - y0) * ph;
      g.moveTo(L, yy); g.lineTo(L + pw, yy);
      g.fillText(this.opts.logy ? "1e" + Math.round(v) : fmt(v), 4, yy + 4);
    }
    for (let i = 0; i <= 5; i++) { const v = x0 + (x1 - x0) * i / 5, xx = X(v); g.moveTo(xx, T); g.lineTo(xx, T + ph); g.fillText(fmt(v), xx - 12, T + ph + 14); }
    g.stroke();
    g.strokeStyle = fg; g.strokeRect(L, T, pw, ph);
    if (this.opts.xlabel) g.fillText(this.opts.xlabel, L + pw / 2 - 20, H - 3);
    let ly = T + 12;
    for (const s of this.series) {
      g.strokeStyle = s.color || "#1f77b4"; g.fillStyle = s.color || "#1f77b4"; g.lineWidth = s.width || 1.6;
      if (this.opts.bars) {
        const w = pw / Math.max(s.data.length, 1) * 0.7;
        for (const [x, y] of s.data) { const yy = Y(Math.max(y, this.opts.logy ? 1e-30 : y)); g.fillRect(X(x) - w / 2, yy, w, T + ph - yy); }
      } else {
        g.beginPath(); let first = true;
        for (const [x, y] of s.data) { if (!isFinite(y) || (this.opts.logy && y <= 0)) continue; const xx = X(x), yy = Y(y); if (first) { g.moveTo(xx, yy); first = false; } else g.lineTo(xx, yy); }
        if (s.dash) g.setLineDash(s.dash); g.stroke(); g.setLineDash([]);
        if (s.markers) for (const [x, y] of s.data) { if (isFinite(y)) { g.beginPath(); g.arc(X(x), Y(y), 2.2, 0, 7); g.fill(); } }
      }
      if (s.label) { g.fillRect(L + pw - 120, ly - 7, 10, 3); g.fillStyle = fg; g.fillText(s.label, L + pw - 104, ly - 2); ly += 14; }
    }
  }
}
export function fmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return (+v.toPrecision(4)).toString();
}

// format with enough digits to distinguish values within [lo, hi]
export function fmtR(v, lo, hi) {
  if (v === null || v === undefined || !isFinite(v)) return "-";
  const span = Math.abs(hi - lo), mag = Math.max(Math.abs(hi), Math.abs(lo), 1e-300);
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(Math.min(8, Math.max(2, Math.ceil(Math.log10(mag / Math.max(span, mag * 1e-12))) + 1)));
  const digits = Math.min(12, Math.max(4, Math.ceil(Math.log10(mag / Math.max(span, mag * 1e-12))) + 3));
  return (+v.toPrecision(digits)).toString();
}
