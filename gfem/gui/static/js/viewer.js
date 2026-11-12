// 3D viewport: CAD faces/edges, mesh, contour results, picking, highlights, legend, triad.
import * as THREE from "three";
import { TrackballControls } from "three/addons/TrackballControls.js";

export function b64f32(s) { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Float32Array(u.buffer); }
export function b64i32(s) { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new Int32Array(u.buffer); }

// ANSYS-like rainbow (blue -> red)
const RAINBOW = ["#0000ff", "#0055ff", "#00aaff", "#00ffff", "#00ffaa", "#00ff00", "#aaff00", "#ffff00", "#ffaa00", "#ff0000"];
const VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"];
const GRAY = ["#202020", "#404040", "#606060", "#808080", "#a0a0a0", "#c0c0c0", "#d8d8d8", "#ececec"];

function lerpColors(list, n) {
  const cs = list.map(c => new THREE.Color(c));
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1) * (cs.length - 1);
    const a = Math.floor(t), b = Math.min(a + 1, cs.length - 1);
    out.push(cs[a].clone().lerp(cs[b], t - a));
  }
  return out;
}

const contourVS = `
attribute float scalar;
uniform float vmin; uniform float vmax;
uniform float scale;
attribute vec3 disp;
varying float vt; varying vec3 vN; varying vec3 vPos;
void main(){
  vt = (scalar - vmin) / max(vmax - vmin, 1e-30);
  vec3 p = position + scale * disp;
  vN = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(p, 1.0);
  vPos = mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
  #include <clipping_planes_vertex>
}`;
const contourFS = `
uniform sampler2D cmap; uniform float bands; uniform float smooth_;
varying float vt; varying vec3 vN; varying vec3 vPos;
#include <clipping_planes_pars_fragment>
void main(){
  #include <clipping_planes_fragment>
  float t = clamp(vt, 0.0, 1.0);
  float tb = smooth_ > 0.5 ? t : (floor(min(t, 0.99999) * bands) + 0.5) / bands;
  vec3 c = texture2D(cmap, vec2(tb, 0.5)).rgb;
  vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
  vec3 L = normalize(vec3(0.3, 0.5, 1.0));
  float d = 0.62 + 0.38 * abs(dot(n, L));
  gl_FragColor = vec4(c * d, 1.0);
}`;
const contourVSclip = contourVS.replace("varying float vt;", "#include <clipping_planes_pars_vertex>\nvarying float vt;");

export class Viewer {
  constructor(el, opts = {}) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.localClippingEnabled = true;
    this.renderer.autoClear = false;
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 1e7);
    this.camera.position.set(1, 1, 1);
    this.controls = new TrackballControls(this.camera, this.renderer.domElement);
    this.controls.rotateSpeed = 3.0; this.controls.zoomSpeed = 1.6; this.controls.panSpeed = 0.9;
    this.controls.staticMoving = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
    this.light = new THREE.DirectionalLight(0xffffff, 1.5); this.scene.add(this.light);
    this.light2 = new THREE.DirectionalLight(0xffffff, 0.5); this.scene.add(this.light2);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.mode = "geometry";
    this.filter = "face";
    this.selection = { dim: 2, ids: new Set() };
    this.groups = { geom: new THREE.Group(), mesh: new THREE.Group(), result: new THREE.Group(), hl: new THREE.Group(), marks: new THREE.Group() };
    for (const g of Object.values(this.groups)) this.root.add(g);
    this.faceColor = new THREE.Color("#b8c4d4");
    this.bodyColors = {};
    this.showEdges = true; this.showMeshLines = true; this.wire = false;
    this.clip = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 1e9); this.clipOn = false;
    this.onSelect = null; this.onProbe = null; this.onHover = null;
    this.probeMode = false;
    this.bands = 9; this.smooth = false; this.cmapName = "rainbow";
    this.result = null; this.deform = 1.0; this.anim = null;
    this._initTriad();
    this._initEvents();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(el);
    this.renderer.setAnimationLoop(() => this._tick());
  }

  // ---------------------------------------------------------------- setup
  _initTriad() {
    this.triScene = new THREE.Scene();
    this.triCam = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10);
    const mk = (dir, color, label) => {
      const a = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 1.0, color, 0.28, 0.16);
      this.triScene.add(a);
      const c = document.createElement("canvas"); c.width = c.height = 64;
      const g = c.getContext("2d"); g.fillStyle = "#" + new THREE.Color(color).getHexString(); g.font = "bold 44px sans-serif";
      g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(label, 32, 34);
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
      s.position.copy(dir.clone().multiplyScalar(1.32)); s.scale.set(0.45, 0.45, 0.45);
      this.triScene.add(s);
    };
    mk(new THREE.Vector3(1, 0, 0), 0xd62728, "X"); mk(new THREE.Vector3(0, 1, 0), 0x2ca02c, "Y"); mk(new THREE.Vector3(0, 0, 1), 0x1f77b4, "Z");
  }

  _initEvents() {
    const dom = this.renderer.domElement;
    let down = null;
    dom.addEventListener("pointerdown", e => { down = { x: e.clientX, y: e.clientY, b: e.button, t: performance.now() }; });
    dom.addEventListener("pointerup", e => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      if (moved < 4 && down.b === 0) this._click(e);
      down = null;
    });
    dom.addEventListener("pointermove", e => { if (!down && this.onHover) this._hover(e); });
    dom.addEventListener("dblclick", () => this.fit());
    dom.addEventListener("contextmenu", e => e.preventDefault());
  }

  resize() {
    const w = this.el.clientWidth || 300, h = this.el.clientHeight || 200;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    this.controls.handleResize && this.controls.handleResize();
  }

  _tick() {
    this.controls.update();
    if (this.anim && this.result) this._animate();
    const cam = this.camera;
    this.light.position.copy(cam.position).add(new THREE.Vector3().copy(cam.up).multiplyScalar(this.radius || 1));
    this.light2.position.copy(cam.position).multiplyScalar(-1);
    const r = this.renderer;
    r.setClearColor(0x000000, 0); r.clear();
    r.setViewport(0, 0, this.el.clientWidth, this.el.clientHeight);
    r.clippingPlanes = [];
    r.render(this.scene, this.camera);
    // triad
    const s = 110;
    r.clearDepth();
    r.setViewport(8, 8, s, s);
    this.triCam.position.copy(cam.position).sub(this.controls.target).normalize().multiplyScalar(4);
    this.triCam.up.copy(cam.up); this.triCam.lookAt(0, 0, 0);
    r.render(this.triScene, this.triCam);
    if (this._marksDirty !== false) this._updateMarks();
  }

  // ---------------------------------------------------------------- content
  clear(group) { const g = this.groups[group]; while (g.children.length) { const c = g.children.pop(); c.geometry && c.geometry.dispose(); } }

  setBodyColors(map) { this.bodyColors = map || {}; if (this.geomData) this.setGeometry(this.geomData, false); }

  setGeometry(data, fit = true) {
    this.geomData = data;
    this.clear("geom");
    const info = data.info; if (!info) { this.faceTri = null; return; }
    const faceBody = {}; for (const f of info.faces) faceBody[f.id] = (f.bodies || [])[0];
    this.faceInfo = {}; for (const f of info.faces) this.faceInfo[f.id] = f;
    this.vertexInfo = info.vertices || [];
    const pos = [], col = [], idx = [], triFace = [];
    let off = 0;
    this.facePos = {};
    for (const f of data.faces) {
      const xyz = f.xyzA || (f.xyzA = b64f32(f.xyz)); const tri = f.triA || (f.triA = b64i32(f.tri));
      const c = new THREE.Color(this.bodyColors[faceBody[f.id]] || this.faceColor);
      for (let i = 0; i < xyz.length; i++) pos.push(xyz[i]);
      for (let i = 0; i < xyz.length / 3; i++) col.push(c.r, c.g, c.b);
      for (let i = 0; i < tri.length; i++) idx.push(tri[i] + off);
      for (let i = 0; i < tri.length / 3; i++) triFace.push(f.id);
      this.facePos[f.id] = { xyz, tri };
      off += xyz.length / 3;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals(); g.computeBoundingSphere();
    const mat = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 30, specular: 0x222222,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, clippingPlanes: [this.clip], clipShadows: true });
    this.faceMesh = new THREE.Mesh(g, mat); this.faceMesh.userData.triFace = triFace;
    this.groups.geom.add(this.faceMesh);
    // edges
    const ep = [], segEdge = [];
    this.edgePos = {};
    for (const e of data.edges) {
      const xyz = e.xyzA || (e.xyzA = b64f32(e.xyz)); this.edgePos[e.id] = xyz;
      for (let i = 0; i + 5 < xyz.length; i += 3) { ep.push(xyz[i], xyz[i + 1], xyz[i + 2], xyz[i + 3], xyz[i + 4], xyz[i + 5]); segEdge.push(e.id); }
    }
    const eg = new THREE.BufferGeometry(); eg.setAttribute("position", new THREE.Float32BufferAttribute(ep, 3));
    this.edgeLines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x101010, clippingPlanes: [this.clip] }));
    this.edgeLines.userData.segEdge = segEdge; this.edgeLines.visible = this.showEdges;
    this.groups.geom.add(this.edgeLines);
    const bb = info.bbox; this.bbox = new THREE.Box3(new THREE.Vector3(bb[0], bb[1], bb[2]), new THREE.Vector3(bb[3], bb[4], bb[5]));
    this.radius = Math.max(info.diag / 2, 1e-9);
    this.camera.near = this.radius / 500; this.camera.far = this.radius * 500; this.camera.updateProjectionMatrix();
    if (fit) this.fit();
    this.refreshHighlights();
  }

  setMesh(m) {
    this.meshData = m;
    this.clear("mesh");
    if (!m || !m.xyz) return;
    const xyz = m.xyzA || (m.xyzA = b64f32(m.xyz)); const tri = m.triA || (m.triA = b64i32(m.tri));
    const fot = m.fotA || (m.fotA = b64i32(m.face_of_tri)); const ed = m.edA || (m.edA = b64i32(m.edges));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3)); g.setIndex(new THREE.BufferAttribute(new Uint32Array(tri), 1));
    g.computeVertexNormals(); g.computeBoundingSphere();
    const mm = new THREE.Mesh(g, new THREE.MeshPhongMaterial({ color: 0x9fb8d0, side: THREE.DoubleSide, shininess: 15, polygonOffset: true,
      polygonOffsetFactor: 1, polygonOffsetUnits: 1, clippingPlanes: [this.clip] }));
    mm.userData.triFace = fot; this.meshMesh = mm; this.groups.mesh.add(mm);
    const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(xyz, 3)); lg.setIndex(new THREE.BufferAttribute(new Uint32Array(ed), 1));
    this.meshLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x1b2b40, transparent: true, opacity: 0.55, clippingPlanes: [this.clip] }));
    this.meshLines.visible = this.showMeshLines;
    this.groups.mesh.add(this.meshLines);
    if (!this.bbox) { g.computeBoundingBox(); this.bbox = g.boundingBox.clone(); this.radius = g.boundingSphere.radius; this.fit(); }
  }

  _cmapTexture() {
    const base = this.cmapName === "viridis" ? VIRIDIS : this.cmapName === "gray" ? GRAY : RAINBOW;
    const n = this.smooth ? 256 : this.bands;
    const cols = lerpColors(base, n);
    const data = new Uint8Array(n * 4);
    cols.forEach((c, i) => { data[i * 4] = c.r * 255; data[i * 4 + 1] = c.g * 255; data[i * 4 + 2] = c.b * 255; data[i * 4 + 3] = 255; });
    const t = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat); t.needsUpdate = true;
    t.magFilter = this.smooth ? THREE.LinearFilter : THREE.NearestFilter; t.minFilter = t.magFilter;
    this.bandColors = cols;
    return t;
  }

  setResult(r, opts = {}) {
    // r: {values(Float32Array|null), U(Float32Array|null), min, max, imin, imax}
    this.result = r;
    this.clear("result");
    const m = this.meshData; if (!m || !m.xyzA) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(m.xyzA, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(m.triA), 1));
    const n = m.xyzA.length / 3;
    const vals = r.values || new Float32Array(n);
    g.setAttribute("scalar", new THREE.BufferAttribute(vals, 1));
    g.setAttribute("disp", new THREE.BufferAttribute(r.U || new Float32Array(n * 3), 3));
    g.computeVertexNormals(); g.computeBoundingSphere();
    let vmin = opts.min ?? r.min ?? 0, vmax = opts.max ?? r.max ?? 1;
    if (vmax - vmin < 1e-30 * Math.max(1, Math.abs(vmax))) { vmax = vmin + 1e-30 + Math.abs(vmin) * 1e-6; }
    this.range = [vmin, vmax];
    const mat = new THREE.ShaderMaterial({
      uniforms: { vmin: { value: vmin }, vmax: { value: vmax }, scale: { value: this.deform }, cmap: { value: this._cmapTexture() },
        bands: { value: this.bands }, smooth_: { value: this.smooth ? 1 : 0 } },
      vertexShader: contourVSclip, fragmentShader: contourFS, side: THREE.DoubleSide, clipping: true,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, clippingPlanes: [this.clip] });
    if (!r.values) { mat.fragmentShader = contourFS.replace("vec3 c = texture2D(cmap, vec2(tb, 0.5)).rgb;", "vec3 c = vec3(0.62,0.72,0.82);"); }
    this.resMesh = new THREE.Mesh(g, mat); this.groups.result.add(this.resMesh);
    // deformed wire (element edges) and undeformed outline
    const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.xyzA), 3));
    lg.setIndex(new THREE.BufferAttribute(new Uint32Array(m.edA), 1));
    this.resLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25, clippingPlanes: [this.clip] }));
    this.resLines.visible = !!opts.showMesh; this.groups.result.add(this.resLines);
    if (opts.undeformed && this.edgeLines) {
      const u = this.edgeLines.clone(); u.material = new THREE.LineBasicMaterial({ color: 0x555555, transparent: true, opacity: 0.35 }); u.visible = true;
      this.groups.result.add(u);
    }
    this._applyDeform(this.deform);
    this._marksDirty = true;
  }

  setDeform(s) { this.deform = s; this._applyDeform(s); }
  _applyDeform(s) {
    if (!this.resMesh) return;
    this.resMesh.material.uniforms.scale.value = s;
    if (this.resLines && this.result && this.result.U) {
      const p = this.resLines.geometry.attributes.position.array, x = this.meshData.xyzA, U = this.result.U;
      for (let i = 0; i < p.length; i++) p[i] = x[i] + s * U[i];
      this.resLines.geometry.attributes.position.needsUpdate = true;
    }
    this._marksDirty = true;
  }
  setBands(n, smooth) { this.bands = n; this.smooth = smooth; if (this.result) this.setResult(this.result, { min: this.range[0], max: this.range[1], showMesh: this.resLines?.visible }); }

  play(kind) { this.anim = kind ? { kind, t0: performance.now() } : null; if (!kind) this._applyDeform(this.deform); }
  _animate() {
    const t = (performance.now() - this.anim.t0) / 1000;
    const f = this.anim.kind === "mode" ? Math.sin(t * 2 * Math.PI * 0.8) : 0.5 - 0.5 * Math.cos(t * 2 * Math.PI * 0.4);
    this._applyDeform(this.deform * f);
  }

  setMode(mode) {
    this.mode = mode;
    this.groups.geom.visible = mode === "geometry";
    this.groups.mesh.visible = mode === "mesh";
    this.groups.result.visible = mode === "result";
    this.groups.hl.visible = mode !== "result";
    this.refreshHighlights();
    this._marksDirty = true;
  }

  setEdges(on) { this.showEdges = on; if (this.edgeLines) this.edgeLines.visible = on; }
  setMeshLines(on) { this.showMeshLines = on; if (this.meshLines) this.meshLines.visible = on; if (this.resLines) this.resLines.visible = on && this.mode === "result" && !!this.result; }
  setWire(on) { for (const m of [this.faceMesh, this.meshMesh, this.resMesh]) if (m) m.material.wireframe = on; }
  setSection(on, axis = 0, frac = 0.5, flip = false) {
    this.clipOn = on;
    if (!this.bbox) return;
    const n = new THREE.Vector3(); n.setComponent(axis, flip ? 1 : -1);
    const lo = this.bbox.min.getComponent(axis), hi = this.bbox.max.getComponent(axis);
    const pos = lo + (hi - lo) * frac;
    this.clip.set(n, on ? (flip ? -pos : pos) : 1e12);
  }

  // ---------------------------------------------------------------- camera
  fit() {
    if (!this.bbox) return;
    const c = this.bbox.getCenter(new THREE.Vector3());
    const r = this.bbox.getSize(new THREE.Vector3()).length() / 2 || 1;
    this.radius = r;
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    if (!isFinite(dir.x) || dir.length() < 0.5) dir.set(1, 0.8, 1.2).normalize();
    const d = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.08;
    this.controls.target.copy(c); this.camera.position.copy(c.clone().add(dir.multiplyScalar(d)));
    this.camera.near = r / 500; this.camera.far = r * 500; this.camera.updateProjectionMatrix();
  }
  view(name) {
    const dirs = { iso: [1, 0.8, 1.2], front: [0, 0, 1], back: [0, 0, -1], top: [0, 1, 0], bottom: [0, -1, 0], right: [1, 0, 0], left: [-1, 0, 0] };
    const ups = { top: [0, 0, -1], bottom: [0, 0, 1] };
    const d = new THREE.Vector3(...dirs[name]).normalize();
    this.camera.up.set(...(ups[name] || [0, 1, 0]));
    this.camera.position.copy(this.controls.target.clone().add(d));
    this.fit();
  }
  screenshot() { return this.renderer.domElement.toDataURL("image/png"); }

  // ---------------------------------------------------------------- picking
  _ray(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    const m = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster(); rc.setFromCamera(m, this.camera);
    rc.params.Line.threshold = (this.radius || 1) * 0.012;
    return rc;
  }
  _pickFace(rc) {
    const target = this.mode === "geometry" ? this.faceMesh : this.mode === "mesh" ? this.meshMesh : this.resMesh;
    if (!target) return null;
    const hits = rc.intersectObject(target).filter(h => !this.clipOn || this.clip.distanceToPoint(h.point) >= 0);
    if (!hits.length) return null;
    const h = hits[0];
    const tf = (this.mode === "geometry" ? this.faceMesh : this.meshMesh)?.userData.triFace;
    return { face: tf ? tf[h.faceIndex] : null, point: h.point, h };
  }
  _click(e) {
    const rc = this._ray(e);
    if (this.probeMode && this.mode === "result" && this.result) {
      const p = this._pickFace(rc);
      if (p && this.onProbe) {
        const tri = this.meshData.triA, i0 = p.h.face.a, i1 = p.h.face.b, i2 = p.h.face.c;
        const x = this.meshData.xyzA; let best = i0, bd = 1e300;
        for (const i of [i0, i1, i2]) { const d = (x[3 * i] - p.point.x) ** 2 + (x[3 * i + 1] - p.point.y) ** 2 + (x[3 * i + 2] - p.point.z) ** 2; if (d < bd) { bd = d; best = i; } }
        this.onProbe(best, p.point);
      }
      return;
    }
    let dim = { body: 3, face: 2, edge: 1, vertex: 0 }[this.filter];
    let id = null;
    if (dim === 1 && this.mode === "geometry" && this.edgeLines) {
      const hits = rc.intersectObject(this.edgeLines);
      if (hits.length) id = this.edgeLines.userData.segEdge[Math.floor(hits[0].index / 2)];
    } else {
      const p = this._pickFace(rc);
      if (p && p.face != null) {
        if (dim === 2) id = p.face;
        else if (dim === 3) id = (this.faceInfo?.[p.face]?.bodies || [])[0] ?? null;
        else if (dim === 0) {
          let best = null, bd = 1e300;
          for (const v of this.vertexInfo || []) { const d = p.point.distanceTo(new THREE.Vector3(...v.xyz)); if (d < bd) { bd = d; best = v.id; } }
          if (best != null && bd < (this.radius || 1) * 0.06) id = best;
        }
      }
    }
    const add = e.ctrlKey || e.metaKey || e.shiftKey;
    if (this.selection.dim !== dim) this.selection = { dim, ids: new Set() };
    if (id == null) { if (!add) this.selection.ids.clear(); }
    else if (add) { this.selection.ids.has(id) ? this.selection.ids.delete(id) : this.selection.ids.add(id); }
    else { this.selection.ids = new Set([id]); }
    this.refreshHighlights();
    this.onSelect && this.onSelect(this.selection);
  }
  _hover(e) {
    if (this._hovT && performance.now() - this._hovT < 60) return; this._hovT = performance.now();
    const p = this._pickFace(this._ray(e)); this.onHover(p);
  }
  setSelection(dim, ids) { this.selection = { dim, ids: new Set(ids) }; this.refreshHighlights(); }
  clearSelection() { this.selection.ids.clear(); this.refreshHighlights(); this.onSelect && this.onSelect(this.selection); }
  selectAll() {
    const dim = { body: 3, face: 2, edge: 1, vertex: 0 }[this.filter];
    const info = this.geomData?.info; if (!info) return;
    const list = dim === 3 ? info.bodies : dim === 2 ? info.faces : dim === 1 ? info.edges : info.vertices;
    this.setSelection(dim, list.map(x => x.id)); this.onSelect && this.onSelect(this.selection);
  }

  // highlights: [{dim, ids, color}] + current selection (green)
  setHighlights(list) { this.hl = list || []; this.refreshHighlights(); }
  refreshHighlights() {
    this.clear("hl");
    const items = [...(this.hl || [])];
    if (this.selection.ids.size) items.push({ dim: this.selection.dim, ids: [...this.selection.ids], color: "#22dd44", sel: true });
    for (const it of items) this._highlight(it);
  }
  _faceIdsOf(dim, ids) {
    if (dim === 2) return ids;
    if (dim === 3 && this.faceInfo) return Object.values(this.faceInfo).filter(f => (f.bodies || []).some(b => ids.includes(b))).map(f => f.id);
    return [];
  }
  _highlight({ dim, ids, color, sel }) {
    const c = new THREE.Color(color);
    if (dim === 2 || dim === 3) {
      const fids = this._faceIdsOf(dim, ids);
      const pos = [], idx = [];
      let off = 0;
      for (const f of fids) {
        const fp = this.facePos?.[f]; if (!fp) continue;
        for (let i = 0; i < fp.xyz.length; i++) pos.push(fp.xyz[i]);
        for (let i = 0; i < fp.tri.length; i++) idx.push(fp.tri[i] + off);
        off += fp.xyz.length / 3;
      }
      if (!pos.length) return;
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
      const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: sel ? 0.55 : 0.5, side: THREE.DoubleSide,
        depthTest: true, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2, clippingPlanes: [this.clip] }));
      m.renderOrder = 2; this.groups.hl.add(m);
    } else if (dim === 1) {
      const ep = [];
      for (const e of ids) { const x = this.edgePos?.[e]; if (!x) continue; for (let i = 0; i + 5 < x.length; i += 3) ep.push(x[i], x[i + 1], x[i + 2], x[i + 3], x[i + 4], x[i + 5]); }
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(ep, 3));
      const l = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: c, linewidth: 3, depthTest: false })); l.renderOrder = 3; this.groups.hl.add(l);
      // thick look: sprites at segment midpoints
    } else if (dim === 0) {
      const pts = []; for (const v of this.vertexInfo || []) if (ids.includes(v.id)) pts.push(...v.xyz);
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      const p = new THREE.Points(g, new THREE.PointsMaterial({ color: c, size: 10, sizeAttenuation: false, depthTest: false })); p.renderOrder = 3; this.groups.hl.add(p);
    }
  }

  // ---------------------------------------------------------------- min/max markers
  setMarks(list) { this.marks = list; this._marksDirty = true; }
  _updateMarks() {
    const layer = this.el.querySelector(".marks"); if (!layer) return;
    layer.innerHTML = "";
    if (this.mode !== "result" || !this.result || !this.result.values || !this.meshData) return;
    const x = this.meshData.xyzA, U = this.result.U;
    const s = this.resMesh ? this.resMesh.material.uniforms.scale.value : 0;
    const put = (i, label, cls) => {
      const v = new THREE.Vector3(x[3 * i] + (U ? s * U[3 * i] : 0), x[3 * i + 1] + (U ? s * U[3 * i + 1] : 0), x[3 * i + 2] + (U ? s * U[3 * i + 2] : 0));
      v.project(this.camera);
      if (v.z > 1) return;
      const d = document.createElement("div"); d.className = "mark " + cls; d.textContent = label;
      d.style.left = ((v.x + 1) / 2 * this.el.clientWidth) + "px"; d.style.top = ((1 - v.y) / 2 * this.el.clientHeight) + "px";
      layer.appendChild(d);
    };
    if (this.showMinMax !== false) { put(this.result.imax, "Max", "max"); put(this.result.imin, "Min", "min"); }
    for (const p of this.probes || []) put(p.i, p.label, "probe");
  }
}
