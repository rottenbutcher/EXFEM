"""Spec -> Model/Steps builder, procedure runner and post-processing.

`run_spec` executes every step of a `Spec` in sequence (static, Riks, implicit/explicit
dynamics, frequency, buckling, heat transfer, topology optimisation), carries the solution
state between steps (including temperature fields from heat-transfer steps into subsequent
structural steps: sequential thermal-stress coupling), records result frames, and writes
ParaView files.
"""
import os
import time
import numpy as np
import jax.numpy as jnp

from .model import Model
from .elements import ELEMENTS
from .materials import from_props
from .surfaces import element_face_surface, Surface
from . import solvers as S
from . import loads as L
from . import constraints as C
from .contact import PenaltyContact, RigidWall


class BuildError(Exception):
    pass


def build(spec, log=print):
    X = np.asarray(spec.nodes)
    model = Model(X)
    model.node_ids = spec.node_ids
    model.units = spec.units
    ne = spec.nelem
    types, conns = spec.element_table()
    emat = np.full(ne, None, dtype=object)
    for sec in spec.sections:
        es = spec.elsets.get(sec["elset"])
        if es is None:
            raise BuildError(f"section refers to unknown elset {sec['elset']}")
        emat[es] = sec["material"]
    if any(x is None for x in emat):
        if len(spec.materials) == 1:
            emat[[i for i in range(ne) if emat[i] is None]] = next(iter(spec.materials))
        else:
            missing = sum(x is None for x in emat)
            spec.warnings.append(f"{missing} elements without section are ignored")
    nlgeom = any(s.get("nlgeom") for s in spec.steps)
    mats = {}
    for name, props in spec.materials.items():
        try:
            mats[name] = from_props(props, nlgeom, name)
        except Exception as e:  # noqa: BLE001
            raise BuildError(f"material {name}: {e}")
    # blocks = (element type, material)
    keyed = {}
    for i, (t, mname) in enumerate(zip(types, emat)):
        if mname is None:
            continue
        if t not in ELEMENTS:
            raise BuildError(f"element type {t} not supported")
        keyed.setdefault((ELEMENTS[t].name, mname, t), []).append(i)
    elem_block = np.full(ne, -1)
    elem_local = np.full(ne, -1)
    for (ename, mname, t), idx in keyed.items():
        conn = np.array([conns[i] for i in idx])
        b = model.add_block(conn, ELEMENTS[t], mats[mname], name=f"{mname}:{t}")
        elem_block[idx] = len(model.blocks) - 1
        elem_local[idx] = np.arange(len(idx))
    eid_all = np.concatenate([g.ids for g in spec.groups]) if spec.groups else np.zeros(0)
    model.elem_ids = [eid_all[np.nonzero(elem_block == bi)[0]] for bi in range(len(model.blocks))]
    model.elem_global = [np.nonzero(elem_block == bi)[0] for bi in range(len(model.blocks))]
    model.nsets = {k: np.asarray(v) for k, v in spec.nsets.items()}
    for k, v in spec.elsets.items():
        v = np.asarray(v, dtype=np.int64)
        v = v[elem_block[v] >= 0]
        model.elsets[k] = [(bi, elem_local[v[elem_block[v] == bi]]) for bi in np.unique(elem_block[v])]
    for name, sd in spec.surfaces.items():
        if "elems" in sd:
            el = np.asarray(sd["elems"], dtype=np.int64)
            ok = elem_block[el] >= 0
            tri = np.stack([elem_block[el[ok]], elem_local[el[ok]], np.asarray(sd["faces"])[ok]], 1)
            model.add_surface(name, element_face_surface(X, model.blocks, tri, name))
        else:
            s = Surface(name=name)
            s.node_only = np.asarray(sd["nodes"], dtype=np.int64)
            model.surfaces[name] = s
    Emod = model.modulus() if model.blocks else 1.0
    for c in spec.constraints:
        if c["type"] == "tie":
            sl, ms = _surf(model, c["slave"]), _surf(model, c["master"])
            slave = sl if len(sl) else getattr(sl, "node_only", sl.nodes())
            mpc = C.tie(X, slave, ms, tol=c.get("tol"), name=c.get("name", "tie"))
            log(f"  tie {c.get('name', '')}: {len(mpc) // 3} slave nodes tied")
            model.add_constraint(mpc)
        elif c["type"] == "normal":
            model.add_constraint(C.normal_constraint(c["nodes"], c["normal"], c.get("name", "normal")))
        elif c["type"] == "mpc":
            model.add_constraint(C.MPC(c["slave"], c["masters"], c["weights"], c.get("name", "equation")))
        elif c["type"] == "coupling":
            model.add_constraint(C.coupling(c["ref"], c["nodes"]))
    for c in spec.contacts:
        if c["type"] == "wall":
            nodes = _nodes(model, spec, c["slave"])
            k = c.get("stiffness") or c.get("fkn", 10.0) * Emod * model.char_length()
            model.add_interaction(RigidWall(nodes, c["point"], c["normal"], k, c.get("motion", (0, 0, 0)),
                                            name=c.get("name", "wall")))
            continue
        ms, sl = _surf(model, c["master"]), _surf(model, c["slave"])
        if c.get("auto_slave", True) and len(sl) and len(ms) and sl.char_length(X) > ms.char_length(X):
            ms, sl = sl, ms                         # the finer mesh becomes the slave
        slave = sl if len(sl) else getattr(sl, "node_only", None)
        model.add_interaction(PenaltyContact(X, ms, slave, E=Emod, fkn=c.get("fkn", 1.0),
                                             augmented=c.get("augmented", True), name=c.get("name", "contact")))
    plan = [_build_step(model, spec, s, elem_block) for s in spec.steps]
    return model, plan


def _surf(model, name):
    if isinstance(name, Surface):
        return name
    if name not in model.surfaces:
        raise BuildError(f"unknown surface {name}")
    return model.surfaces[name]


def _nodes(model, spec, ref):
    if isinstance(ref, str):
        if ref in model.nsets:
            return model.nsets[ref]
        s = model.surfaces[ref]
        return getattr(s, "node_only", s.nodes())
    return np.asarray(ref, dtype=np.int64)


def _blocks_of(model, elset):
    if elset is None or elset not in model.elsets:
        return None
    return [bi for bi, _ in model.elsets[elset]]


def _build_step(model, spec, s, elem_block):
    X = np.asarray(model.X)
    amps = {k: L.Amplitude(v, k) for k, v in spec.amplitudes.items()}
    amp = lambda n: amps.get(n.upper()) if isinstance(n, str) else None
    bcs = [(np.asarray(b["nodes"]), b["dof"], b["value"], amp(b.get("amp"))) for b in s["bcs"]]
    loads = [(np.asarray(c["nodes"]), c["dof"], c["value"], amp(c.get("amp"))) for c in s["cloads"]]
    pressures, dl = [], []
    for d in s["dsloads"]:
        surf = _surf(model, d["surface"])
        kind = d.get("kind", "P").upper()
        a = amp(d.get("amp"))
        if kind == "P":
            pressures.append((surf, d["value"], a, bool(d.get("follower")) and s.get("nlgeom")))
        elif kind in ("TRVEC", "TRACTION"):
            v = np.asarray(d.get("vector", (0, 0, 1)), float)
            v = v / np.linalg.norm(v) * d["value"]
            dl.append((L.traction_vector(X, surf, v), a))
        elif kind == "FORCE":
            dl.append((L.surface_force_vector(X, surf, d["vector"]), a))
        elif kind == "MOMENT":
            dl.append((L.surface_moment_vector(X, surf, d["vector"], d.get("center")), a))
    for d in s["dloads"]:
        kind = d["kind"].upper()
        blocks = _blocks_of(model, d.get("elset"))
        a = amp(d.get("amp"))
        if kind == "GRAV":
            v = np.asarray(d["vector"], float)
            v = v / np.linalg.norm(v) * d["value"]
            dl.append((L.body_force_vector(model, v, blocks), a))
        elif kind == "CENTRIF":
            dl.append((L.centrifugal_vector(model, np.sqrt(d["value"]), d["point"], d["axis"], blocks), a))
    temp = None
    if s["temperatures"]:
        temp = np.zeros(model.nn)
        for nodes, T in s["temperatures"]:
            temp[np.asarray(nodes, dtype=np.int64)] = T
    kind = s["procedure"]
    T = s["time"]
    step = S.Step(s["name"], kind, T, dt_init=s.get("dt_init") or T, dt_min=s.get("dt_min"),
                  dt_max=s.get("dt_max") or T, max_inc=s.get("max_inc", 1000), bcs=bcs, loads=loads,
                  pressures=pressures, dloads=dl, temperature=temp, nmodes=s.get("nmodes", 10),
                  riks=s.get("riks", {}), frames=s.get("frames", 20), damping=s.get("damping", 0.0),
                  alpha=s.get("alpha", -0.05), dt=s.get("dt"), mass_scaling=s.get("mass_scaling"),
                  output_every=s.get("output_every", 1), nlgeom=s.get("nlgeom"))
    step.spec = s
    step.temp_explicit = bool(s["temperatures"])
    return step


# ------------------------------------------------------------------ post-processing
def _nodal_average(model, elem_vals, vols):
    """Volume-weighted average of per-element values onto nodes: list per block (ne, k)."""
    nn = model.nn
    k = elem_vals[0].shape[1] if elem_vals[0].ndim > 1 else 1
    acc = np.zeros((nn, k))
    w = np.zeros(nn)
    for b, v, vol in zip(model.blocks, elem_vals, vols):
        conn = np.asarray(b.conn)
        v = np.asarray(v).reshape(len(conn), k)
        np.add.at(acc, conn, (v * vol[:, None])[:, None, :])
        np.add.at(w, conn, np.broadcast_to(vol[:, None], conn.shape))
    return acc / np.maximum(w, 1e-300)[:, None]


def _voigt(S):
    return np.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 1, 2], S[:, 0, 2]], 1)


def postprocess(model, u, states, fields=None, density=None, T=None, vols=None, extra=None):
    out = dict(extra or {})
    nn = model.nn
    if vols is None:
        vols = [np.asarray(v) for v in model.volumes()]
    if u is not None:
        out["U"] = np.asarray(u).reshape(-1, 3)
        st = model.stresses(u, states, fields)
        Se = [_voigt(np.asarray(x["S"])) for x in st]
        mises = [np.asarray(x["MISES"])[:, None] for x in st]
        peeq = [np.asarray(x["PEEQ"])[:, None] for x in st]
        Fm = [np.asarray(x["F"]) for x in st]
        strain = []
        for b, F in zip(model.blocks, Fm):
            if b.material.small_strain:
                e = 0.5 * (F + F.transpose(0, 2, 1)) - np.eye(3)
            else:
                C_ = np.einsum("eki,ekj->eij", F, F)
                w, V = np.linalg.eigh(C_)
                e = np.einsum("eij,ej,ekj->eik", V, 0.5 * np.log(np.maximum(w, 1e-300)), V)
            strain.append(_voigt(e))
        out["S"] = _nodal_average(model, Se, vols)
        out["MISES"] = _nodal_average(model, mises, vols)[:, 0]
        out["PEEQ"] = _nodal_average(model, peeq, vols)[:, 0]
        out["E"] = _nodal_average(model, strain, vols)
        out["S_elem"] = np.concatenate(Se)
        out["MISES_elem"] = np.concatenate([m[:, 0] for m in mises])
    if fields is not None and fields.get("T") is not None:
        out["T"] = np.asarray(fields["T"])
    if T is not None:
        out["T"] = np.asarray(T)
    if density is not None:
        out["density"] = np.asarray(density)
        dens_blocks, o = [], 0
        for b in model.blocks:
            dens_blocks.append(np.asarray(density[o:o + b.ne])[:, None])
            o += b.ne
        out["density_nodal"] = _nodal_average(model, dens_blocks, vols)[:, 0]
    return out


def principal(Sv):
    """Principal values (n, 3) descending from Voigt (n, 6) [xx yy zz xy yz xz]."""
    M = np.zeros((len(Sv), 3, 3))
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = Sv[:, 0], Sv[:, 1], Sv[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = Sv[:, 3]
    M[:, 1, 2] = M[:, 2, 1] = Sv[:, 4]
    M[:, 0, 2] = M[:, 2, 0] = Sv[:, 5]
    return np.linalg.eigvalsh(M)[:, ::-1]


class Recorder(S.Monitor):
    """Monitor that post-processes and stores frames, forwarding events to `inner`."""

    def __init__(self, model, inner=None, keep=True, max_frames=400):
        super().__init__(False)
        self.model, self.inner = model, inner or S.Monitor(True)
        self.frames = []
        self.keep = keep
        self.max_frames = max_frames
        self.vols = [np.asarray(v) for v in model.volumes()]
        self.step = None

    def log(self, msg):
        self.inner.log(msg)

    def iteration(self, **kw):
        self.inner.iteration(**kw)

    def increment(self, **kw):
        self.inner.increment(**kw)

    def should_stop(self):
        return self.inner.should_stop()

    def frame(self, t, u, states, **kw):
        fields = kw.pop("fields", None)
        T = kw.pop("T", None)
        dens = kw.pop("density", None)
        kw.pop("v", None)
        fr = postprocess(self.model, u, states, fields, density=dens, T=T, vols=self.vols,
                         extra=dict(time=float(t), **{k: v for k, v in kw.items()
                                                      if isinstance(v, (int, float, str))}))
        fr["step"] = kw.get("step", self.step)
        if len(self.frames) >= self.max_frames:
            self.frames = self.frames[::2]
        self.frames.append(fr)
        if hasattr(self.inner, "frame_ready"):
            self.inner.frame_ready(fr, len(self.frames) - 1)


def run_spec(spec, outdir=".", base="job", verbose=True, monitor=None, write=True, options=None):
    """Run every step of `spec`. Returns (model, results) where results is a list of
    dict(step, procedure, frames, info, files)."""
    opts = dict(linear_solver="auto", tangent="ebe")
    opts.update(options or {})
    mon = monitor or S.Monitor(verbose)
    t0 = time.time()
    model, plan = build(spec, mon.log)
    for w in spec.warnings:
        mon.log(f"  warning: {w}")
    import jax
    mon.log(f"{base}: {model.summary()}")
    mon.log(f"  devices: {jax.devices()}  (backend {jax.default_backend()})")
    state = S.State(model)
    if spec.initial_temperature is not None:
        model.T0 = jnp.asarray(spec.initial_temperature, dtype=float)
        state.T = model.T0
    Tcur = None if spec.initial_temperature is None else np.asarray(spec.initial_temperature)
    results = []
    if write:
        os.makedirs(outdir, exist_ok=True)
    for step in plan:
        if mon.should_stop():
            break
        rec = Recorder(model, mon)
        rec.step = step.name
        kind = step.kind
        mon.log(f"* step {step.name} ({kind}, T={step.T})")
        info = {}
        sp = step.spec
        if kind in ("static", "riks", "dynamic", "explicit") and not step.temp_explicit and Tcur is not None:
            step.temperature = Tcur
        if kind == "static":
            u, s, info = S.StaticSolver(model, monitor=rec, linear_solver=opts["linear_solver"],
                                        tangent=opts["tangent"]).solve(step, state=state)
        elif kind == "riks":
            u, s, info = S.RiksSolver(model, monitor=rec, linear_solver=opts["linear_solver"]).solve(step, state=state)
        elif kind == "dynamic":
            u, s, info = S.DynamicSolver(model, monitor=rec, linear_solver=opts["linear_solver"]).solve(step, state=state)
        elif kind == "explicit":
            S.ExplicitSolver(model, monitor=rec).solve(step, state=state)
        elif kind == "frequency":
            info = S.modal(model, step, state, monitor=rec)
        elif kind == "buckle":
            info = S.buckling(model, step, state, monitor=rec)
        elif kind == "heat":
            from .thermal import HeatTransfer
            h = sp["heat"]
            fixed = [(n, T) for n, T in h["fixed"]]
            flux = [(_surf(model, sname), q) for sname, q in h["flux"]]
            film = [(_surf(model, sname), hh, Ti) for sname, hh, Ti in h["film"]]
            gen = {}
            for es, q in h["gen"]:
                for bi in (_blocks_of(model, es) or range(len(model.blocks))):
                    gen[bi] = q
            tr = dict(time=step.T, dt=sp.get("dt_init"), frames=step.frames) if sp.get("transient") else None
            Tcur = HeatTransfer(model).solve(fixed, flux, film, gen, T0=Tcur, transient=tr, monitor=rec)
            state.T = None              # the next structural step ramps from its reference
            info = dict(T=Tcur)
        elif kind == "topopt":
            from .optimize import topology_optimization
            to = sp.get("topopt", {})
            frozen = None
            if to.get("frozen_nodes") is not None:
                fn = set(np.asarray(to["frozen_nodes"]).tolist())
                frozen = np.concatenate([np.array([any(n in fn for n in row) for row in np.asarray(b.conn)])
                                         for b in model.blocks])
            info = topology_optimization(model, step, volfrac=to.get("volfrac", 0.4),
                                         penal=to.get("penal", 3.0), rmin=to.get("rmin"),
                                         iters=to.get("iters", 40), frozen=frozen, monitor=rec,
                                         linear_solver=opts["linear_solver"])
        else:
            raise BuildError(f"procedure {kind} not supported")
        files = []
        if write and rec.frames:
            from .io.vtk import write_vtu, write_pvd
            entries = []
            for i, fr in enumerate(rec.frames):
                f = os.path.join(outdir, f"{base}_{step.name}_{i:04d}.vtu")
                write_vtu(f, model, fr)
                entries.append((fr.get("time", i), os.path.basename(f)))
                files.append(f)
            write_pvd(os.path.join(outdir, f"{base}_{step.name}.pvd"), entries)
        if rec.frames:
            last = rec.frames[-1]
            if "U" in last:
                mon.log(f"  max |U| = {np.linalg.norm(last['U'], axis=1).max():.5g}"
                        + (f", max Mises = {last['MISES'].max():.5g}" if "MISES" in last else ""))
            if "T" in last and kind == "heat":
                mon.log(f"  T range = [{last['T'].min():.5g}, {last['T'].max():.5g}]")
        results.append(dict(step=step.name, procedure=kind, frames=rec.frames, info=info, files=files))
    mon.log(f"analysis completed in {time.time() - t0:.2f}s")
    return model, results
