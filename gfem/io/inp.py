"""Abaqus .inp reader and writer (through the neutral `Spec`).

Reader coverage
  model   : *INCLUDE, *HEADING, *PART/*END PART, *ASSEMBLY, *INSTANCE (translation + rotation),
            *NODE, *ELEMENT (C3D4/C3D10/C3D8/C3D8R/C3D8H/C3D8I/C3D20/C3D20R/C3D6 ...), *NSET, *ELSET
            (GENERATE, INSTANCE=, nested sets), *SURFACE (ELEMENT / NODE), *SOLID SECTION
  material: *MATERIAL, *ELASTIC (ISOTROPIC, ENGINEERING CONSTANTS), *PLASTIC (ISOTROPIC,
            KINEMATIC, COMBINED, JOHNSON COOK), *HYPERELASTIC (NEO HOOKE, MOONEY-RIVLIN, YEOH),
            *DENSITY, *EXPANSION (ZERO=), *CONDUCTIVITY, *SPECIFIC HEAT
  interact: *TIE, *CONTACT PAIR (frictionless penalty / augmented Lagrange), *AMPLITUDE,
            *INITIAL CONDITIONS TYPE=TEMPERATURE
  steps   : *STEP (NLGEOM), *STATIC (RIKS), *DYNAMIC (implicit HHT / EXPLICIT), *FREQUENCY,
            *BUCKLE, *HEAT TRANSFER (STEADY STATE / transient)
  loads   : *BOUNDARY (dof ranges, ENCASTRE/PINNED/XSYMM/..., OP=NEW, AMPLITUDE=, dof 11 = temp.),
            *CLOAD, *DSLOAD (P, TRVEC), *DLOAD (Pn element faces, GRAV, CENTRIF), *TEMPERATURE,
            *DSFLUX, *SFILM, *DFLUX
"""
import os
import numpy as np
from ..elements import ELEMENTS
from ..structural import STRUCT_ELEMENTS

ALL_ELEMENTS = dict(ELEMENTS, **STRUCT_ELEMENTS)
from ..spec import Spec, new_step


def _lines(path):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.upper().startswith("*INCLUDE"):
                p = [x.strip() for x in s.split(",")]
                for q in p[1:]:
                    if q.upper().startswith("INPUT"):
                        fn = q.split("=", 1)[1].strip().strip('"')
                        yield from _lines(os.path.join(base, fn))
                continue
            yield s


def _tokenize(path):
    cards = []
    for s in _lines(path):
        if not s or s.startswith("**"):
            continue
        if s.startswith("*"):
            parts = [p.strip() for p in s[1:].split(",")]
            params = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    params[k.strip().upper()] = v.strip().strip('"')
                elif p:
                    params[p.upper()] = True
            cards.append([parts[0].upper(), params, []])
        elif cards:
            cards[-1][2].append([t.strip() for t in s.rstrip(",").split(",")])
    return cards


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


class _Part:
    def __init__(self, name):
        self.name = name
        self.node_ids, self.coords = [], []
        self.elements = []              # (type, id, nodes)
        self.nsets, self.elsets = {}, {}
        self.surfaces = {}              # name -> [(elset or elem id, face)] or ("NODE", [refs])
        self.sections = []


class InpReader:
    SYM = {"ENCASTRE": [0, 1, 2, 3, 4, 5], "PINNED": [0, 1, 2], "XSYMM": [0, 4, 5], "YSYMM": [1, 3, 5],
           "ZSYMM": [2, 3, 4], "XASYMM": [1, 2, 3], "YASYMM": [0, 2, 4], "ZASYMM": [0, 1, 5]}

    def __init__(self, path):
        self.path = path
        self.spec = Spec(os.path.basename(path))
        self.parts = {}
        self.cur = None                 # current part/assembly collector
        self.root = _Part("")           # flat model (no parts)
        self.instances = []             # (inst name, part, transform)
        self.asm = _Part("ASSEMBLY")
        self.materials = {}
        self.interactions = {}
        self.contact_pairs, self.ties = [], []
        self.amplitudes = {}
        self.init_temp = []
        self.steps = []
        self._parse(_tokenize(path))
        self._assemble()

    # ---------------------------------------------------------- parsing
    def _parse(self, cards):
        tgt = self.root
        mat = None
        step = None
        in_asm = False
        for kw, p, data in cards:
            if kw == "PART":
                tgt = self.parts.setdefault(p["NAME"].upper(), _Part(p["NAME"].upper()))
            elif kw == "END PART":
                tgt = self.root
            elif kw == "ASSEMBLY":
                in_asm = True
                tgt = self.asm
            elif kw == "END ASSEMBLY":
                in_asm = False
                tgt = self.root
            elif kw == "INSTANCE":
                tr = [_f(x) for x in (data[0] if data else [])]
                rot = [_f(x) for x in data[1]] if len(data) > 1 else None
                self.instances.append((p["NAME"].upper(), p["PART"].upper(), tr, rot))
                tgt = _Part("_inst")
            elif kw == "END INSTANCE":
                tgt = self.asm if in_asm else self.root
            elif kw == "NODE":
                for d in data:
                    tgt.node_ids.append(int(d[0]))
                    tgt.coords.append([_f(x) for x in (d[1:4] + ["0"] * 3)[:3]])
                    if "NSET" in p:
                        tgt.nsets.setdefault(p["NSET"].upper(), []).append(int(d[0]))
            elif kw == "ELEMENT":
                et = p["TYPE"].upper()
                if et not in ALL_ELEMENTS:
                    raise ValueError(f"element type {et} not supported")
                flat = [x for d in data for x in d if x]
                nn = ALL_ELEMENTS[et].nnode
                for i in range(0, len(flat), nn + 1):
                    eid = int(flat[i])
                    tgt.elements.append((et, eid, [int(x) for x in flat[i + 1:i + 1 + nn]]))
                    if "ELSET" in p:
                        tgt.elsets.setdefault(p["ELSET"].upper(), []).append(eid)
            elif kw in ("NSET", "ELSET"):
                sets = tgt.nsets if kw == "NSET" else tgt.elsets
                key = p[kw].upper()
                inst = p.get("INSTANCE", "").upper()
                vals = sets.setdefault(key, [])
                if "GENERATE" in p:
                    for d in data:
                        a, b, c = (int(x) for x in (d + ["1"])[:3])
                        vals.extend((inst, i) if inst else i for i in range(a, b + 1, c))
                else:
                    if kw == "NSET" and "ELSET" in p and not data:
                        vals.append(("ELSET_NODES", p["ELSET"].upper()))
                    for x in (x for d in data for x in d if x):
                        if x.lstrip("-").isdigit():
                            vals.append((inst, int(x)) if inst else int(x))
                        else:
                            vals.append(("SET", x.upper()))
            elif kw == "SURFACE":
                name = p["NAME"].upper()
                typ = str(p.get("TYPE", "ELEMENT")).upper()
                rows = [(d[0].upper(), d[1].upper() if len(d) > 1 and d[1] else "") for d in data]
                tgt.surfaces[name] = (typ, rows)
            elif kw == "SOLID SECTION":
                tgt.sections.append((p["ELSET"].upper(), p["MATERIAL"].upper()))
            elif kw == "SHELL SECTION":
                th = _f(data[0][0], 1.0) if data and data[0] and data[0][0] else 1.0
                tgt.sections.append((p["ELSET"].upper(), p["MATERIAL"].upper(), dict(kind="shell", thickness=th)))
            elif kw in ("BEAM SECTION", "BEAM GENERAL SECTION"):
                prof = str(p.get("SECTION", "RECT")).upper()
                dims = [_f(x) for x in data[0] if x] if data else [1.0, 1.0]
                n1 = [_f(x) for x in data[1][:3]] if len(data) > 1 and any(x for x in data[1]) else None
                mat_ = p.get("MATERIAL", "").upper()
                if kw == "BEAM GENERAL SECTION":
                    prof = "GENERAL"
                    A, I11, I12, I22, J = (dims + [0] * 5)[:5]
                    dims = [A, I11, I22, J]
                    if not mat_:
                        E_, G_ = ([_f(x) for x in data[2]] + [0, 0])[:2] if len(data) > 2 else (1.0, 0.4)
                        mat_ = f"_BEAMGEN_{p['ELSET'].upper()}"
                        self.materials[mat_] = dict(E=E_, nu=E_ / (2 * G_) - 1 if G_ else 0.3)
                tgt.sections.append((p["ELSET"].upper(), mat_, dict(kind="beam", profile=prof, dims=dims, n1=n1)))
            elif kw == "MATERIAL":
                mat = self.materials.setdefault(p["NAME"].upper(), {})
            elif kw == "ELASTIC":
                typ = str(p.get("TYPE", "ISOTROPIC")).upper()
                flat = [_f(x) for d in data for x in d if x]
                if typ.startswith("ENGINEERING"):
                    E1, E2, E3, n12, n13, n23, G12, G13, G23 = flat[:9]
                    mat["ortho"] = [E1, E2, E3, n12, n13, n23, G12, G13, G23]
                    mat.setdefault("E", max(E1, E2, E3))
                    mat.setdefault("nu", n12)
                else:
                    mat["E"], mat["nu"] = flat[0], flat[1]
            elif kw == "PLASTIC":
                hard = str(p.get("HARDENING", "ISOTROPIC")).upper()
                if hard.startswith("JOHNSON"):
                    v = [_f(x) for x in data[0]]
                    mat["johnson_cook"] = dict(A=v[0], B=v[1], n=v[2], m=v[3] if len(v) > 3 else 1.0,
                                               Tmelt=v[4] if len(v) > 4 else 1e30,
                                               Ttrans=v[5] if len(v) > 5 else 0.0)
                else:
                    tab = [[_f(d[0]), _f(d[1])] for d in data]
                    if hard == "KINEMATIC":
                        sy0 = tab[0][0]
                        if len(tab) > 1:
                            Hk = (tab[1][0] - tab[0][0]) / max(tab[1][1] - tab[0][1], 1e-30)
                        else:
                            Hk = 0.0
                        mat["Hk"] = Hk
                        tab = [[sy0, 0.0]]
                    mat["plastic"] = tab
            elif kw == "CYCLIC HARDENING":
                pass
            elif kw == "HYPERELASTIC":
                flat = [_f(x) for d in data for x in d if x]
                if "MOONEY-RIVLIN" in p:
                    mat["mooney"] = flat[:3]
                elif "YEOH" in p:
                    mat["yeoh"] = [flat[0], flat[1], flat[2], flat[3] if len(flat) > 3 else 0.0]
                else:
                    mat["neohooke"] = flat[:2] + [0.0] * (2 - len(flat[:2]))
            elif kw == "DENSITY":
                mat["density"] = _f(data[0][0])
            elif kw == "EXPANSION":
                mat["alpha"] = _f(data[0][0])
                mat["tref"] = _f(p.get("ZERO", 0.0))
            elif kw == "CONDUCTIVITY":
                mat["k"] = _f(data[0][0])
            elif kw == "SPECIFIC HEAT":
                mat["cp"] = _f(data[0][0])
            elif kw == "AMPLITUDE":
                flat = [_f(x) for d in data for x in d if x]
                self.amplitudes[p["NAME"].upper()] = np.array(flat).reshape(-1, 2)
            elif kw == "INITIAL CONDITIONS" and str(p.get("TYPE", "")).upper() == "TEMPERATURE":
                for d in data:
                    self.init_temp.append((d[0].upper(), _f(d[1])))
            elif kw == "TIE":
                for d in data:
                    self.ties.append(dict(name=p.get("NAME", "tie"), slave=d[0].upper(), master=d[1].upper(),
                                          tol=_f(p.get("POSITION TOLERANCE"), None)))
            elif kw == "EQUATION":
                i = 0
                rows = [x for d in data for x in d if x != ""]
                while i < len(rows):
                    nt = int(_f(rows[i]))
                    terms = rows[i + 1:i + 1 + 3 * nt]
                    self.__dict__.setdefault("equations", []).append(
                        [(terms[3 * j].upper(), int(_f(terms[3 * j + 1])) - 1, _f(terms[3 * j + 2])) for j in range(nt)])
                    i += 1 + 3 * nt
            elif kw == "SURFACE INTERACTION":
                self.interactions[p["NAME"].upper()] = {}
                self._last_int = p["NAME"].upper()
            elif kw == "FRICTION":
                self.interactions[self._last_int]["mu"] = _f(data[0][0]) if data else 0.0
            elif kw == "CONTACT PAIR":
                for d in data:
                    self.contact_pairs.append(dict(slave=d[0].upper(), master=d[1].upper(),
                                                   interaction=str(p.get("INTERACTION", "")).upper()))
            elif kw == "STEP":
                step = new_step(p.get("NAME", f"Step-{len(self.steps) + 1}"),
                                nlgeom=str(p.get("NLGEOM", "NO")).upper() in ("YES", "TRUE"),
                                max_inc=int(_f(p.get("INC", 1000))))
                step["_bc"], step["_load"], step["_ds"], step["_dl"] = [], [], [], []
                step["_bc_new"] = step["_load_new"] = False
            elif kw == "STATIC":
                vals = [_f(x) for x in data[0] if x] if data else []
                if "RIKS" in p:
                    step["procedure"] = "riks"
                    d = vals + [1.0, 1.0, 1e-5, 1.0, 1.0][len(vals):]
                    step["riks"] = dict(dlam0=d[0], lpf_max=d[4] if len(vals) > 4 and vals[4] else 1.0)
                    step["time"] = d[1]
                else:
                    d = vals + [1.0, 1.0, 1e-5, 1.0][len(vals):]
                    step["dt_init"], step["time"], step["dt_min"], step["dt_max"] = d[:4]
            elif kw == "DYNAMIC":
                vals = [_f(x) for x in data[0] if x] if data else []
                if "EXPLICIT" in p:
                    step["procedure"] = "explicit"
                    step["time"] = vals[1] if len(vals) > 1 else (vals[0] if vals else 1.0)
                else:
                    step["procedure"] = "dynamic"
                    d = vals + [1.0, 1.0, 1e-5, 1.0][len(vals):]
                    step["dt_init"], step["time"], step["dt_min"], step["dt_max"] = d[:4]
                    step["dt"] = d[0]
                    if "ALPHA" in p:
                        step["alpha"] = _f(p["ALPHA"])
            elif kw == "FREQUENCY":
                step["procedure"] = "frequency"
                step["nmodes"] = int(_f(data[0][0], 10)) if data and data[0][0] else 10
            elif kw == "BUCKLE":
                step["procedure"] = "buckle"
                step["nmodes"] = int(_f(data[0][0], 5)) if data and data[0][0] else 5
            elif kw == "HEAT TRANSFER":
                step["procedure"] = "heat"
                vals = [_f(x) for x in data[0] if x] if data else []
                step["transient"] = "STEADY STATE" not in p
                if vals:
                    step["dt_init"] = vals[0]
                    step["time"] = vals[1] if len(vals) > 1 else 1.0
            elif kw == "BOUNDARY" and step is None:
                self.asm.__dict__.setdefault("bcs", []).append((p, data))
            elif kw == "BOUNDARY":
                step["_bc_new"] |= str(p.get("OP", "")).upper() == "NEW"
                step["_bc"].append((p, data))
            elif kw == "CLOAD":
                step["_load_new"] |= str(p.get("OP", "")).upper() == "NEW"
                step["_load"].append((p, data))
            elif kw == "DSLOAD":
                step["_load_new"] |= str(p.get("OP", "")).upper() == "NEW"
                step["_ds"].append((p, data))
            elif kw == "DLOAD":
                step["_load_new"] |= str(p.get("OP", "")).upper() == "NEW"
                step["_dl"].append((p, data))
            elif kw == "TEMPERATURE" and step is not None:
                step.setdefault("_temp", []).extend((d[0].upper(), _f(d[1])) for d in data)
            elif kw == "DSFLUX":
                step.setdefault("_flux", []).extend((d[0].upper(), _f(d[2] if len(d) > 2 else d[1])) for d in data)
            elif kw == "SFILM":
                step.setdefault("_film", []).extend((d[0].upper(), _f(d[2]), _f(d[3])) for d in data)
            elif kw == "DFLUX":
                step.setdefault("_dflux", []).extend((d[0].upper(), _f(d[2] if len(d) > 2 else d[1])) for d in data)
            elif kw == "END STEP":
                self.steps.append(step)
                step = None

    # ---------------------------------------------------------- assembly
    def _assemble(self):
        sp = self.spec
        nid = {}             # (instance, local id) -> idx
        eidx = {}            # (instance, elem id) -> global idx
        X = []
        groups = {}
        nsets, elsets, surfaces = {}, {}, {}
        sections = []

        def add_part(inst, part, tr=None, rot=None):
            P = np.array(part.coords, float).reshape(-1, 3)
            if rot and len(rot) >= 7:
                a, b, ang = np.array(rot[:3]), np.array(rot[3:6]), np.deg2rad(rot[6])
                k = (b - a) / np.linalg.norm(b - a)
                K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
                P = (P - a) @ R.T + a
            if tr:
                P = P + np.array((tr + [0, 0, 0])[:3])
            for i, n in enumerate(part.node_ids):
                nid[(inst, n)] = len(X) + i
            X.extend(P)
            for et, e, nodes in part.elements:
                groups.setdefault(et, []).append(((inst, e), [nid[(inst, n)] for n in nodes]))
            pre = f"{inst}." if inst else ""
            for k, v in part.nsets.items():
                nsets[pre + k] = (inst, v, part)
            for k, v in part.elsets.items():
                elsets[pre + k] = (inst, v, part)
            for k, v in part.surfaces.items():
                surfaces[pre + k] = (inst, v)
            for sec in part.sections:
                sections.append((pre + sec[0],) + tuple(sec[1:]))

        add_part("", self.root)
        for name, pname, tr, rot in self.instances:
            add_part(name, self.parts[pname], tr, rot)
        add_part("", self.asm)
        sp.nodes = np.array(X, float).reshape(-1, 3)
        sp.node_ids = np.zeros(len(X), np.int64)
        for (inst, n), i in nid.items():
            sp.node_ids[i] = n
        if self.instances:          # make ids unique across instances
            sp.node_ids = np.arange(1, len(X) + 1)
        for et, rows in groups.items():
            start = sp.nelem
            for j, (key, _) in enumerate(rows):
                eidx[key] = start + j
            ids = [k[1] for k, _ in rows] if not self.instances else np.arange(start, start + len(rows)) + 1
            sp.add_elements(et, np.array([r for _, r in rows]), ids)
        single = len(self.instances) == 1
        inst0 = self.instances[0][0] if single else ""

        def resolve(name, kind, depth=0):
            name = name.upper()
            table = nsets if kind == "N" else elsets
            cand = [name, f"{inst0}.{name}"] if single else [name]
            if "." in name and single:
                cand.append(name.split(".", 1)[1])
            for c in cand:
                if c in table:
                    inst, vals, part = table[c]
                    out = []
                    for v in vals:
                        if isinstance(v, tuple) and v[0] == "SET":
                            out.extend(resolve((f"{inst}." if inst else "") + v[1], kind, depth + 1))
                        elif isinstance(v, tuple) and v[0] == "ELSET_NODES":
                            for e in resolve((f"{inst}." if inst else "") + v[1], "E", depth + 1):
                                out.extend(sp_conn[e])
                        elif isinstance(v, tuple):
                            out.append(_idx(kind, v[0], v[1]))
                        else:
                            out.append(_idx(kind, inst, v))
                    return out
            if name.lstrip("-").isdigit():
                return [_idx(kind, inst0, int(name))]
            if "." in name:
                inst, rest = name.split(".", 1)
                if rest.lstrip("-").isdigit():
                    return [_idx(kind, inst, int(rest))]
            raise KeyError(f"unknown {'node' if kind == 'N' else 'element'} set {name}")

        def _idx(kind, inst, i):
            d = nid if kind == "N" else eidx
            if (inst, i) in d:
                return d[(inst, i)]
            if (inst0, i) in d:
                return d[(inst0, i)]
            return d[("", i)]

        types, conns = sp.element_table()
        sp_conn = conns
        self._resolve = resolve
        for k in nsets:
            try:
                sp.nsets[k] = np.unique(resolve(k, "N")).astype(np.int64)
            except KeyError:
                pass
        for k in elsets:
            try:
                sp.elsets[k] = np.unique(resolve(k, "E")).astype(np.int64)
            except KeyError:
                pass
        if single:
            for k in list(sp.nsets):
                if k.startswith(inst0 + "."):
                    sp.nsets.setdefault(k.split(".", 1)[1], sp.nsets[k])
            for k in list(sp.elsets):
                if k.startswith(inst0 + "."):
                    sp.elsets.setdefault(k.split(".", 1)[1], sp.elsets[k])
        # surfaces
        for name, (inst, (typ, rows)) in surfaces.items():
            if typ == "NODE":
                nodes = []
                for ref, _ in rows:
                    nodes.extend(resolve((f"{inst}." if inst else "") + ref if not ref.isdigit() else ref, "N"))
                sp.surfaces[name] = dict(nodes=np.unique(nodes))
            else:
                el, fc = [], []
                for ref, face in rows:
                    es = resolve((f"{inst}." if inst and not ref.isdigit() else "") + ref, "E")
                    k = 1 if face == "SNEG" else 0 if face in ("SPOS", "") else int(face[1:]) - 1
                    el.extend(es)
                    fc.extend([k] * len(es))
                sp.surfaces[name] = dict(elems=np.array(el, np.int64), faces=np.array(fc, np.int64))
            if single and name.startswith(inst0 + "."):
                sp.surfaces.setdefault(name.split(".", 1)[1], sp.surfaces[name])
        for sec in sections:
            es, m = sec[0], sec[1]
            key = es if es in sp.elsets else (f"{inst0}.{es}" if f"{inst0}.{es}" in sp.elsets else es)
            d = dict(elset=key, material=m)
            if len(sec) > 2:
                d.update(sec[2])
            sp.sections.append(d)
        for name, props in self.materials.items():
            sp.materials[name] = props
        sp.amplitudes = dict(self.amplitudes)
        if self.init_temp:
            T = np.zeros(len(sp.nodes))
            for ref, v in self.init_temp:
                T[self._nodes(ref)] = v
            sp.initial_temperature = T
        for t in self.ties:
            sp.constraints.append(dict(type="tie", slave=self._surf(t["slave"]), master=self._surf(t["master"]),
                                       tol=t["tol"], name=t["name"]))
        for eq in getattr(self, "equations", []):
            # first term is the dependent dof: u_s = -sum(c_j / c_s u_j)
            (n0, d0, c0), rest = eq[0], eq[1:]
            s_nodes = self._nodes(n0)
            for k, sn in enumerate(s_nodes):
                md, w = [], []
                for (n, d, cj) in rest:
                    mn = self._nodes(n)
                    md.append(int(mn[k if len(mn) == len(s_nodes) else 0]) * 3 + d)
                    w.append(-cj / c0)
                sp.constraints.append(dict(type="mpc", slave=[int(sn) * 3 + d0], masters=[md], weights=[w]))
        for c in self.contact_pairs:
            mu = self.interactions.get(c["interaction"], {}).get("mu", 0.0)
            if mu:
                sp.warnings.append(f"contact pair {c['slave']}/{c['master']}: friction mu={mu} "
                                   f"ignored (frictionless)")
            sp.contacts.append(dict(type="frictionless", slave=self._surf(c["slave"]),
                                    master=self._surf(c["master"]), auto_slave=False))
        self._steps()

    def _surf(self, name):
        sp = self.spec
        if name in sp.surfaces:
            return name
        for k in sp.surfaces:
            if k.endswith("." + name) or name.endswith("." + k):
                return k
        raise KeyError(f"unknown surface {name}")

    def _nodes(self, ref):
        return np.asarray(self._resolve(ref, "N"), dtype=np.int64)

    def _elems(self, ref):
        return np.asarray(self._resolve(ref, "E"), dtype=np.int64)

    def _bc_items(self, data):
        out = {}
        temps = {}
        for d in data:
            if len(d) >= 2 and d[1].upper() in self.SYM:
                dofs, val = self.SYM[d[1].upper()], 0.0
            else:
                a = int(_f(d[1]))
                b = int(_f(d[2])) if len(d) > 2 and d[2] else a
                val = _f(d[3]) if len(d) > 3 and d[3] else 0.0
                dofs = [k - 1 for k in range(a, b + 1) if k <= 6]
                if a <= 11 <= b:
                    for n in self._nodes(d[0]):
                        temps[n] = val
            for n in self._nodes(d[0]):
                for k in dofs:
                    out[(n, k)] = val
        return out, temps

    def _steps(self):
        sp = self.spec
        bcs0, tb0 = {}, {}
        for p, data in getattr(self.asm, "bcs", []):
            b, t = self._bc_items(data)
            bcs0.update(b)
            tb0.update(t)
        bcs, bamp = dict(bcs0), {}
        loads, lamp = {}, {}
        ds, dl = [], []
        temps = {}
        for s in self.steps:
            if s["_bc_new"]:
                bcs, bamp = dict(bcs0), {}
            tb = dict(tb0)
            for p, data in s["_bc"]:
                b, t = self._bc_items(data)
                bcs.update(b)
                tb.update(t)
                a = p.get("AMPLITUDE")
                for k in b:
                    bamp[k] = a.upper() if a else None
            if s["_load_new"]:
                loads, lamp, ds, dl = {}, {}, [], []
            for p, data in s["_load"]:
                a = p.get("AMPLITUDE")
                for d in data:
                    for n in self._nodes(d[0]):
                        loads[(n, int(_f(d[1])) - 1)] = _f(d[2])
                        lamp[(n, int(_f(d[1])) - 1)] = a.upper() if a else None
            for p, data in s["_ds"]:
                a = p.get("AMPLITUDE")
                fol = str(p.get("FOLLOWER", "YES")).upper() != "NO"
                for d in data:
                    kind = d[1].upper()
                    rec = dict(surface=self._surf(d[0].upper()), kind="P" if kind == "P" else "TRVEC",
                               value=_f(d[2]), amp=a.upper() if a else None, follower=fol)
                    if kind == "TRVEC":
                        rec["vector"] = [_f(x) for x in d[3:6]]
                    ds = [x for x in ds if not (x["surface"] == rec["surface"] and x["kind"] == rec["kind"])]
                    ds.append(rec)
            for p, data in s["_dl"]:
                a = p.get("AMPLITUDE")
                for d in data:
                    kind = d[1].upper()
                    if kind == "GRAV":
                        dl.append(dict(kind="GRAV", elset=d[0].upper() if d[0] else None, value=_f(d[2]),
                                       vector=[_f(x) for x in d[3:6]], amp=a))
                    elif kind == "CENTRIF":
                        dl.append(dict(kind="CENTRIF", elset=d[0].upper() if d[0] else None, value=_f(d[2]),
                                       point=[_f(x) for x in d[3:6]], axis=[_f(x) for x in d[6:9]], amp=a))
                    elif kind == "P" or kind == "PNEG":
                        el = self._elems(d[0])
                        name = f"_DLOAD_{d[0].upper()}_{kind}"
                        sp.surfaces[name] = dict(elems=el, faces=np.full(len(el), 0 if kind == "P" else 1))
                        ds.append(dict(surface=name, kind="P", value=_f(d[2]), amp=a, follower=True))
                    elif kind.startswith("P") and kind[1:].isdigit():
                        # element face pressure -> anonymous surface
                        el = self._elems(d[0])
                        name = f"_DLOAD_{d[0].upper()}_{kind}"
                        sp.surfaces[name] = dict(elems=el, faces=np.full(len(el), int(kind[1:]) - 1))
                        ds.append(dict(surface=name, kind="P", value=_f(d[2]), amp=a, follower=True))
            for ref, v in s.get("_temp", []):
                for n in self._nodes(ref):
                    temps[n] = v
            st = {k: v for k, v in s.items() if not k.startswith("_")}
            st["bcs"] = [dict(nodes=[n], dof=k, value=v, amp=bamp.get((n, k))) for (n, k), v in bcs.items()]
            st["cloads"] = [dict(nodes=[n], dof=k, value=v, amp=lamp.get((n, k))) for (n, k), v in loads.items()]
            st["dsloads"] = list(ds)
            st["dloads"] = list(dl)
            st["temperatures"] = [([n], v) for n, v in temps.items()]
            if st["procedure"] == "heat":
                st["heat"] = dict(fixed=[([n], v) for n, v in tb.items()],
                                  flux=[(self._surf(r), q) for r, q in s.get("_flux", [])],
                                  film=[(self._surf(r), h, Ti) for r, Ti, h in s.get("_film", [])],
                                  gen=[(r, q) for r, q in s.get("_dflux", [])])
            _compress(st)
            sp.steps.append(st)


def _compress(st):
    """Group per-node BC/load records with identical (dof, value, amp) for speed."""
    for key in ("bcs", "cloads"):
        grp = {}
        for r in st[key]:
            grp.setdefault((r["dof"], r["value"], r.get("amp")), []).extend(r["nodes"])
        st[key] = [dict(nodes=np.array(v, np.int64), dof=k[0], value=k[1], amp=k[2]) for k, v in grp.items()]
    grp = {}
    for nodes, v in st["temperatures"]:
        grp.setdefault(v, []).extend(nodes)
    st["temperatures"] = [(np.array(n, np.int64), v) for v, n in grp.items()]
    if "heat" in st:
        grp = {}
        for nodes, v in st["heat"]["fixed"]:
            grp.setdefault(v, []).extend(nodes)
        st["heat"]["fixed"] = [(np.array(n, np.int64), v) for v, n in grp.items()]


def read_inp_spec(path):
    return InpReader(path).spec


def read_inp(path):
    """Backwards compatible: returns (Model, [Step]) for an input deck."""
    from ..analysis import build
    return build(read_inp_spec(path))


# ======================================================================== writer
def _fmt_rows(ids, per=16):
    ids = list(map(str, ids))
    return "\n".join(", ".join(ids[i:i + per]) for i in range(0, len(ids), per))


def write_inp(spec, path):
    """Write a Spec as a flat (part-less) Abaqus input deck."""
    nid = spec.node_ids
    eids = np.concatenate([g.ids for g in spec.groups]) if spec.groups else np.zeros(0, np.int64)
    L = [f"*HEADING\n{spec.title} (written by gfem)"]
    if spec.units:
        L.append(f"** units: {spec.units}")
    L.append("*NODE")
    L += [f"{i}, {x:.10g}, {y:.10g}, {z:.10g}" for i, (x, y, z) in zip(nid, spec.nodes)]
    for g in spec.groups:
        L.append(f"*ELEMENT, TYPE={g.etype}" + (f", ELSET={g.elset}" if g.elset else ""))
        for e, row in zip(g.ids, g.conn):
            vals = [str(e)] + [str(nid[n]) for n in row]
            L.append(",\n".join(", ".join(vals[i:i + 16]) for i in range(0, len(vals), 16)))
    elset_written = {g.elset for g in spec.groups if g.elset}
    for k, v in spec.nsets.items():
        L.append(f"*NSET, NSET={_safe(k)}\n{_fmt_rows(nid[np.asarray(v, np.int64)])}")
    for k, v in spec.elsets.items():
        if k in elset_written:
            continue
        L.append(f"*ELSET, ELSET={_safe(k)}\n{_fmt_rows(eids[np.asarray(v, np.int64)])}")
    for k, s in spec.surfaces.items():
        if "elems" in s:
            L.append(f"*SURFACE, NAME={_safe(k)}, TYPE=ELEMENT")
            el, fc = np.asarray(s["elems"]), np.asarray(s["faces"])
            types_ = spec.element_table()[0]
            shell = np.array([types_[e] in STRUCT_ELEMENTS for e in el], bool)
            for isshell in (False, True):
                for f in np.unique(fc[shell == isshell]):
                    sel = (fc == f) & (shell == isshell)
                    lab = ("SPOS" if f == 0 else "SNEG") if isshell else f"S{f + 1}"
                    name = f"_{_safe(k)}_{lab}"
                    L.insert(len(L) - 1, f"*ELSET, ELSET={name}\n{_fmt_rows(eids[el[sel]])}")
                    L.append(f"{name}, {lab}")
        else:
            L.append(f"*NSET, NSET=_{_safe(k)}_N\n{_fmt_rows(nid[np.asarray(s['nodes'])])}")
            L.append(f"*SURFACE, NAME={_safe(k)}, TYPE=NODE\n_{_safe(k)}_N, 1.0")
    for sec in spec.sections:
        k = sec.get("kind", "solid")
        if k == "shell":
            L.append(f"*SHELL SECTION, ELSET={_safe(sec['elset'])}, MATERIAL={_safe(sec['material'])}\n{sec['thickness']:.10g}, 5")
        elif k == "beam":
            prof = sec.get("profile", "RECT").upper()
            n1 = sec.get("n1") or [0.0, 0.0, -1.0]
            if prof == "GENERAL":
                A, I2, I3, J = sec["dims"][:4]
                m = spec.materials[sec["material"]]
                L.append(f"*BEAM GENERAL SECTION, ELSET={_safe(sec['elset'])}, SECTION=GENERAL\n{A:.10g}, {I2:.10g}, 0., {I3:.10g}, {J:.10g}"
                         f"\n{n1[0]:.6g}, {n1[1]:.6g}, {n1[2]:.6g}\n{m['E']:.10g}, {m['E'] / (2 * (1 + m['nu'])):.10g}")
            else:
                L.append(f"*BEAM SECTION, ELSET={_safe(sec['elset'])}, MATERIAL={_safe(sec['material'])}, SECTION={prof}\n"
                         + ", ".join(f"{x:.10g}" for x in sec["dims"]) + f"\n{n1[0]:.6g}, {n1[1]:.6g}, {n1[2]:.6g}")
        else:
            L.append(f"*SOLID SECTION, ELSET={_safe(sec['elset'])}, MATERIAL={_safe(sec['material'])}\n,")
    for name, p in spec.materials.items():
        L.append(f"*MATERIAL, NAME={_safe(name)}")
        if "density" in p:
            L.append(f"*DENSITY\n{p['density']:.10g},")
        if "neohooke" in p or p.get("model") == "neohooke":
            c = p.get("neohooke") or [p.get("C10"), p.get("D1", 0.0)]
            L.append(f"*HYPERELASTIC, NEO HOOKE\n{c[0]:.10g}, {c[1] or 0:.10g}")
        elif "mooney" in p:
            c = p["mooney"]
            L.append(f"*HYPERELASTIC, MOONEY-RIVLIN\n{c[0]:.10g}, {c[1]:.10g}, {c[2]:.10g}")
        elif "ortho" in p:
            L.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS\n" + ", ".join(f"{x:.10g}" for x in p["ortho"][:8])
                     + f",\n{p['ortho'][8]:.10g}")
        elif "E" in p:
            L.append(f"*ELASTIC\n{p['E']:.10g}, {p['nu']:.10g}")
        if p.get("alpha"):
            L.append(f"*EXPANSION, ZERO={p.get('tref', 0.0):.10g}\n{p['alpha']:.10g},")
        if "k" in p:
            L.append(f"*CONDUCTIVITY\n{p['k']:.10g},")
        if "cp" in p:
            L.append(f"*SPECIFIC HEAT\n{p['cp']:.10g},")
        tab = p.get("plastic")
        if tab is None and p.get("model") == "plastic" and "yield" in p:
            E, Et = p["E"], p.get("tangent", 0.0)
            Hp = E * Et / (E - Et) if Et else 0.0
            tab = [[p["yield"], 0.0], [p["yield"] + Hp, 1.0]]
        if tab is not None:
            L.append("*PLASTIC\n" + "\n".join(f"{a:.10g}, {b:.10g}" for a, b in tab))
    for name, t in spec.amplitudes.items():
        L.append(f"*AMPLITUDE, NAME={_safe(name)}\n" + _fmt_rows([f"{x:.10g}" for x in np.ravel(t)], 8))
    for c in spec.constraints:
        if c["type"] == "tie":
            L.append(f"*TIE, NAME={_safe(c.get('name', 'TIE'))}, ADJUST=NO\n{_safe(c['slave'])}, {_safe(c['master'])}")
    for c in spec.constraints:
        if c["type"] == "normal":
            n = np.asarray(c["normal"], float)
            n = n / np.linalg.norm(n)
            L.append(f"** frictionless support {c.get('name', '')}: n.u = 0\n*EQUATION")
            k = int(np.argmax(np.abs(n)))
            order = [k] + [j for j in range(3) if j != k]
            for node in np.asarray(c["nodes"]):
                terms = [(j, n[j]) for j in order if abs(n[j]) > 1e-12]
                L.append(f"{len(terms)}\n" + ", ".join(f"{nid[node]}, {j + 1}, {v:.10g}" for j, v in terms))
    for c in spec.contacts:
        if c["type"] == "wall":
            L.append(f"** rigid wall {c.get('name', '')} is not exported (model it as an analytical rigid surface)")
    if any(c["type"] != "wall" for c in spec.contacts):
        L.append("*SURFACE INTERACTION, NAME=FRICTIONLESS\n1.0,")
        for c in spec.contacts:
            if c["type"] != "wall":
                L.append(f"*CONTACT PAIR, INTERACTION=FRICTIONLESS, TYPE=SURFACE TO SURFACE\n"
                         f"{_safe(c['slave'])}, {_safe(c['master'])}")
    if spec.initial_temperature is not None:
        L.append("*INITIAL CONDITIONS, TYPE=TEMPERATURE")
        L += [f"{nid[i]}, {T:.10g}" for i, T in enumerate(spec.initial_temperature)]
    for s in spec.steps:
        L.append(f"*STEP, NAME={_safe(s['name'])}, NLGEOM={'YES' if s.get('nlgeom') else 'NO'}, INC={s.get('max_inc', 1000)}")
        pr = s["procedure"]
        T = s.get("time", 1.0)
        if pr == "static":
            L.append(f"*STATIC\n{s.get('dt_init') or T:.6g}, {T:.6g}, {s.get('dt_min') or 1e-5 * T:.6g}, {s.get('dt_max') or T:.6g}")
        elif pr == "riks":
            L.append(f"*STATIC, RIKS\n{s['riks'].get('dlam0', 0.1)}, {T}, , , {s['riks'].get('lpf_max', 1.0)}")
        elif pr == "dynamic":
            L.append(f"*DYNAMIC, ALPHA={s.get('alpha', -0.05)}\n{s.get('dt') or s.get('dt_init') or T / 100:.6g}, {T:.6g}")
        elif pr == "explicit":
            L.append(f"*DYNAMIC, EXPLICIT\n, {T:.6g}")
        elif pr == "frequency":
            L.append(f"*FREQUENCY, EIGENSOLVER=LANCZOS\n{s.get('nmodes', 10)},")
        elif pr == "buckle":
            L.append(f"*BUCKLE\n{s.get('nmodes', 5)},")
        elif pr == "heat":
            if s.get("transient"):
                L.append(f"*HEAT TRANSFER\n{s.get('dt_init') or T / 20:.6g}, {T:.6g}")
            else:
                L.append(f"*HEAT TRANSFER, STEADY STATE\n1., 1.")
        elif pr == "topopt":
            L.append("** gfem topology optimisation step: exported as a linear static step")
            L.append("*STATIC\n1., 1.")
        if s["bcs"]:
            L.append("*BOUNDARY, OP=NEW")
            for b in s["bcs"]:
                for n in np.asarray(b["nodes"]):
                    L.append(f"{nid[n]}, {b['dof'] + 1}, {b['dof'] + 1}, {b['value']:.10g}")
        if pr == "heat":
            h = s["heat"]
            if h["fixed"]:
                L.append("*BOUNDARY")
                for nodes, v in h["fixed"]:
                    L += [f"{nid[n]}, 11, 11, {v:.10g}" for n in np.asarray(nodes)]
            for sname, q in h["flux"]:
                L.append(f"*DSFLUX\n{_safe(sname)}, S, {q:.10g}")
            for sname, hh, Ti in h["film"]:
                L.append(f"*SFILM\n{_safe(sname)}, F, {Ti:.10g}, {hh:.10g}")
            for es, q in h["gen"]:
                L.append(f"*DFLUX\n{_safe(es) if es else ''}, BF, {q:.10g}")
        if s["cloads"]:
            L.append("*CLOAD, OP=NEW")
            for c in s["cloads"]:
                for n in np.asarray(c["nodes"]):
                    L.append(f"{nid[n]}, {c['dof'] + 1}, {c['value']:.10g}")
        for d in s["dsloads"]:
            k = d.get("kind", "P")
            if k == "P":
                L.append(f"*DSLOAD\n{_safe(d['surface'])}, P, {d['value']:.10g}")
            elif k in ("TRVEC", "TRACTION"):
                v = d.get("vector", (0, 0, 1))
                L.append(f"*DSLOAD\n{_safe(d['surface'])}, TRVEC, {d['value']:.10g}, {v[0]:.6g}, {v[1]:.6g}, {v[2]:.6g}")
            elif k in ("FORCE", "MOMENT"):
                pass                 # written below as exact equivalent nodal loads
        eq = spec.equivalent_nodal_loads(s)
        if eq:
            L.append("** surface FORCE / MOMENT loads as equivalent nodal forces\n*CLOAD")
            for n, f in eq.items():
                for k in range(3):
                    if f[k] != 0:
                        L.append(f"{nid[n]}, {k + 1}, {f[k]:.12g}")
        for d in s["dloads"]:
            if d["kind"] == "GRAV":
                v = d["vector"]
                L.append(f"*DLOAD\n{_safe(d['elset']) if d.get('elset') else ''}, GRAV, {d['value']:.10g}, {v[0]:.6g}, {v[1]:.6g}, {v[2]:.6g}")
        if s["temperatures"]:
            L.append("*TEMPERATURE")
            for nodes, v in s["temperatures"]:
                L += [f"{nid[n]}, {v:.10g}" for n in np.asarray(nodes)]
        L.append("*OUTPUT, FIELD, VARIABLE=PRESELECT\n*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def _safe(name):
    return str(name).replace(" ", "_").replace(",", "_")
