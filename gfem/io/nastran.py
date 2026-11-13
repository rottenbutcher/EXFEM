"""Nastran bulk data (.bdf / .nas / .dat) reader and writer.

Reader: free, small (8) and large (16, '*') field formats with continuations.
  GRID, CTETRA (4/10), CHEXA (8/20), CPENTA (6), PSOLID, MAT1, MAT4 (conductivity),
  SPC, SPC1 (THRU), FORCE, PLOAD4 (solid faces), GRAV, LOAD (combinations), TEMP/TEMPD,
  EIGRL, case control SUBCASE / SPC= / LOAD= / METHOD= / TEMP(LOAD)=, SOL 101/103/105/106.
Writer: GRID/CTETRA/CHEXA/CPENTA/PSOLID/MAT1/SPC1/FORCE/PLOAD4/GRAV for SOL 101/103/105.
"""
import numpy as np
from ..spec import Spec, new_step

_H20_N2A = list(range(12)) + list(range(16, 20)) + list(range(12, 16))   # Nastran -> Abaqus CHEXA20
_H20_A2N = list(range(12)) + list(range(16, 20)) + list(range(12, 16))


def _num(s):
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    t = s.upper().replace("D", "E")
    if "E" not in t:      # Nastran exponent shorthand: 1.0-3, 2.5+4
        for i in range(len(t) - 1, 0, -1):
            if t[i] in "+-" and t[i - 1] not in "+-E":
                t = t[:i] + "E" + t[i:]
                break
    try:
        return float(t)
    except ValueError:
        return s.upper()


def _cards(path):
    """Yield (name, fields) with continuations merged; also returns case control lines."""
    case, bulk, sol = [], [], None
    in_bulk = False
    cur = None
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    if not any(l.upper().startswith("BEGIN BULK") for l in lines):
        in_bulk = True
    for raw in lines:
        line = raw.split("$")[0].rstrip()
        if not line.strip():
            continue
        up = line.upper().strip()
        if not in_bulk:
            if up.startswith("SOL "):
                sol = up.split()[1]
            if up.startswith("BEGIN BULK"):
                in_bulk = True
            else:
                case.append(up)
            continue
        if up.startswith("ENDDATA"):
            break
        if "," in line:
            fields = [x.strip() for x in line.split(",")]
            first = fields[0]
            data = fields[1:]
            large = first.endswith("*")
        elif line[:8].rstrip().endswith("*") or (line[:1] == "*"):
            first = line[:8].strip()
            data = [line[8 + 16 * i: 24 + 16 * i] for i in range(4)]
            large = True
        else:
            first = line[:8].strip()
            data = [line[8 + 8 * i: 16 + 8 * i] for i in range(8)]
            large = False
        is_cont = (not first) or first.startswith("+") or (first.startswith("*") and len(first) <= 8
                                                             and not first[1:2].isalpha())
        if is_cont and cur is not None:
            cur[1].extend(data)
        else:
            if cur is not None:
                bulk.append(cur)
            cur = [first.rstrip("*").upper(), list(data)]
    if cur is not None:
        bulk.append(cur)
    return case, bulk, sol


def read_bdf(path):
    case, bulk, sol = _cards(path)
    sp = Spec(path)
    grids, elems, props, mats, mat4 = {}, [], {}, {}, {}
    spcs, forces, ploads, gravs, combos, temps, eigrl = {}, {}, {}, {}, {}, {}, {}
    shellp, beamp, moments = {}, {}, {}
    for name, f in bulk:
        v = [_num(x) if isinstance(x, str) else x for x in f]
        v = [x for x in v]
        g = lambda i, d=None: v[i] if i < len(v) and v[i] is not None else d
        if name == "GRID":
            grids[g(0)] = [g(2, 0.0), g(3, 0.0), g(4, 0.0)]
        elif name in ("CTETRA", "CHEXA", "CPENTA"):
            nodes = [x for x in v[2:] if isinstance(x, int)]
            elems.append((name, g(0), g(1), nodes))
        elif name == "PSOLID":
            props[g(0)] = g(1)
        elif name in ("CQUAD4", "CTRIA3", "CQUADR", "CTRIAR"):
            nn = 4 if name.startswith("CQUAD") else 3
            elems.append((name, g(0), g(1), [int(x) for x in v[2:2 + nn]]))
        elif name in ("CBAR", "CBEAM"):
            x1, x2, x3 = g(4), g(5), g(6)
            orient = ("G0", x1) if isinstance(x1, int) and x2 is None else ("V", (x1 or 0.0, x2 or 0.0, x3 or 0.0))
            elems.append((name, g(0), g(1), [g(2), g(3)], orient))
        elif name == "PSHELL":
            shellp[g(0)] = (g(1), g(2, 1.0))
        elif name in ("PBARL", "PBEAML"):
            typ = str(v[3] if name == "PBARL" else v[3]).upper()
            dims = [float(x) for x in v[4:] if isinstance(x, (int, float))]
            beamp[g(0)] = (g(1), typ, dims)
        elif name in ("PBAR", "PBEAM"):
            beamp[g(0)] = (g(1), "PBAR", [g(2, 1.0), g(3, 1.0), g(4, 1.0), g(5, 1.0)])
        elif name == "MOMENT":
            mv = g(3, 0.0) * np.array([g(4, 0.0), g(5, 0.0), g(6, 0.0)])
            moments.setdefault(g(0), []).append((g(1), mv))
        elif name == "MAT1":
            E, G, nu = g(1), g(2), g(3)
            if nu is None and E and G:
                nu = E / (2 * G) - 1
            if E is None and G and nu is not None:
                E = 2 * G * (1 + nu)
            mats[g(0)] = dict(E=E, nu=nu if nu is not None else 0.3, density=g(4, 1.0),
                              alpha=g(5, 0.0), tref=g(6, 0.0))
        elif name == "MAT4":
            mat4[g(0)] = dict(k=g(1, 1.0), cp=g(2, 1.0))
        elif name == "SPC":
            for j in range(0, len(v) - 1, 3):
                if g(1 + j) is None:
                    continue
                spcs.setdefault(g(0), []).append((g(1 + j), str(g(2 + j)), g(3 + j, 0.0)))
        elif name == "SPC1":
            ids = [x for x in v[2:] if x is not None]
            nodes = []
            k = 0
            while k < len(ids):
                if k + 2 < len(ids) and ids[k + 1] == "THRU":
                    nodes.extend(range(ids[k], ids[k + 2] + 1))
                    k += 3
                else:
                    if isinstance(ids[k], int):
                        nodes.append(ids[k])
                    k += 1
            for n in nodes:
                spcs.setdefault(g(0), []).append((n, str(g(1)), 0.0))
        elif name == "FORCE":
            fv = g(3, 0.0) * np.array([g(4, 0.0), g(5, 0.0), g(6, 0.0)])
            forces.setdefault(g(0), []).append((g(1), fv))
        elif name == "PLOAD4":
            ploads.setdefault(g(0), []).append((g(1), g(2, 0.0), g(6), g(7)))
        elif name == "GRAV":
            gravs.setdefault(g(0), []).append(g(2, 0.0) * np.array([g(3, 0.0), g(4, 0.0), g(5, 0.0)]))
        elif name == "LOAD":
            S = g(1, 1.0)
            combos[g(0)] = [(S * v[i], v[i + 1]) for i in range(2, len(v) - 1, 2) if v[i] is not None]
        elif name == "TEMP":
            for j in range(1, len(v) - 1, 2):
                if g(j) is not None:
                    temps.setdefault(g(0), {})[g(j)] = g(j + 1)
        elif name == "TEMPD":
            temps.setdefault(g(0), {})["*"] = g(1)
        elif name == "EIGRL":
            eigrl[g(0)] = g(3, 10)
    ids = sorted(grids)
    idx = {n: i for i, n in enumerate(ids)}
    sp.add_nodes(np.array([grids[n] for n in ids]), ids)
    by = {}
    beams = {}
    for rec in elems:
        name, eid, pid, nodes = rec[:4]
        n = len(nodes)
        if name in ("CBAR", "CBEAM"):
            o = rec[4]
            if o[0] == "G0":
                vec = np.asarray(grids[o[1]]) - np.asarray(grids[nodes[0]])
            else:
                vec = np.asarray(o[1], float)
            key = (pid, tuple(np.round(vec / max(np.linalg.norm(vec), 1e-300), 6)))
            beams.setdefault(key, []).append((eid, [idx[x] for x in nodes]))
            continue
        if name.startswith("CQUAD") or name.startswith("CTRIA"):
            by.setdefault(("S4" if n == 4 else "S3", pid), []).append((eid, [idx[x] for x in nodes]))
            continue
        if name == "CTETRA":
            et = "C3D10" if n >= 10 else "C3D4"
            nodes = nodes[:10] if n >= 10 else nodes[:4]
        elif name == "CHEXA":
            et = "C3D20" if n >= 20 else "C3D8"
            nodes = [nodes[i] for i in _H20_N2A] if n >= 20 else nodes[:8]
        else:
            et = "C3D6"
            nodes = nodes[:6]
        by.setdefault((et, pid), []).append((eid, [idx[x] for x in nodes]))
    eglob = {}
    for (et, pid), rows in by.items():
        es = f"PID_{pid}"
        gi = sp.add_elements(et, np.array([r for _, r in rows]), [e for e, _ in rows], elset=es)
        for (e, _), i in zip(rows, gi):
            eglob[e] = (i, et, len(sp.groups) - 1)
        mid = props.get(pid, pid)
        if f"MAT_{mid}" not in sp.materials and mid in mats:
            m = dict(mats[mid])
            m.update(mat4.get(mid, {}))
            sp.materials[f"MAT_{mid}"] = m
        if et in ("S4", "S3"):
            mid, t = shellp.get(pid, (pid, 1.0))
            if f"MAT_{mid}" not in sp.materials and mid in mats:
                sp.materials[f"MAT_{mid}"] = dict(mats[mid])
            sp.sections.append(dict(elset=es, material=f"MAT_{mid}", kind="shell", thickness=float(t)))
            continue
        sp.sections.append(dict(elset=es, material=f"MAT_{mid}"))
    for k, ((pid, n1), rows) in enumerate(beams.items()):
        es = f"PID_{pid}_B{k}"
        gi = sp.add_elements("B31", np.array([r for _, r in rows]), [e for e, _ in rows], elset=es)
        for (e, _), i in zip(rows, gi):
            eglob[e] = (i, "B31", len(sp.groups) - 1)
        mid, typ, d = beamp.get(pid, (pid, "BAR", [1.0, 1.0]))
        d = list(d) + [1.0] * (6 - len(d))
        prof = {"BAR": ("RECT", d[:2]), "ROD": ("CIRC", d[:1]), "TUBE": ("PIPE", [d[0], d[0] - d[1]]),
                "BOX": ("BOX", [d[0], d[1], d[2]]), "I": ("I", [d[0], d[1], d[4], d[3]]),
                "PBAR": ("GENERAL", [d[0], d[2], d[1], d[3]])}.get(typ, ("RECT", d[:2]))
        if f"MAT_{mid}" not in sp.materials and mid in mats:
            sp.materials[f"MAT_{mid}"] = dict(mats[mid])
        sp.sections.append(dict(elset=es, material=f"MAT_{mid}", kind="beam", profile=prof[0],
                                dims=[float(x) for x in prof[1]], n1=list(n1)))
    # case control
    sub = {"SPC": None, "LOAD": None, "METHOD": None, "TEMP": None}
    for line in case:
        for k in sub:
            if line.replace(" ", "").startswith(k + "=") or line.replace(" ", "").startswith(k + "(LOAD)="):
                try:
                    sub[k] = int(line.split("=")[1].strip())
                except ValueError:
                    pass
    proc = {"101": "static", "103": "frequency", "105": "buckle", "106": "static",
            "SESTATIC": "static", "SEMODES": "frequency", "SEBUCKL": "buckle", "NLSTATIC": "static"}
    st = new_step("SUBCASE_1", proc.get(str(sol), "static"), nlgeom=str(sol) in ("106", "NLSTATIC"),
                  dt_init=0.1 if str(sol) in ("106", "NLSTATIC") else None)
    spc_id = sub["SPC"] or (next(iter(spcs)) if len(spcs) == 1 else None)
    grp = {}
    for n, comp, val in spcs.get(spc_id, []):
        for c in str(comp):
            if c in "123456":
                grp.setdefault((int(c) - 1, float(val or 0.0)), []).append(idx[n])
    st["bcs"] = [dict(nodes=np.array(v), dof=k[0], value=k[1]) for k, v in grp.items()]
    lid = sub["LOAD"] or (next(iter({**forces, **ploads, **gravs, **combos})) if (forces or ploads or gravs) else None)
    parts = combos.get(lid, [(1.0, lid)])
    fg = {}
    from ..elements import ELEMENTS
    for scale, sid in parts:
        for n, mv in moments.get(sid, []):
            for d in range(3):
                if mv[d]:
                    fg.setdefault(d + 3, {}).setdefault(idx[n], 0.0)
                    fg[d + 3][idx[n]] += scale * mv[d]
        for n, fv in forces.get(sid, []):
            for d in range(3):
                if fv[d]:
                    fg.setdefault(d, {}).setdefault(idx[n], 0.0)
                    fg[d][idx[n]] += scale * fv[d]
        for eid, p, g1, g34 in ploads.get(sid, []):
            gi, et, _ = eglob[eid]
            conn = sp.element_table()[1][gi]
            face = _pload_face(ELEMENTS[et], conn, idx.get(g1), idx.get(g34) if g34 else None)
            name = f"PLOAD4_{sid}_{p}"
            s = sp.surfaces.setdefault(name, dict(elems=[], faces=[]))
            s["elems"].append(gi)
            s["faces"].append(face)
            if not any(d["surface"] == name for d in st["dsloads"]):
                st["dsloads"].append(dict(surface=name, kind="P", value=scale * p))
        for gv in gravs.get(sid, []):
            st["dloads"].append(dict(kind="GRAV", value=float(np.linalg.norm(gv)) * scale,
                                     vector=list(gv / max(np.linalg.norm(gv), 1e-300))))
    for s in sp.surfaces.values():
        s["elems"], s["faces"] = np.array(s["elems"]), np.array(s["faces"])
    st["cloads"] = [dict(nodes=np.array([n]), dof=d, value=v) for d, m in fg.items() for n, v in m.items()]
    if sub["METHOD"] in eigrl:
        st["nmodes"] = eigrl[sub["METHOD"]]
    tid = sub["TEMP"]
    if tid in temps:
        T = np.full(len(ids), temps[tid].get("*", 0.0))
        for n, v in temps[tid].items():
            if n != "*":
                T[idx[n]] = v
        st["temperatures"] = [(np.nonzero(T == v)[0], float(v)) for v in np.unique(T)]
    if st["procedure"] == "buckle" and sub["METHOD"] in eigrl:
        st["nmodes"] = eigrl[sub["METHOD"]]
    sp.steps.append(st)
    return sp


def _pload_face(et, conn, g1, g34):
    conn = list(conn)
    for k, f in enumerate(et.faces):
        cs = [conn[i] for i in f[:len(f) if len(f) <= 4 else (3 if len(f) == 6 else 4)]]
        if et.ncorner == 4 and et.name in ("C3D4", "C3D10"):
            if g1 in cs and (g34 is None or g34 not in cs):
                return k
        else:
            if g1 in cs and (g34 is None or g34 in cs):
                if g34 is None and len(cs) == 3:
                    return k
                if g34 is not None and len(cs) == 4 and abs(cs.index(g1) - cs.index(g34)) == 2:
                    return k
    return 0


def write_bdf(spec, path):
    """Write a Spec as Nastran bulk data (free field)."""
    nid = spec.node_ids
    eids = np.concatenate([g.ids for g in spec.groups])
    L = ["$ written by gfem"]
    st = spec.steps[0] if spec.steps else None
    sol = {"static": 101, "frequency": 103, "buckle": 105}.get(st["procedure"] if st else "static", 101)
    if st and st.get("nlgeom") and sol == 101:
        sol = 106
    L += [f"SOL {sol}", "CEND", "ECHO = NONE", "DISPLACEMENT = ALL", "STRESS = ALL", "SPCFORCE = ALL",
          "SUBCASE 1", "  SPC = 1", "  LOAD = 1"]
    if sol in (103, 105):
        L.append("  METHOD = 1")
    L.append("BEGIN BULK")
    for i, (x, y, z) in zip(nid, spec.nodes):
        L.append(f"GRID,{i},,{x:.9g},{y:.9g},{z:.9g}")
    mid = {m: k + 1 for k, m in enumerate(spec.materials)}
    ematname = {}
    for sec in spec.sections:
        for e in spec.elsets.get(sec["elset"], []):
            ematname[e] = sec["material"]
    off = 0
    pid_of = {}
    for g in spec.groups:
        if g.etype in ("S4", "S4R", "S3", "S3R", "B31", "B33"):
            off += len(g.conn)
            continue
        name = {"C3D4": "CTETRA", "C3D10": "CTETRA", "C3D8": "CHEXA", "C3D8R": "CHEXA", "C3D8H": "CHEXA",
                "C3D8I": "CHEXA", "C3D20": "CHEXA", "C3D20R": "CHEXA", "C3D6": "CPENTA", "C3D10M": "CTETRA"}[g.etype]
        for k, (e, row) in enumerate(zip(g.ids, g.conn)):
            mname = ematname.get(off + k, next(iter(spec.materials)))
            pid = pid_of.setdefault(mname, len(pid_of) + 1)
            if g.etype.startswith("C3D20"):
                row = [row[i] for i in _H20_A2N]
            vals = [str(e), str(pid)] + [str(nid[n]) for n in row]
            L.append(name + "," + ",".join(vals[:8]))
            for j in range(8, len(vals), 8):
                L.append("," + ",".join(vals[j:j + 8]))
        off += len(g.conn)
    for mname, pid in pid_of.items():
        L.append(f"PSOLID,{pid},{mid[mname]}")
    # shells and beams: one property per section
    secs = {sec["elset"]: sec for sec in spec.sections}
    pid_next = len(pid_of) + 100
    estart = np.cumsum([0] + [len(g.conn) for g in spec.groups])
    eset_of = {}
    for es_name, idxs in spec.elsets.items():
        for e in np.asarray(idxs):
            eset_of.setdefault(int(e), es_name)
    done_pid = {}
    for gi_, g in enumerate(spec.groups):
        if g.etype not in ("S4", "S4R", "S3", "S3R", "B31", "B33"):
            continue
        for k, (e, row) in enumerate(zip(g.ids, g.conn)):
            sec = secs.get(eset_of.get(int(estart[gi_] + k)), None) or next(x for x in spec.sections if x.get("kind") in ("shell", "beam"))
            key = sec["elset"]
            if key not in done_pid:
                pid_next += 1
                done_pid[key] = pid_next
                mm = mid[sec["material"]]
                if sec.get("kind") == "shell":
                    L.append(f"PSHELL,{pid_next},{mm},{sec['thickness']:.9g},{mm},,{mm}")
                else:
                    prof, d = sec.get("profile", "RECT").upper(), sec.get("dims", [1, 1])
                    if prof == "GENERAL":
                        L.append(f"PBAR,{pid_next},{mm},{d[0]:.9g},{d[2]:.9g},{d[1]:.9g},{d[3]:.9g}")
                    else:
                        t, dd = {"RECT": lambda: ("BAR", d[:2]), "CIRC": lambda: ("ROD", d[:1]),
                                 "PIPE": lambda: ("TUBE", [d[0], d[0] - d[1]]), "BOX": lambda: ("BOX", [d[0], d[1], d[2], d[2]]),
                                 "I": lambda: ("I", [d[0], d[1], d[1], d[3], d[2], d[2]])}[prof]()
                        L.append(f"PBARL,{pid_next},{mm},,{t}")
                        L.append(",".join([""] + [f"{x:.9g}" for x in dd]))
            p_ = done_pid[key]
            if g.etype.startswith("S"):
                L.append(f"{'CQUAD4' if len(row) == 4 else 'CTRIA3'},{e},{p_}," + ",".join(str(nid[n]) for n in row))
            else:
                n1 = sec.get("n1") or [0.0, 0.0, 1.0]
                L.append(f"CBAR,{e},{p_},{nid[row[0]]},{nid[row[1]]},{n1[0]:.9g},{n1[1]:.9g},{n1[2]:.9g}")
    for mname, k in mid.items():
        p = spec.materials[mname]
        L.append(f"MAT1,{k},{p.get('E', 0):.9g},,{p.get('nu', 0.3):.9g},{p.get('density', 0):.9g},"
                 f"{p.get('alpha', 0):.9g},{p.get('tref', 0):.9g}")
    if st:
        for b in st["bcs"]:
            if b["value"] == 0:
                L.append(f"SPC1,1,{b['dof'] + 1}," + ",".join(str(nid[n]) for n in np.asarray(b["nodes"])))
            else:
                for n in np.asarray(b["nodes"]):
                    L.append(f"SPC,1,{nid[n]},{b['dof'] + 1},{b['value']:.9g}")
        for c in st["cloads"]:
            v = [0.0, 0.0, 0.0]
            v[c["dof"] % 3] = 1.0
            card = "FORCE" if c["dof"] < 3 else "MOMENT"
            for n in np.asarray(c["nodes"]):
                L.append(f"{card},1,{nid[n]},0,{c['value']:.9g},{v[0]},{v[1]},{v[2]}")
        for n, f in spec.equivalent_nodal_loads(st).items():
            mag = float(np.linalg.norm(f))
            L.append(f"FORCE,1,{nid[n]},0,{mag:.12g},{f[0] / mag:.12g},{f[1] / mag:.12g},{f[2] / mag:.12g}")
        etab = spec.element_table()[1]
        for d in st["dsloads"]:
            if d.get("kind", "P") != "P":
                continue
            s = spec.surfaces[d["surface"]]
            from ..elements import ELEMENTS
            types = spec.element_table()[0]
            for e, f in zip(s["elems"], s["faces"]):
                et = ELEMENTS[types[e]]
                fn = [etab[e][i] for i in et.faces[f]]
                if et.name in ("C3D4", "C3D10"):
                    opp = [n for n in etab[e][:4] if n not in fn[:3]][0]
                    L.append(f"PLOAD4,1,{eids[e]},{d['value']:.9g},,,,{nid[fn[0]]},{nid[opp]}")
                elif len(et.faces[f]) in (4, 8):
                    L.append(f"PLOAD4,1,{eids[e]},{d['value']:.9g},,,,{nid[fn[0]]},{nid[fn[2]]}")
                else:
                    L.append(f"PLOAD4,1,{eids[e]},{d['value']:.9g},,,,{nid[fn[0]]},")
        for d in st["dloads"]:
            if d["kind"] == "GRAV":
                v = d["vector"]
                L.append(f"GRAV,1,0,{d['value']:.9g},{v[0]},{v[1]},{v[2]}")
        if sol in (103, 105):
            L.append(f"EIGRL,1,,,{st.get('nmodes', 10)}")
    L.append("ENDDATA")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
