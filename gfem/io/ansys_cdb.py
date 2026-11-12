"""ANSYS Mechanical APDL .cdb (CDWRITE) reader: NBLOCK, EBLOCK (SOLID185/186/187/285 and
degenerate forms), ET, MPDATA/MP, CMBLOCK (named selections), D, F, SFE/SF pressure."""
import re
import numpy as np
from ..spec import Spec, new_step


def _fortran_widths(fmt):
    """'(3i9,6e21.13e3)' -> [9,9,9,21,...]."""
    out = []
    for rep, kind, w in re.findall(r"(\d*)([iIeEfFgGaA])(\d+)", fmt):
        out += [int(w)] * (int(rep) if rep else 1)
    return out


def _split(line, widths):
    vals, p = [], 0
    for w in widths:
        s = line[p:p + w]
        if not s.strip():
            break
        vals.append(s)
        p += w
    return vals


DOF = {"UX": 0, "UY": 1, "UZ": 2}


def read_cdb(path):
    sp = Spec(path)
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    nodes, elems, etypes, mats, comps = {}, [], {}, {}, {}
    ds, fs, pres = [], [], []
    i = 0
    while i < len(lines):
        L = lines[i].strip()
        U = L.upper()
        if U.startswith("NBLOCK"):
            w = _fortran_widths(lines[i + 1])
            i += 2
            while i < len(lines) and not lines[i].strip().upper().startswith("N,R5") \
                    and lines[i].strip() not in ("-1", ""):
                v = _split(lines[i], w)
                nid = int(v[0])
                xyz = [float(x) for x in v[3:6]] + [0.0] * (3 - len(v[3:6]))
                nodes[nid] = xyz[:3]
                i += 1
        elif U.startswith("EBLOCK"):
            w = _fortran_widths(lines[i + 1])
            solid = "SOLID" in U.split(",")[3].upper() if len(U.split(",")) > 3 else True
            i += 2
            while i < len(lines) and lines[i].strip() != "-1":
                v = [int(x) for x in _split(lines[i], w)]
                if len(v) < 11:
                    i += 1
                    continue
                mat, typ, nn, eid = v[0], v[1], v[8], v[10]
                conn = v[11:]
                while len(conn) < nn:
                    i += 1
                    conn += [int(x) for x in _split(lines[i], w)]
                elems.append((eid, mat, typ, conn[:nn]))
                i += 1
        elif U.startswith("ET,"):
            p = [x.strip() for x in L.split(",")]
            etypes[int(p[1])] = int(re.sub(r"\D", "", p[2]) or 0)
        elif U.startswith("MPDATA"):
            p = [x.strip() for x in L.split(",")]
            # MPDATA,R5.0, 1,EX  ,  1, 1,  value
            prop, m = p[3].upper(), int(p[4])
            mats.setdefault(m, {})[prop] = float(p[6])
        elif U.startswith("MP,"):
            p = [x.strip() for x in L.split(",")]
            mats.setdefault(int(p[2]), {})[p[1].upper()] = float(p[3])
        elif U.startswith("CMBLOCK"):
            p = [x.strip() for x in L.split(",")]
            name, kind, cnt = p[1].upper(), p[2].upper(), int(p[3].split("!")[0])
            w = _fortran_widths(lines[i + 1])
            ids = []
            i += 2
            while len(ids) < cnt and i < len(lines):
                ids += [int(x) for x in _split(lines[i], w)]
                i += 1
            out = []
            for k, x in enumerate(ids):
                if x < 0:
                    out.extend(range(abs(ids[k - 1]) + 1, abs(x) + 1))
                else:
                    out.append(x)
            comps[name] = (kind, out)
            continue
        elif U.startswith("D,"):
            p = [x.strip() for x in L.split(",")]
            lab = p[2].upper()
            val = float(p[3]) if len(p) > 3 and p[3] else 0.0
            dofs = [0, 1, 2] if lab == "ALL" else ([DOF[lab]] if lab in DOF else [])
            for d in dofs:
                ds.append((int(p[1]), d, val))
        elif U.startswith("F,"):
            p = [x.strip() for x in L.split(",")]
            lab = p[2].upper()
            if lab in ("FX", "FY", "FZ"):
                fs.append((int(p[1]), "XYZ".index(lab[1]), float(p[3])))
        elif U.startswith("SFE,"):
            p = [x.strip() for x in L.split(",")]
            if p[3].upper().startswith("PRES"):
                vals = [float(x) for x in lines[i + 1].split()] if i + 1 < len(lines) else []
                pres.append((int(p[1]), int(p[2]), vals[0] if vals else 0.0))
                i += 1
        i += 1
    ids = sorted(nodes)
    idx = {n: k for k, n in enumerate(ids)}
    sp.add_nodes(np.array([nodes[n] for n in ids]), ids)
    groups = {}
    for eid, mat, typ, conn in elems:
        et_num = etypes.get(typ, 185)
        c = conn
        if et_num in (187,) or (et_num in (186,) and len(c) == 10):
            et, row = "C3D10", c[:10]
        elif et_num in (285,) or len(c) == 4:
            et, row = "C3D4", c[:4]
        elif et_num == 186 or len(c) == 20:
            if c[2] == c[3] and c[4] == c[5] == c[6] == c[7]:            # degenerate tetra (186)
                et, row = "C3D10", [c[0], c[1], c[2], c[4], c[8], c[9], c[11], c[16], c[17], c[18]]
            elif c[2] == c[3] and c[6] == c[7]:                           # degenerate wedge -> skip mids
                et, row = "C3D6", [c[0], c[1], c[2], c[4], c[5], c[6]]
                sp.warnings.append("SOLID186 wedge converted to linear C3D6")
            else:
                et, row = "C3D20", c[:20]
        else:  # SOLID185 and degenerate
            if c[2] == c[3] and c[4] == c[5] == c[6] == c[7]:
                et, row = "C3D4", [c[0], c[1], c[2], c[4]]
            elif c[2] == c[3] and c[6] == c[7]:
                et, row = "C3D6", [c[0], c[1], c[2], c[4], c[5], c[6]]
            elif c[4] == c[5] == c[6] == c[7]:                            # pyramid -> 2 tets
                groups.setdefault(("C3D4", mat), []).append((eid, [idx[n] for n in (c[0], c[1], c[2], c[4])]))
                groups[("C3D4", mat)].append((-eid, [idx[n] for n in (c[0], c[2], c[3], c[4])]))
                continue
            else:
                et, row = "C3D8", c[:8]
        groups.setdefault((et, mat), []).append((eid, [idx[n] for n in row]))
    eglob = {}
    for (et, mat), rows in groups.items():
        es = f"MAT_{mat}_{et}"
        gi = sp.add_elements(et, np.array([r for _, r in rows]), [abs(e) for e, _ in rows], elset=es)
        for (e, _), g in zip(rows, gi):
            if e > 0:
                eglob[e] = g
        m = mats.get(mat, {})
        name = f"MAT_{mat}"
        if name not in sp.materials:
            E = m.get("EX", 200e3)
            nu = m.get("PRXY", m.get("NUXY", 0.3))
            sp.materials[name] = dict(E=E, nu=nu, density=m.get("DENS", 1.0), alpha=m.get("ALPX", 0.0),
                                      k=m.get("KXX", 1.0), cp=m.get("C", 1.0))
        sp.sections.append(dict(elset=es, material=name))
    for name, (kind, vals) in comps.items():
        if kind.startswith("NODE"):
            sp.nsets[name] = np.array([idx[n] for n in vals if n in idx], np.int64)
        else:
            sp.elsets[name] = np.array([eglob[e] for e in vals if e in eglob], np.int64)
    st = new_step("LS_1", "static")
    grp = {}
    for n, d, v in ds:
        if n in idx:
            grp.setdefault((d, v), []).append(idx[n])
    st["bcs"] = [dict(nodes=np.array(v), dof=k[0], value=k[1]) for k, v in grp.items()]
    st["cloads"] = [dict(nodes=np.array([idx[n]]), dof=d, value=v) for n, d, v in fs if n in idx]
    if pres:
        # ANSYS face numbers -> Abaqus faces for hexes (SOLID185/186) and tets (187)
        from ..elements import ELEMENTS
        types, conns = sp.element_table()
        surf = dict(elems=[], faces=[])
        val = pres[0][2]
        for eid, face, v in pres:
            g = eglob.get(eid)
            if g is None:
                continue
            et = ELEMENTS[types[g]]
            amap = ({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5} if et.ncorner == 8 else {1: 0, 2: 1, 3: 2, 4: 3})
            surf["elems"].append(g)
            surf["faces"].append(amap.get(face, 0))
        sp.surfaces["SFE_PRES"] = dict(elems=np.array(surf["elems"]), faces=np.array(surf["faces"]))
        st["dsloads"].append(dict(surface="SFE_PRES", kind="P", value=val))
    sp.steps.append(st)
    return sp
