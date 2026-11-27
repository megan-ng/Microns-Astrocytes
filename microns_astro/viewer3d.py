"""Write a self-contained interactive 3D screening page for one astrocyte.

The page shows the astrocyte mesh, every candidate tile as a box, and (for the
tiles you switch on) the part of the mesh inside them plus the other
astrocytes they contain. Switching tiles on/off updates a live bounding box
(min/max in voxels) that can be exported as a selection file for
examples/download_selection.py.

Depends only on numpy + json. plotly.js is embedded from the `plotly` package
(plotly="inline") or loaded from a CDN (plotly="cdn"); nothing else is needed.
"""

import json
import os

import numpy as np

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------------------------------------------------------------------------
# data preparation (pure numpy)
# ---------------------------------------------------------------------------

def assign_faces_to_tiles(verts_nm, faces, res_nm, origin, tile_size, tile_starts):
    """Tile id for every mesh face (by face centroid), -1 if in no tile.

    verts_nm: (N, 3) nm; faces: (M, 3); res_nm: nm per voxel;
    origin/tile_size: the tile grid; tile_starts: (T, 3) voxel starts.
    """
    origin = np.asarray(origin, dtype=np.int64)
    tile_size = np.asarray(tile_size, dtype=np.int64)
    grid_idx = (np.asarray(tile_starts, dtype=np.int64) - origin) // tile_size
    shape = grid_idx.max(axis=0) + 1
    lookup = np.full(tuple(shape), -1, dtype=np.int32)
    lookup[tuple(grid_idx.T)] = np.arange(len(grid_idx), dtype=np.int32)

    centroid_vox = verts_nm[faces].mean(axis=1) / np.asarray(res_nm, dtype=float)
    idx = np.floor((centroid_vox - origin) / tile_size).astype(np.int64)
    valid = (idx >= 0).all(axis=1) & (idx < shape).all(axis=1)
    face_tile = np.full(len(faces), -1, dtype=np.int32)
    face_tile[valid] = lookup[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
    return face_tile


def _flat_um(verts_nm):
    return np.round(np.asarray(verts_nm, dtype=float) / 1000.0, 2).ravel().tolist()


def build_viewer_data(data):
    """Convert screen.build_screening_data output into the JSON the page reads.

    Root IDs are strings: they exceed JavaScript's exact-integer range.
    """
    res = np.asarray(data["resolution_nm"], dtype=float)
    mins = np.asarray(data["mins"], dtype=np.int64)
    tile = np.asarray(data["tile_size"], dtype=np.int64)
    tiles = data["tiles"]
    starts = np.array([t["start"] for t in tiles], dtype=np.int64)

    tv, tf = data["target_mesh"]["verts_nm"], data["target_mesh"]["faces"]
    face_tile = assign_faces_to_tiles(tv, tf, res, mins, tile, starts)

    grid = (starts - mins) // tile
    out_tiles = [dict(
        id=int(t["id"]), start=t["start"], end=t["end"], grid=grid[i].tolist(),
        status=t["status"], target_frac=round(float(t["target_frac"]), 4),
        other_proofread=[str(k) for k in t["other_proofread"]],
        unproofread=[str(k) for k in t["unproofread"]])
        for i, t in enumerate(tiles)]

    tiles_of = {}
    for t in tiles:
        for k in list(t["other_proofread"]) + list(t["unproofread"]):
            tiles_of.setdefault(int(k), []).append(int(t["id"]))
    others = [dict(
        id=str(k), kind=m["kind"], verts=_flat_um(m["verts_nm"]),
        faces=np.asarray(m["faces"]).astype(int).ravel().tolist(),
        tiles=tiles_of.get(int(k), []))
        for k, m in data["other_meshes"].items()]

    return dict(
        astro_id=str(data["astro_id"]), datastack_version=data["datastack_version"],
        mip=data["mip"], scan_mip=data["scan_mip"],
        resolution_nm=data["resolution_nm"], tile_size=data["tile_size"],
        tiles=out_tiles,
        target=dict(verts=_flat_um(tv),
                    faces=np.asarray(tf).astype(int).ravel().tolist(),
                    face_tile=face_tile.tolist()),
        others=others)


def write_html(data, path, plotly="inline"):
    """Write the viewer page. plotly: "inline" | "cdn" | path to a plotly.min.js."""
    payload = build_viewer_data(data)
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    if plotly == "cdn":
        tag = f'<script src="{PLOTLY_CDN}" charset="utf-8"></script>'
    elif plotly == "inline":
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as e:
            raise ImportError(
                'plotly="inline" needs `pip install plotly` (or use --plotlyjs cdn).'
            ) from e
        tag = "<script>" + get_plotlyjs() + "</script>"
    else:
        with open(plotly, encoding="utf-8") as f:
            tag = "<script>" + f.read() + "</script>"

    html = _TEMPLATE.replace("__DATA__", blob).replace("__PLOTLY__", tag)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Astrocyte screening</title>
<style>
:root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--panel:#f6f8fa;--accent:#0969da;--warn:#cf222e}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--line:#30363d;--panel:#161b22;--accent:#58a6ff;--warn:#ff7b72}}
html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
#app{display:flex;height:100%}
#plot{flex:1;min-width:0}
#panel{width:370px;flex:none;border-left:1px solid var(--line);background:var(--panel);overflow-y:auto;padding:14px;box-sizing:border-box}
h1{font-size:15px;margin:0 0 2px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:16px 0 6px}
.sub{color:var(--muted);font-size:12px;word-break:break-all}
.btns{display:flex;flex-wrap:wrap;gap:6px}
button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
label.opt{display:flex;gap:6px;align-items:center;margin:3px 0}
.legend span{display:inline-flex;align-items:center;gap:5px;margin-right:12px;font-size:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;flex:none}
#summary{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px 10px;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
.warn{color:var(--warn)}
.range{display:grid;grid-template-columns:auto 1fr 1fr;gap:4px 6px;align-items:center;margin-bottom:6px}
.range input{width:100%;box-sizing:border-box;font:inherit;padding:2px 5px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
#list{max-height:300px;overflow-y:auto;border:1px solid var(--line);border-radius:6px;background:var(--bg)}
.row{display:flex;gap:6px;align-items:center;padding:3px 8px;border-bottom:1px solid var(--line);cursor:pointer}
.row:last-child{border-bottom:0}.row:hover{background:var(--panel)}
.row .meta{color:var(--muted);font-size:11px;margin-left:auto;white-space:nowrap}
textarea{width:100%;height:140px;box-sizing:border-box;font:11px ui-monospace,Menlo,Consolas,monospace;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px}
details summary{cursor:pointer;color:var(--muted);margin-top:8px}
@media (max-width:800px){#app{flex-direction:column}#plot{height:55vh;flex:none}#panel{width:auto;border-left:0;border-top:1px solid var(--line)}}
</style>
__PLOTLY__
</head>
<body>
<div id="app">
  <div id="plot"></div>
  <aside id="panel">
    <h1>Astrocyte screening</h1>
    <div class="sub" id="meta"></div>

    <h2>Select tiles</h2>
    <div class="sub" style="margin-bottom:6px">Click a tile's dot in the 3D view (diamond = selected), tick a row below, or use a quick select. Only selected tiles show their box and contents.</div>
    <div class="btns">
      <button id="bClean">Clean tiles</button>
      <button id="bNonEmpty">Non-empty tiles</button>
      <button id="bAll">All</button>
      <button id="bNone">None</button>
      <button id="bInvert">Invert</button>
    </div>

    <h2>Select by grid range</h2>
    <div class="range" id="range"></div>
    <div class="btns"><button id="bAdd">Add range</button><button id="bRem">Remove range</button></div>

    <h2>Display</h2>
    <label class="opt"><input type="checkbox" id="oGrid" checked> all tile boxes (faint)</label>
    <label class="opt"><input type="checkbox" id="oContext" checked> full astrocyte mesh (context)</label>
    <label class="opt"><input type="checkbox" id="oNbr" checked> other astrocytes inside selected tiles</label>
    <div class="legend" style="margin-top:6px" id="legend"></div>

    <h2>Selection</h2>
    <div id="summary"></div>
    <div class="btns" style="margin-top:8px">
      <button class="primary" id="bDownload">Download selection.json</button>
      <button id="bCopy">Copy JSON</button>
    </div>
    <details><summary>Show JSON</summary><textarea id="json" readonly></textarea></details>

    <h2>Tiles</h2>
    <div id="list"></div>
  </aside>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
"use strict";
const D = JSON.parse(document.getElementById("data").textContent);
const T = D.tiles, res = D.resolution_nm;
const $ = id => document.getElementById(id);
const dark = !!(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
const fg = dark ? "#e6edf3" : "#1f2328", gridc = dark ? "#30363d" : "#d0d7de";
const STATUS = {
  clean:       {color: "#1a9850", label: "clean"},
  unproofread: {color: "#d73027", label: "unproofread astrocyte present"},
  empty:       {color: "#8c959f", label: "no proofread astrocyte"}
};
const um = (v, a) => v * res[a] / 1000;                 // voxels -> micrometers
const sel = new Set();

// ---- geometry helpers ---------------------------------------------------
function addBox(start, end, out) {
  const lo = start.map((v, a) => um(v, a)), hi = end.map((v, a) => um(v, a));
  const c = [];
  for (const z of [lo[2], hi[2]])
    for (const [x, y] of [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]) c.push([x, y, z]);
  const E = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  for (const [a, b] of E) {
    out.x.push(c[a][0], c[b][0], null);
    out.y.push(c[a][1], c[b][1], null);
    out.z.push(c[a][2], c[b][2], null);
  }
}
function meshArrays(verts, faces) {
  const n = verts.length / 3, m = faces.length / 3;
  const x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
  for (let i = 0; i < n; i++) { x[i] = verts[3*i]; y[i] = verts[3*i+1]; z[i] = verts[3*i+2]; }
  const I = new Int32Array(m), J = new Int32Array(m), K = new Int32Array(m);
  for (let f = 0; f < m; f++) { I[f] = faces[3*f]; J[f] = faces[3*f+1]; K[f] = faces[3*f+2]; }
  return {x, y, z, i: I, j: J, k: K};
}

// ---- traces -------------------------------------------------------------
const traces = [];
const add = tr => (traces.push(tr), traces.length - 1);
const light = {ambient: 0.75, diffuse: 0.5, specular: 0.1, roughness: 0.8};

const tgt = meshArrays(D.target.verts, D.target.faces);
const iContext = add({type: "mesh3d", ...tgt, color: "#8b949e", opacity: 0.13, flatshading: true,
                      lighting: light, hoverinfo: "skip", name: "target (context)"});
// Selected fragments reuse the same vertices; only the face list changes.
const iFrag = add({type: "mesh3d", x: tgt.x, y: tgt.y, z: tgt.z, i: [0], j: [0], k: [0],
                   color: "#0aa6a6", opacity: 1, flatshading: true, lighting: light,
                   hoverinfo: "skip", visible: false, name: "target in selected tiles"});

const tileFaces = T.map(() => []);
D.target.face_tile.forEach((t, f) => { if (t >= 0) tileFaces[t].push(f); });

const allBoxes = {x: [], y: [], z: []};
T.forEach(t => addBox(t.start, t.end, allBoxes));
const iGrid = add({type: "scatter3d", mode: "lines", ...allBoxes, hoverinfo: "skip",
                   line: {color: dark ? "#6e7681" : "#8c959f", width: 2}, opacity: 0.7});

const iSel = {};
for (const s of Object.keys(STATUS))
  iSel[s] = add({type: "scatter3d", mode: "lines", x: [null], y: [null], z: [null], hoverinfo: "skip",
                 line: {color: STATUS[s].color, width: 6}, visible: false});

const cx = T.map(t => (um(t.start[0], 0) + um(t.end[0], 0)) / 2);
const cy = T.map(t => (um(t.start[1], 1) + um(t.end[1], 1)) / 2);
const cz = T.map(t => (um(t.start[2], 2) + um(t.end[2], 2)) / 2);
const hover = T.map(t => {
  let h = `<b>tile #${t.id}</b> grid (${t.grid.join(", ")})<br>voxels ${t.start.join(", ")} &rarr; ${t.end.join(", ")}` +
          `<br>${STATUS[t.status].label}<br>target covers ${(t.target_frac * 100).toFixed(1)}% of tile`;
  if (t.other_proofread.length) h += `<br>+${t.other_proofread.length} other proofread`;
  if (t.unproofread.length) h += `<br>${t.unproofread.length} unproofread`;
  return h + "<br><i>click to toggle</i>";
});
const iCent = add({type: "scatter3d", mode: "markers", x: cx, y: cy, z: cz, text: hover, hoverinfo: "text",
                   marker: {size: T.map(() => 7), symbol: T.map(() => "circle"), opacity: 0.9,
                            color: T.map(t => STATUS[t.status].color), line: {width: 0}}});

const iUnion = add({type: "scatter3d", mode: "lines", x: [null], y: [null], z: [null], hoverinfo: "skip",
                    line: {color: "#bf8700", width: 4, dash: "dot"}, visible: false});

const iHi = add({type: "scatter3d", mode: "lines", x: [null], y: [null], z: [null], hoverinfo: "skip",
                 line: {color: "#8250df", width: 9}, visible: false});   // row-hover highlight

const nbrs = D.others.map(o => {
  const m = meshArrays(o.verts, o.faces);
  const idx = add({type: "mesh3d", ...m, color: o.kind === "unproofread" ? "#e8590c" : "#5c7cfa",
                   opacity: 0.55, flatshading: true, lighting: light, visible: false,
                   hoverinfo: "name", name: `${o.kind} astrocyte ${o.id}`});
  return {idx, tiles: o.tiles, id: o.id, kind: o.kind};
});

const layout = {
  margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
  paper_bgcolor: "rgba(0,0,0,0)", font: {color: fg},
  scene: {
    aspectmode: "data",
    xaxis: {title: "x (µm)", gridcolor: gridc, showbackground: false},
    yaxis: {title: "y (µm)", gridcolor: gridc, showbackground: false, autorange: "reversed"},
    zaxis: {title: "z (µm)", gridcolor: gridc, showbackground: false}
  }
};

// ---- panel --------------------------------------------------------------
$("meta").textContent = `astrocyte ${D.astro_id} · v${D.datastack_version} · mip ${D.mip} (${res.join("×")} nm/voxel) · tile ${D.tile_size.join("×")} voxels · contents scanned at mip ${D.scan_mip}`;
const counts = {clean: 0, unproofread: 0, empty: 0};
T.forEach(t => counts[t.status]++);
$("legend").innerHTML = Object.entries(STATUS).map(([k, s]) =>
  `<span><i class="dot" style="background:${s.color}"></i>${s.label} (${counts[k]})</span>`).join("");

$("range").innerHTML = ["x", "y", "z"].map((a, i) => {
  const vals = T.map(t => t.grid[i]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return `<span>${a}</span><input id="r${a}0" type="number" placeholder="min (${lo})"><input id="r${a}1" type="number" placeholder="max (${hi})">`;
}).join("");

$("list").innerHTML = T.map(t => {
  const bits = [`target ${(t.target_frac * 100).toFixed(0)}%`];
  if (t.other_proofread.length) bits.push(`+${t.other_proofread.length}`);
  if (t.unproofread.length) bits.push(`!${t.unproofread.length}`);
  return `<label class="row"><input type="checkbox" data-i="${t.id}">` +
         `<i class="dot" style="background:${STATUS[t.status].color}"></i>` +
         `<span>#${t.id} (${t.grid.join(",")})</span><span class="meta">${bits.join(" · ")}</span></label>`;
}).join("");

// ---- selection state ----------------------------------------------------
const LS_KEY = `astro_screen:${D.astro_id}:${D.mip}:${D.tile_size.join("x")}:${T.length}`;
const save = () => { try { localStorage.setItem(LS_KEY, JSON.stringify([...sel])); } catch (e) {} };
const restore = () => { try {
  JSON.parse(localStorage.getItem(LS_KEY) || "[]").forEach(i => { if (Number.isInteger(i) && i >= 0 && i < T.length) sel.add(i); });
} catch (e) {} };

function unionBox() {
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (const i of sel) for (let a = 0; a < 3; a++) {
    mn[a] = Math.min(mn[a], T[i].start[a]); mx[a] = Math.max(mx[a], T[i].end[a]);
  }
  return sel.size ? {min: mn, max: mx} : null;
}
function selectionJSON() {
  const ids = [...sel].sort((a, b) => a - b), b = unionBox();
  return {
    format: "microns_astro.selection", version: 1,
    astro_id: D.astro_id, datastack_version: D.datastack_version, mip: D.mip,
    resolution_nm: res, tile_size: D.tile_size,
    bbox: b ? {min: b.min, max: b.max} : null,
    tiles: ids.map(i => ({start: T[i].start, end: T[i].end, status: T[i].status}))
  };
}

let gd = null;
function refresh() {
  const ids = [...sel].sort((a, b) => a - b), b = unionBox();
  const per = {clean: {x: [], y: [], z: []}, unproofread: {x: [], y: [], z: []}, empty: {x: [], y: [], z: []}};
  const I = [], J = [], K = [];
  for (const i of ids) {
    addBox(T[i].start, T[i].end, per[T[i].status]);
    for (const f of tileFaces[i]) { I.push(D.target.faces[3*f]); J.push(D.target.faces[3*f+1]); K.push(D.target.faces[3*f+2]); }
  }
  if (gd) {
    const keys = Object.keys(STATUS);
    Plotly.restyle(gd, {
      x: keys.map(k => per[k].x.length ? per[k].x : [null]),
      y: keys.map(k => per[k].y.length ? per[k].y : [null]),
      z: keys.map(k => per[k].z.length ? per[k].z : [null]),
      visible: keys.map(k => per[k].x.length > 0)
    }, keys.map(k => iSel[k]));
    Plotly.restyle(gd, {i: [I.length ? Int32Array.from(I) : [0]], j: [J.length ? Int32Array.from(J) : [0]],
                        k: [K.length ? Int32Array.from(K) : [0]], visible: [I.length > 0]}, [iFrag]);
    const u = {x: [], y: [], z: []};
    if (b) addBox(b.min, b.max, u);
    Plotly.restyle(gd, {x: [u.x.length ? u.x : [null]], y: [u.y.length ? u.y : [null]],
                        z: [u.z.length ? u.z : [null]], visible: [!!b]}, [iUnion]);
    Plotly.restyle(gd, {"marker.size": [T.map(t => sel.has(t.id) ? 11 : 7)],
                        "marker.symbol": [T.map(t => sel.has(t.id) ? "diamond" : "circle")]}, [iCent]);
    if (nbrs.length) {
      const on = $("oNbr").checked;
      Plotly.restyle(gd, {visible: nbrs.map(n => on && n.tiles.some(t => sel.has(t)))}, nbrs.map(n => n.idx));
    }
  }
  document.querySelectorAll("#list input").forEach(el => { el.checked = sel.has(+el.dataset.i); });

  // summary text
  let s;
  if (!b) {
    s = "Nothing selected.";
  } else {
    const size = b.max.map((v, a) => v - b.min[a]);
    const umSize = size.map((v, a) => (v * res[a] / 1000).toFixed(1));
    const gb = size.reduce((p, v) => p * v, 1) * 2 / 1e9;
    const unp = new Set(), oth = new Set();
    ids.forEach(i => { T[i].unproofread.forEach(k => unp.add(k)); T[i].other_proofread.forEach(k => oth.add(k)); });
    const nUnpTiles = ids.filter(i => T[i].status === "unproofread").length;
    s = `${ids.length} tile(s) selected\n` +
        `bbox min  (voxels): ${b.min.join(", ")}\n` +
        `bbox max  (voxels): ${b.max.join(", ")}  (exclusive)\n` +
        `size      (voxels): ${size.join(" × ")}\n` +
        `size      (µm)    : ${umSize.join(" × ")}\n` +
        `~${gb < 0.1 ? gb.toFixed(3) : gb.toFixed(1)} GB uncompressed (image + mask) at mip ${D.mip}\n` +
        `other proofread astrocytes: ${oth.size}\n`;
    if (unp.size) s += `<span class="warn">${nUnpTiles} tile(s) contain unproofread astrocytes (${unp.size} distinct)</span>\n`;
    s += `\nNote: the bbox is the union of the selected tiles; it can include unselected tiles in between.`;
  }
  $("summary").innerHTML = s;
  $("json").value = JSON.stringify(selectionJSON(), null, 2);
  save();
}

function toggle(i) { sel.has(i) ? sel.delete(i) : sel.add(i); refresh(); }
function setWhere(pred, on) { T.forEach(t => { if (pred(t)) on ? sel.add(t.id) : sel.delete(t.id); }); refresh(); }
function rangePred() {
  const lim = ["x", "y", "z"].map(a => {
    const lo = $(`r${a}0`).value, hi = $(`r${a}1`).value;
    return [lo === "" ? -Infinity : +lo, hi === "" ? Infinity : +hi];
  });
  return t => t.grid.every((g, a) => g >= lim[a][0] && g <= lim[a][1]);
}

$("bClean").onclick = () => { sel.clear(); setWhere(t => t.status === "clean", true); };
$("bNonEmpty").onclick = () => { sel.clear(); setWhere(t => t.status !== "empty", true); };
$("bAll").onclick = () => setWhere(() => true, true);
$("bNone").onclick = () => { sel.clear(); refresh(); };
$("bInvert").onclick = () => { T.forEach(t => sel.has(t.id) ? sel.delete(t.id) : sel.add(t.id)); refresh(); };
$("bAdd").onclick = () => setWhere(rangePred(), true);
$("bRem").onclick = () => setWhere(rangePred(), false);
$("list").addEventListener("change", e => { if (e.target.dataset.i !== undefined) toggle(+e.target.dataset.i); });
function highlight(i) {          // outline tile i in 3D (null = off)
  if (!gd) return;
  const h = {x: [], y: [], z: []};
  if (i !== null) addBox(T[i].start, T[i].end, h);
  Plotly.restyle(gd, {x: [h.x.length ? h.x : [null]], y: [h.y.length ? h.y : [null]],
                      z: [h.z.length ? h.z : [null]], visible: [i !== null]}, [iHi]);
}
$("list").addEventListener("mouseover", e => {
  const row = e.target.closest(".row");
  if (row) highlight(+row.querySelector("input").dataset.i);
});
$("list").addEventListener("mouseleave", () => highlight(null));
$("oNbr").onchange = refresh;
$("oGrid").onchange = () => Plotly.restyle(gd, {visible: [$("oGrid").checked]}, [iGrid]);
$("oContext").onchange = () => Plotly.restyle(gd, {visible: [$("oContext").checked]}, [iContext]);

$("bDownload").onclick = () => {
  if (!sel.size) { alert("Select at least one tile first."); return; }
  const blob = new Blob([JSON.stringify(selectionJSON(), null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `selection_${D.astro_id}.json`;
  document.body.appendChild(a); a.click(); a.remove();
};
$("bCopy").onclick = async () => {
  const text = JSON.stringify(selectionJSON(), null, 2);
  try { await navigator.clipboard.writeText(text); $("bCopy").textContent = "Copied"; }
  catch (e) { $("json").value = text; $("json").closest("details").open = true; $("json").select(); $("bCopy").textContent = "Select + copy below"; }
  setTimeout(() => { $("bCopy").textContent = "Copy JSON"; }, 1800);
};

Plotly.newPlot("plot", traces, layout, {responsive: true, displaylogo: false}).then(g => {
  gd = g;
  g.on("plotly_click", ev => {
    const p = ev.points && ev.points[0];
    if (p && p.curveNumber === iCent) toggle(p.pointNumber !== undefined ? p.pointNumber : p.pointIndex);
  });
  restore();
  refresh();
  window._screen = {sel, toggle, refresh, selectionJSON, unionBox, tiles: T};   // handy for debugging
});
})();
</script>
</body></html>
"""
