#!/usr/bin/env python3
"""
visualize_network.py
Generates interactive D3.js network visualizations from the causal DAG.
"""

import os, json, argparse
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NETWORK = os.path.join(BASE_DIR, "results", "causal_dag_full.csv")
DEFAULT_OUTDIR  = os.path.join(BASE_DIR, "outputs")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>%%TITLE%%</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:#0a0e1a;color:#c8d6e5;overflow:hidden;height:100vh}
#header{background:linear-gradient(135deg,#0f1628 0%,#1a1f3a 100%);border-bottom:1px solid rgba(99,132,255,0.15);padding:12px 20px;display:flex;align-items:center;justify-content:space-between;z-index:100;position:relative}
#header h1{font-size:1.15rem;font-weight:700;background:linear-gradient(135deg,#6c8cff,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.02em}
#controls{display:flex;gap:10px;align-items:center}
#search{background:#131830;border:1px solid #2a3158;border-radius:8px;padding:7px 14px;color:#c8d6e5;font-size:0.8rem;width:220px;outline:none;transition:border 0.2s}
#search:focus{border-color:#6c8cff}
.btn{background:#1a2040;border:1px solid #2a3158;border-radius:8px;padding:7px 14px;color:#8892b0;font-size:0.75rem;cursor:pointer;transition:all 0.2s}
#canvas-wrap{position:relative;width:100%;height:calc(100vh - 52px)}
svg{width:100%;height:100%;cursor:grab}
#tooltip{position:fixed;pointer-events:none;background:rgba(15,22,40,0.96);backdrop-filter:blur(12px);border:1px solid rgba(99,132,255,0.25);border-radius:10px;padding:14px 18px;font-size:0.78rem;z-index:200;display:none;box-shadow:0 8px 32px rgba(0,0,0,0.5)}
#legend{position:fixed;bottom:16px;left:16px;background:rgba(15,22,40,0.92);backdrop-filter:blur(12px);border:1px solid rgba(99,132,255,0.15);border-radius:10px;padding:14px 18px;z-index:150;font-size:0.75rem}
.leg-item { display: flex; align-items: center; margin-bottom: 5px; }
.box { width: 10px; height: 10px; margin-right: 8px; border-radius: 50%; }
</style>
</head>
<body>
<div id="header">
  <h1>%%HEADING%%</h1>
  <div id="controls">
    <input id="search" type="text" placeholder="🔍 Search gene..." autocomplete="off">
    <button class="btn" onclick="resetView()">Reset View</button>
  </div>
</div>
<div id="canvas-wrap"><svg id="graph-svg"></svg></div>
<div id="tooltip"></div>
<div id="legend">
  <div class="leg-item"><div class="box" style="background:#27ae60"></div> Activation</div>
  <div class="leg-item"><div class="box" style="background:#e74c3c"></div> Repression</div>
  <div class="leg-item"><div class="box" style="background:#f39c12"></div> Hub Gene</div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = %%JSON_DATA%%;
const nodes = data.nodes, links = data.links;

const svg = d3.select('#graph-svg');
const width = window.innerWidth, height = window.innerHeight - 52;
const g = svg.append('g');

const zoom = d3.zoom().scaleExtent([0.05, 10]).on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

// Arrow markers
const defs = svg.append('defs');
['#27ae60','#e74c3c'].forEach((c,i) => {
  defs.append('marker').attr('id','arrow-'+i).attr('viewBox','0 -5 10 10')
    .attr('refX',20).attr('refY',0).attr('markerWidth',6).attr('markerHeight',6)
    .attr('orient','auto').append('path').attr('d','M0,-4L10,0L0,4').attr('fill',c);
});

// Simulation
const sim = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d=>d.id).distance(100))
  .force('charge', d3.forceManyBody().strength(-200))
  .force('center', d3.forceCenter(width/2, height/2))
  .force('collision', d3.forceCollide().radius(d => 10 + (d.out/5)))
  .on('tick', () => {
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('transform', d=>`translate(${d.x},${d.y})`);
  });

const link = g.append('g').selectAll('line').data(links).join('line')
  .attr('stroke', d => d.reg==='act'?'#27ae60':'#e74c3c')
  .attr('stroke-width', 1).attr('stroke-opacity', 0.4)
  .attr('marker-end', d => d.reg==='act'?'url(#arrow-0)':'url(#arrow-1)');

const node = g.append('g').selectAll('g').data(nodes).join('g').attr('cursor','pointer')
  .call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended));

node.append('circle')
  .attr('r', d => 5 + (d.out/5))
  .attr('fill', d => d.hub ? '#f39c12' : '#6c8cff')
  .attr('stroke', '#0a0e1a').attr('stroke-width', 1);

node.append('text').text(d=>d.id).attr('dy', -12).attr('text-anchor','middle').attr('fill','#fff').attr('font-size','10px');

// Drag functions
function dragstarted(e,d) { if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }
function dragged(e,d) { d.fx=e.x; d.fy=e.y; }
function dragended(e,d) { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }

// Search
document.getElementById('search').addEventListener('input', function(e){
  const q = e.target.value.toUpperCase();
  if(!q) { node.attr('opacity', 1); link.attr('opacity', 0.4); return; }
  node.attr('opacity', n => n.id.includes(q) ? 1 : 0.1);
  link.attr('opacity', l => (l.source.id.includes(q) || l.target.id.includes(q)) ? 0.8 : 0.05);
});

function resetView(){
  svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity);
  document.getElementById('search').value = '';
  node.attr('opacity', 1); link.attr('opacity', 0.4);
}
</script>
</body>
</html>"""

def build_graph_json(df, hub_set=None):
    out_deg = Counter(df["Cause"])
    in_deg  = Counter(df["Effect"])
    all_genes = set(df["Cause"]) | set(df["Effect"])
    nodes = []
    for g in all_genes:
        nodes.append({"id": g, "out": out_deg.get(g,0), "indeg": in_deg.get(g,0), "hub": g in hub_set if hub_set else out_deg.get(g,0) >= 10})
    links = []
    for _, row in df.iterrows():
        links.append({"source": row["Cause"], "target": row["Effect"], "es": float(row["EffectSize"]), "reg": "act" if "Positive" in str(row["Regulation"]) else "rep"})
    return {"nodes": nodes, "links": links}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--outdir",  default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if not os.path.exists(args.network): return
    df = pd.read_csv(args.network)
    out_deg = Counter(df["Cause"])
    top50 = [g for g,_ in out_deg.most_common(50)]
    
    # Generate D3 Full
    full_json = build_graph_json(df, top50)
    html_full = HTML_TEMPLATE.replace("%%JSON_DATA%%", json.dumps(full_json)).replace("%%TITLE%%", "Full Network").replace("%%HEADING%%", "Causal Gene Regulatory Network")
    with open(os.path.join(args.outdir, "network_full_d3.html"), "w") as f: f.write(html_full)

    # Generate D3 Top Hubs
    subnet = df[df["Cause"].isin(top50)].copy()
    hub_json = build_graph_json(subnet, set(top50))
    html_hub = HTML_TEMPLATE.replace("%%JSON_DATA%%", json.dumps(hub_json)).replace("%%TITLE%%", "Top 50 Hubs").replace("%%HEADING%%", "Master Regulators (Top 50)")
    with open(os.path.join(args.outdir, "network_top50_d3.html"), "w") as f: f.write(html_hub)
    print("D3.js visualizations saved to", args.outdir)

if __name__ == "__main__":
    main()
