#!/usr/bin/env python3
"""Product-facing TShape-Zero+ checkpoint, CLI, and local web demo."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT_CHECKPOINT = MODELS / "tshape_zero_plus_release.pt"
DEFAULT_DEMO_CASE = ROOT / "Results" / "RENE" / "product_demo_case_scores.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from layyer import TShape_model  # noqa: E402
from rene_experiments import (  # noqa: E402
    DEFAULT_DATASETS,
    PATTERN_BANK_VERSION,
    iter_series,
    scale01,
    train_tshape_model,
    train_tshape_hybrid_model,
    tshape_scores,
    tshape_zero_plus_variants,
)


def parse_values(text: str) -> np.ndarray:
    cleaned = text.replace(",", " ").replace("\n", " ").replace("\t", " ")
    vals = [float(tok) for tok in cleaned.split() if tok.strip()]
    if len(vals) < 24:
        raise ValueError("Need at least 24 numeric values for a useful score trace.")
    return np.asarray(vals, dtype=np.float32)


def load_values(path: Path | None, values: str | None) -> np.ndarray:
    if values:
        return parse_values(values)
    if path is None:
        raise ValueError("Provide --values or --input.")
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    return parse_values(path.read_text(encoding="utf-8"))


def experiment_difference(values: np.ndarray, order: int) -> np.ndarray:
    """Use the exact drop-first differencing contract used by the evaluation harness."""
    output = np.asarray(values, dtype=np.float32).reshape(-1)
    for _ in range(order):
        if output.size < 2:
            return np.empty(0, dtype=np.float32)
        output = np.diff(output).astype(np.float32)
    return output


def prepare_target_context(
    values: np.ndarray,
    diff_order: int,
    calibration_values: np.ndarray | None = None,
) -> tuple[np.ndarray, int, str]:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    raw_len = len(series)
    series = experiment_difference(series, diff_order)
    if calibration_values is None:
        calibration = series
        calibration_scope = "uploaded unlabeled history"
    else:
        calibration = experiment_difference(
            np.asarray(calibration_values, dtype=np.float32).reshape(-1),
            diff_order,
        )
        if len(calibration) == 0:
            raise ValueError("Calibration history is empty after preprocessing.")
        calibration_scope = "separate unlabeled calibration history"
    lo = float(np.nanmin(calibration))
    hi = float(np.nanmax(calibration))
    if not np.isfinite(hi - lo) or hi - lo < 1e-8:
        scaled = np.zeros_like(series, dtype=np.float32)
    else:
        scaled = ((series - lo) / (hi - lo)).astype(np.float32)
    return (
        np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0),
        raw_len,
        calibration_scope,
    )


def load_model(checkpoint: Path, device: str = "cpu"):
    import torch

    payload = torch.load(checkpoint, map_location=device)
    meta = payload.get("metadata", {})
    p = int(meta.get("p", 16))
    model = TShape_model(p)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, meta


def score_array(
    values: np.ndarray,
    checkpoint: Path,
    device: str = "cpu",
    calibration_values: np.ndarray | None = None,
) -> dict:
    model, meta = load_model(checkpoint, device=device)
    p = int(meta.get("p", 16))
    diff_order = int(meta.get("diff_order", 1))
    prepared, raw_len, calibration_scope = prepare_target_context(
        values,
        diff_order,
        calibration_values,
    )
    if len(prepared) <= p:
        raise ValueError(f"Need more than {p + diff_order} points after preprocessing.")
    tscore = tshape_scores(model, prepared, p, batch_size=4096)
    variants = tshape_zero_plus_variants(tscore, prepared, p)
    alpha = float(meta.get("fusion_alpha", 0.05))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid checkpoint fusion_alpha={alpha}")
    residual_guard = variants["TShape-zero_plus_residual_only_minmax"]
    tshape_channel = scale01(tscore)
    final = alpha * tshape_channel + (1.0 - alpha) * residual_guard
    alignment_pad = raw_len - len(final)
    if alignment_pad < 0:
        raise ValueError("Scorer produced more values than the raw input history.")

    def align(scores: np.ndarray) -> np.ndarray:
        if alignment_pad == 0:
            return np.asarray(scores, dtype=np.float32)
        floor = float(np.min(scores)) if len(scores) else 0.0
        return np.concatenate(
            [np.full(alignment_pad, floor, dtype=np.float32), np.asarray(scores, dtype=np.float32)]
        )

    final = align(final)
    tshape_channel = align(tshape_channel)
    residual_guard = align(residual_guard)
    top_k = min(10, len(final))
    top = np.argsort(-final)[:top_k]
    return {
        "metadata": meta,
        "raw_length": raw_len,
        "scored_length": int(len(final)),
        "score_index_offset": 0,
        "differencing_warmup_points": alignment_pad,
        "calibration_scope": calibration_scope,
        "fusion_alpha": alpha,
        "scores": [float(x) for x in final],
        "tshape_scores": [float(x) for x in tshape_channel],
        "residual_guard_scores": [float(x) for x in residual_guard],
        "top_anomalies": [
            {
                "rank": i + 1,
                "index": int(idx),
                "score": float(final[idx]),
                "tshape_channel": float(tshape_channel[idx]),
                "residual_guard": float(residual_guard[idx]),
            }
            for i, idx in enumerate(top)
        ],
    }


def train_release(args: argparse.Namespace) -> None:
    import torch

    MODELS.mkdir(parents=True, exist_ok=True)
    local = argparse.Namespace(
        p=args.p,
        diff_order=args.diff_order,
        windows_per_series=args.windows_per_series,
        max_train_windows=args.max_train_windows,
        balance_source_datasets=args.balance_source_datasets,
        synthetic_ratio=args.synthetic_ratio,
        pattern_bank_fraction=args.pattern_bank_fraction,
        pattern_bank_variants=args.pattern_bank_variants,
        pattern_bank_noise_std=args.pattern_bank_noise_std,
        pattern_bank_event_rate=args.pattern_bank_event_rate,
        pattern_bank_event_scale=args.pattern_bank_event_scale,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    refs = iter_series(args.datasets)
    if args.include_pattern_bank:
        model, run_info = train_tshape_hybrid_model(local, refs)
    else:
        model, run_info = train_tshape_model(local, refs)
    metadata = {
        "name": "TShape-Zero+ release checkpoint",
        "version": "2026-07-14",
        "checkpoint_role": args.checkpoint_role,
        "datasets": list(args.datasets),
        "excluded_target": args.excluded_target or None,
        "p": args.p,
        "diff_order": args.diff_order,
        "preprocessing": "np.diff drop-first; min-max fitted on unlabeled history/calibration values",
        "output_alignment": "prepend a minimum-score warm-up value per differencing order",
        "windows_per_series": args.windows_per_series,
        "max_train_windows": args.max_train_windows,
        "balance_source_datasets": bool(args.balance_source_datasets),
        "source_anomaly_filtering": False,
        "synthetic_ratio": args.synthetic_ratio,
        "pattern_bank_fraction": args.pattern_bank_fraction,
        "pattern_bank_included": bool(args.include_pattern_bank),
        "pattern_bank_variants": args.pattern_bank_variants,
        "pattern_bank_noise_std": args.pattern_bank_noise_std,
        "pattern_bank_event_rate": args.pattern_bank_event_rate,
        "pattern_bank_event_scale": args.pattern_bank_event_scale,
        "pattern_bank_version": PATTERN_BANK_VERSION,
        "epochs": args.epochs,
        "seed": args.seed,
        "fusion_alpha": args.fusion_alpha,
        "score_formula": (
            f"{args.fusion_alpha:.2f} minmax(Pattern-TShape) + "
            f"{1.0 - args.fusion_alpha:.2f} residual_guard; "
            "residual_guard=(0.55 median + 0.25 spectral + 0.05 MAD)/0.85"
        ),
        "fusion_selection": "source-only cross-dataset EasyTSAD validation; no evaluated target labels",
        "calibration": "label-free per-series score output; caller chooses threshold or uses top-k.",
        "run_info": run_info,
    }
    torch.save({"metadata": metadata, "state_dict": model.cpu().state_dict()}, args.output)
    print(f"Wrote {args.output}")


def score_command(args: argparse.Namespace) -> None:
    values = load_values(args.input, args.values)
    calibration_values = None
    if args.calibration_input is not None or args.calibration_values:
        calibration_values = load_values(args.calibration_input, args.calibration_values)
    result = score_array(
        values,
        args.checkpoint,
        device=args.device,
        calibration_values=calibration_values,
    )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(result, indent=2))


DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TShape-Zero+ Open-Box Scoring Demo</title>
<style>
*{box-sizing:border-box}
body{font-family:Inter,ui-sans-serif,system-ui,Arial,sans-serif;margin:0;background:#f4f7fa;color:#172033}
main{position:relative;max-width:1180px;margin:0 auto;padding:18px 22px}
.hero{display:flex;justify-content:space-between;gap:18px;align-items:end;margin-bottom:11px}
h1{margin:0 0 6px;font-size:31px;letter-spacing:0;font-weight:850}
.subtitle{max-width:820px;color:#526178;font-size:14px;line-height:1.45;margin:0}
.status{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.pill{background:#fff;color:#177d72;border:1px solid #ccebe7;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:800;white-space:nowrap}
.shell{display:grid;grid-template-columns:360px 1fr;gap:14px;align-items:start}
.panel{background:#fff;border:1px solid #d9e2ec;border-radius:8px;padding:12px;box-shadow:0 8px 24px rgba(23,32,51,.08)}
.results{min-width:0}
.panel h2{font-size:15px;margin:0 0 10px;font-weight:850}
textarea{width:100%;height:245px;border:1px solid #cbd5e1;border-radius:8px;padding:10px;font-family:Menlo,Consolas,monospace;font-size:11px;line-height:1.35;background:#fbfcff;color:#172033;resize:vertical}
button{border:0;border-radius:8px;background:#2a9d8f;color:white;font-weight:850;padding:10px 14px;margin-right:8px;cursor:pointer;box-shadow:0 6px 14px rgba(42,157,143,.20)}
button.secondary{background:#6c5ce7;box-shadow:0 8px 18px rgba(108,92,231,.22)}
button.ghost{background:#f4a261;color:#172033;box-shadow:0 8px 18px rgba(244,162,97,.22)}
.muted{color:#64748b;font-size:12px;line-height:1.45}
.charts{display:grid;grid-template-columns:1fr;gap:8px}
.chart-card{border:1px solid #dbe5ef;border-radius:8px;background:#fbfcff;padding:8px}
.chart-title{display:flex;justify-content:space-between;align-items:center;margin:0 0 6px;color:#263244;font-size:12px;font-weight:850}
canvas{width:100%;height:172px;border-radius:8px;background:white}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:8px 0}
.metric{border:1px solid #dbe5ef;border-radius:8px;background:#ffffff;padding:8px}
.metric b{display:block;font-size:18px}.metric span{display:block;color:#64748b;font-size:11px;margin-top:2px}
.evidence{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin:0 0 8px;padding:7px 9px;border-left:4px solid #6c5ce7;background:#f7f6ff;color:#344054;font-size:11px;font-weight:750}
.evidence.hidden{display:none}.evidence b{color:#6c5ce7}.legend{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:#64748b;font-size:10px;font-weight:750;margin:-1px 0 5px}
.legend i{display:inline-block;width:15px;height:3px;border-radius:2px;margin-right:4px;vertical-align:middle}
table{width:100%;border-collapse:collapse;font-size:11px;background:white;border-radius:8px;overflow:hidden}
td,th{border-bottom:1px solid #e5edf5;padding:4px 6px;text-align:right}td:first-child,th:first-child{text-align:left}
th{background:#f7fafc;color:#526178;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.channels{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.channel{padding:8px 0 2px;border-top:1px solid #dbe5ef;background:transparent}
.channel strong{display:block;font-size:13px}.channel p{margin:4px 0 0;color:#64748b;font-size:12px;line-height:1.35}
@media(max-width:900px){.hero{display:block}.status{justify-content:flex-start;margin-top:10px}.shell{grid-template-columns:1fr}.metrics{grid-template-columns:1fr}}
</style>
</head>
<body>
<main>
<section class="hero">
<div>
<h1>TShape-Zero+ Open-Box Scoring Demo</h1>
<p class="subtitle">Paste a univariate history. One frozen Pattern-Bank checkpoint combines shape residuals with transparent reliability guards and returns aligned anomaly scores plus the top-ranked suspicious points.</p>
</div>
<div class="status"><span class="pill">synthetic-only checkpoint</span><span class="pill">no benchmark fitting</span><span class="pill">open-box channels</span></div>
</section>
<div class="shell">
<section class="panel">
<h2>Input history</h2>
<textarea id="values" oninput="markCustom()"></textarea>
<p><button onclick="score()">Score series</button><button class="secondary" onclick="loadExample()">Example</button><button class="ghost" onclick="clearAll()">Clear</button></p>
<p class="muted">Input accepts comma, space, tab, or newline separated values. Experiment-matched first differencing plus one non-alarming warm-up pad keeps returned indices aligned with the upload.</p>
<div class="channels">
<div class="channel"><strong>TShape channel</strong><p>Pattern-Bank checkpoint residual, exposed instead of hidden.</p></div>
<div class="channel"><strong>Residual guard</strong><p>Median, spectral, and MAD evidence for safer zero-shot use.</p></div>
</div>
</section>
<section class="results">
<div class="metrics">
<div class="metric"><b id="npoints">-</b><span>input points</span></div>
<div class="metric"><b id="scored">-</b><span>scored points</span></div>
<div class="metric"><b id="maxscore">-</b><span>max Zero+ score</span></div>
</div>
<div id="caseEvidence" class="evidence hidden"></div>
<div class="charts">
<div class="chart-card"><div class="chart-title"><span>1. User input time series</span><span id="seriesLabel">waiting for input</span></div><canvas id="seriesPlot" width="760" height="220"></canvas></div>
<div class="chart-card"><div class="chart-title"><span>2. Open-box anomaly score channels</span><span id="scoreLabel">model output</span></div><div class="legend"><span><i style="background:#e76f51"></i>Zero+</span><span><i style="background:#6c5ce7"></i>TShape</span><span><i style="background:#2a9d8f"></i>Residual guard</span><span><i style="background:rgba(231,111,81,.22)"></i>Retrospective event</span></div><canvas id="scorePlot" width="760" height="220"></canvas></div>
</div>
<h2>Top anomalies</h2>
<table><thead><tr><th>Rank</th><th>Index</th><th>Zero+ score</th><th>TShape channel</th><th>Guard channel</th></tr></thead><tbody id="tops"></tbody></table>
</section>
</div>
</main>
<script>
let ex = "", exLabels = [], exampleMeta = {}, exampleActive = false;
function showEvidence(){const e=document.getElementById('caseEvidence');if(!exampleActive||!exampleMeta.anomaly_events){e.classList.add('hidden');return}e.innerHTML=`<b>Real AIOPS stress case</b><span>Synthetic Pattern checkpoint: no benchmark fitting</span><span>${exampleMeta.anomaly_events} event phases</span><span>Point-F1 ${Number(exampleMeta.local_point_f1_pa).toFixed(3)}</span><span>Event-F1 ${Number(exampleMeta.local_event_f1_pa_log).toFixed(3)}</span>`;e.classList.remove('hidden')}
function markCustom(){exampleActive=false;showEvidence()}
async function fetchExample(){try{const res=await fetch('/example');const data=await res.json();ex=(data.values||[]).map(x=>Number(x).toFixed(6)).join(", ");exLabels=data.labels||[];exampleMeta=data.case||{};document.getElementById('seriesLabel').textContent=data.label||'AIOPS real example'}catch(e){ex=Array.from({length:240},(_,i)=>Math.sin(i/9)+0.08*Math.cos(i/3)+(i===92?3.1:0)+(i===156?-2.4:0)).map(x=>x.toFixed(4)).join(", ");exLabels=[];exampleMeta={}}}
async function loadExample(){if(!ex){await fetchExample()}document.getElementById('values').value=ex;exampleActive=true;showEvidence();document.getElementById('seriesLabel').textContent=exampleMeta.case_name||'AIOPS complex KPI case'}
function clearAll(){exampleActive=false;showEvidence();document.getElementById('values').value='';['seriesPlot','scorePlot'].forEach(id=>{const c=document.getElementById(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height)});document.getElementById('tops').innerHTML=''}
function parseVals(text){return text.replace(/[,\\n\\t]/g,' ').split(' ').filter(Boolean).map(Number).filter(Number.isFinite)}
function shadeEvents(ctx,c,labels){if(!labels?.length)return;ctx.fillStyle='rgba(231,111,81,.11)';let start=-1;for(let i=0;i<=labels.length;i++){if(i<labels.length&&labels[i]&&start<0)start=i;if((i===labels.length||!labels[i])&&start>=0){let x0=start/Math.max(1,labels.length-1)*(c.width-22)+11,x1=(i-1)/Math.max(1,labels.length-1)*(c.width-22)+11;ctx.fillRect(x0,0,Math.max(3,x1-x0+2),c.height);start=-1}}}
function drawLine(id, vals, color, tops=[],labels=[]){const c=document.getElementById(id),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#ffffff';ctx.fillRect(0,0,c.width,c.height);shadeEvents(ctx,c,labels);ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;for(let y=36;y<c.height;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(c.width,y);ctx.stroke()}if(!vals.length)return;let lo=Math.min(...vals),hi=Math.max(...vals);if(Math.abs(hi-lo)<1e-9){hi=lo+1}ctx.strokeStyle=color;ctx.lineWidth=2.4;ctx.beginPath();vals.forEach((s,i)=>{let x=i/Math.max(1,vals.length-1)*(c.width-22)+11,y=c.height-18-(s-lo)/(hi-lo)*(c.height-36);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();ctx.fillStyle='#ef476f';tops.forEach(t=>{let i=Math.min(vals.length-1,Math.max(0,t.index));let x=i/Math.max(1,vals.length-1)*(c.width-22)+11,y=c.height-18-(vals[i]-lo)/(hi-lo)*(c.height-36);ctx.beginPath();ctx.arc(x,y,4.2,0,Math.PI*2);ctx.fill()})}
function drawScores(data,labels=[]){const c=document.getElementById('scorePlot'),ctx=c.getContext('2d'),sets=[[data.residual_guard_scores,'#2a9d8f',1.2],[data.tshape_scores,'#6c5ce7',1.5],[data.scores,'#e76f51',2.5]];ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fff';ctx.fillRect(0,0,c.width,c.height);shadeEvents(ctx,c,labels);ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1;for(let y=36;y<c.height;y+=36){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(c.width,y);ctx.stroke()}sets.forEach(([vals,color,width])=>{if(!vals?.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();vals.forEach((s,i)=>{let x=i/Math.max(1,vals.length-1)*(c.width-22)+11,y=c.height-18-s*(c.height-36);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke()});ctx.fillStyle='#ef476f';data.top_anomalies.forEach(t=>{let i=Math.min(data.scores.length-1,Math.max(0,t.index)),x=i/Math.max(1,data.scores.length-1)*(c.width-22)+11,y=c.height-18-data.scores[i]*(c.height-36);ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill()})}
async function score(){const values=document.getElementById('values').value;const series=parseVals(values),labels=exampleActive?exLabels:[];drawLine('seriesPlot',series,'#2a9d8f',[],labels);const res=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({values})});const data=await res.json();if(data.error){alert(data.error);return}drawScores(data,labels);document.getElementById('tops').innerHTML=data.top_anomalies.map(r=>`<tr><td>${r.rank}</td><td>${r.index}</td><td>${r.score.toFixed(3)}</td><td>${r.tshape_channel.toFixed(3)}</td><td>${r.residual_guard.toFixed(3)}</td></tr>`).join('');document.getElementById('npoints').textContent=series.length;document.getElementById('scored').textContent=data.scored_length;document.getElementById('maxscore').textContent=Math.max(...data.scores).toFixed(3);document.getElementById('seriesLabel').textContent=exampleActive?(exampleMeta.case_name||'AIOPS complex KPI case'):`${series.length} values`;document.getElementById('scoreLabel').textContent=`top index ${data.top_anomalies[0]?.index ?? '-'}`}
fetchExample().then(()=>{loadExample();drawLine('seriesPlot',parseVals(ex),'#2a9d8f',[],exLabels)})
</script>
</body>
</html>"""


def serve(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint
    device = args.device

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/example"):
                if DEFAULT_DEMO_CASE.exists():
                    payload = json.loads(DEFAULT_DEMO_CASE.read_text(encoding="utf-8"))
                    case = payload.get("case", {})
                    body = {
                        "values": payload.get("input_values", []),
                        "labels": payload.get("labels", []),
                        "label": f"{case.get('dataset', 'AIOPS')} real KPI window",
                        "case": case,
                    }
                else:
                    values = [
                        float(np.sin(i / 9) + 0.08 * np.cos(i / 3) + (3.1 if i == 92 else 0.0) - (2.4 if i == 156 else 0.0))
                        for i in range(220)
                    ]
                    body = {"values": values, "label": "synthetic fallback"}
                self._send(200, json.dumps(body).encode("utf-8"), "application/json")
                return
            self._send(200, DEMO_HTML.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                values = payload.get("values", "")
                calibration_text = payload.get("calibration_values", "")
                calibration_values = parse_values(calibration_text) if calibration_text else None
                result = score_array(
                    parse_values(values),
                    checkpoint,
                    device=device,
                    calibration_values=calibration_values,
                )
                self._send(200, json.dumps(result).encode("utf-8"), "application/json")
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TShape-Zero+ demo: http://{args.host}:{args.port}")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train-release")
    p_train.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    p_train.add_argument("--p", type=int, default=16)
    p_train.add_argument("--diff-order", type=int, default=1)
    p_train.add_argument("--windows-per-series", type=int, default=256)
    p_train.add_argument("--max-train-windows", type=int, default=60000)
    p_train.add_argument("--balance-source-datasets", action="store_true")
    p_train.add_argument("--synthetic-ratio", type=float, default=0.25)
    p_train.add_argument("--pattern-bank-fraction", type=float, default=1.00)
    p_train.add_argument("--pattern-bank-variants", type=int, default=8)
    p_train.add_argument("--pattern-bank-noise-std", type=float, default=0.025)
    p_train.add_argument("--pattern-bank-event-rate", type=float, default=0.35)
    p_train.add_argument("--pattern-bank-event-scale", type=float, default=0.75)
    p_train.add_argument("--include-pattern-bank", action="store_true")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=20260704)
    p_train.add_argument("--fusion-alpha", type=float, default=0.05)
    p_train.add_argument("--excluded-target", default="")
    p_train.add_argument(
        "--checkpoint-role",
        default="future-unseen product checkpoint",
    )
    p_train.add_argument("--device", default="auto", choices=["cpu", "mps", "auto"])
    p_train.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    p_train.set_defaults(func=train_release)

    p_score = sub.add_parser("score")
    p_score.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_score.add_argument("--input", type=Path)
    p_score.add_argument("--values")
    p_score.add_argument("--calibration-input", type=Path)
    p_score.add_argument("--calibration-values")
    p_score.add_argument("--output", type=Path)
    p_score.add_argument("--device", default="cpu")
    p_score.set_defaults(func=score_command)

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument("--device", default="cpu")
    p_serve.set_defaults(func=serve)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
