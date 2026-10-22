#!/usr/bin/env python3
"""Luko studio: run one generation job against the Gemini API and save every output to out/.

The job arrives as JSON in the JOB environment variable; the key comes from GEMINI_API_KEY (a repository secret).
Only the Python standard library is used. Job shape:

  {"name": "ui-kit", "model": "gemini-3.1-flash-image", "style": "text added to the front of every prompt",
   "refs": ["refs/style_game.jpg"],                  # reference images for every item (paths under .github/studio)
   "config": {...},                                  # optional, merged into generationConfig (e.g. for music models)
   "items": [{"id": "banner", "prompt": "...", "refs": [...], "n": 2, "aspect": "16:9", "size": "2K"}]}
"""
import base64, concurrent.futures as cf, json, mimetypes, os, pathlib, sys, time, urllib.error, urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_OUTPUTS = 24   # a guard on cost: one run never asks for more than this many generations
EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
       "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav"}
HERE = pathlib.Path(__file__).parent
OUT = pathlib.Path("out")


def ref_part(path):
    p = (HERE / path).resolve()
    if HERE.resolve() not in p.parents:
        raise ValueError(f"reference outside .github/studio: {path}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(p.read_bytes()).decode()}}


def call(model, body, key):
    req = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:3000]
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(10 * (attempt + 1)); continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except urllib.error.URLError:
            if attempt < 4:
                time.sleep(10 * (attempt + 1)); continue
            raise


def run_one(job, item, k, key):
    model = item.get("model") or job["model"]
    prompt = " ".join(x for x in [job.get("style", ""), item["prompt"]] if x).strip()
    parts = [{"text": prompt}] + [ref_part(r) for r in job.get("refs", []) + item.get("refs", [])]
    cfg = {}
    if "image" in model:
        cfg["responseModalities"] = ["TEXT", "IMAGE"]
        ic = {}
        if item.get("aspect"): ic["aspectRatio"] = item["aspect"]
        if item.get("size"): ic["imageSize"] = item["size"]
        if ic: cfg["imageConfig"] = ic
    cfg.update(job.get("config", {})); cfg.update(item.get("config", {}))
    body = {"contents": [{"role": "user", "parts": parts}]}
    if cfg: body["generationConfig"] = cfg
    t0 = time.time()
    res = call(model, body, key)
    saved, texts = [], []
    for cand in res.get("candidates", []):
        for p in cand.get("content", {}).get("parts", []):
            d = p.get("inlineData") or p.get("inline_data")
            if d:
                mime = d.get("mimeType") or d.get("mime_type") or "application/octet-stream"
                name = f"{item['id']}_{k + 1}{'' if not saved else '_' + str(len(saved) + 1)}{EXT.get(mime, '.bin')}"
                (OUT / name).write_bytes(base64.b64decode(d["data"])); saved.append(name)
            elif p.get("text") and not p.get("thought"):
                texts.append(p["text"])
    if texts:
        (OUT / f"{item['id']}_{k + 1}.txt").write_text("\n".join(texts))
    return {"id": item["id"], "k": k + 1, "model": model, "files": saved, "seconds": round(time.time() - t0, 1),
            "finish": [c.get("finishReason") for c in res.get("candidates", [])], "feedback": res.get("promptFeedback")}


def fatal(msg):
    """Stop the run, leaving the reason where Claude can read it (the studio branch and a check annotation)."""
    OUT.mkdir(exist_ok=True)
    (OUT / "fatal.txt").write_text(msg)
    print(f"::error::{msg}", flush=True)
    sys.exit(1)


def main():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        fatal("GEMINI_API_KEY secret is not set: add it under Settings > Secrets and variables > Actions")
    try:
        job = json.loads(os.environ["JOB"])
    except Exception as e:
        fatal(f"the job is not valid JSON: {e}")
    tasks = [(it, k) for it in job["items"] for k in range(int(it.get("n", 1)))]
    if len(tasks) > MAX_OUTPUTS:
        fatal(f"{len(tasks)} generations asked for; the limit is {MAX_OUTPUTS} per run")
    OUT.mkdir(exist_ok=True)
    (OUT / "job.json").write_text(json.dumps(job, indent=2))
    log, failed = [], 0
    with cf.ThreadPoolExecutor(max_workers=int(job.get("parallel", 4))) as ex:
        futs = {ex.submit(run_one, job, it, k, key): (it["id"], k + 1) for it, k in tasks}
        for f in cf.as_completed(futs):
            try:
                r = f.result(); log.append(r); print("ok", r["id"], r["k"], r["files"], r["seconds"], "s", flush=True)
            except Exception as e:
                failed += 1; i, k = futs[f]; log.append({"id": i, "k": k, "error": str(e)}); print("FAILED", i, k, str(e)[:1500], flush=True)
    (OUT / "log.json").write_text(json.dumps(sorted(log, key=lambda r: (r["id"], r["k"])), indent=2))
    if failed == len(tasks):
        fatal("every generation failed: " + "; ".join(r.get("error", "")[:400] for r in log[:3]))


if __name__ == "__main__":
    main()
