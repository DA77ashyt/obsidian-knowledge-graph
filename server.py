"""
Obsidian Knowledge Graph Platform - Server v0.3.0
FastAPI backend: vault parser + graph builder + DeepSeek AI multi-analysis
"""
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import frontmatter
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("obsidian-kg")

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    cfg = {
        "vault_path": "",
        "deepseek_api_key": "",
        "deepseek_model": "deepseek-chat",
        "deepseek_base_url": "https://api.deepseek.com/v1/chat/completions",
        "server_host": "127.0.0.1",
        "server_port": 8765,
    }
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    else:
        example = CONFIG_PATH.parent / "config.example.json"
        if example.exists():
            import shutil
            shutil.copy(example, CONFIG_PATH)
            logger.warning("config.json created from template")
    if os.environ.get("DEEPSEEK_API_KEY"):
        cfg["deepseek_api_key"] = os.environ["DEEPSEEK_API_KEY"]
    if os.environ.get("OBSIDIAN_VAULT_PATH"):
        cfg["vault_path"] = os.environ["OBSIDIAN_VAULT_PATH"]
    if os.environ.get("DEEPSEEK_MODEL"):
        cfg["deepseek_model"] = os.environ["DEEPSEEK_MODEL"]
    if os.environ.get("SERVER_PORT"):
        cfg["server_port"] = int(os.environ["SERVER_PORT"])
    return cfg


cfg = load_config()
VAULT_PATH = Path(cfg["vault_path"]) if cfg.get("vault_path") else None
if not VAULT_PATH or not VAULT_PATH.exists():
    logger.warning("vault_path not configured")
    VAULT_PATH = None

DEEPSEEK_KEY = cfg.get("deepseek_api_key", "")
if DEEPSEEK_KEY in ("", "sk-your-deepseek-api-key", "your-deepseek-api-key-here"):
    DEEPSEEK_KEY = ""
DEEPSEEK_MODEL = cfg.get("deepseek_model", "deepseek-chat")
DEEPSEEK_URL = cfg.get("deepseek_base_url", "https://api.deepseek.com/v1/chat/completions")
AI_ENABLED = bool(DEEPSEEK_KEY)
if not AI_ENABLED:
    logger.warning("DeepSeek API key not configured - AI features disabled")

_http_client = None


async def _get_client():
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10))
    return _http_client


app = FastAPI(title="Obsidian Knowledge Graph", version="0.3.0")


def _check_key():
    if not AI_ENABLED:
        raise HTTPException(status_code=400, detail="AI disabled. Configure DEEPSEEK_API_KEY.")


def _extract_json(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    for pat in [r"```json\s*\n?([\s\S]*?)\n?```", r"```\s*\n?([\s\S]*?)\n?```"]:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return text


async def _call_deepseek(prompt: str, max_tokens: int = 2048) -> str:
    client = await _get_client()
    resp = await client.post(DEEPSEEK_URL,
        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
        json={"model": DEEPSEEK_MODEL, "messages": [
            {"role": "system", "content": "You are a knowledge management expert. Always output valid JSON."},
            {"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": max_tokens})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"DeepSeek API error (HTTP {resp.status_code})")
    result = resp.json()
    choices = result.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="DeepSeek API returned empty choices")
    content = choices[0].get("message", {}).get("content")
    if content is None:
        raise HTTPException(status_code=502, detail="DeepSeek API returned null content")
    return _extract_json(content)


class AskRequest(BaseModel):
    question: str


def extract_wiki_links(content: str) -> list:
    pattern = r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]"
    return [m.strip() for m in re.findall(pattern, content) if m.strip()]


def extract_tags_from_frontmatter(fm_tags) -> list:
    if not fm_tags:
        return []
    if isinstance(fm_tags, str):
        return [fm_tags.strip().lstrip("#")]
    if isinstance(fm_tags, list):
        return [str(t).strip().lstrip("#") for t in fm_tags if t]
    return []


def extract_inline_tags(content: str) -> list:
    clean = re.sub(r"```[^`]*```", "", content, flags=re.DOTALL)
    clean = re.sub(r"`[^`]+`", "", clean)
    matches = re.findall(r"(?<!\w)#([a-zA-Z一-鿿][a-zA-Z0-9一-鿿_/-]*)", clean)
    seen = set()
    result = []
    for m in matches:
        if len(m) >= 2 and m.lower() not in seen:
            seen.add(m.lower())
            result.append(m)
    return result


def extract_summary(content: str, max_chars: int = 200) -> str:
    clean = re.sub(r"```[^`]*```", "", content, flags=re.DOTALL)
    clean = re.sub(r"`[^`]+`", "", clean)
    clean = re.sub(r"#{1,6}\s+", "", clean)
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
    clean = re.sub(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]", r"\1", clean)
    clean = re.sub(r"[*_~>|]", "", clean)
    clean = re.sub(r"\n{2,}", "\n", clean).strip()
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    summary = " ".join(lines[:3])
    if len(summary) > max_chars:
        summary = summary[:max_chars - 3] + "..."
    return summary


# Vault scan cache
_vault_cache = None
_vault_cache_mtime = 0.0


def _get_vault_mtime():
    if not VAULT_PATH or not VAULT_PATH.exists():
        return 0.0
    try:
        return VAULT_PATH.stat().st_mtime
    except OSError:
        return 0.0


def get_cached_scan():
    global _vault_cache, _vault_cache_mtime
    cur = _get_vault_mtime()
    if _vault_cache is not None and cur == _vault_cache_mtime:
        return _vault_cache
    data = scan_vault()
    _vault_cache = data
    _vault_cache_mtime = cur
    return data


def invalidate_cache():
    global _vault_cache, _vault_cache_mtime
    _vault_cache = None
    _vault_cache_mtime = 0.0


def scan_vault():
    if not VAULT_PATH or not VAULT_PATH.exists():
        return {"nodes": [], "edges": [], "total": 0}
    nodes = []
    node_index = {}
    node_id_counter = 0
    for md_file in sorted(VAULT_PATH.rglob("*.md")):
        rel_path = md_file.relative_to(VAULT_PATH)
        parts = rel_path.parts
        if any(p.startswith(".") for p in parts):
            continue
        if ".agents" in parts:
            continue
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception as e:
            logger.warning("Skipping %s: %s", rel_path, e)
            continue
        try:
            post = frontmatter.loads(raw)
            fm, body = post.metadata, post.content
        except Exception:
            fm, body = {}, raw
        title = fm.get("title", "")
        if not title:
            h1 = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if h1:
                title = h1.group(1).strip()
        if not title:
            title = md_file.stem
        fm_tags = extract_tags_from_frontmatter(fm.get("tags", []))
        inline_tags = extract_inline_tags(body)
        all_tags = list(dict.fromkeys(fm_tags + inline_tags))
        wiki_links = extract_wiki_links(body)
        headings = []
        for m in re.finditer(r"^(#{1,2})\s+(.+)$", body, re.MULTILINE):
            headings.append({"level": len(m.group(1)), "text": m.group(2).strip()})
        summary = extract_summary(body)
        domain = parts[0] if len(parts) > 1 else "root"
        mtime = md_file.stat().st_mtime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        node_id_counter += 1
        node = {"id": f"n{node_id_counter}", "title": title,
                "path": str(rel_path).replace("\\", "/"), "tags": all_tags,
                "links": wiki_links, "headings": headings[:10], "domain": domain,
                "mtime": mtime_str, "mtime_ts": mtime, "summary": summary}
        nodes.append(node)
        node_index[str(rel_path).replace("\\", "/")] = f"n{node_id_counter}"
    edges = []
    edge_set = set()
    def add_edge(src, tgt, etype):
        key = f"{src}-{tgt}-{etype}"
        if key not in edge_set and src != tgt:
            edge_set.add(key)
            edges.append({"source": src, "target": tgt, "type": etype})
    stem_to_id = {}
    for n in nodes:
        stem = Path(n["path"]).stem
        stem_to_id[stem.lower()] = n["id"]
        stem_to_id[n["path"].replace(".md", "").lower()] = n["id"]
    for node in nodes:
        nid = node["id"]
        for link in node["links"]:
            tid = stem_to_id.get(link.lower())
            if not tid:
                ll = link.lower().replace(" ", "-")
                for s, sid in stem_to_id.items():
                    if s.endswith(ll) or ll in s:
                        tid = sid
                        break
            if tid and tid != nid:
                add_edge(nid, tid, "wiki-link")
        for other in nodes:
            if other["id"] <= nid:
                continue
            shared = set(node["tags"]) & set(other["tags"])
            if len([t for t in shared if len(t) >= 3]) >= 1:
                add_edge(nid, other["id"], "tag-shared")
        for other in nodes:
            if other["id"] <= nid:
                continue
            if node["domain"] == other["domain"] and node["domain"] != "root":
                if str(Path(node["path"]).parent) == str(Path(other["path"]).parent):
                    add_edge(nid, other["id"], "same-folder")
    for node in nodes:
        lc = sum(1 for e in edges if e["source"] == node["id"] or e["target"] == node["id"])
        node["importance"] = max(1, lc + len(node["tags"]) + len(node["links"]))
    logger.info("Scanned %d notes, %d edges", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges, "total": len(nodes)}


def _build_notes_context(nodes, max_summary_len=150, include_content=False):
    ctx = []
    for n in nodes:
        item = {"title": n["title"], "domain": n["domain"],
                "tags": n["tags"], "summary": n["summary"][:max_summary_len]}
        if include_content:
            item["headings"] = n.get("headings", [])[:5]
        ctx.append(item)
    return ctx


# ======================== API Endpoints ========================

@app.get("/api/vault/scan")
async def api_scan():
    try:
        invalidate_cache()
        return JSONResponse({"success": True, "data": get_cached_scan()})
    except Exception as e:
        logger.exception("Scan failed")
        raise HTTPException(status_code=500, detail="Vault scan failed")


@app.get("/api/file/content")
async def api_file_content(path: str = Query(..., description="Relative path within vault")):
    if not VAULT_PATH:
        raise HTTPException(status_code=503, detail="Vault not configured")
    fp = (VAULT_PATH / path).resolve()
    try:
        fp.relative_to(VAULT_PATH.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not fp.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(fp, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        logger.exception("Failed to read %s", path)
        raise HTTPException(status_code=500, detail="Read failed")
    try:
        post = frontmatter.loads(raw)
        body, fm = post.content, post.metadata
    except Exception:
        body, fm = raw, {}
    return {"success": True, "path": path, "content": body, "raw": raw,
            "frontmatter": fm, "title": fm.get("title", Path(path).stem)}


@app.get("/api/search")
async def api_search(q: str = Query(""), tags: str = Query("")):
    data = get_cached_scan()
    nodes = data["nodes"]
    ft = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    kw = q.strip().lower()
    results = []
    for n in nodes:
        if ft:
            nl = [t.lower() for t in n["tags"]]
            if not any(f.lower() in nl for f in ft):
                continue
        if kw:
            if kw not in f"{n['title']} {' '.join(n['tags'])} {n['summary']} {n['domain']}".lower():
                continue
        results.append(n)
    return {"success": True, "results": results, "total": len(results)}


@app.get("/api/stats")
async def api_stats():
    data = get_cached_scan()
    nodes, edges = data["nodes"], data["edges"]
    if not nodes:
        return {"success": True, "total": 0, "edges": 0, "domains": [], "top_tags": [], "recent": []}
    dc = defaultdict(int)
    for n in nodes:
        dc[n["domain"]] += 1
    domains = sorted(dc.items(), key=lambda x: x[1], reverse=True)
    tc = defaultdict(int)
    for n in nodes:
        for t in n["tags"]:
            if len(t) >= 2:
                tc[t] += 1
    top_tags = sorted(tc.items(), key=lambda x: x[1], reverse=True)[:15]
    recent = sorted(nodes, key=lambda n: n["mtime_ts"], reverse=True)[:8]
    return {"success": True, "total": len(nodes), "edges": len(edges),
            "domains": [{"name": d, "count": c} for d, c in domains],
            "top_tags": [{"name": t, "count": c} for t, c in top_tags],
            "recent": [{"title": n["title"], "domain": n["domain"],
                         "mtime": n["mtime"], "tags": n["tags"][:3]} for n in recent]}


@app.post("/api/analysis/cluster")
async def api_cluster():
    _check_key()
    data = get_cached_scan()
    nodes = data["nodes"]
    if not nodes:
        return {"success": True, "clusters": [], "insights": None}
    ctx = _build_notes_context(nodes, include_content=True)
    prompt = (
        "You are a knowledge management expert. Analyze the following notes and provide TWO outputs:\n\n"
        "--- PART 1: Thematic Clusters ---\n"
        "Group notes into 3-5 thematic clusters with cluster_name, description, notes list.\n\n"
        "--- PART 2: Knowledge Health Check ---\n"
        "Identify orphans (up to 3 isolated notes), hubs (up to 3 central notes), "
        "blindspots (up to 2 under-explored topics), bridges (up to 2 cross-domain hidden connections).\n\n"
        "Notes:\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n\n"
        "Output strictly as JSON:\n"
        '{"clusters":[{"cluster_name":"...","description":"...","notes":["title1"]}],'
        '"insights":{"orphans":[{"title":"...","reason":"..."}],'
        '"hubs":[{"title":"...","connections":3,"reason":"..."}],'
        '"blindspots":[{"topic":"...","evidence":"..."}],'
        '"bridges":[{"note_a":"title1","note_b":"title2","reason":"..."}]}}'
    )
    try:
        result = json.loads(await _call_deepseek(prompt, max_tokens=3072))
        tid = {n["title"].lower(): n["id"] for n in nodes}
        for c in result.get("clusters", []):
            c["node_ids"] = []
            for nt in c.get("notes", []):
                nid = tid.get(nt.lower())
                if nid:
                    c["node_ids"].append(nid)
                else:
                    for n in nodes:
                        if nt.lower() in n["title"].lower() or n["title"].lower() in nt.lower():
                            c["node_ids"].append(n["id"])
                            break
        return {"success": True, "clusters": result.get("clusters", []), "insights": result.get("insights")}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Cluster failed")
        raise HTTPException(status_code=500, detail="Analysis failed")


@app.post("/api/analysis/timeline")
async def api_timeline():
    _check_key()
    data = get_cached_scan()
    nodes = data["nodes"]
    if not nodes:
        return {"success": True, "quarters": []}
    quarters = defaultdict(lambda: {"count": 0, "titles": [], "tags": [], "domains": defaultdict(int)})
    for n in nodes:
        ts = n.get("mtime_ts", 0)
        if ts == 0:
            continue
        dt = datetime.fromtimestamp(ts)
        q = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        quarters[q]["count"] += 1
        quarters[q]["titles"].append(n["title"])
        quarters[q]["tags"].extend(n["tags"])
        quarters[q]["domains"][n["domain"]] += 1
    qlist = []
    for q in sorted(quarters.keys()):
        dq = quarters[q]
        tcs = defaultdict(int)
        for t in dq["tags"]:
            tcs[t] += 1
        qlist.append({"quarter": q, "count": dq["count"],
                       "top_tags": sorted(tcs, key=tcs.get, reverse=True)[:5],
                       "top_domains": sorted(dq["domains"], key=dq["domains"].get, reverse=True)[:3],
                       "sample_titles": dq["titles"][:8]})
    if len(qlist) <= 1:
        return {"success": True, "quarters": qlist, "ai_labels": []}
    qctx = [{"quarter": q["quarter"], "count": q["count"],
             "top_tags": q["top_tags"], "samples": q["sample_titles"][:5]} for q in qlist]
    prompt = (
        "Given note-writing activity by quarter, give each quarter a 3-6 word theme label "
        "and a 1-sentence evolution story.\nQuarters:\n" +
        json.dumps(qctx, ensure_ascii=False, indent=2) +
        "\n\nOutput JSON: {\"labels\":[{\"quarter\":\"2024-Q1\",\"theme\":\"...\"}],\"evolution\":\"...\"}"
    )
    try:
        ar = json.loads(await _call_deepseek(prompt, max_tokens=1024))
        return {"success": True, "quarters": qlist,
                "ai_labels": ar.get("labels", []), "evolution": ar.get("evolution", "")}
    except (httpx.TimeoutException, json.JSONDecodeError):
        logger.warning("Timeline AI labeling failed")
        return {"success": True, "quarters": qlist, "ai_labels": [], "evolution": ""}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Timeline failed")
        raise HTTPException(status_code=500, detail="Timeline analysis failed")


@app.post("/api/analysis/hidden-links")
async def api_hidden_links():
    _check_key()
    data = get_cached_scan()
    nodes = data["nodes"]
    if len(nodes) < 3:
        return {"success": True, "links": []}
    sn = sorted(nodes, key=lambda n: n["importance"], reverse=True)[:80]
    ctx = _build_notes_context(sn, max_summary_len=120)
    prompt = (
        "Find up to 8 pairs of notes that discuss related concepts but are probably NOT linked. "
        "For each: note_a, note_b, connection (one sentence), strength ('strong'|'weak').\n"
        "Notes:\n" + json.dumps(ctx, ensure_ascii=False, indent=2) +
        "\n\nOutput JSON: {\"links\":[{\"note_a\":\"...\",\"note_b\":\"...\",\"connection\":\"...\",\"strength\":\"strong\"}]}"
    )
    try:
        return {"success": True, "links": json.loads(await _call_deepseek(prompt, max_tokens=2048)).get("links", [])}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hidden links failed")
        raise HTTPException(status_code=500, detail="Hidden link analysis failed")


@app.post("/api/analysis/ask")
async def api_ask(req: AskRequest):
    _check_key()
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    data = get_cached_scan()
    nodes = data["nodes"]
    if not nodes:
        return {"success": True, "answer": "Vault is empty.", "sources": []}
    kws = re.findall(r"[a-zA-Z一-鿿]+", q.lower())
    scored = []
    for n in nodes:
        s = 0
        for kw in kws:
            if kw in n["title"].lower(): s += 10
            if kw in " ".join(n["tags"]).lower(): s += 5
            if kw in n["summary"].lower(): s += 2
        if s > 0:
            scored.append((s, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]
    if not top:
        return {"success": True, "answer": f"No notes found for '{q}'.", "sources": []}
    sources = []
    for score, n in top:
        try:
            with open(VAULT_PATH / n["path"], "r", encoding="utf-8") as f:
                raw = f.read()
            try:
                body = frontmatter.loads(raw).content
            except Exception:
                body = raw
            if len(body) > 1500:
                body = body[:1500] + "\n...(truncated)"
            sources.append({"title": n["title"], "path": n["path"],
                            "tags": n["tags"][:5], "relevance": score, "content": body})
        except Exception:
            sources.append({"title": n["title"], "path": n["path"],
                            "tags": n["tags"][:5], "relevance": score, "content": n["summary"]})
    cbs = [f"[DOC {i+1}] Title: {s['title']}\nContent:\n{s['content']}" for i, s in enumerate(sources)]
    rag = ("Answer based ONLY on provided documents. Cite titles. Max 3 paragraphs.\n\n" +
           "\n\n".join(cbs) + f"\n\nQuestion: {q}")
    try:
        client = await _get_client()
        resp = await client.post(DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": [
                {"role": "system", "content": "Answer based on documents."},
                {"role": "user", "content": rag}],
                "temperature": 0.3, "max_tokens": 1024})
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"DeepSeek API error (HTTP {resp.status_code})")
        result = resp.json()
        choices = result.get("choices", [])
        answer = choices[0].get("message", {}).get("content", "") if choices else ""
        return {"success": True, "answer": answer,
                "sources": [{"title": s["title"], "path": s["path"], "relevance": s["relevance"]} for s in sources]}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Q&A failed")
        raise HTTPException(status_code=500, detail="Q&A failed")


# ======================== Startup & Static ========================

@app.on_event("startup")
async def startup():
    print(f"\n{'='*50}")
    print(f"  Obsidian Knowledge Graph v0.3.0")
    print(f"  Vault: {VAULT_PATH}")
    print(f"  Server: http://{cfg['server_host']}:{cfg['server_port']}")
    print(f"  DeepSeek: {'Configured' if AI_ENABLED else 'Not configured (AI disabled)'}")
    print(f"  Model: {DEEPSEEK_MODEL}")
    print(f"{'='*50}\n")


@app.on_event("shutdown")
async def shutdown():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    if not VAULT_PATH or not VAULT_PATH.exists():
        print("\n  vault_path not configured.")
        print("  Set OBSIDIAN_VAULT_PATH env var or edit config.json\n")
        sys.exit(1)
    uvicorn.run("server:app", host=cfg["server_host"], port=cfg["server_port"], reload=True)
