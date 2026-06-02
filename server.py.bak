"""
Obsidian Knowledge Graph Platform - Server v0.2.0
FastAPI backend: vault parser + graph builder + DeepSeek AI multi-analysis
"""
import json
import os
import re
import time
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

CONFIG_PATH = Path(__file__).parent / "config.json"

def load_config():
    """Load config.json; if missing, copy from config.example.json and exit with instructions."""
    if not CONFIG_PATH.exists():
        example = CONFIG_PATH.parent / "config.example.json"
        if example.exists():
            import shutil
            shutil.copy(example, CONFIG_PATH)
            print(f"\n  ⚠  config.json 已从模板创建: {CONFIG_PATH}")
            print(f"  ⚠  请编辑该文件，填入你的 Obsidian Vault 路径和 DeepSeek API Key")
            print(f"  ⚠  获取 Key: https://platform.deepseek.com/api_keys\n")
        else:
            print(f"\n  ⚠  未找到 config.json，请复制 config.example.json 并重命名为 config.json")
        import sys
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 检查是否仍为占位符
    if cfg.get("deepseek_api_key") in ("", "sk-your-deepseek-api-key"):
        print(f"\n  ⚠  config.json 中的 API Key 仍为占位符，请编辑填入真实 Key")
        print(f"  ⚠  获取 Key: https://platform.deepseek.com/api_keys\n")
        import sys
        sys.exit(1)

    if cfg.get("vault_path") in ("", "/path/to/your/Obsidian/Vault"):
        print(f"\n  ⚠  config.json 中的 vault_path 仍为占位符，请填入实际路径\n")
        import sys
        sys.exit(1)

    return cfg

cfg = load_config()
VAULT_PATH = Path(cfg["vault_path"])
DEEPSEEK_KEY = cfg.get("deepseek_api_key", "")
DEEPSEEK_MODEL = cfg.get("deepseek_model", "deepseek-chat")
DEEPSEEK_URL = cfg.get("deepseek_base_url", "https://api.deepseek.com/chat/completions")

app = FastAPI(title="Obsidian Knowledge Graph", version="0.2.0")


# ═══════════════════════════════════════════════
#  Shared Helpers
# ═══════════════════════════════════════════════

def _check_key():
    if not DEEPSEEK_KEY or DEEPSEEK_KEY == "your-deepseek-api-key-here":
        raise HTTPException(status_code=400, detail="Please configure DeepSeek API Key in config.json")


async def _call_deepseek(prompt: str, max_tokens: int = 2048) -> str:
    """Call DeepSeek API and return cleaned text response."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a knowledge management expert. Always output valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            },
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"DeepSeek API error (HTTP {resp.status_code}): {resp.text[:500]}")

    result = resp.json()
    ai_text = result["choices"][0]["message"]["content"]

    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", ai_text)
    if json_match:
        ai_text = json_match.group(1)
    return ai_text


class AskRequest(BaseModel):
    question: str


# ═══════════════════════════════════════════════
#  Vault Parsing
# ═══════════════════════════════════════════════

def extract_wiki_links(content: str) -> list:
    pattern = r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]"
    matches = re.findall(pattern, content)
    return [m.strip() for m in matches if m.strip()]


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
    pattern = r"(?<!\w)#([a-zA-Z一-鿿][a-zA-Z0-9一-鿿_/-]*)"
    matches = re.findall(pattern, clean)
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


def scan_vault() -> dict:
    nodes = []
    node_index = {}
    node_id_counter = [0]

    def get_node_id():
        node_id_counter[0] += 1
        return f"n{node_id_counter[0]}"

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
        except Exception:
            continue

        try:
            post = frontmatter.loads(raw)
            fm = post.metadata
            body = post.content
        except Exception:
            fm = {}
            body = raw

        title = fm.get("title", "")
        if not title:
            h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
        if not title:
            title = md_file.stem

        fm_tags = extract_tags_from_frontmatter(fm.get("tags", []))
        inline_tags = extract_inline_tags(body)
        all_tags = list(dict.fromkeys(fm_tags + inline_tags))

        wiki_links = extract_wiki_links(body)

        headings = []
        for m in re.finditer(r"^(#{1,2})\s+(.+)$", body, re.MULTILINE):
            level = len(m.group(1))
            headings.append({"level": level, "text": m.group(2).strip()})

        summary = extract_summary(body)
        domain = parts[0] if len(parts) > 1 else "root"
        mtime = md_file.stat().st_mtime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        node_id = get_node_id()
        node = {
            "id": node_id,
            "title": title,
            "path": str(rel_path).replace("\\", "/"),
            "tags": all_tags,
            "links": wiki_links,
            "headings": headings[:10],
            "domain": domain,
            "mtime": mtime_str,
            "mtime_ts": mtime,
            "summary": summary,
        }
        nodes.append(node)
        node_index[str(rel_path).replace("\\", "/")] = node_id

    edges = []
    edge_set = set()

    def add_edge(src, tgt, etype):
        key = f"{src}-{tgt}-{etype}"
        if key not in edge_set and src != tgt:
            edge_set.add(key)
            edges.append({"source": src, "target": tgt, "type": etype})

    stem_to_id = {}
    for node in nodes:
        stem = Path(node["path"]).stem
        stem_to_id[stem.lower()] = node["id"]
        stem_to_id[node["path"].replace(".md", "").lower()] = node["id"]

    for node in nodes:
        nid = node["id"]

        for link in node["links"]:
            target_id = stem_to_id.get(link.lower())
            if not target_id:
                link_lower = link.lower().replace(" ", "-")
                for stem, sid in stem_to_id.items():
                    if stem.endswith(link_lower) or link_lower in stem:
                        target_id = sid
                        break
            if target_id and target_id != nid:
                add_edge(nid, target_id, "wiki-link")

        for other in nodes:
            if other["id"] <= nid:
                continue
            shared = set(node["tags"]) & set(other["tags"])
            meaningful_shared = [t for t in shared if len(t) >= 3]
            if len(meaningful_shared) >= 1:
                add_edge(nid, other["id"], "tag-shared")

        for other in nodes:
            if other["id"] <= nid:
                continue
            if node["domain"] == other["domain"] and node["domain"] != "root":
                n_parent = str(Path(node["path"]).parent)
                o_parent = str(Path(other["path"]).parent)
                if n_parent == o_parent:
                    add_edge(nid, other["id"], "same-folder")

    for node in nodes:
        link_count = sum(1 for e in edges if e["source"] == node["id"] or e["target"] == node["id"])
        node["importance"] = max(1, link_count + len(node["tags"]) + len(node["links"]))

    return {"nodes": nodes, "edges": edges, "total": len(nodes)}


def _build_notes_context(nodes: list, max_summary_len: int = 150) -> list:
    """Build lightweight context list for AI prompts."""
    return [{
        "title": n["title"],
        "domain": n["domain"],
        "tags": n["tags"],
        "summary": n["summary"][:max_summary_len],
    } for n in nodes]


# ═══════════════════════════════════════════════
#  API Endpoints — Core
# ═══════════════════════════════════════════════

@app.get("/api/vault/scan")
async def api_scan():
    try:
        data = scan_vault()
        return JSONResponse({"success": True, "data": data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")


@app.get("/api/file/content")
async def api_file_content(path: str = Query(..., description="Relative path")):
    file_path = VAULT_PATH / path
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(VAULT_PATH.resolve())):
            raise HTTPException(status_code=403, detail="Path traversal denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"success": True, "path": path, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Read failed: {str(e)}")


@app.get("/api/search")
async def api_search(
    q: str = Query("", description="Search keyword"),
    tags: str = Query("", description="Comma-separated tag filter"),
):
    data = scan_vault()
    nodes = data["nodes"]
    filter_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    keyword = q.strip().lower()

    results = []
    for n in nodes:
        if filter_tags:
            node_tags_lower = [t.lower() for t in n["tags"]]
            if not any(ft.lower() in node_tags_lower for ft in filter_tags):
                continue
        if keyword:
            searchable = f"{n['title']} {' '.join(n['tags'])} {n['summary']} {n['domain']}".lower()
            if keyword not in searchable:
                continue
        results.append(n)

    return {"success": True, "results": results, "total": len(results)}


# ═══════════════════════════════════════════════
#  P0: 知识聚类 + 知识体检
# ═══════════════════════════════════════════════

@app.post("/api/analysis/cluster")
async def api_cluster():
    _check_key()

    data = scan_vault()
    nodes = data["nodes"]
    if not nodes:
        return {"success": True, "clusters": [], "insights": None, "message": "No notes in vault"}

    notes_context = _build_notes_context(nodes)

    prompt = (
        "You are a knowledge management expert. Analyze the following notes and provide TWO outputs:\n\n"
        "--- PART 1: Thematic Clusters ---\n"
        "Group notes into 3-5 thematic clusters.\n"
        "For each cluster provide:\n"
        "  - cluster_name: A short, meaningful name\n"
        "  - description: One sentence describing this cluster\n"
        "  - notes: List of note titles belonging to this cluster\n\n"
        "--- PART 2: Knowledge Health Check ---\n"
        "Analyze the knowledge base and identify:\n"
        "  - orphans: Up to 3 notes that have no apparent connection to others (knowledge islands)\n"
        "  - hubs: Up to 3 notes that seem to be central hubs connecting many concepts\n"
        "  - blindspots: Up to 2 topics that appear under-explored — mentioned briefly or implied but lacking depth\n"
        "  - bridges: Up to 2 pairs of notes from different domains that discuss related concepts but aren't linked\n\n"
        "Notes:\n" + json.dumps(notes_context, ensure_ascii=False, indent=2) + "\n\n"
        "Output strictly as JSON (no markdown outside the JSON block):\n"
        '{\n'
        '  "clusters": [\n'
        '    {"cluster_name": "...", "description": "...", "notes": ["title1"]}\n'
        '  ],\n'
        '  "insights": {\n'
        '    "orphans": [{"title": "...", "reason": "..."}],\n'
        '    "hubs": [{"title": "...", "connections": 3, "reason": "..."}],\n'
        '    "blindspots": [{"topic": "...", "evidence": "..."}],\n'
        '    "bridges": [{"note_a": "title1", "note_b": "title2", "reason": "..."}]\n'
        '  }\n'
        '}'
    )

    try:
        ai_text = await _call_deepseek(prompt, max_tokens=3072)
        result = json.loads(ai_text)

        title_to_id = {n["title"].lower(): n["id"] for n in nodes}

        for cluster in result.get("clusters", []):
            cluster["node_ids"] = []
            for note_title in cluster.get("notes", []):
                nid = title_to_id.get(note_title.lower())
                if nid:
                    cluster["node_ids"].append(nid)
                else:
                    for n in nodes:
                        if note_title.lower() in n["title"].lower() or n["title"].lower() in note_title.lower():
                            cluster["node_ids"].append(n["id"])
                            break

        return {"success": True, "clusters": result.get("clusters", []), "insights": result.get("insights")}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ═══════════════════════════════════════════════
#  P2: 知识时间线
# ═══════════════════════════════════════════════

@app.post("/api/analysis/timeline")
async def api_timeline():
    _check_key()

    data = scan_vault()
    nodes = data["nodes"]
    if not nodes:
        return {"success": True, "quarters": [], "message": "No notes in vault"}

    # Group by quarter
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

    quarter_list = []
    for q in sorted(quarters.keys()):
        dq = quarters[q]
        tag_counts = defaultdict(int)
        for t in dq["tags"]:
            tag_counts[t] += 1
        top_tags = sorted(tag_counts, key=tag_counts.get, reverse=True)[:5]
        top_domains = sorted(dq["domains"], key=dq["domains"].get, reverse=True)[:3]
        quarter_list.append({
            "quarter": q,
            "count": dq["count"],
            "top_tags": top_tags,
            "top_domains": top_domains,
            "sample_titles": dq["titles"][:8],
        })

    # Ask AI to label each quarter
    if len(quarter_list) <= 1:
        return {"success": True, "quarters": quarter_list, "ai_labels": []}

    q_context = [{
        "quarter": q["quarter"],
        "count": q["count"],
        "top_tags": q["top_tags"],
        "samples": q["sample_titles"][:5],
    } for q in quarter_list]

    prompt = (
        "Given a user's note-writing activity by quarter, give each quarter a 3-6 word theme label "
        "that captures the main focus. Also provide a 1-sentence 'evolution story' describing how "
        "the user's interests shifted over time.\n\n"
        "Quarters:\n" + json.dumps(q_context, ensure_ascii=False, indent=2) + "\n\n"
        "Output strictly as JSON:\n"
        '{"labels": [{"quarter": "2024-Q1", "theme": "..."}], "evolution": "One sentence story of interest evolution"}'
    )

    try:
        ai_text = await _call_deepseek(prompt, max_tokens=1024)
        ai_result = json.loads(ai_text)
        return {
            "success": True,
            "quarters": quarter_list,
            "ai_labels": ai_result.get("labels", []),
            "evolution": ai_result.get("evolution", ""),
        }
    except (httpx.TimeoutException, json.JSONDecodeError, HTTPException):
        # If AI fails, still return quarters without labels
        return {"success": True, "quarters": quarter_list, "ai_labels": [], "evolution": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline analysis failed: {str(e)}")


# ═══════════════════════════════════════════════
#  P1: 隐含关联发现
# ═══════════════════════════════════════════════

@app.post("/api/analysis/hidden-links")
async def api_hidden_links():
    _check_key()

    data = scan_vault()
    nodes = data["nodes"]
    if len(nodes) < 3:
        return {"success": True, "links": [], "message": "Need at least 3 notes for hidden link analysis"}

    # Sort by importance, take top 80 to keep prompt manageable
    sorted_nodes = sorted(nodes, key=lambda n: n["importance"], reverse=True)[:80]
    notes_context = _build_notes_context(sorted_nodes, max_summary_len=120)

    prompt = (
        "You are a knowledge graph expert. Below are notes from a personal knowledge base. "
        "Find up to 8 pairs of notes that appear to discuss related or complementary concepts "
        "but are probably NOT explicitly linked to each other.\n\n"
        "For each pair provide:\n"
        "  - note_a: title of first note\n"
        "  - note_b: title of second note\n"
        "  - connection: one short sentence describing the hidden connection\n"
        "  - strength: 'strong' or 'weak'\n\n"
        "Notes:\n" + json.dumps(notes_context, ensure_ascii=False, indent=2) + "\n\n"
        "Output strictly as JSON:\n"
        '{"links": [{"note_a": "title1", "note_b": "title2", "connection": "...", "strength": "strong"}]}'
    )

    try:
        ai_text = await _call_deepseek(prompt, max_tokens=2048)
        result = json.loads(ai_text)
        return {"success": True, "links": result.get("links", [])}
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hidden link analysis failed: {str(e)}")


# ═══════════════════════════════════════════════
#  P3: 知识库智能问答 (RAG-light)
# ═══════════════════════════════════════════════

@app.post("/api/analysis/ask")
async def api_ask(req: AskRequest):
    _check_key()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    data = scan_vault()
    nodes = data["nodes"]
    if not nodes:
        return {"success": True, "answer": "Your vault is empty. Add some notes first!", "sources": []}

    # Simple keyword relevance scoring
    keywords = re.findall(r"[a-zA-Z一-鿿]+", question.lower())
    scored = []
    for n in nodes:
        searchable = f"{n['title']} {' '.join(n['tags'])} {n['summary']}".lower()
        score = 0
        for kw in keywords:
            if kw in n["title"].lower():
                score += 10
            if kw in " ".join(n["tags"]).lower():
                score += 5
            if kw in n["summary"].lower():
                score += 2
        if score > 0:
            scored.append((score, n))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_notes = scored[:5]

    if not top_notes:
        return {
            "success": True,
            "answer": f"No notes found related to your question about \"{question}\". Try different keywords.",
            "sources": [],
        }

    # Read full content of top notes
    sources = []
    for score, n in top_notes:
        file_path = VAULT_PATH / n["path"]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Strip frontmatter for cleaner context
            try:
                post = frontmatter.loads(raw)
                body = post.content
            except Exception:
                body = raw
            # Limit content length per note
            if len(body) > 1500:
                body = body[:1500] + "\n...(truncated)"
            sources.append({
                "title": n["title"],
                "path": n["path"],
                "tags": n["tags"][:5],
                "relevance": score,
                "content": body,
            })
        except Exception:
            sources.append({
                "title": n["title"],
                "path": n["path"],
                "tags": n["tags"][:5],
                "relevance": score,
                "content": n["summary"],
            })

    # Build RAG prompt
    context_blocks = []
    for i, s in enumerate(sources):
        context_blocks.append(f"[DOC {i + 1}] Title: {s['title']}\nContent:\n{s['content']}")

    rag_prompt = (
        "You are a helpful knowledge assistant. Answer the user's question based ONLY on "
        "the provided documents from their personal notes. If the documents don't contain "
        "enough information, say so honestly. Cite document titles when referencing them.\n\n"
        "Documents:\n\n" + "\n\n".join(context_blocks) + "\n\n"
        f"Question: {question}\n\n"
        "Answer (be concise but thorough, max 3 paragraphs). Also note if any important "
        "info seems to be missing from the provided documents."
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                DEEPSEEK_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful knowledge assistant. Answer based on provided documents. Be concise."},
                        {"role": "user", "content": rag_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"DeepSeek API error (HTTP {resp.status_code}): {resp.text[:500]}")

        result = resp.json()
        answer = result["choices"][0]["message"]["content"]

        return {
            "success": True,
            "answer": answer,
            "sources": [{"title": s["title"], "path": s["path"], "relevance": s["relevance"]} for s in sources],
        }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Q&A failed: {str(e)}")


# ═══════════════════════════════════════════════
#  Startup & Static Files
# ═══════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print(f"\n{'='*50}")
    print(f"  Obsidian Knowledge Graph v0.2.0")
    print(f"  Vault: {VAULT_PATH}")
    print(f"  Server: http://{cfg['server_host']}:{cfg['server_port']}")
    key_status = "Configured" if DEEPSEEK_KEY and DEEPSEEK_KEY != "your-deepseek-api-key-here" else "Not configured"
    key_preview = f"{DEEPSEEK_KEY[:10]}..." if DEEPSEEK_KEY and DEEPSEEK_KEY != "your-deepseek-api-key-here" else "(empty)"
    print(f"  DeepSeek Key:  {key_status} ({key_preview})")
    print(f"  DeepSeek URL:  {DEEPSEEK_URL}")
    print(f"  DeepSeek Model: {DEEPSEEK_MODEL}")
    print(f"  PID: {os.getpid()}")
    print(f"{'='*50}\n")


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host=cfg["server_host"], port=cfg["server_port"], reload=True)
