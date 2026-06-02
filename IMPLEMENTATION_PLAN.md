# Implementation Plan: Obsidian Knowledge Graph Platform (MVP)

> **版本**：v1.0 | **日期**：2026-06-02 | **状态**：开发中

## Context

Build a local web application that reads an Obsidian vault and visualizes it as an interactive knowledge graph, with AI-powered analysis via DeepSeek API. Lightweight, convenient tech — no heavy frameworks, no build steps.

- **Vault**: `~/Documents/Obsidian Vault/` — 11 user notes (35 total including templates)
- **User**: Tech professional focused on AI agents, automation
- **Goal**: MVP with vault parsing + knowledge graph + AI clustering

---

## Architecture

```
┌──────────────────────────────────────────┐
│  Browser (index.html)                    │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Sidebar │ │ D3 Graph │ │ AI Panel  │ │
│  │ Search  │ │ (Canvas) │ │ (Modal)   │ │
│  │ Filters │ │          │ │           │ │
│  └─────────┘ └──────────┘ └───────────┘ │
├──────────────────────────────────────────┤
│  FastAPI Server (server.py)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Vault    │ │ Graph    │ │ DeepSeek │ │
│  │ Parser   │ │ Builder  │ │ Client   │ │
│  └──────────┘ └──────────┘ └──────────┘ │
├──────────────────────────────────────────┤
│  Obsidian Vault (filesystem)             │
└──────────────────────────────────────────┘
```

---

## Project Structure

```
obsidian-knowledge-graph/
├── server.py              # FastAPI backend
├── static/
│   └── index.html         # SPA frontend
├── requirements.txt       # Python dependencies
└── config.json            # vault_path, deepseek_api_key
```

---

## Implementation Steps

### Step 1: Project scaffold + config
- Create project directory
- Write `requirements.txt`
- Write `config.json`
- Write minimal `server.py`

### Step 2: Vault parser module
- Parse `.md` files with `python-frontmatter`
- Extract: title, tags, wiki-links, headings, mtime, summary
- Edge types: wiki-link, tag-shared, same-folder

### Step 3: API endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vault/scan` | GET | Full vault scan → graph JSON |
| `/api/file/content?path=` | GET | Single file content |
| `/api/analysis/cluster` | POST | AI clustering via DeepSeek |
| `/api/search?q=&tags=` | GET | Search notes |

### Step 4-8: Frontend
- Pure HTML5 + D3.js CDN, dark theme
- Canvas force-directed graph with drag/zoom/hover/click
- Sidebar: search, tag filter, AI button
- Modal for AI cluster results

---

## Verification

1. `pip install -r requirements.txt`
2. Edit `config.json` with DeepSeek API key
3. `python server.py` → http://localhost:8765
4. Scan vault, interact with graph, run AI analysis
