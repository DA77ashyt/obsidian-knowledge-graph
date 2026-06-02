# 🧠 Obsidian Knowledge Graph

> 将你的 Obsidian Vault 可视化为交互式知识图谱，并借助 DeepSeek AI 进行多维知识分析。

![](https://img.shields.io/badge/python-3.10+-blue)
![](https://img.shields.io/badge/fastapi-0.100+-green)
![](https://img.shields.io/badge/license-MIT-yellow)

## 功能

### 🔍 知识图谱可视化
- 扫描 Obsidian Vault，自动解析 Markdown 笔记
- 基于 Wiki 链接 `[[...]]`、标签 `#tag` 和文件夹结构构建知识图谱
- D3.js 力导向图交互：拖拽、缩放、悬停预览、点击查看笔记详情
- 支持按标签过滤和关键词搜索

### 🤖 AI 多维分析（需 DeepSeek API Key）

| 分析模块 | 说明 |
|---------|------|
| **知识聚类** | AI 将笔记分为 3-5 个主题组，发现知识结构 |
| **知识体检** | 识别知识孤岛、枢纽节点、知识盲区、跨域桥接 |
| **时间线** | 按季度展示笔记产出趋势，AI 标注每季度主题和兴趣演化 |
| **隐含关联** | 发现内容相关但尚未链接的笔记对 |
| **智能问答** | 基于笔记内容回答你的问题（RAG） |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

项目首次启动时会自动从 `config.example.json` 创建 `config.json`，或你也可以手动复制：

```bash
cp config.example.json config.json
```

编辑 `config.json`，填入你的配置：

```json
{
  "vault_path": "/Users/you/Documents/Obsidian Vault",
  "deepseek_api_key": "sk-your-deepseek-api-key",
  "deepseek_model": "deepseek-chat",
  "deepseek_base_url": "https://api.deepseek.com/v1/chat/completions",
  "server_host": "127.0.0.1",
  "server_port": 8765
}
```

| 字段 | 说明 |
|------|------|
| `vault_path` | Obsidian Vault 的绝对路径 |
| `deepseek_api_key` | [DeepSeek API Key](https://platform.deepseek.com/api_keys)（AI 分析必需） |
| `deepseek_model` | 模型名称：`deepseek-chat`（V3）或 `deepseek-reasoner`（R1） |
| `deepseek_base_url` | API 端点，一般无需修改 |
| `server_host` / `server_port` | 本地服务地址，一般无需修改 |

> 💡 **不配置 API Key 也能使用知识图谱的可视化、搜索和标签过滤功能**，只是 AI 分析模块会提示配置。

### 3. 启动

```bash
python server.py
```

打开浏览器访问 `http://127.0.0.1:8765`，点击「扫描 Vault」加载笔记。

### 4. 使用

- **图谱交互**：拖拽节点、滚轮缩放、点击节点查看详情
- **标签过滤**：点击侧边栏标签筛选相关笔记
- **关键词搜索**：搜索笔记标题和内容
- **AI 分析**：点击左下角「AI 多维分析」，在弹窗中切换 Tab 查看不同分析维度

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python · FastAPI · uvicorn |
| 前端 | 原生 HTML/CSS/JS · D3.js v7 |
| AI | DeepSeek API（chat-completions） |
| 笔记解析 | python-frontmatter · 正则表达式 |

## 项目结构

```
obsidian-knowledge-graph/
├── server.py              # FastAPI 后端
├── static/
│   └── index.html         # 前端单页应用
├── config.example.json    # 配置模板（可提交）
├── config.json            # 实际配置（已在 .gitignore 中排除）
├── requirements.txt       # Python 依赖
├── .gitignore
└── README.md
```

## License

MIT
