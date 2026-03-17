## TODO (For Sprint) —— Tech Stack Pivot & Refinement

### 0. Docs Initialization

- [x] Initialize the README_zh.md and the README.md

### 1. Environment Changes (include the .env.example changes)

- [x] Clean all deprecated settings
- [x] Change the embedding model platform (huggingface -> siliconflow)
- [x] Change the search engine api (tavily -> duckduckgo)
- [ ] Done the new settings tests

### 2. Inspect and Refactor some modules

- [x] Ensure indexer.py correctly delegates `bge-m3` embedding to SiliconFlow instead of locally downloading models.

### 3. Integration tests and ci test

- [x] Rewrite tests

### 4. Complete the version 0.1

- [ ] Tag and Release MVP `v0.1` adhering to standard GitFlow.

### 5. Organize the files then to Version 0.2

- [ ] Organize the docs and integrate them to README.md
