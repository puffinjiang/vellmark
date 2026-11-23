# Vellmark

**Vellmark** 是一款用 Rust 编写的跨平台、纯原生 GPU 渲染的所见即所得（WYSIWYG）Markdown 编辑器。

> 让 Markdown 写作"像打字一样轻"——快速、专注、所见即所得、永不卡顿。

## 核心特性（规划与实施中）

- **单栏实时渲染 + 源码标记按需展开**：仅展开当前块级前缀与光标所在的行内元素标记，消除双栏割裂感。
- **源码为唯一真相（Round-trip 保真）**：内部基于 Rope（`ropey`）存储源文本，未修改保存 **100% 字节级一致**，局部编辑仅改动被操作块，保证 Git diff 干净。
- **纯原生高性能**：冷启动 `< 300 ms`，空闲内存 `< 80 MB`，10 MB 文档按键延迟 `P99 < 16 ms`。
- **完整的数据安全与冲突状态机**：完备的 `Clean / Dirty / Conflict / Missing` 状态机、写入前基准校验、版本绑定串行写入与双槽原子崩溃恢复快照（RPO ≤ 3 秒）。
- **CSS 子集主题系统**：支持标准 `.css` 语法子集与热重载，编辑视图与 HTML / PDF 导出共享同一套主题样式。

详细产品需求与交互规范请参见：**[docs/PRD.md](docs/PRD.md)**。

## 仓库结构（Cargo Workspace）

```text
vellmark/
├── crates/
│   ├── vellmark-core/      # 文本缓冲(Rope)、文档模型、撤销栈、编码检测、状态机与崩溃恢复
│   ├── vellmark-parser/    # CommonMark 0.31.2 / GFM 增量解析与字节偏移映射
│   ├── vellmark-render/    # 原生文本排版、高亮、公式、图表与块级布局
│   ├── vellmark-theme/     # CSS 子集解析、选择器匹配、样式级联
│   ├── vellmark-export/    # HTML（原样 / 安全模式）、PDF、DOCX 导出
│   ├── vellmark-app/       # 跨平台桌面 GUI 应用入口
│   └── vellmark-cli/       # 命令行工具（打开文件、无界面导出）
├── themes/                 # 内置 CSS 主题
└── docs/                   # 产品需求文档（PRD）与架构设计文档
```

## 构建与测试

```bash
# 编译整个 Workspace
cargo build --workspace

# 运行全部单元测试与规范测试
cargo test --workspace

# 运行 CLI
cargo run -p vellmark-cli -- --help
```

## 提交前检查

首次克隆后，在仓库根目录启用纳入版本控制的 Git 钩子。

```bash
rustup component add rustfmt clippy
git config --local core.hooksPath .githooks
```

每次提交前，当暂存区包含 Rust、Cargo 配置或内置主题 CSS 文件时，钩子会检查当前工作树的 Rust 格式、运行 Clippy（`--locked`）和工作区测试；纯文档或配置提交会自动跳过 Rust 检查。任一检查失败都会中止提交：格式检查失败时请运行 `cargo fmt --all` 并重新暂存，修改 `Cargo.toml` 依赖后请先运行 `cargo check` 更新并暂存 `Cargo.lock`。

## 开源协议

本项目基于 [GPL-3.0](LICENSE) 协议开源。
