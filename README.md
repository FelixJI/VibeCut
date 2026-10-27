# VibeCut

VibeCut 把单目 equirectangular（360° 全景）SDR MP4 剪成**固定视角的普通平面视频**。
本仓库是任务 #2 的最小完整静态流水线：探测 → 工程 → 固定视角导出 1920×1080 H.264/AAC/BT.709，
不依赖任何 SDK、模型、GPU、Qt 或网络服务，核心包只用 Python 标准库。

## 快速开始

```bash
# Python 3.12 + uv；ffmpeg/ffprobe 需在 PATH，或用环境变量指定：
#   VIBECUT_FFMPEG=C:\tools\ffmpeg\bin\ffmpeg.exe
#   VIBECUT_FFPROBE=C:\tools\ffmpeg\bin\ffprobe.exe
uv sync --frozen             # 安装锁定环境（运行时零第三方依赖，dev 组为 pytest）
uv run vibecut init ./proj   # 建工程（vibecut.project.json, project schema v1, revision 1）
uv run vibecut inspect 输入.mp4 --projection equirect   # 打印 MediaSource 合同
uv run vibecut add-source ./proj 输入.mp4 --projection equirect
uv run vibecut export ./proj 输入.mp4 成片.mp4 --projection equirect --yaw 90 --pitch 0 --hfov 90
```

出口码：`0` 成功；`2` 用法/不支持的源；`3` ffmpeg/ffprobe 缺失；`4` 渲染/验证失败；`5` 冲突（输出已存在/源被改动/工程锁）。

## 时间与几何合同（v1，已冻结）

- 所有源时间/输出时间用 `Fraction` 精确有理数；序列化为 `"num/den"`。
- **source-relative 时间 0 = 源首个显示帧**（`start_pts * time_base`）；音频与视频按同一**绝对容器时间线**裁切；调用 FFmpeg 时先减去容器起点，将绝对窗口换成 demuxer 的 seek 时间。
- 真 VFR（真实包 PTS 不均匀，或与声明帧率矛盾）在探测阶段 fail-fast 拒绝，绝不按 frame_index 猜测。
- canonical 全景语义：yaw 0 = equirect 中心列，**正 yaw 向右**，**正 pitch 向上**，正 roll = 相机顺时针滚转；**水平 FOV** 为合同 FOV，垂直 FOV 由画幅推导（`tan(v/2)=tan(h/2)·高/宽`）。FFmpeg `v360` 的符号约定与 canonical 一致（映射为恒等）。
- 手工锁定（用户显式给的 yaw/pitch/roll/FOV）永远优先于任何自动推导值。

## 版本化合同对象（`vibecut.core.contracts`，schema **v2**，随本 PR 冻结）

`MediaSource`（稳定 `source_id` = 内容指纹前缀、路径、投影、**稳定化 provenance（v2 新增，#2 仅允许 "none"）**、色彩 primaries/transfer/matrix/range/bit depth（仅接受显式 bt709/tv/8bit）、start PTS/time_base/fps 有理数、音频流、VFR 标记）、`NormalizedAsset`（规范资产/代理 + `TimeMap` 时间映射 + 稳定化 provenance）、`EditPlan`（source-relative 入出点 + **manual cut 锁定标志**（v2）、工程帧率、`CameraPath`）、`CameraPath`（v2 新增 **provenance="user"/"derived"**：手工锁定值永远优先于自动推导；静态导出仅接受 "user"）、`JobEvent`（job_id + project_revision）。

帧时序验证（v2 纠偏）：CFR 由视频流真实包 PTS 的均匀间隔验证；`r_frame_rate==avg_frame_rate` 不再作为充分条件（存在同均值真 VFR 夹具回归），PTS 非均匀或与 avg 帧率矛盾均拒绝；首个显示帧 PTS 取自实际包，不盲信 `stream.start_pts`。投影检测（v2 纠偏）仅信任结构化标记（流 side_data `projection` / `ProjectionType` 标签）或显式声明；自由文本/文件名永不参与，立体布局与矛盾标记直接拒绝。

### 内容指纹的生产者/消费者与失败处理（任务 #2 明确契约，无默认扩散）

- **生产者**：`media.probe.probe_source` 对源文件字节做 SHA-256，存入 `MediaSource.content_fingerprint`。
- **消费者**：① `core.cache` 以指纹入键 —— 同名不同内容立即失效缓存；② `render.static` 在每次渲染前后复核源哈希，变动即中止且不提交产物。
- **失败处理**：文件不可读 → `ProbeError`（不产出 MediaSource、不写缓存）；渲染中源变化 → `SourceChangedError`（删除临时产物，不发成功事件）。
- 不做任何下载/交付哈希。

## 持久化与作业

- 工程文件 `vibecut.project.json`：schema/revision，写临时文件 → fsync → 同目录 `os.replace` 原子替换；中断只留下 `*.tmp-*`，最后一版完整工程永远可读。
- 同工程跨进程独占锁（`vibecut.project.json.lock`，msvcrt/flock）；CLI 导入/代理和导出从探测开始一直持锁到最后保存，第二个作业立即拒绝。保存复用当前作业的锁，锁内复核 on-disk revision，迟到写入者被 `StaleRevisionError` 拒绝。
- 导出的 `EditPlan` 在渲染前存入工程，包含入出点、帧率和人工视角，重新 `Project.load` 后可直接再次渲染。素材内容在已导入路径发生变化会拒绝；MVP 不隐式迁移同内容的路径引用，需使用原路径或新建工程。
- `JobEvent` 带 job_id/revision；迟到事件不能覆盖新版本；`succeeded` 只在“产物探测验证通过 + 无覆盖提交”之后发布。
- 缓存（`.vibecut-cache/`）按**源内容 + 工具构建 + 流水线配置 + 色彩合同**为键，绝不绑定 edit revision；改剪辑版本不会清掉规范母版（proxy）。代理在冷/热缓存路径均核对源内容与取消；命中还检查媒体合同并完整解码。使用唯一 payload + 原子替换 manifest 同时发布元数据与 payload 引用，失败保留上一完整条目。崩溃可能留下不可命中的临时或未引用 payload；旧 payload 保留供已打开的读者使用，MVP 尚无缓存垃圾回收。

## 有界进程管理

`core.proc.run_bounded`：argv 直传（绝不经 shell）；总超时 + 可取消事件；stdout/stderr 并发排空到有界尾部缓冲；等待**真实进程退出**；后代持有管道不无限等待（有界 join，`pipes_drained=False` 如实上报）；Windows 用专用 Job Object（KILL_ON_JOB_CLOSE）只终止自家进程树，绝不按进程名杀进程。

## 支持范围（MVP 边界）

- 仅单目 equirect **显式标记 BT.709 / tv-range / 8bit SDR** 的 MP4/MOV，H.264/HEVC：未知/未标记色彩、非 bt709、全范围（pc/jpeg）、10bit、HDR 一律提前拒绝——绝不做未经真实转换的色彩重标；真 VFR、非 2:1、未知投影（未声明）、立体/双鱼眼标记、矛盾元数据、多视频流、不可剪音轨同样提前可识别地拒绝。CFR 由真实帧包 PTS 均匀性验证（而非 r/avg 帧率字段），首个显示帧 PTS 取自实际包。
- 无音频不失败（输出无音轨）；有音频转 AAC 192k，输出移除全部 spherical metadata 并带 faststart。
- 源与既有成片**永不覆盖**：目标冲突失败（`--unique-suffix` 另建唯一输出）。

## 测试与 CI

```bash
uv run pytest          # 全部测试：纯核心 + 真实 ffmpeg 合成夹具（无 ffmpeg 自动跳过媒体测试）
```

合成夹具由测试内生成（方向标记全景图、luma 阶跃视频、正弦音轨经 stdin 喂给 ffmpeg），不读取任何用户素材。
CI（`.github/workflows/ci.yml`）：从 Ubuntu 官方软件包安装 ffmpeg 并打印真实版本，不依赖 runner 预装；`permissions: contents: read`，无 secret、无发布、无 `pull_request_target`、无模型下载，仅 CPU 合成测试。

## 布局

```
src/vibecut/core/     contracts(合同)/geometry(几何)/project(工程)/jobs/cache/proc(进程)
src/vibecut/media/    ffmpeg(工具发现)/probe(探测+产物验证)/proxy(规范代理)
src/vibecut/render/   static(静态导出流水线)
src/vibecut/cli.py    headless CLI（惰性导入，核心无 ffmpeg 依赖）
tests/                纯核心测试 + ffmpeg 合成媒体测试
```

## 依赖与许可记录（任务 #2）

- Python 3.12：PSF License；本机验证使用 CPython 3.12.12。
- 运行时无第三方 Python 包。开发测试依赖 pytest（MIT）；构建依赖 hatchling（MIT），dev 的精确解析结果在 `uv.lock`，构建后端由隔离构建环境安装。
- FFmpeg/ffprobe 是外部工具。本机实际验证为 9.0.2 essentials build，配置包含 `--enable-gpl --enable-version3`，因此该构建按 GPLv3 分发。CI 使用 Ubuntu 软件包版本并在日志打印真实版本；不同构建的完整 version/configuration 文本参与缓存身份。
- 此处记录依赖许可，不替 VibeCut 项目选择开源许可；本任务不打包或再分发 FFmpeg。

本机验证入口（仓库根目录，已准备 `.venv`，无需重新解析依赖）：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:VIBECUT_REQUIRE_FFMPEG = "1"
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
# 与 uv run vibecut 使用相同 CLI 入口：
.venv/Scripts/python.exe -m vibecut.cli --help
```

Windows 产物提交需要目标文件系统支持硬链接且运行环境允许创建硬链接；不满足时明确失败并保留现有文件。受限 AppContainer 可拒绝这一操作；本机媒体验收使用正常 NTFS 进程权限。
