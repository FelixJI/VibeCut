# VibeCut

VibeCut 把单目 equirectangular（360° 全景）SDR MP4 剪成**普通平面视频**。
支持固定视角和动态虚拟相机：探测 → 工程 → 路径/多源顺序剪辑 → 1080p H.264/AAC/BT.709，
不依赖任何 SDK、模型、GPU、Qt 或网络服务，核心包只用 Python 标准库。

## 快速开始

```bash
# Python 3.12 + uv；ffmpeg/ffprobe 需在 PATH，或用环境变量指定：
#   VIBECUT_FFMPEG=C:\tools\ffmpeg\bin\ffmpeg.exe
#   VIBECUT_FFPROBE=C:\tools\ffmpeg\bin\ffprobe.exe
uv sync --frozen             # 安装锁定环境（运行时零第三方依赖，dev 组为 pytest）
uv run vibecut init ./proj   # 建工程（vibecut.project.json, project schema v2, revision 1）
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

`MediaSource`（稳定 `source_id` = 内容指纹前缀、路径、投影、稳定化 provenance（仅允许 "none"）、色彩 primaries/transfer/matrix/range/bit depth（显式 bt709/tv/8bit 或 10bit SDR）、start PTS/time_base/fps 有理数、音频流、VFR 标记）、`NormalizedAsset`（规范资产/代理 + `TimeMap` 时间映射 + 稳定化 provenance）、`EditPlan`（source-relative 入出点 + manual cut 锁定标志、工程帧率、`CameraPath`）、`CameraPath`（provenance="user"/"derived"：手工锁定值优先；渲染仅接受 "user"）、`JobEvent`（job_id + project_revision）。

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

- 仅单目 equirect **显式标记 BT.709 / tv-range / 8bit 或 10bit SDR** 的 MP4/MOV，H.264/HEVC；10bit 在 FFmpeg 像素处理中真实转换为 yuv420p 8bit，代理合同同样记录实际 8bit。未知/未标记色彩、非 bt709、全范围（pc/jpeg）、12/16bit、HDR/HLG/PQ/Log/Dolby 一律提前拒绝；真 VFR、非 2:1、未知投影（未声明）、立体/双鱼眼标记、矛盾元数据、多视频流、不可剪音轨同样提前拒绝。
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
src/vibecut/render/   static(唯一产物事务)/dynamic(动态相机与序列图)
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

## 动态相机与多源剪辑（任务 #4）

旧 `CameraPath` / `EditPlan` schema v2、静态 API 和旧 schema1 工程保持可读。工程保存升级为 project schema2，使旧读取器明确拒绝，而非忽略并丢失 `sequences`；contract schema 仍为 v2。`core.sequence` 独立提供 schema v1：
`BoundCameraPath(asset_id, aspect, camera, cuts, geometry_revision=1)`、`SequenceClip(source_id, cut_in_rel, cut_out_rel, path)` 和 `SequencePlan(clips, project_fps, aspect, audio=True)`。
这里 asset_id 绑定已探测原素材的 source_id（其内容身份已由 #2 指纹合同验证），geometry revision 1 固定绝对 Ry(yaw) Rx(pitch) Rz(roll) 世界参考。

路径键时间仍是源首显示帧起算的 Fraction。路径须覆盖完整片段且所有键位于实际源范围内；键时间严格递增、数值有限、pitch 在 ±90°、水平 FOV 在 (0,180)。yaw 和 roll 沿最短圆弧线性插值；pitch/FOV 线性插值。`cuts` 必须命名非首个现有键；切点之前保持上一键，切点处立即采用新键。片段边界本身也是硬切。每段时长必须是整数个输出帧，避免多段舍入累积；不同源帧率由实际 PTS 校验后的源统一到 project_fps。

横屏路径必须绑定 `16:9`，竖屏路径必须另建并绑定 `9:16`。默认最终尺寸分别为 1920×1080 和 1080×1920；预览分别为 640×360 和 360×640。预览/最终共用同一评估器、绝对源时间、v360 投影与每帧运行时命令（`reset_rot=1,rorder=ypr`）；只降低分辨率和编码质量。每帧命令在 PTS 归零及 fps 统一之后应用。

在工程导入素材后，可以用标准库 API 写入可检查的序列 JSON：

```python
import json
from fractions import Fraction as F
from pathlib import Path
from vibecut.core import CONTRACT_VERSION, CameraPath, CameraView, BoundCameraPath, SequenceClip, SequencePlan
from vibecut.core.project import Project

project = Project.load(Path("proj/vibecut.project.json"))
source = next(iter(project.sources.values()))
camera = CameraPath(CONTRACT_VERSION, ((F(0), CameraView(178)), (F(1), CameraView(-178))))
path = BoundCameraPath(source.source_id, "16:9", camera)
plan = SequencePlan((SequenceClip(source.source_id, F(0), F(1), path),), F(30))
Path("sequence.json").write_text(json.dumps(plan.to_json(), indent=2), encoding="utf-8")
```

```bash
uv run vibecut render-sequence ./proj sequence.json preview.mp4 --preview
uv run vibecut render-sequence ./proj sequence.json final.mp4
# --cpu 禁用 GPU 探测；--unique-suffix 保留既有输出并另取名字
```

序列随 job_id 写入工程可选 `sequences` 字段，持同一项目锁至验证/保存结束。所有来源按同一绝对 PTS 窗口裁切音画；音频统一 48kHz stereo AAC，无音轨的片段补静音，`audio=False` 明确输出无音轨。预览缓存包含不可变 payload，以源内容、工具 build、路径、cut、画幅、质量和色彩入键，命中后仍经 probe 与全部轨道解码验证，最后无覆盖提交；工程 revision 不入键。

默认动态渲染实际生成小型 H.264，分别尝试 NVDEC 解码和 NVENC 编码，记录探测结果及最终 decode/projection/encode 后端。不可用则用 CPU；实际 GPU 作业初始化失败也以**同一裁切和路径**重新 CPU 渲染。v360 始终 CPU；事件记录小样 CPU v360 时间、完整渲染时间与 EOF 解码时间，它们不是 4K 性能承诺。CPU-only CI 不要求显卡。

提交前必须同时通过媒体尺寸/像素格式/帧率/色彩/旋转/轨道检查及全部流 EOF 解码。取消、坏源、缺滤镜、写失败或解码错误均无 succeeded、有效新缓存或部分成片。错误级 stderr 即使 FFmpeg 返回 0 也按失败处理。测试只使用合成方位标记、视频阶跃、音频标记和 10bit SDR 数值斜坡。


## 本机 AI 候选分析（任务 #5）

可选 `analyze` 入口只连接显式配置的回环 HTTP 服务，不需要 SDK；模型不可用时保留人工可审阅候选并报告具体阶段，不把规则结果标为 AI 成功。先自行在 LM Studio 启动已安装的视觉模型和本地服务；不会下载模型、修改全局配置或回退云端。

```powershell
.venv/Scripts/python.exe -m vibecut.cli init ./project
.venv/Scripts/python.exe -m vibecut.cli analyze ./project ./canonical.mp4 --projection equirect --focus "人物优先，剪出20秒，排除模糊" --endpoint http://127.0.0.1:1234 --model qwen3.8-27b@q4_k_xl --out ./analysis.json
# --json 同时在 stdout 输出结果；--scenery-priority / --person-priority 可显式指定偏好。
# 已选模型尚未加载时，只有显式 --allow-jit-load 才允许首个推理触发加载。
```

默认全片 12 个均匀区间中各选一个实际显示帧，每个时刻 14 张重叠视图（6 个赤道方向及上下各 4 个极区方向），768×432 JPEG；通常为 24 次观察 + 1 次选片。默认推理硬预算 32 次（含每批至多一次修复），每次至多 8 图、6,000,000 输入字节；观察最多 2,048 输出 token，文字选片（含其至多一次修复）独立最多 4,096 输出 token，单请求总截止时间 300 s；全部模型请求累计预算 900 s，含至多两次只读模型探测。请求数、探测数、修复数、finish_reason、token 预算、输入大小、耗时和有效采样数写入计量。`--samples`、`--max-requests`、`--timeout` 可调整 CLI 预算；预算较小时把可负担采样重新均布全片，为选择保留请求余量，而不是只看前段。达到限制输出有诊断的 partial 结果，不无限重试。

观察缓存与重点无关，改重点/目标时长仅重算选片；源内容、规范资产、工具、采样视图、模型、推理参数和 schema/prompt 版本参与相应缓存身份。结果中的候选、观察和证据保留 source/time/view ID；分析结果 schema v2 的选片只能引用该候选窗口内实际观察的 img_ids；时间/source/view 由应用从已验证图像派生，文字阶段禁止生成 bbox、任意时间或主体精确位置。输出方向明确为 view-center 视图方向，不声称主体定位。观察描述合同/prompt 保持 v1；视觉缓存升为 v2，因为旧毫秒取帧不能证明图片与所记时间等价。采样由真实包 PTS 验证的 CFR 映射到显示帧，时间用精确有理数，重复帧去重；规范资产首 PTS 可非零，输入侧 seek 后按保留的真实 PTS 精确取帧，廉价特征另按 asset_id 隔离；选片 prompt、算法版本及独立 token 预算入选片缓存键。finish_reason=length 明确报告 output-truncated，不接受截断或空白内容为成功。AI 草稿不覆盖工程、人工锁定选择或既有结果文件；`to_edit_plan` 生成 derived 相机路径，导出前仍需用户确认。一个工程从探测到结果提交一直持独占作业锁，提交前复核 revision。

离线诊断先运行 `tests/test_analysis_transport.py`、`tests/test_analysis_model.py` 和 `tests/test_analysis_service.py`：这些只使用回环假服务与合成素材，不证明真实模型语义准确。未启动/未加载/无视觉能力、结构化输出失败、超时、OOM、取消都有 stage/code；断网运行需已就绪的本地模型。提示和 JPEG 仅发往配置的回环服务，禁止外部重定向；计量不记录提示、图像或模型思维文本。结果 JSON 本身含用户重点与内容描述，应由用户保存在本机。

语义验收 AC2 尚未放行：需先在本机准备至少 3 段源片，冻结至少 12 个候选窗口的人工标签（source_id、准确起止时间、view_id、人物/风景相关性、不确定项），再分别运行人物优先/风景优先并核对两组相关率是否各 ≥80%，同时记录差异、漏检和不确定项。不得根据模型结果反写标签，也不上传用户视频；缺少这组人工数据时只能报告能力与合同测试证据。
