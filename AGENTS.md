# AGENTS.md — VibeCut 仓库工作规则（ai-flow-contract v5.1）

## 流程

- **Issue-first**：所有实现、修复、整改必须挂在 FelixJI/VibeCut 的 Issue/Task 上；不接受无 Issue 的游离改动。
- **本地 route-subagents**：主代理定界后按 `route-subagents-auto` 路由有界实施；子代理不自行改变仓库保护、CI 权限或共享契约。
- 高风险取舍（跨任务契约、CI 信任边界、进程终止语义、源/产物覆盖策略）影响主代理冻结的不变量时，先交回主代理，不就地改约。

## 冻结不变量（违反即缺陷）

1. 核心包仅标准库；CLI 不依赖 Qt/SDK/模型；重模块惰性导入。
2. 源时间与输出时间一律 `Fraction` 有理数；source-relative 0 = 源首个显示帧（PTS 取自实际包验证）；音视频同绝对时间线裁切；CFR 由真实包 PTS 均匀性验证（r/avg 帧率字段不作充分条件），真 VFR 提前 fail-fast，禁止 frame_index/fps 猜测。
3. canonical 几何：yaw0=equirect 中心、正 yaw 向右、正 pitch 向上、roll 显式顺时针、合同 FOV 为水平 FOV；手工锁定优先（CameraPath provenance="user"，静态导出仅接受 user）。
3a. 色彩：仅接受显式标记 BT.709/tv/8bit SDR；未知/非 709/全范围/HDR 一律拒绝，绝无未经真实转换的重标；投影检测仅信任结构化标记或显式声明，矛盾/立体元数据拒绝。
4. 源/既有输出永不覆盖：同目录临时写 → 子进程真实退出 → probe 验证 → 无覆盖提交 → 成功事件；工程原子 replace + 同项目跨进程独占锁 + revision 拒绝迟到事件。
5. 缓存绑定源内容/工具/配置/色彩空间，不绑定 edit revision。
6. 子进程：argv 直传不经 shell；有界超时/取消/管道排空/等待真实退出；Windows Job Object（或等价）只杀自家进程树；完成事件必须在产物验证后发布。
7. 内容哈希仅限 #2 契约（探测生产者 → 缓存/源不变性消费者，失败即拒绝），不做额外下载/交付 hash。

## 安全与验证

- 只读取仓库内代码与测试生成的合成素材；不上传用户视频；不读取凭据或无关私人文件。
- 不引入 secrets、发布动作、`pull_request_target` 不可信执行、私有 SDK 或大模型下载；CI 保持 `contents: read`、CPU-only 合成测试。
- 验证命令（cwd = 仓库根）：
  - `uv sync --frozen`
  - `uv run pytest`（ffmpeg/ffprobe 在 PATH 或经 `VIBECUT_FFMPEG`/`VIBECUT_FFPROBE` 提供；缺失时媒体测试自动跳过且必须在报告中如实说明）
- 不用真实素材“验证”，不把未运行的命令写成已执行；工具缺失即报告，不假冒验证结果。
- Git 写操作（commit/push/merge）由主代理统一执行。

## 代码约定

- Python 3.12，`src/` 布局，新公共 API 进 `core/contracts.py` 前必须考虑 schema 版本化。
- 修改共享契约/锁文件属于当前契约持有者任务范围，其他任务不得并行改动。
