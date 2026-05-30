# SS-BSN 移植改动记录

基于 AP-BSN 框架（已修复版本）对 SS-BSN 进行的适配与改进。

---

## 一、项目概述

SS-BSN 与 AP-BSN 的唯一实质区别在于**模型架构**：

| | AP-BSN | SS-BSN |
|--|--------|--------|
| 顶层模型 | `APBSN` | `SSBSN` |
| 主干网络 | `DBSNl` → `DCl` (空洞残差) | `SSBSNl` → `SSBlock` (自相似注意力) |
| 注意力机制 | 无 | Cosine similarity attention in pixel-unshuffle space |

其余基础设施（datahandler、trainer、util、loss）完全同源。但 SS-BSN 代码基于 AP-BSN 早期版本，存在若干已修复的 bug。

---

## 二、修改清单

### 1. `src/util/util.py` — 覆盖

**修复内容：**

- **`np2tensor()` 灰度图 bug**：原版对 2D 输入执行 `np.transpose(n, (2,0,1))` 会 IndexError，改为 `n.unsqueeze(0)`
- **`tensor2np()` 灰度图 bug**：原版对 2D tensor 执行 `t.permute(1,2,0)` 会错误，改为 `t.numpy()`；新增 `(1,h,w)` → `(h,w,1)` 分支
- **`psnr()` / `ssim()` data_range 自动检测**：原版硬编码 `data_range=255`，改为自动检测 [0,1]→1.0, [0,255]→255, [0,65535]→65535

### 2. `src/util/config_parse.py` — 覆盖

**修复内容：**

- 新增 `get(name, default=None)` 方法，安全访问不存在的配置项
- CLI args 仅在显式传入（非 None）时才覆盖 YAML 值，避免 None 覆盖有效配置

### 3. `src/util/file_manager.py` — 覆盖

**修复内容：**

- 新增 `save_img_tensor_denorm()` 方法，根据 `norm_factor` 确定 8-bit 或 16-bit 存储
- 修复原版靠像素最大值猜测位深导致的"后期训练全黑图" bug

### 4. `src/trainer/base.py` — 合并

以 AP-BSN 修复版为底，保留 SS-BSN 独有功能：

| 来源 | 内容 |
|------|------|
| AP-BSN | `norm_factor` 支持（逐张归一化/反归一化） |
| AP-BSN | `test_img()` 位深自动检测 + 归一化 |
| AP-BSN | `save_img_tensor_denorm` 保存逻辑 |
| AP-BSN | `cfg.get()` 安全访问 |
| AP-BSN | TensorBoard (`SummaryWriter`) |
| AP-BSN | DataLoader 使用 `cfg['thread']` 而非硬编码 0 |
| AP-BSN | 泛化的 GPU 数据传输（`isinstance` 检查） |
| SS-BSN | `setup_determinism()` — 固定随机种子 |
| SS-BSN | `_log_configs()` — 递归打印配置项 |
| SS-BSN | `_set_dataloader()` 的 `drop_last` 参数 |

### 5. `src/trainer/trainer.py` — 覆盖

**修复内容：**

- `test_img` / `test_dir` 使用 `cfg.get()` 安全访问
- `test_dataloader_process()` 和 `validation()` 传递 `norm_factor` 参数

### 6. `train.py` / `test.py` — 覆盖

**修复内容：**

- 移除 `trainer.set_device()`，改用 `os.environ['CUDA_VISIBLE_DEVICES'] = cfg.get('gpu')`（进程级设置）
- `cfg.get('gpu')` 安全访问

### 7. `src/datahandler/CONFOCAL.py` — 新增

共聚焦显微图像数据集类，支持：

- 混合 8-bit / 16-bit 单通道灰度 PNG/TIF
- 逐张自动检测位深（`norm_factor = 255.0` 或 `65535.0`）
- `Confocal`：单目录全图加载
- `prep_confocal`：预处理后的裁剪 patch 加载

### 8. `conf/SSBSN_CONFOCAL.yaml` — 新增

SS-BSN 的共聚焦训练/验证/测试配置，关键参数：

```yaml
model:
  type: SSBSN
  kwargs:
    in_ch: 1            # 灰度单通道
    mode: [na, na, na, na, na, na, ss, ss, ss]
    f_scale: 2
    ss_exp_factor: 1

training:
  dataset: Confocal
  crop_size: [255, 255]  # 必须被 pd_a=5 整除
  init_lr: 1e-4
  scheduler: {step_size: 8, gamma: 0.1}
```

### 9. `train_ssbsn_confocal.sh` — 新增

一键训练脚本。

---

## 三、未改动的文件

以下 SS-BSN 文件保持原样（模型核心 + 与 AP-BSN 一致的代码）：

- `src/model/SSBSN.py` — 顶层模型
- `src/model/SSBSNl.py` — 含 SSBlock 的主干网络
- `src/model/SSBlock.py` — 自相似注意力模块（核心创新）
- `src/model/__init__.py`
- `src/datahandler/__init__.py` / `denoise_dataset.py` / `SIDD.py` / `custom.py`
- `src/loss/__init__.py` / `recon_self.py`
- `src/trainer/__init__.py`
- `src/util/logger.py` / `progress_msg.py`
- `src/util/dnd_submission/` / `sidd_submission/`
- `conf/SSBSN_SIDD.yaml`
- `prep.py`

---

## 四、使用方法

```bash
# 训练
python train.py -c SSBSN_CONFOCAL -g 0 --thread 8
# 或
bash train_ssbsn_confocal.sh

# 推理（单张）
python test.py -c SSBSN_CONFOCAL -g 0 --pretrained SSBSN_CONFOCAL.pth --test_img ./test.png

# 推理（文件夹）
python test.py -c SSBSN_CONFOCAL -g 0 --pretrained SSBSN_CONFOCAL.pth --test_dir ./test_origin/

# 推理（用训练检查点）
python test.py -c SSBSN_CONFOCAL -g 0 -e 50 --test_dir ./test_origin/
```
