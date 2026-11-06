# EGDER

Code for **Evidence Guided Dual Expert Memory with Joint Routing for VLLMs Online Correction**.

![Overview of EGDER](assets/egder_overview.png)

[PDF version](assets/framework.pdf)

EGDER combines evidence-guided visual localization, paired support/suppress experts,
quality estimation, and joint sparse routing for online multimodal knowledge editing.

## Implementation

- [egder/egder.py](egder/egder.py): the `EGDER` editor, expert memory, editing, and training.
- [egder/modules.py](egder/modules.py): visual localization, dual expert generation, quality heads, and joint routing.
- [configs/egder/blip2-opt-2.7b.yaml](configs/egder/blip2-opt-2.7b.yaml): the editor configuration.
- [evaluate.py](evaluate.py): the default EGDER and LiveEdit comparison command.

## Setup

Use Python 3.10 and a CUDA-compatible PyTorch installation.

```bash
pip install -r requirements.txt
git clone https://github.com/qizhou000/LiveEdit.git ../runtime
git -C ../runtime checkout 3615a37b05294509f411df045621940f276a5e6b
```

The pinned checkout provides the shared backbone and evaluation runtime.
Provide the BLIP2 OPT-2.7B backbone, MMEdit EVQA data, and compatible editor
checkpoints separately. The data root must contain `vqa/vqa_eval.json` and
`images/`. Model weights and datasets are not included.

## Code verification

```bash
CUDA_VISIBLE_DEVICES=2 python -m unittest discover -s tests -v
python evaluate.py --help
```

## Performance verification

Set the runtime, backbone, data, two checkpoint paths, and output directory in
[configs/evaluation.yaml](configs/evaluation.yaml), then run:

```bash
python evaluate.py
```

By default, this evaluates EGDER and official LiveEdit in separate processes using
the same EVQA cases, FP32 backbone, seed 2026, and official scoring functions.
Each method processes two groups of 1000 sequential edits. Outputs are saved
under a new run directory within the configured output path.

## Training integration

To register EGDER for training, use a separate pinned runtime checkout:

```bash
git clone https://github.com/qizhou000/LiveEdit.git ../training-runtime
git -C ../training-runtime checkout 3615a37b05294509f411df045621940f276a5e6b
python install_egder.py ../training-runtime
```

Select `--editor_name egder` in that runtime's training entrypoint.

## Citation

**Evidence Guided Dual Expert Memory with Joint Routing for VLLMs Online Correction**

Lintong Zhang, Yearang Lee, Seong-Whan Lee\*

Dept. of Artificial Intelligence, Korea University, Seoul, Korea

`{zhanglintong, yr_lee, sw.lee}@korea.ac.kr`

### Code contribution acknowledgment

Code contributions from [LiveEdit](https://github.com/qizhou000/LiveEdit) are acknowledged
with its paper: Qizhou Chen et al.,
[Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Lifelong_Knowledge_Editing_for_Vision_Language_Models_with_Low-Rank_Mixture-of-Experts_CVPR_2025_paper.html),
CVPR 2025, pp. 9455-9466.

The upstream copyright and permission notice are preserved in `LICENSE.upstream`.
