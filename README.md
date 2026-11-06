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
- [evaluate.py](evaluate.py): the default performance evaluation command.

## Setup

Use Python 3.10 and a CUDA-compatible PyTorch installation.

```bash
pip install -r requirements.txt
git clone https://github.com/qizhou000/LiveEdit.git ../runtime
python install_egder.py ../runtime --runtime-only
```

The preparation command selects the compatible shared backbone and evaluation runtime.
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

To register EGDER for training, use a separate runtime checkout:

```bash
git clone https://github.com/qizhou000/LiveEdit.git ../training-runtime
python install_egder.py ../training-runtime
```

Select `--editor_name egder` in that runtime's training entrypoint.

## Citation

```bibtex
@misc{Zhang_EGDER,
  author = {Zhang, Lintong and Lee, Yearang and Lee, Seong-Whan},
  title = {Evidence Guided Dual Expert Memory with Joint Routing for VLLMs Online Correction}
}
```

Dept. of Artificial Intelligence, Korea University, Seoul, Korea

`{zhanglintong, yr_lee, sw.lee}@korea.ac.kr`

### Code contribution acknowledgment

We acknowledge code contributions from [LiveEdit](https://github.com/qizhou000/LiveEdit):

```bibtex
@InProceedings{Chen_2025_CVPR,
  author = {Chen, Qizhou and Wang, Chengyu and Wang, Dakan and Zhang, Taolin and Li, Wangyue and He, Xiaofeng},
  title = {Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
  pages = {9455--9466}
}
```

The upstream copyright and permission notice are preserved in `LICENSE.upstream`.
