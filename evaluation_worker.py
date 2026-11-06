"""Run the pinned official EVQA evaluator with a strictly loaded editor."""
import argparse
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys

PINNED_REVISION = '3615a37b05294509f411df045621940f276a5e6b'
PACKAGE = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--method', choices=('liveedit','egder'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.upstream = args.upstream.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.backbone = args.backbone.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    revision = subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip()
    if revision != PINNED_REVISION:
        raise RuntimeError('Use the documented pinned official revision.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor, BlipImageProcessor, GPT2Tokenizer
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0,str(args.upstream))
    import utils
    import utils.GLOBAL as global_config
    global_config.ROOT_PATH = utils.ROOT_PATH = str(args.upstream)
    global_config.model_path_map['blip2-opt-2.7b'] = str(args.backbone)
    utils.model_path_map = global_config.model_path_map
    from editor.vllms_for_edit.blip2.blip2 import BLIP2OPTForEdit
    from editor.vllms_for_edit.base import BaseVLLMForEdit
    def initialize(self, model_path, device='cuda:0'):
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path,device_map=device,
            torch_dtype=torch.float32,local_files_only=True)
        self.processor = Blip2Processor(
            image_processor=BlipImageProcessor.from_pretrained(model_path,local_files_only=True),
            tokenizer=GPT2Tokenizer.from_pretrained(model_path,local_files_only=True))
        self.model.eval().requires_grad_(False)
        BaseVLLMForEdit.__init__(self,self.model,device,False)
    BLIP2OPTForEdit.__init__ = initialize
    if args.method == 'egder':
        from egder.egder import EGDER, EGDERConfig
        config = EGDERConfig.from_yaml(PACKAGE/'configs/egder/blip2-opt-2.7b.yaml')
        editor = EGDER(utils.load_vllm_for_edit('blip2','cuda:0'),config,'cuda:0')
    else:
        editor = utils.load_vllm_editor('liveedit','blip2','cuda:0',None,None,False)
    state = torch.load(str(args.checkpoint),map_location='cpu',weights_only=True,mmap=True)['train_modules']
    modules = editor.get_modules_for_training()
    if set(state) != set(modules):
        raise RuntimeError('The checkpoint does not contain exactly the compatible editor modules.')
    for name, module in modules.items():
        module.load_state_dict(state[name],strict=True)
    del state
    editor.set_train(False)
    editor.restore_to_original_model()
    utils.load_vllm_editor = lambda *unused, **kwargs: editor
    import dataset.vllm as datasets
    original_evqa = datasets.EVQA
    datasets.EVQA = lambda data_path,img_root_dir,data_n=None: original_evqa(
        str(args.data_root/'vqa/vqa_eval.json'),str(args.data_root/'images'),data_n)
    import evaluation.vllm_editor_eval as evaluation
    original_evaluator = evaluation.VLLMEditorEvaluation
    evaluation.VLLMEditorEvaluation = lambda ed,data,name,results: original_evaluator(
        ed,data,name,str(args.output))
    args.output.mkdir(parents=True,exist_ok=False)
    os.chdir(args.output)
    sys.argv = [str(args.upstream/'test_vllm_edit.py'),'-en',args.method,'-mn','blip2',
        '-sen','1000','-dvc','cuda:0','-dn','EVQA','-ckpt',str(args.checkpoint),'-enp',args.method]
    runpy.run_path(str(args.upstream/'test_vllm_edit.py'),run_name='__main__')

if __name__ == '__main__':
    main()
