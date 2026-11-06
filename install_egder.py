"""Register EGDER in a dedicated pinned runtime checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess

PINNED_REVISION = '3615a37b05294509f411df045621940f276a5e6b'
FILES = (
    'egder/__init__.py',
    'egder/egder.py',
    'egder/modules.py',
    'configs/egder/blip2-opt-2.7b.yaml',
)
ANCHOR = "    if editor_name == 'liveedit':\n"
REGISTRATION = """    if editor_name == 'egder':
        from egder.egder import EGDER, EGDERConfig
        data_proc_device = 'cuda:%s'%extra_devices[0] if for_train else None
        vllm_data_proc = load_vllm_for_edit(edit_model_name, data_proc_device) if for_train else None
        config = EGDERConfig.from_yaml(config_path)
        editor = EGDER(vllm, config, device, vllm_data_proc, data_proc_device)
    elif editor_name == 'liveedit':
"""

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--runtime-only', action='store_true',
                        help='Prepare the evaluation runtime without registering EGDER.')
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PINNED_REVISION:
        changes = subprocess.check_output(
            ['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'],
            text=True).strip()
        if changes:
            raise RuntimeError('The runtime has local changes; preserve them before preparing it.')
        subprocess.run(
            ['git', '-C', str(checkout), 'checkout', '--quiet', '--detach', PINNED_REVISION],
            check=True)
    if args.runtime_only:
        print('Evaluation runtime ready.')
        return
    package = Path(__file__).resolve().parent
    loader = checkout/'utils/__init__.py'
    original = loader.read_text()
    if REGISTRATION in original:
        updated = original
    elif original.count(ANCHOR) == 1:
        updated = original.replace(ANCHOR, REGISTRATION, 1)
    else:
        raise RuntimeError('The runtime editor loader has changed.')
    for relative in FILES:
        source, destination = package/relative, checkout/relative
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError('Refusing to overwrite modified file: ' + str(destination))
    for relative in FILES:
        destination = checkout/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package/relative, destination)
    if updated != original:
        loader.write_text(updated)
    print('EGDER registered. Use -en egder with the runtime training entrypoint.')

if __name__ == '__main__':
    main()
