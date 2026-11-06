"""Check the default two-method command without loading models."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate


class EvaluationCommandTests(unittest.TestCase):
    def prepare(self, root):
        settings = {'gpu': 2}
        for key in ('runtime', 'backbone', 'data_root'):
            path = root/key
            path.mkdir()
            settings[key] = str(path)
        for key in ('egder_checkpoint', 'baseline_checkpoint'):
            path = root/(key + '.pt')
            path.touch()
            settings[key] = str(path)
        settings['output'] = str(root/'output')
        config = root/'settings.yaml'
        config.write_text(json.dumps(settings))
        return config, settings

    def test_default_command_runs_both_methods_with_identical_shared_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            config, settings = self.prepare(Path(directory))
            with patch.object(evaluate, 'DEFAULT_CONFIG', config), \
                 patch.object(sys, 'argv', ['evaluate.py']), \
                 patch.object(evaluate.subprocess, 'run') as run, \
                 redirect_stdout(io.StringIO()):
                evaluate.main()
            self.assertEqual(run.call_count, 2)
            options = []
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(Path(command[1]).name, 'evaluation_worker.py')
                options.append(dict(zip(command[2::2], command[3::2])))
                self.assertTrue(call.kwargs['check'])
                self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
                self.assertEqual(call.kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
            self.assertEqual([x['--method'] for x in options], ['egder', 'liveedit'])
            for key in ('--upstream', '--backbone', '--data-root'):
                self.assertEqual(options[0][key], options[1][key])
            self.assertEqual(options[0]['--checkpoint'], settings['egder_checkpoint'])
            self.assertEqual(options[1]['--checkpoint'], settings['baseline_checkpoint'])
            outputs = [Path(x['--output']) for x in options]
            self.assertEqual(outputs[0].parent, outputs[1].parent)
            self.assertNotEqual(outputs[0], outputs[1])

    def test_missing_comparison_checkpoint_starts_neither_method(self):
        with tempfile.TemporaryDirectory() as directory:
            config, settings = self.prepare(Path(directory))
            Path(settings['baseline_checkpoint']).unlink()
            with patch.object(evaluate.subprocess, 'run') as run:
                with self.assertRaisesRegex(FileNotFoundError, 'baseline_checkpoint'):
                    evaluate.run_comparison(config)
            run.assert_not_called()

    def test_failed_method_does_not_report_a_completed_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.prepare(Path(directory))
            error = evaluate.subprocess.CalledProcessError(1, ['evaluation_worker.py'])
            with patch.object(evaluate.subprocess, 'run', side_effect=error) as run, \
                 redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(evaluate.subprocess.CalledProcessError):
                    evaluate.run_comparison(config)
            self.assertEqual(run.call_count, 1)
            self.assertNotIn('Evaluation completed', output.getvalue())


if __name__ == '__main__':
    unittest.main()
