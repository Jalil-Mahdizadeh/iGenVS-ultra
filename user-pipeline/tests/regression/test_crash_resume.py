"""Process-crash and orchestration regressions; run in the dedicated iGenVS SIF."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')

from igenvs_ultra import workflow, generation_worker
from igenvs_ultra.cli import build_parser

REPO = Path(__file__).resolve().parents[3]


def shard(directory, index, count, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'manifest.json').write_text(json.dumps({
        'status': 'complete',
        'config': {'num_shards': count, 'shard_index': index},
        'validation': {'valid': len(rows)},
        'counts': {'prepared': len(rows), 'docked': len(rows)},
        'timings': {},
    }))
    with (directory / 'results.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_row', 'molecule_id', 'status'])
        for row in rows:
            writer.writerow([row, f'mol-{row}', 'success'])


@pytest.mark.parametrize('gpu_count', [1, 2])
def test_actual_launcher_and_merger_preserve_legacy_partitions(tmp_path, monkeypatch, gpu_count):
    validated = tmp_path / 'library/validated.csv'
    validated.parent.mkdir()
    validated.write_text('source_row\n' + ''.join(f'{n}\n' for n in range(1, 9)))
    validation = {'valid_rows': 8, 'validated': str(validated)}
    (validated.parent / 'validation-manifest.json').write_text(json.dumps(validation))
    for index in [0, 1]:
        shard(tmp_path / f'docking/runs/shard-{index}', index, 4, [index+1, index+5])
    args = build_parser().parse_args([
        'dock', '--target', str(tmp_path/'target'), '--input', str(validated),
        '--output-dir', str(tmp_path), '--pose-output', 'none',
    ])
    launched = []

    def popen(command, **kwargs):
        index = int(command[command.index('--shard-index') + 1])
        count = int(command[command.index('--num-shards') + 1])
        destination = Path(command[command.index('--output-dir') + 1])
        launched.append(index)
        assert count == 4
        shard(destination, index, count, [n for n in range(1, 9) if (n-1)%count == index])
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(workflow.subprocess, 'Popen', popen)
    runtime = SimpleNamespace(execution='docker', _wrap=lambda tool, command, **kw: command)
    source = {'kind': 'external', 'path': str(validated)}
    gpus = [str(n) for n in range(gpu_count)]
    dirs, elapsed, plan, active = workflow._launch_regular_docking_shards(args, runtime, tmp_path, source, validated, gpus)
    assert launched == [2, 3]
    assert plan['logical_shards'] == 4
    result = workflow._merge_regular_docking_shards(
        args, tmp_path, dirs, active, elapsed, validation,
        validated=validated, logical_shards=4,
    )
    with Path(result['outputs']['results']).open() as f:
        assert [int(row['source_row']) for row in csv.DictReader(f)] == list(range(1, 9))


@pytest.mark.parametrize('source_rows', [[1, 2, 3], [1, 5, 9]])
def test_empty_partitions_never_launch(tmp_path, monkeypatch, source_rows):
    validated = tmp_path / 'library/validated.csv'
    validated.parent.mkdir()
    validated.write_text('source_row\n' + ''.join(f'{n}\n' for n in source_rows))
    validation = {'valid_rows': 3, 'validated': str(validated)}
    (validated.parent / 'validation-manifest.json').write_text(json.dumps(validation))
    args = build_parser().parse_args([
        'dock', '--target', str(tmp_path/'target'), '--input', str(validated),
        '--output-dir', str(tmp_path), '--pose-output', 'none',
    ])
    launched = []

    def popen(command, **kwargs):
        index = int(command[command.index('--shard-index') + 1])
        count = int(command[command.index('--num-shards') + 1])
        rows = [n for n in source_rows if (n-1)%count == index]
        assert rows
        launched.append(index)
        shard(Path(command[command.index('--output-dir') + 1]), index, count, rows)
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(workflow.subprocess, 'Popen', popen)
    runtime = SimpleNamespace(execution='docker', _wrap=lambda tool, command, **kw: command)
    dirs, elapsed, plan, active = workflow._launch_regular_docking_shards(
        args, runtime, tmp_path, {'kind': 'external', 'path': str(validated)},
        validated, ['0','1','2','3'],
    )
    assert launched == sorted({(n-1)%4 for n in source_rows})
    result = workflow._merge_regular_docking_shards(
        args, tmp_path, dirs, active, elapsed, validation, validated=validated, logical_shards=4,
    )
    assert result['counts']['docked'] == 3


@pytest.mark.parametrize('boundary', ['before_marker', 'after_marker', 'after_commit'])
def test_admission_survives_process_exit(tmp_path, boundary):
    code = r'''
import os, sys
from pathlib import Path
from igenvs_ultra import workflow as w
root = Path(sys.argv[1])
boundary = sys.argv[2]
original = w.atomic_json
def publish(path, value):
    if path.name == 'admission.json' and boundary == 'before_marker':
        os._exit(73)
    original(path, value)
    if path.name == 'admission.json' and boundary == 'after_marker':
        os._exit(73)
w.atomic_json = publish
connection = w.open_dedup_database(root)
w.admit_batch(connection, root, 1, [dict(molecule_id='a', canonical_smiles='CC', original_smiles='CC', source_row='1')], 'external', 0)
os._exit(73)
'''
    process = subprocess.run([sys.executable, '-c', code, str(tmp_path), boundary])
    assert process.returncode == 73
    connection = workflow.open_dedup_database(tmp_path)
    try:
        workflow.replay_completed_admissions(connection, tmp_path)
        result = workflow.admit_batch(connection, tmp_path, 1, [dict(
            molecule_id='a', canonical_smiles='CC', original_smiles='CC', source_row='1',
        )], 'external', 0)
        assert result['accepted_rows'] == 1
        assert result['duplicate_rows'] == 0
        assert connection.execute('SELECT value, owner FROM smiles').fetchall() == [('CC', 1)]
    finally:
        connection.close()


def test_generator_rolls_back_output_before_state_commit(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('generator_test_fixture', REPO/'user-pipeline/tests/test_generation_worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(generation_worker, 'generate_de_novo_batch', lambda generator, count, **kw:
                        torch.tensor([[int(torch.initial_seed()), n] for n in range(count)]))
    database = tmp_path/'state.sqlite3'
    engine = module._engine(database)
    monkeypatch.setattr(engine, '_commit_state', lambda *args, **kw: (_ for _ in ()).throw(RuntimeError('injected commit failure')))
    first = {'output':str(tmp_path/'first.smi'), 'count':1, 'seed':13}
    with pytest.raises(RuntimeError, match='injected'):
        engine.generate(first)
    engine.close()
    resumed = module._engine(database)
    resumed.generate(first)
    resumed.generate({'output':str(tmp_path/'second.smi'), 'count':1, 'seed':14})
    assert (tmp_path/'first.smi').read_text() == 'S13_0\n'
    assert (tmp_path/'second.smi').read_text() == 'S13_1\n'
    resumed.close()
