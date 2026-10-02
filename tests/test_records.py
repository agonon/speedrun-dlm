import csv
import importlib.util
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


records = module('record_builder', 'records/make_record.py')
reproduce = module('record_reproduction', 'records/reproduce.py')
prefix = module('prefix_builder', 'data/prepare_training_prefix.py')


def measurements():
    with (ROOT / 'figures/checkpoint-evaluations.csv').open() as f:
        return list(csv.DictReader(f))


def test_records_reproduce_the_measurement_averages():
    data = measurements()
    for path in (ROOT / 'records').glob('*/hellaswag.json'):
        result = json.loads(path.read_text())
        rows = records.select(data, result['recipe'], result['position'])
        assert result['average_accuracy'] == statistics.mean(float(r['accuracy']) for r in rows)
        assert result['average_training_seconds'] == statistics.mean(float(r['timed_training_seconds']) for r in rows)
        assert all(len(r['checkpoint_sha256']) == 64 for r in rows)


def test_incomplete_training_sets_are_not_pooled():
    rows = records.select(measurements(), 'subs_mask', 'minute:012')
    with pytest.raises(ValueError, match='ten independent'):
        records.select(rows[:-1], 'subs_mask', 'minute:012')


def test_partial_larger_draw_count_does_not_replace_complete_set():
    rows = [dict(row, mc_samples='32') for row in
            records.select(measurements(), 'duo_simple', 'minute:065')]
    extra = dict(rows[0], mc_samples='128', accuracy='1')
    selected = records.select(rows+[extra], 'duo_simple', 'minute:065')
    assert len(selected) == 10
    assert {int(r['mc_samples']) for r in selected} == {32}


def test_reproduction_uses_recorded_batch_and_noise(monkeypatch, tmp_path):
    monkeypatch.setenv('NNODES','2')
    monkeypatch.setenv('NPROC_PER_NODE','4')
    monkeypatch.setenv('NODE_RANK','1')
    monkeypatch.setenv('MASTER_ADDR','training-node-0')
    cmd = reproduce.command(ROOT/'records/0001-subs-mask', 9, tmp_path, 'train/*.bin', 'val/*.bin')
    assert '--nnodes=2' in cmd and '--nproc_per_node=4' in cmd
    assert '--no_time_conditioning' in cmd
    assert cmd[cmd.index('--batch_size')+1] == '64'
    assert cmd[cmd.index('--seed')+1] == '9'
    assert cmd[cmd.index('--checkpoint_every_timed_minutes')+1] == '1'


def test_prefix_keeps_the_concatenated_token_order(tmp_path):
    inspected = []
    for i, values in enumerate([np.arange(6),np.arange(6,12)]):
        path = tmp_path/f'input-{i}.bin'
        header = np.zeros(256, dtype='<i4')
        header[:3] = [prefix.MAGIC,prefix.VERSION,len(values)]
        path.write_bytes(header.tobytes()+values.astype('<u2').tobytes())
        inspected.append((path,*prefix.inspect_shard(path)))
    # The last token of the first physical shard is the first token of the next;
    # a loader consuming four tokens per shard sees 0..7 exactly once.
    first = np.empty(5,dtype='<u2'); second = np.empty(5,dtype='<u2')
    prefix.copy_global_range(inspected,0,5,first)
    prefix.copy_global_range(inspected,4,5,second)
    np.testing.assert_array_equal(np.concatenate([first[:-1],second[:-1]]), np.arange(8))


def test_finish_line_does_not_depend_on_a_text_sampler():
    rule = json.loads((ROOT/'finish_line_rule.json').read_text())
    assert [k for k in rule if k.endswith('_min')] == ['hellaswag_accuracy_min']


def test_all_six_reported_crossings_have_complete_adjacent_checks():
    data = measurements()
    with (ROOT/'figures/crossing-confirmations.csv').open() as f:
        checks = list(csv.DictReader(f))
    confirmed = [r for r in checks if r['confirmation_complete'] == 'True']
    assert len(confirmed) == 6
    assert next(r for r in confirmed if r['recipe'] == 'duo_simple')['time_key'] == 'minute:043'
    for row in confirmed:
        minute = int(row['time_key'].split(':')[1])
        values = []
        for m in [minute-1,minute,minute+1]:
            selected = records.select(data, row['recipe'], f'minute:{m:03d}')
            assert {int(r['mc_samples']) for r in selected} == {128}
            assert {int(r['seed']) for r in selected} == set(range(1,11))
            values.append(statistics.mean(float(r['accuracy']) for r in selected))
        assert values[0] < .275 <= values[1] and values[2] >= .275
