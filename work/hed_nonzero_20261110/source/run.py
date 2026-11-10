"""Select nonzero HED+RBCM sensitivity arms from existing BIPED counts only."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_CODE = ROOT / 'work/hed_rbcm_20261110/source/run.py'
PREVIOUS_OUT = ROOT / 'outputs/RBCM_HED_AUG_20261110/source'
OUT = ROOT / 'outputs/RBCM_HED_NONZERO_20261110/source'

spec = importlib.util.spec_from_file_location('previous_hed_rbcm_source', PREVIOUS_CODE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
hed = previous.hed


def select(options, configs, source_f, grid):
    selected = {}
    for family, choices in options.items():
        ranked = [(score, -configs[name]['multiplier'], -threshold, name)
                  for name in choices for score, threshold in zip(source_f[name], grid)]
        value, _, negative_threshold, name = max(ranked)
        threshold = -negative_threshold
        selected[family] = dict(config=configs[name], threshold=threshold, selected_name=name,
                                F_source_development=value, threshold_on_grid_boundary=threshold in (grid[0], grid[-1]))
    return selected


def run():
    original, configs, entries = previous.verify_inputs()
    original_protocol = PREVIOUS_OUT / 'protocol.json'
    original_freeze = PREVIOUS_OUT / 'source_frozen.json'
    if hed.read(original_protocol) != original:
        raise ValueError('Previous BIPED source protocol could not be reconstructed')
    original_protocol_sha = hed.digest(original_protocol)
    historical = hed.read(original_freeze)
    if historical != previous.choose(entries, original, original_protocol_sha, configs):
        raise ValueError('Historical source selection does not match the validated BIPED source counts')
    if historical['target_score_accessed_for_selection']:
        raise ValueError('Historical source selection was not target-blind')

    grid = original['source_threshold_grid']
    nonzero = (0.5, 1.0)
    options = {family: [name for name in previous.OPTIONS[family] if configs[name]['multiplier'] in nonzero]
               for family in previous.FAMILIES}
    if any(tuple(configs[name]['multiplier'] for name in choices) != nonzero for choices in options.values()):
        raise ValueError('Each sensitivity family must have the same two nonzero amplitudes')
    original_frozen_sha = hed.digest(original_freeze)
    inherited = {name: historical['selected'][name] for name in (previous.PLAIN, previous.BASE, previous.LEGACY)}
    protocol = dict(
        scope='Source-only, nonzero-amplitude sensitivity with the existing BIPED development counts',
        source=historical['source'], source_entries=entries, source_threshold_grid=grid,
        sensitivity_amplitudes=list(nonzero), sensitivity_options=options, inherited_controls=inherited,
        selection='Max pooled F, then lower m, then lower threshold. Original HED, calibration and legacy are inherited.',
        target_score_accessed_for_selection=False, prior_source_protocol_sha256=original_protocol_sha,
        prior_source_frozen_sha256=original_frozen_sha, hed_provenance=original['hed_provenance'],
        runner_sha256=hed.digest(__file__),
    )
    binding = hed.freeze(OUT / 'protocol.json', protocol)
    source_f = historical['source_F_by_threshold']
    selected = {**select(options, configs, source_f, grid), **inherited}
    result = dict(
        source=historical['source'], selected=selected, source_n_images=len(entries),
        protocol_sha256=binding, prior_source_frozen_sha256=original_frozen_sha,
        target_score_accessed_for_selection=False, source_F_by_threshold=source_f,
    )
    hed.freeze(OUT / 'source_frozen.json', result)
    total = len(entries) * len(previous.MEASURED)
    hed.save(OUT / 'status.json', dict(status='complete', phase='source_frozen', completed=total, total=total,
                                      eta_seconds=0, updated_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
    for name, arm in selected.items():
        print(f'{name}: {arm["selected_name"]} τ={arm["threshold"]:.2f} F={arm["F_source_development"]:.9f}')


if __name__ == '__main__':
    run()
