"""Structural check of a COCOA smoke inference output.

Compares the source ROOT entries [entry_start, entry_stop), the Mean Shift
segmented file (EventTree) and the merged prediction file (event_tree) and
writes a JSON report. output_valid is 1 only when every check passes.

Run from the repository root:
    python smoke/check_cocoa_smoke.py --source ... --entry-start 0 --entry-stop 10 \
        --seg ... --pred ... --config-v ... --output report.json
"""

import argparse
import hashlib
import json
import os
import sys

import awkward as ak
import numpy as np
import uproot
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from hgpflow_v2.utility.helper_dicts import pdgid_class_dict  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for block in iter(lambda: fp.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def is_finite(arr):
    flat = np.asarray(ak.flatten(arr, axis=None), dtype=float)
    return bool(np.isfinite(flat).all()), int(flat.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-tree', default='Out_Tree')
    parser.add_argument('--entry-start', type=int, required=True)
    parser.add_argument('--entry-stop', type=int, required=True)
    parser.add_argument('--event-number-offset', type=int, default=0)
    parser.add_argument('--seg', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--config-v', required=True)
    parser.add_argument('--ind-threshold-loose', type=float, default=-0.1)
    parser.add_argument('--ind-threshold-plot', type=float, default=0.48)
    parser.add_argument('--hash-source', action='store_true',
                        help='also compute SHA-256 of the (large) source file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    max_particles = yaml.safe_load(open(args.config_v))['max_particles']
    checks = {}
    failures = []

    def check(name, ok, detail=None):
        checks[name] = {'ok': bool(ok), 'detail': detail}
        if not ok:
            failures.append(name)

    # source entries
    src_tree = uproot.open(args.source)[args.source_tree]
    src = src_tree.arrays(['track_theta', 'topo_bary_eta', 'particle_pt'], library='ak',
                          entry_start=args.entry_start, entry_stop=args.entry_stop)
    n_src = len(src)
    check('source_entry_count', n_src == args.entry_stop - args.entry_start,
          {'read': n_src, 'requested': args.entry_stop - args.entry_start})
    expected_events = np.arange(n_src) + args.entry_start + args.event_number_offset
    src_ntrack = ak.to_numpy(ak.num(src.track_theta))
    src_ntopo = ak.to_numpy(ak.num(src.topo_bary_eta))
    src_npart = ak.to_numpy(ak.num(src.particle_pt))
    # the splitter skips events without any track or topocluster
    events_with_nodes = expected_events[(src_ntrack + src_ntopo) > 0]

    # segmented file: every node of every source event must be in some partition
    seg = uproot.open(args.seg)['EventTree'].arrays(
        ['eventNumber', 'track_eta', 'topo_eta', 'particle_pdgid'], library='ak')
    seg_ntrack = ak.to_numpy(ak.num(seg.track_eta))
    seg_ntopo = ak.to_numpy(ak.num(seg.topo_eta))
    seg_npart = ak.to_numpy(ak.num(seg.particle_pdgid))
    seg_ev = ak.to_numpy(seg.eventNumber)
    check('seg_event_set', set(np.unique(seg_ev).tolist()) == set(events_with_nodes.tolist()),
          {'seg': sorted(np.unique(seg_ev).tolist()), 'expected': events_with_nodes.tolist()})

    flat_cls = np.array([pdgid_class_dict[int(x)] for x in ak.flatten(seg.particle_pdgid)], dtype=int)
    seg_cls = ak.unflatten(flat_cls, ak.num(seg.particle_pdgid))
    seg_ntruth = ak.to_numpy(ak.sum((seg_cls >= 0) & (seg_cls <= 4), axis=1))

    per_event = []
    node_mismatch = []
    for i, ev in enumerate(expected_events):
        m = seg_ev == ev
        row = {
            'event_number': int(ev),
            'source_ntrack': int(src_ntrack[i]), 'source_ntopo': int(src_ntopo[i]),
            'source_nparticle': int(src_npart[i]),
            'n_partitions': int(m.sum()),
            'seg_ntrack': int(seg_ntrack[m].sum()), 'seg_ntopo': int(seg_ntopo[m].sum()),
            'seg_nparticle': int(seg_npart[m].sum()),
            'expected_ntruth': int(seg_ntruth[m].sum()),
            'max_partition_nparticle': int(seg_npart[m].max()) if m.any() else 0,
        }
        if row['seg_ntrack'] != row['source_ntrack'] or row['seg_ntopo'] != row['source_ntopo']:
            node_mismatch.append(int(ev))
        per_event.append(row)
    check('seg_nodes_cover_source', not node_mismatch, {'mismatched_events': node_mismatch})
    over = [r['event_number'] for r in per_event if r['max_partition_nparticle'] > max_particles]
    check('seg_partition_nparticle_le_max', not over,
          {'max_particles': max_particles, 'events_over': over})

    # prediction file
    pred = uproot.open(args.pred)['event_tree'].arrays(library='ak')
    pred_ev = ak.to_numpy(pred.event_number)
    check('pred_events_unique_sorted', bool(np.all(np.diff(pred_ev) > 0)), pred_ev.tolist())
    check('pred_event_set', set(pred_ev.tolist()) == set(events_with_nodes.tolist()),
          {'pred': pred_ev.tolist(), 'expected': events_with_nodes.tolist()})

    groups = {
        'truth': ['pt', 'eta', 'phi', 'class'],
        'proxy': ['pt', 'eta', 'phi'],
        'hgpflow': ['pt', 'eta', 'phi', 'class'],
    }
    length_bad = []
    for g, vs in groups.items():
        n = ak.to_numpy(pred[f'n{g}'])
        for v in vs:
            if not np.array_equal(ak.to_numpy(ak.num(pred[f'{g}_{v}'])), n):
                length_bad.append(f'{g}_{v}')
    n_hg = ak.to_numpy(pred.nhgpflow)
    if not (np.array_equal(ak.to_numpy(pred.nproxy), n_hg)
            and np.array_equal(ak.to_numpy(pred.npred_ind), n_hg)
            and np.array_equal(ak.to_numpy(ak.num(pred.pred_ind)), n_hg)):
        length_bad.append('proxy/hgpflow/pred_ind count mismatch')
    check('pred_array_lengths_consistent', not length_bad, length_bad)

    finite = {}
    for name in pred.fields:
        if name.startswith('n') or name == 'event_number':
            continue
        ok, size = is_finite(pred[name])
        finite[name] = {'finite': ok, 'size': size}
    check('pred_values_finite', all(x['finite'] for x in finite.values()), finite)

    ind = np.asarray(ak.flatten(pred.pred_ind), dtype=float)
    check('pred_ind_in_unit_interval', bool(((ind >= 0) & (ind <= 1)).all()),
          {'min': float(ind.min()), 'max': float(ind.max())})
    hg_cls = np.unique(np.asarray(ak.flatten(pred.hgpflow_class)))
    tr_cls = np.unique(np.asarray(ak.flatten(pred.truth_class)))
    check('class_values_valid', set(hg_cls.tolist()) <= {0, 1, 2, 3, 4} and set(tr_cls.tolist()) <= {0, 1, 2, 3, 4},
          {'hgpflow': hg_cls.tolist(), 'truth': tr_cls.tolist()})

    # every partition was inferred: with the loose threshold below the sigmoid range
    # each partition contributes exactly max_particles slots
    by_ev = {r['event_number']: r for r in per_event}
    slot_bad, truth_bad = [], []
    for j, ev in enumerate(pred_ev):
        r = by_ev.get(int(ev))
        if r is None:
            continue
        r['pred_nhgpflow'] = int(n_hg[j])
        r['pred_ntruth'] = int(pred.ntruth[j])
        r['pred_n_above_plot_threshold'] = int(ak.sum(pred.pred_ind[j] > args.ind_threshold_plot))
        if args.ind_threshold_loose < 0 and n_hg[j] != max_particles * r['n_partitions']:
            slot_bad.append(int(ev))
        if pred.ntruth[j] != r['expected_ntruth']:
            truth_bad.append(int(ev))
    check('pred_slots_match_partitions', not slot_bad,
          {'rule': 'nhgpflow == max_particles * n_partitions (loose threshold < 0)', 'events': slot_bad}
          if args.ind_threshold_loose < 0 else 'skipped: loose threshold >= 0')
    check('pred_truth_matches_seg', not truth_bad,
          {'rule': 'ntruth == partition particles with class 0-4', 'events': truth_bad})

    report = {
        'output_valid': int(not failures),
        'failures': failures,
        'inputs': {
            'source': {'path': os.path.abspath(args.source), 'tree': args.source_tree,
                       'entry_start': args.entry_start, 'entry_stop': args.entry_stop,
                       'num_entries_total': int(src_tree.num_entries),
                       'size_bytes': os.path.getsize(args.source),
                       'sha256': sha256(args.source) if args.hash_source else None},
            'seg': {'path': os.path.abspath(args.seg), 'size_bytes': os.path.getsize(args.seg),
                    'sha256': sha256(args.seg), 'entries': len(seg)},
            'pred': {'path': os.path.abspath(args.pred), 'size_bytes': os.path.getsize(args.pred),
                     'sha256': sha256(args.pred), 'entries': len(pred)},
        },
        'summary': {
            'source_events': n_src,
            'events_with_nodes': len(events_with_nodes),
            'partitions': len(seg),
            'truth_particles': int(ak.sum(pred.ntruth)),
            'hgpflow_slots': int(n_hg.sum()),
            'hgpflow_above_plot_threshold': int(np.sum(ind > args.ind_threshold_plot)),
            'ind_threshold_plot': args.ind_threshold_plot,
        },
        'per_event': per_event,
        'checks': checks,
    }
    with open(args.output, 'w') as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps({'output_valid': report['output_valid'], 'failures': failures,
                      'summary': report['summary']}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
