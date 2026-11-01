#!/usr/bin/env python3
"""Offline before/after consumer smoke; --server selects an unmodified or fixed server checkout.

Only synthetic geometry is used. No game, profile, network or map-cache access. JSON output is evidence of the
selected implementation, not a claim that any reported location has been rendered or driven successfully.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.server.resolve()))
    import furniture
    import geodesy
    import mapgeom
    import powerlines
    import roads

    own = (4., np.array([[0., -100.], [0., 100.]]))
    crossing = (4., np.array([[-5000., 6.5], [5000., 6.5]]))
    q = furniture.signal_spot(np.zeros(2), np.array([0., 1.]), np.array([1., 0.]), 4., 4., [own, crossing])
    signal = {'placed_xy': None if q is None else q.tolist(),
              'pivot_to_crossing_road_edge_m': None if q is None else abs(float(q[1])-6.5)-4.}

    spec = roads.Spec(8., 'paved', True, 'residential', False)
    fid = 'clearance-fixture'
    line = np.array([[-300., 50., 0.], [600., 50., 0.]])
    base = (fid, spec, line, 0.)
    side = 1 if furniture._h01(fid, 'side') < .5 else -1
    walks = [{'fid': fid, 'pts': roads.offset_polyline(line, side*5.25), 'side': side, 'w': 2.5,
              'caps': (False, False)}]
    box = (0., 0., 512., 256.)
    initial = furniture.place([base], walks, box)
    target = next(r for r in initial if r[0] == furniture.G_LAMP)
    cross = ('crossing', roads.Spec(8., 'paved', False, 'service', False),
             np.array([[target[1], -100., 0.], [target[1], 300., 0.]]), 0.)
    lamps = [r for r in furniture.place([base, cross], walks, box) if r[0] == furniture.G_LAMP]
    sidewalk = {'target': target, 'target_still_emitted': target in lamps, 'remaining_lamps': len(lamps),
                'min_pivot_to_crossing_edge_m': min((abs(r[1]-target[1])-4. for r in lamps), default=None)}

    deck = (np.array([[-100., 0., 8.], [100., 0., 8.]]), 5.5)
    high_wire = np.array([[-20., 0., 20.], [20., 0., 20.]])
    wires = [high_wire]
    furniture.under_decks([], [deck], wires)

    frame = geodesy.EnuFrame(39.2834, -76.5934)
    la, lo, _ = frame.enu_to_geodetic(np.array([100., 200.]), np.array([100., 100.]), 0.)
    infra = mapgeom.Features(
        layer='infrastructure', ids=np.array(['line']), kind=np.array([mapgeom.LINE], np.uint8),
        part_off=np.array([0, 1], np.int64), ring_off=np.array([0, 1], np.int64),
        coord_off=np.array([0, 2], np.int64), lonlat=np.column_stack([lo, la]),
        attrs={'class': np.array(['power_line'])}, rules={})

    class Md:
        def cell(self, key, layers, kind):
            return SimpleNamespace(layers={'infrastructure': infra} if 'infrastructure' in layers else {'roads': None})

    service = ('service', roads.Spec(8., 'paved', False, 'service', False),
               np.array([[100., 0., 0.], [100., 256., 0.]]), 0.)
    props, conductors = [], []
    params = inspect.signature(powerlines.tile_lines).parameters
    if 'clearance' in params:
        kwargs = {'clearance': furniture.RoadClearance([(4., service[2])], bounds=(0., 0., 256., 256.))}
    else:
        kwargs = {'road_context': [service]} if 'road_context' in params else {}
    powerlines.tile_lines(frame, 0, 0, Md(), lambda la, lo: np.zeros(np.shape(la)), props, conductors, **kwargs)

    stacked = [(np.array([[-100., 0., z], [100., 0., z]]), 5.5) for z in (8., 20.)]
    through_upper = np.array([[-20., 0., 20.], [20., 0., 20.]])
    above_all = through_upper + [0., 0., 10.]
    stacked_wires = [through_upper, above_all]
    furniture.under_decks([], stacked, stacked_wires)

    seam_id = 'seam-support'
    phase = furniture._h01(seam_id, 'pphase') * furniture.POLE_SPACING
    start = 250. - phase - 5 * furniture.POLE_SPACING
    seam_road = (seam_id, roads.Spec(8., 'paved', False, 'tertiary', False),
                 np.array([[start, 100., 0.], [600., 100., 0.]]), 0.)
    seam_cross = ('next-support', roads.Spec(8., 'paved', False, 'service', False),
                  np.array([[295., -100., 0.], [295., 300., 0.]]), 0.)
    whole_wires, left_wires, right_wires = [], [], []
    furniture.place([seam_road, seam_cross], [], (0., 0., 512., 256.), wires=whole_wires)
    furniture.place([seam_road, seam_cross], [], (0., 0., 256., 256.), wires=left_wires)
    furniture.place([seam_road, seam_cross], [], (256., 0., 512., 256.), wires=right_wires)
    split_spans_match = (sorted(tuple(w.ravel()) for w in whole_wires)
                        == sorted(tuple(w.ravel()) for w in left_wires + right_wires))

    dev = {'available': hasattr(roads, 'taper_widths')}
    if dev['available']:
        import navgraph

        def features(lines, widths):
            xy = np.concatenate([np.asarray(p, np.float64) for p in lines])
            la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.)
            n = len(lines)
            return mapgeom.Features(
                layer='roads', ids=np.array([f'r{i}' for i in range(n)]), kind=np.full(n, mapgeom.LINE, np.uint8),
                part_off=np.arange(n + 1, dtype=np.int64), ring_off=np.arange(n + 1, dtype=np.int64),
                coord_off=np.cumsum([0] + [len(p) for p in lines]).astype(np.int64), lonlat=np.column_stack([lo, la]),
                attrs={'class': np.array(['primary'] * n), 'osm_width': np.asarray(widths, np.float64)}, rules={})

        f = features([[(0., 0.), (0., -100.)], [(0., 0.), (100., 0.)], [(0., 0.), (-100., 0.)],
                      [(-100., 0.), (100., 0.)]], [8., 8., 8., 80.])
        f.rules['flags'] = mapgeom.RuleTable(np.array([3], np.int32), np.array(['is_bridge']),
                                            np.array([0.], np.float32), np.array([1.], np.float32))
        f.rules['connectors'] = mapgeom.RuleTable(np.arange(3, dtype=np.int32), np.array(['J'] * 3),
                                                 np.zeros(3, np.float32), np.zeros(3, np.float32))
        urban = lambda la, lo: np.ones(np.shape(la), bool)
        specs = roads.road_specs(f, urban)
        static_roads = []
        for i in range(3):
            ll = np.asarray(f.parts(i)[0][0], np.float64)
            x, y, _ = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], 0.)
            z = frame.geodetic_to_enu(ll[:, 1], ll[:, 0], np.full(len(ll), 10.))[2]
            static_roads.append((str(f.ids[i]), specs[i], np.column_stack([x, y, z]), float(ll[:, 0].mean())))
        xy = np.array([[-100., 0.], [100., 0.]])
        z = np.full(2, 50.)
        mapped_signals = np.zeros((1, 2))
        static_heads = furniture.junction_props(
            static_roads, (-50., -50., 50., 50.), signals=mapped_signals, decks=[(np.column_stack([xy, z]), 40.)])
        graph = navgraph.build(frame, [f], lambda la, lo: np.full(np.shape(la), 10.), (0., 0.),
                               urban_fn=urban, decks={'r3': [(xy, z)]}, signals=mapped_signals)
        heads = [a[7:11] for junction in graph['signals'] for a in junction['app']]
        dev['signals_below_high_bridge'] = {
            'static_heads': len(static_heads), 'switchable_heads': sum(h[0] is not None for h in heads),
            'all_transforms_match': sorted(tuple(h) for h in heads if h[0] is not None)
                                    == sorted(tuple(r[1:5]) for r in static_heads)}

        narrowed = (fid, spec._replace(width=12.), line, 0.)
        kwargs = {}
        if 'clearance' in inspect.signature(furniture.place).parameters:
            kwargs['clearance'] = furniture.RoadClearance([(4., line)], bounds=box)
        narrow_rows = furniture.place([narrowed], walks, box, **kwargs)
        dev['narrowed_sidewalk'] = {'nominal_width_m': 12., 'actual_width_m': 8.,
                                   'shared_context_available': bool(kwargs),
                                   'lamps': sum(r[0] == furniture.G_LAMP for r in narrow_rows)}
    result = {'server': str(args.server.resolve()), 'long_crossing_signal': signal, 'crossing_sidewalk': sidewalk,
              'wire_above_bridge_kept': len(wires) == 1,
              'stacked_decks': {'wire_through_upper_deck_retained': any(np.array_equal(w, through_upper) for w in stacked_wires),
                                'wire_above_both_decks_retained': any(np.array_equal(w, above_all) for w in stacked_wires)},
              'split_tile_span': {'matches_whole_tile': split_spans_match, 'whole_spans': len(whole_wires),
                                  'split_spans': len(left_wires) + len(right_wires)},
              'dev_context': dev,
              'native_pylon_layout': {'towers': len(props), 'conductors': len(conductors),
                                      'towers_whose_pivots_obstruct_service_road': sum(abs(r[1]-100.) < 4. for r in props)},
              'scope': 'Synthetic source geometry through actual placement/native layout APIs; not an in-game check.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
