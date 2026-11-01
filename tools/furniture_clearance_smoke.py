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
    kwargs = {'road_context': [service]} if 'road_context' in inspect.signature(powerlines.tile_lines).parameters else {}
    powerlines.tile_lines(frame, 0, 0, Md(), lambda la, lo: np.zeros(np.shape(la)), props, conductors, **kwargs)
    result = {'server': str(args.server.resolve()), 'long_crossing_signal': signal, 'crossing_sidewalk': sidewalk,
              'wire_above_bridge_kept': len(wires) == 1,
              'native_pylon_layout': {'towers': len(props), 'conductors': len(conductors),
                                      'towers_whose_pivots_obstruct_service_road': sum(abs(r[1]-100.) < 4. for r in props)},
              'scope': 'Synthetic source geometry through actual public placement/native layout APIs; not an in-game check.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
