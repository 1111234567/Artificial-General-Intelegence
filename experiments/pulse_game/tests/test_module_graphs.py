import numpy as np

from experiments.pulse_game import module_graphs as graph
from experiments.pulse_game.free_graph_model import strongly_connected


def test_cycle_birth_and_inter_ring_edges_on_multiple_starts():
    for seed in range(64,80):
        dense=graph.sample(seed)
        sparse=graph.sample(seed,"minimum")
        assert dense.within==sparse.within and dense.rings==sparse.rings
        internal=np.asarray(dense.within)
        for col in (0,1):
            np.testing.assert_array_equal(np.bincount(internal[:,col]),np.ones(32))
        group={node:i for i,ring in enumerate(dense.rings) for node in ring}
        assert set(group)==set(range(32)) and min(map(len,dense.rings))>=2 and len(dense.rings)>=2
        for candidate in (dense,sparse):
            arcs=candidate.arcs();info=candidate.record()
            assert len(set(map(tuple,arcs)))==len(arcs)
            assert all(group[s]!=group[t] for s,t in candidate.bridges)
            assert min(info["ring_out_degree"])>=1 and min(info["ring_in_degree"])>=1
        assert len(dense.bridges)==32
        assert len(sparse.bridges)==len(dense.rings)
        assert strongly_connected(32,sparse.arcs())
        assert len(graph.single_sparse(seed))==len(sparse.arcs())


def test_ring_level_floor_does_not_imply_one_connected_network():
    # Four isolated two-node cycles joined in two closed pairs of cycles.
    internal=((0,1),(1,0),(2,3),(3,2),(4,5),(5,4),(6,7),(7,6))
    graph_pair=graph.ModuleGraph(((0,1),(2,3),(4,5),(6,7)),internal,((1,2),(3,0),(5,6),(7,4)))
    info=graph_pair.record()
    assert info["ring_in_degree"]==info["ring_out_degree"]==[1]*4
    assert not info["strongly_connected"]
