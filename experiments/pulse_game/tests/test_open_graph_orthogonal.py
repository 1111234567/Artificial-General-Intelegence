import numpy as np

from experiments.pulse_game import open_graph_orthogonal as ortho


def test_new_direction_is_orthogonal_and_has_matched_public_inputs():
    roots=[56,78,56,78]
    for root in roots:
        a,b=ortho.private_vectors(root)
        np.testing.assert_allclose(a @ b,0,atol=1e-12)
    for phase in range(6):
        foods,noise,sham=ortho.make_menu(roots,phase,7)
        np.testing.assert_array_equal(foods.chemicals[:2],foods.chemicals[2:])
        np.testing.assert_array_equal(foods.body.wind[:2],foods.body.wind[2:])
        np.testing.assert_array_equal(noise[:2],noise[2:])
        np.testing.assert_array_equal(sham[:2],sham[2:])
        sign = -1 if phase in (2,3) else 1
        np.testing.assert_array_equal(foods.realized[:2],sign*foods.realized[2:])
        if sign==-1: np.testing.assert_allclose(foods.probability[:2]+foods.probability[2:],1.,atol=2e-7)


def test_silent_branch_has_no_credit_or_parameter_edits():
    seeds=[12,13,12,13]
    graph=ortho.net.Graph([1235]*4)
    menu,noise,_=ortho.make_menu(seeds,3,5)
    old_w=graph.weight.copy()
    archive,events=ortho.live(graph,menu,noise,None,False)
    np.testing.assert_array_equal(graph.weight,old_w)
    assert (events==-1).all() and not graph.receipts.any()
    np.testing.assert_array_equal(archive[:2,:,:2],archive[2:,:,:2])
    np.testing.assert_array_equal(archive[:2,:,2],-archive[2:,:,2])
