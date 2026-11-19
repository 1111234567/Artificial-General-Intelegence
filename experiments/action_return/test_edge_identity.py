import torch
from experiments.action_return.edge_identity import silence
from experiments.continuity.run import state_digest
from experiments.undirected.model import UndirectedModel

def test_exact_identity_silencing_removes_both_physical_directions_and_preserves_source():
    model=UndirectedModel(input_size=2,n_jobs=1,n_actions=3,width=4,input_scale=[1,1]).double()
    with torch.no_grad():model.edge_shared.copy_(torch.tensor([1.,2.,3.,4.,5.,6.],dtype=torch.float64))
    before=state_digest(model.state_dict());copy,stats=silence(model,[1,4])
    old=model.recurrent_matrix();new=copy.recurrent_matrix();selected=model.pair_index[:,[1,4]]
    expected=old.clone();expected[selected[0],selected[1]]=0;expected[selected[1],selected[0]]=0
    assert torch.equal(new,expected)
    assert state_digest(model.state_dict())==before
    assert torch.equal(copy.edge_shared.detach(),torch.tensor([1.,0.,3.,4.,0.,6.],dtype=torch.float64))
    assert abs(stats['l2']-float((old-new).detach().norm()))<1e-14
    assert stats['source_nonzero_selected']==2
