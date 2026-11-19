import torch
from experiments.action_return.finish_readout import replace_other


def test_conditional_policy_keeps_begin_gate_and_version_including_ties():
    generator=torch.Generator().manual_seed(991)
    logits=torch.randint(-2,3,(500,2,11),generator=generator).double()
    alternate=torch.randint(-3,3,(500,2,8),generator=generator).double()
    merged=replace_other(logits,alternate)
    before=logits.argmax(-1)
    in_begin=(before==1)|(before==2)
    collapsed=torch.cat((torch.zeros_like(alternate[...,:1]),alternate),-1).argmax(-1)
    desired=torch.where(in_begin,before,torch.where(collapsed>0,collapsed+2,0))
    assert torch.equal(merged.argmax(-1),desired)
