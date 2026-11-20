import torch

from experiments.pulse_game import native_rehearsal_graph as game


def test_copy_starts_equal_but_has_reciprocal_paths_and_separate_fast_weights():
    graph=game.NativeGraph(731,"copy")
    torch.testing.assert_close(graph.weight[:20,:20],graph.weight[20:,20:],rtol=0,atol=0)
    assert torch.equal(graph.mask[:20,:20],graph.mask[20:,20:])
    assert int(graph.mask[:20,20:].sum())==int(graph.mask[20:,:20].sum())==80
    shuffled=game.NativeGraph(731,"permuted")
    torch.testing.assert_close(graph.weight[:20,20:],shuffled.weight[:20,20:],rtol=0,atol=0)
    torch.testing.assert_close(graph.weight[20:,:20],shuffled.weight[20:,:20],rtol=0,atol=0)
    assert graph.mask.sum()==shuffled.mask.sum()
    assert not torch.equal(graph.weight[20:,20:],shuffled.weight[20:,20:])


def test_blank_continuation_uses_only_current_neural_state():
    graph=game.NativeGraph(731,"copy").requires_grad_(False)
    e=game.scenes(2103,3,evaluation=True)
    _,state,_,visits=game.support(graph,e)
    for i in range(3):
        torch.testing.assert_close(game.recorded_favorable(visits,i),e.vectors[:,i]*e.laws[:,i,None])
    x=state.h.new_zeros((len(state.h),10))
    restored=game.State(*(v.clone() for v in vars(state).values()))
    for _ in range(12):restored,_=graph.tick(restored,x)
    blank,h,back=game.silence(graph,state)
    for field in ("h","trace","fast"):
        torch.testing.assert_close(getattr(restored,field),getattr(blank,field),rtol=0,atol=0)
    assert torch.count_nonzero(blank.fast[:,~graph.mask])==0
    assert back.shape==(12,len(state.h),20) and h.shape==(12,len(state.h),40)


def test_feedback_override_using_own_current_is_an_identity_within_float32_roundoff():
    graph=game.NativeGraph(733,"copy").requires_grad_(False)
    e=game.scenes(1971,7)
    _,s,_,_=game.support(graph,e)
    x=s.h.new_zeros((len(e.laws),10))
    expected,own=graph.tick(s,x)
    actual,_=graph.tick(s,x,feedback_override=own.clone())
    torch.testing.assert_close(expected.h,actual.h,rtol=0,atol=3e-7)
    torch.testing.assert_close(expected.fast,actual.fast,rtol=0,atol=3e-7)
