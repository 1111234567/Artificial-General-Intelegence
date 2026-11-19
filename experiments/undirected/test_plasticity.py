"""Equal-budget pair probes, independent acceptance, and complete recovery."""
from copy import deepcopy
import unittest

import torch

from experiments.undirected.model import UndirectedModel
from experiments.undirected.plasticity import MODES, PairConfig, PairOperator, _digest


def fixture(mode="both", *, window=1, candidates=1, tiny=True, direction="undirected", device="cpu"):
    model = UndirectedModel(input_size=2, n_jobs=1, n_actions=2, width=6, mode=direction,
                             initialization_seed=19).double().to(device)
    mask, _ = model.initialize_birth(71, 83, degree=2, gain=.7)
    mask = mask.cpu()
    first = mask.nonzero().flatten()[0]
    with torch.no_grad():
        model.edge_shared[first] = 1e-5 if tiny else 1.2
    operator = PairOperator(model, mask, PairConfig(mode=mode, window_updates=window, candidates=candidates, seed=617))
    optimizer = torch.optim.Adam(model.parameters(), lr=.05, weight_decay=0.)
    return model, operator, optimizer


def raw_gradients(model, mask):
    result = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
    result["edge_shared"][~mask.to(model.edge_shared.device)] = -1.
    if "edge_skew" in result:
        result["edge_skew"].fill_(.1)
    return result


def adam_step(model, optimizer, gradients):
    optimizer.zero_grad(set_to_none=True)
    for name, value in model.named_parameters():
        value.grad = gradients[name].clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def step(model, operator, optimizer, probe=None):
    gradients = raw_gradients(model, operator.mask)
    adam_step(model, optimizer, gradients)
    return operator.after_optimizer(optimizer, gradients, 1., probe=probe)


def scripted(values, model=None):
    count = [0]

    def probe():
        if model is not None:
            matrix = model.recurrent_matrix()
            assert torch.equal(matrix, matrix.T)
            assert not bool(matrix.diag().ne(0).any())
            assert not torch.is_grad_enabled()
        value = values[count[0]]
        count[0] += 1
        if isinstance(value, BaseException):
            raise value
        live, history = value if isinstance(value, tuple) else (value, value)
        return {"live": live, "history": history, "mixed": .5 * (live + history)}

    return probe, count


class PairOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_configuration_and_untied_fixed_boundary(self):
        for change in (dict(mode="random"), dict(window_updates=0), dict(candidates=True), dict(seed=-1)):
            with self.assertRaises(ValueError):
                PairConfig(**change)
        model, operator, optimizer = fixture("fixed", direction="untied")
        step(model, operator, optimizer)
        self.assertEqual(operator.updates, 1)
        self.assertEqual(int(model.edge_shared[~operator.mask].count_nonzero()), 0)
        self.assertEqual(int(model.edge_skew[~operator.mask].count_nonzero()), 0)
        for mode in MODES[1:]:
            with self.assertRaises(ValueError):
                PairOperator(model, operator.mask, PairConfig(mode=mode))

    def test_four_policies_have_equal_four_call_budget_and_independent_flags(self):
        proposals = []
        for mode in MODES[1:]:
            model, operator, optimizer = fixture(mode)
            first = step(model, operator, optimizer)
            self.assertIsNone(first["probe"])
            proposals.append(deepcopy(operator.pending))
            closure, count = scripted([10., 8., 15., 7.], model)
            event = step(model, operator, optimizer, closure)
            self.assertEqual(count[0], 4)
            self.assertEqual(event["probe_calls"], 4)
            self.assertEqual(event["probe"]["birth_accepted"], mode in ("grow", "both"))
            self.assertEqual(event["probe"]["fade_accepted"], mode in ("retire", "both"))
            self.assertEqual(event["births"], int(mode in ("grow", "both")))
            self.assertEqual(event["deaths"], int(mode in ("retire", "both")))
            expected_baseline = 8. if mode in ("grow", "both") else 10.
            self.assertEqual(event["probe"]["fade_baseline"]["mixed"], expected_baseline)
            model.validate_eligibility(operator.mask)
        self.assertTrue(all(_digest(proposal) == _digest(proposals[0]) for proposal in proposals))

    def test_fixed_probe_keeps_actual_parameter_and_adam_trajectory_identical(self):
        fixed = fixture("fixed", tiny=False)
        probing = fixture("fixed_probe", tiny=False)
        for _ in range(4):
            a = step(*fixed)
            closure, count = scripted([10., 2., 1., 3.], probing[0])
            b = step(*probing, closure)
            self.assertIsNone(a["probe"])
            self.assertEqual(count[0], 0 if fixed[1].updates == 1 else 4)
            self.assertEqual(_digest(fixed[0].state_dict()), _digest(probing[0].state_dict()))
            self.assertEqual(_digest(fixed[2].state_dict()), _digest(probing[2].state_dict()))
            self.assertEqual(a["eligible_pairs"], b["eligible_pairs"])
        self.assertEqual(fixed[1].probe_calls, 0)
        self.assertEqual(probing[1].probe_calls, 12)

    def test_growth_and_retirement_are_not_equal_count_replacement(self):
        for mode in ("grow", "retire", "both"):
            model, operator, optimizer = fixture(mode, candidates=2)
            initial = int(operator.mask.sum())
            step(model, operator, optimizer)
            fade = operator.pending["fade_indices"]
            with torch.no_grad():
                # Both fade candidates remain eligible, but only the first is
                # small enough for this finite trial to reach exactly zero.
                model.edge_shared[fade[1]] = 1.3
            closure, _ = scripted([10., 8., 15., 7.], model)
            event = step(model, operator, optimizer, closure)
            births = 2 if mode in ("grow", "both") else 0
            deaths = 1 if mode in ("retire", "both") else 0
            self.assertEqual((event["births"], event["deaths"]), (births, deaths))
            self.assertEqual(int(operator.mask.sum()), initial + births - deaths)

    def test_full_ablation_is_diagnostic_and_half_fade_is_not_deletion(self):
        model, operator, optimizer = fixture("retire", tiny=False)
        step(model, operator, optimizer)
        index = int(operator.pending["fade_indices"][0])
        original = float(model.edge_shared[index].detach())

        def loss():
            theta = float(model.edge_shared[index])
            value = (theta - .6 * original) ** 2
            return {"live": value, "history": value, "mixed": value}

        event = step(model, operator, optimizer, loss)
        probe = event["probe"]
        self.assertGreater(probe["full_ablation_damage"]["mixed"], 0.)
        self.assertLess(probe["half_fade_damage"]["mixed"], 0.)
        self.assertTrue(probe["fade_accepted"])
        self.assertEqual(event["deaths"], 0)
        self.assertTrue(bool(operator.mask[index]))
        self.assertAlmostEqual(float(model.edge_shared[index].detach()), original / 2)
        # Even a favorable full deletion never replaces the prescribed finite
        # acceptance test when that finite test itself fails.
        model, operator, optimizer = fixture("retire", tiny=False)
        step(model, operator, optimizer)
        closure, _ = scripted([10., 10., 2., 12.], model)
        event = step(model, operator, optimizer, closure)
        self.assertLess(event["probe"]["full_ablation_damage"]["mixed"], 0.)
        self.assertFalse(event["probe"]["fade_accepted"])
        self.assertEqual(event["deaths"], 0)

    def test_mixed_acceptance_can_increase_history_loss(self):
        model, operator, optimizer = fixture("retire", tiny=False)
        step(model, operator, optimizer)
        index = int(operator.pending["fade_indices"][0])
        old = float(model.edge_shared[index].detach())

        def loss():
            value = float(model.edge_shared[index])
            live, history = value ** 2, (value - 1.25 * old) ** 2
            return {"live": live, "history": history, "mixed": .5 * (live + history)}

        event = step(model, operator, optimizer, loss)
        self.assertTrue(event["probe"]["fade_accepted"])
        self.assertGreater(event["probe"]["half_fade_damage"]["history"], 0.)
        self.assertLess(event["probe"]["half_fade_damage"]["mixed"], 0.)

    def test_atomic_shared_pair_probes_and_moments_are_never_reset(self):
        model, operator, optimizer = fixture("both")
        step(model, operator, optimizer)
        gradients = raw_gradients(model, operator.mask)
        adam_step(model, optimizer, gradients)
        moments = deepcopy(optimizer.state_dict())
        closure, _ = scripted([10., 8., 15., 7.], model)
        result = operator.after_optimizer(optimizer, gradients, 1., closure)
        self.assertEqual(_digest(moments), _digest(optimizer.state_dict()))
        self.assertEqual(len(result["probe"]["actual_birth_indices"]), 1)
        self.assertEqual(len(result["probe"]["actual_retire_indices"]), 1)
        for index in result["probe"]["actual_birth_indices"]:
            u, v = model.pair_index[:, index]
            matrix = model.recurrent_matrix().detach()
            self.assertNotEqual(float(matrix[u, v]), 0.)
            self.assertEqual(float(matrix[u, v]), float(matrix[v, u]))
        self.assertEqual(float(optimizer.state[model.edge_shared]["step"]), 2.)
        self.assertGreater(float(optimizer.state[model.edge_shared]["exp_avg"][~operator.initial_mask.to(model.edge_shared.device)].norm()), 0.)

    def test_probe_failure_rolls_back_after_adam_boundary_and_clears_closure(self):
        model, operator, optimizer = fixture("both")
        step(model, operator, optimizer)
        gradients = raw_gradients(model, operator.mask)
        adam_step(model, optimizer, gradients)
        operator.project_and_validate()
        model_state, optimizer_state = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
        operator_state = operator.state_dict()
        closure, count = scripted([10., 8., RuntimeError("probe failure")], model)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            operator.after_optimizer(optimizer, gradients, 1., closure)
        self.assertEqual(count[0], 3)
        self.assertEqual(_digest(model.state_dict()), _digest(model_state))
        self.assertEqual(_digest(optimizer.state_dict()), _digest(optimizer_state))
        self.assertEqual(_digest(operator.state_dict()), _digest(operator_state))
        self.assertIsNone(operator._probe)
        good, _ = scripted([10., 8., 15., 7.], model)
        self.assertEqual(operator.after_optimizer(optimizer, gradients, 1., good)["updates"], 2)

    def test_complete_ranking_vectors_reconstruct_selected_coordinates(self):
        model, operator, optimizer = fixture("both", candidates=2)
        step(model, operator, optimizer)
        proposal = operator.pending
        mean, delta, weight, mask = (proposal[name] for name in ("gmean", "delta_all", "weight_at_proposal", "mask_at_proposal"))
        births = (~mask).nonzero().flatten()
        exits = mask.nonzero().flatten()
        birth_order = torch.argsort((-mean * delta.double())[births], descending=True, stable=True)[:2]
        fade_order = torch.argsort((-mean * weight.double())[exits], stable=True)[:2]
        self.assertTrue(torch.equal(proposal["birth_indices"], births[birth_order]))
        self.assertTrue(torch.equal(proposal["fade_indices"], exits[fade_order]))
        self.assertEqual(proposal["execute_update"], proposal["generated_update"] + 1)

    def test_accepted_multi_pair_score_order_differs_from_canonical_event_order(self):
        # Regression for the width-80 mechanical smoke: four accepted birth
        # candidates were stored in score order, but committed mask events
        # were canonical.  Both representations must restore without changing
        # which proposal-ordered value belongs to each actual pair.
        model, operator, optimizer = fixture("both", candidates=4)
        gradients = raw_gradients(model, operator.mask)
        inactive = (~operator.mask).nonzero().flatten()
        active = operator.mask.nonzero().flatten()
        with torch.no_grad():
            model.edge_shared[active] = torch.linspace(1e-5, 3e-5, len(active), dtype=torch.double)
        gradients["edge_shared"][inactive] = -torch.linspace(1., 3., len(inactive), dtype=torch.double)
        gradients["edge_shared"][active] = torch.linspace(1., 3., len(active), dtype=torch.double) * 1e-14
        adam_step(model, optimizer, gradients)
        operator.after_optimizer(optimizer, gradients, 1.)
        proposal_indices = operator.pending["birth_indices"].tolist()
        fade_indices = operator.pending["fade_indices"].tolist()
        self.assertNotEqual(proposal_indices, sorted(proposal_indices))
        self.assertNotEqual(fade_indices, sorted(fade_indices))
        closure, _ = scripted([10., 8., 15., 7.], model)
        event = step(model, operator, optimizer, closure)["probe"]
        self.assertEqual(event["actual_birth_indices"], sorted(proposal_indices))
        self.assertEqual(event["birth_indices"], proposal_indices)
        self.assertEqual(event["birth_final_weights"], event["birth_trial_weights"])
        self.assertEqual(event["actual_retire_indices"], sorted(fade_indices))
        restored = PairOperator.from_state(model, operator.state_dict())
        self.assertEqual(_digest(restored.state_dict()), _digest(operator.state_dict()))

    def _roundtrip(self, device, window):
        model, operator, optimizer = fixture("both", window=window, device=device)
        step(model, operator, optimizer)
        state = operator.state_dict()
        restored_model = UndirectedModel(**model.config()).double().to(device)
        restored_model.load_state_dict(deepcopy(model.state_dict()))
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=.05, weight_decay=0.)
        restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        restored = PairOperator.from_state(restored_model, state)
        self.assertEqual(_digest(restored.state_dict()), _digest(state))
        for _ in range(4):
            a, _ = scripted([10., 8., 15., 7.], model)
            b, _ = scripted([10., 8., 15., 7.], restored_model)
            actual, expected = step(model, operator, optimizer, a), step(restored_model, restored, restored_optimizer, b)
            self.assertEqual(_digest(actual), _digest(expected))
            self.assertEqual(_digest(model.state_dict()), _digest(restored_model.state_dict()))
            self.assertEqual(_digest(operator.state_dict()), _digest(restored.state_dict()))
            self.assertEqual(_digest(optimizer.state_dict()), _digest(restored_optimizer.state_dict()))
            PairOperator.from_state(model, operator.state_dict())

    def test_pending_partial_window_and_lifetime_traits_resume_exactly(self):
        self._roundtrip("cpu", 1)
        self._roundtrip("cpu", 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_pair_pending_resume(self):
        self._roundtrip("cuda", 2)

    def test_rehashed_corrupt_schema_traits_ranking_and_clocks_are_rejected(self):
        model, operator, optimizer = fixture("both")
        step(model, operator, optimizer)
        closure, _ = scripted([10., 8., 15., 7.], model)
        step(model, operator, optimizer, closure)
        original = operator.state_dict()
        mutations = [lambda s: s.update(format="foreign-schema"), lambda s: s.pop("window_gradient_sum"),
            lambda s: s["activation_counts"].add_(1), lambda s: s.update(probe_calls=0),
            lambda s: s["pending"]["birth_indices"].fill_(0),
            lambda s: s["event_history"][0]["proposal"]["fade_indices"].fill_(0),
            lambda s: s.update(window_count=99), lambda s: s["last_birth"].fill_(99),
            lambda s: s["initial_survival"].logical_not_(), lambda s: s["update_history"].pop(),
            lambda s: s["event_history"][0]["event"].update(birth_accepted=False),
            lambda s: s["event_history"][0]["event"].update(fade_trial_weights=[.731]),
            lambda s: s["event_history"][0]["event"]["half_fade_damage"].update(history=99.)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                state = deepcopy(original)
                mutate(state)
                state["state_hash"] = _digest({key: value for key, value in state.items() if key != "state_hash"})
                with self.assertRaises(ValueError):
                    PairOperator.from_state(model, state)


if __name__ == "__main__":
    unittest.main()
