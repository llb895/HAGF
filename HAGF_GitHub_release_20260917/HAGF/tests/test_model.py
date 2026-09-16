import torch

from hagf.model import AblationConfig, CrossModalFusionClassifier, entmax_bisect


def test_entmax_mass_and_gradient() -> None:
    logits = torch.tensor([[3.0, 1.0, -8.0, 0.5]], requires_grad=True)
    probs = entmax_bisect(logits, alpha=1.1)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.count_nonzero(probs == 0) >= 1
    probs.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_entmax_bisection_convergence() -> None:
    generator = torch.Generator().manual_seed(20260909)
    logits = torch.randn(8, 4, 257, generator=generator)
    estimate = entmax_bisect(logits, alpha=1.1, n_iter=20)
    reference = entmax_bisect(logits, alpha=1.1, n_iter=80)
    assert torch.max(torch.abs(estimate - reference)).item() < 1e-5


def build_model(n_modalities: int, ablation: AblationConfig | None = None):
    return CrossModalFusionClassifier(
        input_sizes=[20 + idx for idx in range(n_modalities)],
        num_layers=2,
        hidden_size=20,
        output_size=2,
        num_masks=4,
        group_ratio=0.2,
        dropout=0.1,
        num_heads=4,
        ablation=ablation,
    )


def test_full_model_forward_backward() -> None:
    model = build_model(4)
    inputs = [torch.randn(3, 20 + idx) for idx in range(4)]
    logits = model(inputs)
    assert logits.shape == (3, 2)
    logits.sum().backward()

    for branch in model.branches:
        for layer in branch.layers:
            assert layer.concatenated_masks().shape[1] == layer.input_size


def test_leave_one_modality_forward() -> None:
    model = build_model(3)
    inputs = [torch.randn(3, 20 + idx) for idx in range(3)]
    assert model(inputs).shape == (3, 2)


def test_component_ablation_forward() -> None:
    ablation = AblationConfig(
        use_grouping=False,
        use_sparse_masks=False,
        use_transformer=False,
        use_positional_embedding=False,
        use_fidelity_path=False,
        use_cross_modal_fusion=False,
    )
    model = build_model(4, ablation=ablation)
    inputs = [torch.randn(3, 20 + idx) for idx in range(4)]
    assert model(inputs).shape == (3, 2)


def test_no_transformer_removes_transformer_parameters() -> None:
    full_model = build_model(4)
    no_transformer_model = build_model(
        4, ablation=AblationConfig(use_transformer=False)
    )
    assert any("group_transformer" in name for name, _ in full_model.named_parameters())
    assert not any(
        "group_transformer" in name
        for name, _ in no_transformer_model.named_parameters()
    )
    assert sum(parameter.numel() for parameter in no_transformer_model.parameters()) < sum(
        parameter.numel() for parameter in full_model.parameters()
    )


if __name__ == "__main__":
    test_entmax_mass_and_gradient()
    test_entmax_bisection_convergence()
    test_full_model_forward_backward()
    test_leave_one_modality_forward()
    test_component_ablation_forward()
    test_no_transformer_removes_transformer_parameters()
    print("ALL_MODEL_TESTS_PASSED")
