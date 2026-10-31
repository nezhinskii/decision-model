"""Small CPU checks: no pretrained models, datasets, downloads or GPU."""

from dataclasses import replace
import pytest

import torch

from src.features import FeatureBatch
from src.feature_cache.dataset import collate_features
from src.heads import HeadConfig, NormType, ProcessorType, ReadoutMode, build_head
from src.heads.processors import JointBlock, pack_rows, prepare_context_attention
from src.heads.readout import LayerReadout


def features():
    """Three questions share two contexts and have 2/3/1 options respectively."""
    return FeatureBatch(
        questions=torch.randn(3, 3, 8), options=torch.randn(6, 3, 8),
        option_to_question=torch.tensor([0, 0, 1, 1, 1, 2]),
        question_to_context=torch.tensor([0, 0, 1]),
        contexts=[torch.randn(5, 8), torch.randn(2, 8)],
        question_ids=["a", "b", "c"], option_ids=[str(i) for i in range(6)], metadata={},
    )


class TestHeads:
    def test_pack_restores_values_and_gradients(self):
        rows = torch.randn(6, 8, requires_grad=True)
        groups = torch.tensor([0, 0, 2, 2, 2, 3])  # Includes an empty group.
        packed, valid, positions = pack_rows(rows, groups, 4)
        assert positions.tolist() == [0, 1, 0, 1, 2, 0]
        assert valid.sum(1).tolist() == [2, 0, 3, 1]
        restored = packed[groups, positions]
        torch.testing.assert_close(restored, rows)
        restored.sum().backward()
        torch.testing.assert_close(rows.grad, torch.ones_like(rows))

    def test_grouped_attention_and_gradients_match_unpadded_reference(self):
        data = features()
        valid = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        states = torch.randn(3, 4, 8, requires_grad=True)
        context = prepare_context_attention(data, valid)
        block = JointBlock(8, 2, 16, 0.0, True)
        actual = block(states, valid, context)
        references = []
        # Independent reference: one unpadded question at a time, no pack_rows.
        for q in range(3):
            x = states[q, valid[q]].unsqueeze(0)
            memory = data.contexts[int(data.question_to_context[q])].unsqueeze(0)
            read, _ = block.cross(block.cross_norm(x), memory, memory, need_weights=False)
            x = x + read
            normalized = block.self_norm(x)
            update, _ = block.self_attention(normalized, normalized, normalized, need_weights=False)
            references.append(block.ffn(x + update).squeeze(0))
        reference = torch.cat(references)
        torch.testing.assert_close(actual[valid], reference, atol=2e-6, rtol=2e-5)
        parameters = (states, *block.parameters())
        actual_grads = torch.autograd.grad(actual[valid].square().sum(), parameters)
        reference_grads = torch.autograd.grad(reference.square().sum(), parameters)
        for left, right in zip(actual_grads, reference_grads):
            torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-4)

    def test_padding_cannot_influence_real_states(self):
        data = features()
        valid = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        states = torch.randn(3, 3, 8)
        context = prepare_context_attention(data, valid)
        block = JointBlock(8, 2, 16, 0.0, True)
        original = block(states, valid, context)
        changed_states = states.clone()
        changed_states[~valid] = torch.randn_like(changed_states[~valid]) * 100
        memory = context.memory.clone()
        memory[context.padding_mask] = 100
        changed = block(changed_states, valid, replace(context, memory=memory))
        torch.testing.assert_close(original[valid], changed[valid])
        torch.testing.assert_close(changed[~valid], torch.zeros_like(changed[~valid]))

    def test_joint_batch_matches_isolated_questions(self):
        data = features()
        config = HeadConfig(dim=8, hidden_dim=16, attention_heads=2, score_dim=4, num_slots=2)
        head = build_head(8, config, num_feature_layers=3)
        actual = head(data)
        for q in range(3):
            indices = (data.option_to_question == q).nonzero().flatten()
            context = int(data.question_to_context[q])
            isolated = replace(
                data, questions=data.questions[q:q+1], options=data.options[indices],
                option_to_question=torch.zeros(len(indices), dtype=torch.long),
                question_to_context=torch.zeros(1, dtype=torch.long),
                contexts=[data.contexts[context]], question_ids=[data.question_ids[q]],
                option_ids=[data.option_ids[i] for i in indices.tolist()],
            )
            torch.testing.assert_close(actual[indices], head(isolated), atol=2e-6, rtol=2e-5)

    def test_cache_collation_rebases_links_after_context_reordering(self):
        data = features()
        records = []
        for context in range(2):
            qs = (data.question_to_context == context).nonzero().flatten()
            vs = torch.isin(data.option_to_question, qs).nonzero().flatten()
            records.append({
                "context_id": str(context), "context": data.contexts[context],
                "questions": data.questions[qs], "options": data.options[vs],
                "option_to_question": data.option_to_question[vs] - qs[0],
                "question_ids": [data.question_ids[i] for i in qs.tolist()],
                "option_ids": [data.option_ids[i] for i in vs.tolist()],
                "targets": torch.arange(6, dtype=torch.float32)[vs],
                "categories": ["demo"] * len(qs), "tasks": ["choice"] * len(qs),
                "target_kinds": ["hard"] * len(qs), "score_values": [None] * len(qs),
                "feature_metadata": {},
            })
        head = build_head(8, HeadConfig(
            dim=8, hidden_dim=16, attention_heads=2, score_dim=4, num_slots=2,
        ), num_feature_layers=3)
        original = head(data)
        batch = collate_features(records[::-1])
        assert batch.features.option_to_question.tolist() == [0, 1, 1, 2, 2, 2]
        assert batch.features.question_to_context.tolist() == [0, 1, 1]
        assert batch.targets.tolist() == [5, 0, 1, 2, 3, 4]
        torch.testing.assert_close(head(batch.features), original[[5, 0, 1, 2, 3, 4]], atol=2e-6, rtol=2e-5)

    @pytest.mark.parametrize('processor,cross,slots', [
        (ProcessorType.identity, False, 0), (ProcessorType.mlp, False, 0),
        *[(ProcessorType.joint, cross, slots) for cross in (False, True) for slots in (0, 2)],
    ])
    @pytest.mark.parametrize('norm', list(NormType))
    @pytest.mark.parametrize('readout', list(ReadoutMode))
    def test_ablations_norms_readouts_backward(self, processor, cross, slots, norm, readout):
        data = features()
        config = HeadConfig(
            processor=processor, dim=8, hidden_dim=16, depth=2, attention_heads=2,
            score_dim=4, num_slots=slots, use_cross_attention=cross,
            norm_type=norm, readout=readout, router_dim=4,
        )
        head = build_head(8, config, num_feature_layers=3)
        inputs = replace(
            data, contexts=data.contexts if cross else None,
            questions=data.questions.bfloat16(), options=data.options.bfloat16(),
        )
        logits = head(inputs)
        assert logits.shape == (6,)
        assert torch.isfinite(logits).all()
        logits.square().mean().backward()
        for name, parameter in head.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        if processor == ProcessorType.joint:
            assert (head.processor.slots is None) == (slots == 0)
            assert (head.processor.blocks[0].cross is None) == (not cross)
        assert (head.projection.context is None) == (not cross)

    def test_readout_matches_explicit_question_loop(self):
        data = features()
        for mode in ReadoutMode:
            readout = LayerReadout(8, 3, mode, indices=(0, 2), router_dim=4)
            if mode == ReadoutMode.static:
                with torch.no_grad():
                    readout.logits.copy_(torch.tensor([-1.0, 1.0]))
            q_out, o_out = readout(data)
            for q in range(3):
                indices = (data.option_to_question == q).nonzero().flatten()
                qs = data.questions[q, [0, 2]]
                os = data.options[indices][:, [0, 2]]
                if mode == ReadoutMode.last:
                    weights = torch.tensor([0.0, 1.0])
                elif mode == ReadoutMode.mean:
                    weights = torch.tensor([0.5, 0.5])
                elif mode == ReadoutMode.static:
                    weights = readout.logits.softmax(0)
                else:
                    weights = readout.router(torch.cat((qs, os.mean(0)), dim=1)).squeeze(1).softmax(0)
                torch.testing.assert_close(q_out[q], (qs * weights[:, None]).sum(0))
                torch.testing.assert_close(o_out[indices], (os * weights[None, :, None]).sum(1))


