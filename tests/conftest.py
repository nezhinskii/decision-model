"""Deterministic tiny CPU fixtures; network and pretrained files are never needed."""

import pytest
import torch

from src.data.schema import Example, Option, Question, Target, TargetKind, Task


class CharacterTokenizer:
    """One character per token makes expected spans independent of tokenizer libraries."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.template_calls = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 2 for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.template_calls += 1
        return '<user>' + messages[0]['content'] + '</user>'


def make_example(identity='context', context='Facts', questions=2):
    return Example(identity, context, [
        Question(
            id=f'{identity}/q{i}', source_row=i, text=f'Q{i}?', category='demo',
            task=Task.choice,
            options=[Option('a', 'yes', 'rule'), Option('b', 'longer answer')],
            target=Target(TargetKind.soft, [0.25, 0.75]),
        ) for i in range(questions)
    ])


@pytest.fixture(scope='session', autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture
def tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def example():
    return make_example()
