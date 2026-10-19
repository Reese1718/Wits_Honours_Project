"""Prompt parity tests: synthetic records only; no models or predictions.

Extract the real entry-point functions with AST to avoid importing GPU-only
libraries. Tokenizer spies stop both batch paths before tensor/model work.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import yaml

from common import build_recommendation_messages

HERE = Path(__file__).resolve().parent


def functions_from(filename, names):
    tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing functions in {filename}")
    for node in selected:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *selected], type_ignores=[])
    namespace = dict(build_recommendation_messages=build_recommendation_messages,
                     json=json, time=time)
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    return SimpleNamespace(**namespace)


class BeforeModelWork(Exception):
    pass


class TokenizerSpy:
    """Record canonical messages while retaining distinct template identities."""
    unk_token_id = -1
    eos_token_id = 1
    pad_token_id = None

    def __init__(self, template):
        self.template = template
        self.messages = []
        self.rendered = []

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 1, "<|eot_id|>": 2}.get(token, -1)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages.append(messages)
        text = self.template + json.dumps(messages)
        self.rendered.append(text)
        return text

    def __call__(self, prompts, **kwargs):
        assert prompts == self.rendered
        raise BeforeModelWork


class RecommendationPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step06 = functions_from("06_generate_test_recommendations.py", {
            "build_chat_prompt", "nested_value", "configure_tokenizer",
            "first_present", "normalise_item_list", "test_record_values",
            "generate_batch",
        })
        cls.step05 = functions_from("05_select_best_checkpoint.py", {"evaluate_adapter"})
        cls.step02 = functions_from("02_build_sft_datasets.py", {"build_messages_with_answer"})
        cls.step04 = functions_from("04_train_qlora.py", {"convert_to_prompt_completion"})

    def capture(self, history, target="HELD_OUT_SENTINEL", template="qwen"):
        metadata = {item: {"title": f"Title {item[-1]}"} for item in history}
        record = dict(user_id="synthetic-user", history_item_ids=history,
                      target_item_id=target)
        validation = TokenizerSpy(template)
        test = TokenizerSpy(template)
        with self.assertRaises(BeforeModelWork):
            self.step05.evaluate_adapter(
                model=None, tokenizer=validation, validation_records=[record],
                metadata_lookup=metadata,
                catalogue_constraint=SimpleNamespace(item_to_token_sequence={"x": [1]}),
                history_constraint=None, max_history_length=5,
                max_sequence_length=512, batch_size=1, top_k=10,
                num_beams=10, length_penalty=1.0)
        with self.assertRaises(BeforeModelWork):
            self.step06.generate_batch(
                model=None, tokenizer=test, trie=None, catalogue_set=set(),
                records=[(0, record)], metadata=metadata, max_history_length=5,
                top_k=10, num_beams=10, max_new_tokens=16,
                maximum_input_length=496, exclude_history_items=True,
                length_penalty=1.0)
        self.assertEqual(validation.messages, test.messages)
        self.assertEqual(validation.rendered, test.rendered)
        return test.messages[0], metadata

    def test_training_validation_and_test_messages_match(self):
        history = ["itemA", "itemB", "itemC"]
        messages, metadata = self.capture(history)
        training = self.step02.build_messages_with_answer(history, "target", metadata, 5)
        converted = self.step04.convert_to_prompt_completion(
            dict(messages=training, target_item_id="target"))
        self.assertEqual(converted["prompt"], messages)
        self.assertEqual(converted["completion"], [{"role": "assistant", "content": "target"}])

    def test_exact_history_format(self):
        messages, _ = self.capture(["itemA", "itemB", "itemC"])
        self.assertIn("1. itemA | Title A\n2. itemB | Title B\n3. itemC | Title C",
                      messages[1]["content"])
        self.assertNotIn("(Item ID:", messages[1]["content"])

    def test_recent_five_chronological(self):
        messages, _ = self.capture([f"item{i}" for i in range(8)])
        content = messages[1]["content"]
        for i in range(3):
            self.assertNotIn(f"item{i}", content)
        expected = "\n".join(f"{i - 2}. item{i} | Title {i}" for i in range(3, 8))
        self.assertIn(expected, content)

    def test_target_does_not_influence_prompt(self):
        first, _ = self.capture(["itemA", "itemB"], "HELD_OUT_SENTINEL")
        second, _ = self.capture(["itemA", "itemB"], "DIFFERENT_TARGET")
        self.assertEqual(first, second)
        self.assertNotIn("HELD_OUT_SENTINEL", json.dumps(first))
        self.assertNotIn("DIFFERENT_TARGET", json.dumps(second))

    def test_model_specific_templates_and_eos(self):
        captured = []
        for model, eos, name in (
            ("qwen", "<|im_end|>", "Qwen/Qwen2.5-3B-Instruct"),
            ("llama", "<|eot_id|>", "meta-llama/Llama-3.2-3B-Instruct"),
        ):
            with self.subTest(model=model):
                path = HERE.parent / "configs" / "llm" / f"{model}_video_games_recent5_three_qlora.yaml"
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["model"]["name"], name)
                self.assertEqual(config["model"]["eos_token"], eos)
                self.assertEqual(config["generation"]["max_history_length"], 5)
                self.assertEqual(config["checkpoint_selection"]["max_history_length"], 5)
                tokenizer = TokenizerSpy(model)
                self.step06.configure_tokenizer(tokenizer, config)
                self.assertEqual(tokenizer.eos_token, eos)
                self.assertEqual(tokenizer.pad_token, eos)
                self.assertEqual(tokenizer.padding_side, "left")
                self.assertEqual(tokenizer.truncation_side, "left")
                messages, _ = self.capture(["itemA", "itemB"], template=model)
                captured.append(messages)
        self.assertEqual(*captured)


if __name__ == "__main__":
    unittest.main()
