import importlib
import types
from typing import Any, Dict, List

import pytest
import torch


def make_fake_tokenizer():
    class FakeTokenizer:
        def __init__(self):
            self._id2tok = {}
            self._next_id = 1
            self.eos_token = ""
            self.eos_token_id = 0

        def _tokenize_to_ids(self, text: str) -> List[int]:
            toks = text.split()
            ids = []
            for t in toks:
                for k, v in self._id2tok.items():
                    if v == t:
                        ids.append(k)
                        break
                else:
                    _id = self._next_id
                    self._id2tok[_id] = t
                    self._next_id += 1
                    ids.append(_id)
            return ids

        def __call__(self, text: Any, add_special_tokens: bool = False, return_tensors: str = None, **kwargs):
            if isinstance(text, list):
                text = " \n ".join([m.get("content", "") for m in text])
            ids = self._tokenize_to_ids(str(text))
            if return_tensors == "pt":
                tensor = torch.tensor([ids], dtype=torch.long)
                class BatchLike(dict):
                    def __init__(self, input_ids, attention_mask):
                        super().__init__()
                        self.input_ids = input_ids
                        self.attention_mask = attention_mask
                        super().__setitem__("input_ids", input_ids)
                        super().__setitem__("attention_mask", attention_mask)
                    def to(self, device):
                        self.input_ids = self.input_ids.to(device)
                        self.attention_mask = self.attention_mask.to(device)
                        super().__setitem__("input_ids", self.input_ids)
                        super().__setitem__("attention_mask", self.attention_mask)
                        return self
                return BatchLike(tensor, torch.ones_like(tensor))
            return types.SimpleNamespace(input_ids=ids)

        def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True):
            parts = []
            for m in messages:
                parts.append(f"[{m['role'].upper()}]\n{m['content']}")
            return "\n\n".join(parts)

        def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
            words = []
            for i in ids:
                words.append(self._id2tok.get(int(i), f"<GEN_{i}>") )
            return " ".join(words)

    return FakeTokenizer()


def make_fake_model():
    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.last_generate_kwargs = None

        def generate(self, *args, **kwargs):
            self.last_generate_kwargs = kwargs
            inputs = None
            if "input_ids" in kwargs:
                inputs = kwargs["input_ids"]
            elif args:
                if isinstance(args[0], dict) and "input_ids" in args[0]:
                    inputs = args[0]["input_ids"]
            if inputs is None:
                inp = torch.tensor([[1, 2, 3]])
            else:
                inp = inputs
            gen = torch.randint(low=900, high=20000, size=(1, 2), dtype=torch.long)
            return torch.cat([inp, gen], dim=1)

    return FakeModel()


@pytest.fixture(params=[
    "generation.generate_llama",
    "generation.generate_qwen",
])
def patched_module(request, monkeypatch):
    """Import module under test, patch tokenizer/model loading and device availability."""
    module_path = request.param
    mod = importlib.import_module(module_path)
    fake_tokenizer = make_fake_tokenizer()
    fake_model = make_fake_model()

    monkeypatch.setattr(mod, "AutoTokenizer", types.SimpleNamespace(from_pretrained=lambda mid: fake_tokenizer))
    monkeypatch.setattr(mod, "AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=lambda mid, **kw: fake_model))
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(mod.torch.backends.mps, "is_available", lambda: False)

    return mod, fake_model, fake_tokenizer


def make_pr_sample():
    return {
        "info": {"id": "PR-1", "repository_name_with_owner": "owner/repo", "title": "Fix bug", "issue_titles": "", "issue_bodies": ""},
        "files": [{"filename": "src/foo.py", "patch": "+x\n-y", "file_content": "def foo(): pass"}],
        "issues": [{"title": "Bug report", "body": "Details"}]
    }


def test_prompt_and_json_instruction(patched_module):
    mod, fake_model, fake_tokenizer = patched_module
    pr = make_pr_sample()
    for var in mod.CONTEXT_MAPPING.keys():
        ctx = mod.PromptBuilder.construct_context_block(pr, var)
        assert "You are a helpful assistant" in mod.PromptBuilder.SYSTEM_PROMPT
        messages = mod.PromptBuilder.wrap_context_in_prompt(ctx)
        roles = [m["role"] for m in messages]
        assert "system" in roles and "user" in roles
        assert "Respond with ONLY a JSON object" in mod.PromptBuilder.SYSTEM_PROMPT


def test_no_per_file_patch_truncation(patched_module):
    """Ensure per-file patch content is included in full by the PromptBuilder."""
    mod, _, _ = patched_module
    pr = make_pr_sample()
    long_patch = "X" * 5000
    pr["files"][0]["patch"] = long_patch
    ctx = mod.PromptBuilder.construct_context_block(pr, "P-3")
    assert long_patch in ctx


def test_truncation_and_limits(patched_module):
    mod, fake_model, fake_tokenizer = patched_module
    monkey = pytest.MonkeyPatch()
    monkey.setattr(mod, "MAX_INPUT_TOKENS", 5000)
    try:
        pr = make_pr_sample()
        pr["files"][0]["file_content"] = "w " * 4000
        rec = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-4", seed=42, experiment_id="m1")
        assert "You are a helpful assistant" in rec.full_prompt
        assert isinstance(rec.prompt_token_count, int) and rec.prompt_token_count <= mod.MAX_INPUT_TOKENS
        assert rec.context_token_count >= 0
    finally:
        monkey.undo()


def test_seed_determinism_and_variation(patched_module, monkeypatch):
    mod, fake_model, fake_tokenizer = patched_module
    recorded = {}
    def fake_set_seed(s):
        recorded["last_seed"] = s
        torch.manual_seed(s)
    monkeypatch.setattr(mod, "set_seed", fake_set_seed)
    pr = make_pr_sample()
    r1 = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-6", seed=11, experiment_id="m2")
    r2 = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-6", seed=11, experiment_id="m2")
    assert recorded.get("last_seed") == 11
    assert r1.generated_text == r2.generated_text
    r3 = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-6", seed=99, experiment_id="m3")
    assert r3.generated_text != r1.generated_text


def test_decoding_params_forwarded(patched_module):
    mod, fake_model, fake_tokenizer = patched_module
    pr = make_pr_sample()
    rec = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-6", seed=3, experiment_id="m4")
    assert isinstance(fake_model.last_generate_kwargs, dict)
    for k, v in mod.DECODING_PARAMS.items():
        assert fake_model.last_generate_kwargs.get(k) == v
    assert "pad_token_id" in fake_model.last_generate_kwargs


def test_record_integrity_and_hash(patched_module):
    mod, fake_model, fake_tokenizer = patched_module
    pr = make_pr_sample()
    rec = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-6", seed=5, experiment_id="m5")
    for field in ["experiment_id", "timestamp", "model", "pr_id", "variation", "seed", "prompt_hash", "prompt_token_count", "context_token_count", "generated_token_count", "full_prompt", "generated_text"]:
        assert hasattr(rec, field)
    assert rec.model == mod.MODEL_ID
    h1 = rec.prompt_hash
    pr2 = make_pr_sample(); pr2["info"]["title"] = "Different"
    rec2 = mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr2, "P-6", seed=5, experiment_id="m5")
    assert h1 != rec2.prompt_hash


def test_oom_clears_cache_and_raises(patched_module, monkeypatch):
    mod, fake_model, fake_tokenizer = patched_module
    def raise_oom(*a, **k):
        raise RuntimeError("CUDA out of memory")
    fake_model.generate = raise_oom
    called = {}
    def fake_empty_cache():
        called["emptied"] = True
    monkeypatch.setattr(mod.torch.cuda, "empty_cache", fake_empty_cache)
    pr = make_pr_sample()
    with pytest.raises(RuntimeError):
        mod.run_experiment(fake_model, fake_tokenizer, "devmap", pr, "P-6", seed=0, experiment_id="oom_m")
    assert called.get("emptied", False) is True


def test_main_periodic_and_final_saves(patched_module, tmp_path, monkeypatch):
    mod, fake_model, fake_tokenizer = patched_module
    class FakeLoader:
        def get_comprehensive_data(self):
            out = {}
            for i in range(10):
                out[f"pr{i}"] = make_pr_sample()
            return out
    monkeypatch.setattr(mod, "DataLoader", lambda: FakeLoader())
    monkeypatch.setattr(mod, "load_model", lambda device: (fake_model, fake_tokenizer, "devmap"))
    calls = []
    def fake_to_csv(self, path, index=False):
        calls.append(path)
    monkeypatch.setattr("pandas.DataFrame.to_csv", fake_to_csv, raising=False)
    monkeypatch.setattr(mod, "LOG_DIR", str(tmp_path))
    # set argv differently depending on module name
    script_name = "run_llama_experiment.py" if "llama" in mod.__name__ else "run_qwen_experiment.py"
    monkeypatch.setattr("sys.argv", [script_name, "--limit", "10"]) 
    mod.main()
    assert len(calls) >= 1
