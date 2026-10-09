"""The CrossEncoder backend must work with both generations of the sentence-transformers
API: ``automodel_args``/``activation_fct`` (3.x) and ``model_kwargs``/``activation_fn``
(4.0 and later). Stubs stand in for the library; no model is loaded."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("torch")


class _OldCrossEncoder:
    def __init__(
        self, model_name, num_labels=None, max_length=None, device=None, automodel_args=None
    ):
        self.calls = {
            "init": dict(max_length=max_length, device=device, automodel_args=automodel_args)
        }

    def predict(
        self,
        sentences,
        batch_size=32,
        show_progress_bar=None,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy=True,
    ):
        self.calls["predict"] = {"activation_fct": activation_fct, "n": len(sentences)}
        return np.arange(len(sentences), dtype=np.float32)


class _NewCrossEncoder:
    def __init__(
        self, model_name_or_path, device=None, model_kwargs=None, max_length=None, **kwargs
    ):
        self.calls = {"init": dict(max_length=max_length, device=device, model_kwargs=model_kwargs)}

    def predict(
        self,
        inputs,
        batch_size=32,
        show_progress_bar=None,
        activation_fn=None,
        convert_to_numpy=True,
        **kwargs,
    ):
        self.calls["predict"] = {"activation_fn": activation_fn, "n": len(inputs)}
        return np.arange(len(inputs), dtype=np.float32)


@pytest.fixture(params=["old", "new"])
def stubbed(monkeypatch, request):
    cls = _OldCrossEncoder if request.param == "old" else _NewCrossEncoder
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = cls
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return request.param


def test_backend_adapts_to_the_installed_api(stubbed):
    from text2ipc.rerank.st import CrossEncoderReranker

    r = CrossEncoderReranker("any/model", device="cpu", max_length=512)
    init = r._model.calls["init"]
    assert init["device"] == "cpu" and init["max_length"] == 512
    passed = init["automodel_args"] if stubbed == "old" else init["model_kwargs"]
    assert "torch_dtype" in passed
    scores = r.score("q", ["a", "b", "c"])
    assert scores.tolist() == [0.0, 1.0, 2.0] and scores.dtype == np.float32
    activation = r._model.calls["predict"]
    key = "activation_fct" if stubbed == "old" else "activation_fn"
    assert type(activation[key]).__name__ == "Identity"
    assert r.name == "ce:any/model"
    assert r.score("q", []).shape == (0,)


def test_unknown_api_is_reported(monkeypatch):
    class _Odd:
        def __init__(self, model_name, device=None, max_length=None): ...

        def predict(self, sentences): ...

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _Odd
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    from text2ipc.rerank.st import CrossEncoderReranker

    with pytest.raises(ImportError, match="sentence-transformers>=3.0"):
        CrossEncoderReranker("any/model", device="cpu")
