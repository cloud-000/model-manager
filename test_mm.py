#!/usr/bin/env python3
"""Tests for the parts of mm that guess: classification, slugs, sources, flags.

Stdlib only, like mm itself. Run with `python3 test_mm.py`.

Everything here builds a fake cache or a fake root out of empty files, because
the questions worth testing are all about layout and naming rather than about
weights - none of these fixtures contain a single real tensor.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mmlib import runners, scan as scan_mod
from mmlib.runners import hf, litert
from mmlib.classify import classify_dir
from mmlib.entry import Entry
from mmlib.resolve import resolve
from mmlib.runtimes import Runtime, parse_flag, resolve_runtime
from mmlib.scan import make_slug, scan
from mmlib.settings import (parse_value, resolve_settings, runner_name,
                            scopes, settings_flags)

MLX_CONFIG = {"model_type": "llama", "max_position_embeddings": 128000,
              "quantization": {"bits": 4, "group_size": 64}}


def write(path: Path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def make_model_dir(d: Path, config=None, tokenizer=True) -> Path:
    write(d / "config.json", config if config is not None else MLX_CONFIG)
    write(d / "model.safetensors", "x" * 100)
    if tokenizer:
        write(d / "tokenizer.json", "{}")
    return d


def make_hub_repo(hub: Path, repo: str, sha="abc123") -> Path:
    """A repo laid out the way huggingface_hub lays one out."""
    repo_dir = hub / ("models--" + repo.replace("/", "--"))
    write(repo_dir / "refs/main", sha)
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    return snap


class TestClassify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_quantised_mlx_is_chat(self):
        e = classify_dir(make_model_dir(self.d / "m"), "org/M", None)
        self.assertEqual((e.fmt, e.role, e.quant), ("mlx", "chat", "4bit"))
        self.assertEqual(e.ctx, 128000)
        self.assertEqual(e.model_type, "llama")
        self.assertFalse(e.custom_code)

    def test_unquantised_is_hf_not_mlx(self):
        cfg = {"model_type": "llama", "max_position_embeddings": 4096}
        e = classify_dir(make_model_dir(self.d / "m", cfg), "org/M", None)
        self.assertEqual(e.fmt, "hf")
        self.assertIsNone(e.quant)

    def test_mlx_in_repo_name_is_enough(self):
        cfg = {"model_type": "llama"}
        e = classify_dir(make_model_dir(self.d / "m", cfg), "org/M-MLX", None)
        self.assertEqual(e.fmt, "mlx")

    def test_no_tokenizer_is_not_runnable(self):
        e = classify_dir(make_model_dir(self.d / "m", tokenizer=False), "org/M", None)
        self.assertEqual(e.role, "none")
        self.assertFalse(e.runnable)
        self.assertIn("no tokenizer", e.reason)

    def test_vision_config_is_vision(self):
        cfg = dict(MLX_CONFIG, vision_config={"hidden_size": 8})
        e = classify_dir(make_model_dir(self.d / "m", cfg), "org/M", None)
        self.assertEqual(e.role, "vision")

    def test_assistant_suffix_is_a_draft(self):
        cfg = dict(MLX_CONFIG, model_type="gemma4_assistant")
        e = classify_dir(make_model_dir(self.d / "m", cfg), "org/M", None)
        self.assertEqual(e.role, "draft")

    def test_model_file_marks_custom_code(self):
        """A checkpoint carrying its architecture needs no runtime support."""
        cfg = dict(MLX_CONFIG, model_type="maple", model_file="maple.py")
        e = classify_dir(make_model_dir(self.d / "m", cfg), "org/M", None)
        self.assertTrue(e.custom_code)

    def test_context_from_nested_text_config(self):
        cfg = {"model_type": "gemma4", "quantization": {"bits": 4},
               "text_config": {"max_position_embeddings": 262144}}
        e = classify_dir(make_model_dir(self.d / "m", cfg), "org/M", None)
        self.assertEqual(e.ctx, 262144)

    def test_ctranslate2_is_named_not_skipped(self):
        write(self.d / "m/config.json", {"model_type": "whisper"})
        write(self.d / "m/model.bin", "x")
        e = classify_dir(self.d / "m", "org/M", None)
        self.assertEqual(e.fmt, "ct2")
        self.assertIn("faster-whisper", e.reason)


class TestScan(unittest.TestCase):
    """scan() reads module-level HUB, so each test points it at a fixture."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.hub = self.root / "hub"
        self.hub.mkdir()
        self._real_hub = scan_mod.HUB
        scan_mod.HUB = self.hub
        # llama.cpp is not involved in any of these fixtures.
        self._real_specs = scan_mod.llama_specs
        scan_mod.llama_specs = lambda: {}

    def tearDown(self):
        scan_mod.HUB = self._real_hub
        scan_mod.llama_specs = self._real_specs
        self.tmp.cleanup()

    def test_format_namespaces_prevent_collision(self):
        """The whole reason handles carry a format prefix."""
        make_model_dir(make_hub_repo(self.hub, "org/Weird-3B-MLX"))
        write(make_hub_repo(self.hub, "org/Weird-3B-GGUF", "def456")
              / "weird-3b-Q4_0.gguf", "x")
        handles = {e.handle for e in scan({"roots": []})}
        self.assertIn("mlx/weird-3b", handles)
        self.assertIn("gguf/weird-3b:Q4_0", handles)

    def test_collision_inside_one_format_falls_back_to_org(self):
        make_model_dir(make_hub_repo(self.hub, "alpha/Same-3B"))
        make_model_dir(make_hub_repo(self.hub, "beta/Same-3B", "def456"))
        handles = {e.handle for e in scan({"roots": []})}
        self.assertEqual(handles, {"mlx/alpha-same-3b", "mlx/beta-same-3b"})

    def test_variant_subdirectories_become_variants(self):
        snap = make_hub_repo(self.hub, "org/Multi-MLX")
        make_model_dir(snap / "4bit")
        make_model_dir(snap / "8bit")
        handles = {e.handle for e in scan({"roots": []})}
        self.assertEqual(handles, {"mlx/multi:4bit", "mlx/multi:8bit"})

    def test_refs_stub_with_no_snapshot_is_explained(self):
        """`hf download --local-dir` leaves exactly this behind."""
        write(self.hub / "models--org--Gone/refs/main", "abc123")
        entries = scan({"roots": []})
        self.assertEqual(len(entries), 1)
        self.assertIn("no snapshot", entries[0].reason)
        self.assertFalse(entries[0].runnable)

    def test_registered_root_supersedes_the_stub(self):
        write(self.hub / "models--deepgrove--maple-2bit-mlx/refs/main", "sha999")
        local = self.root / "project"
        make_model_dir(local / "maple-2bit-mlx")
        entries = scan({"roots": [str(local)]})
        self.assertEqual([e.handle for e in entries], ["mlx/maple-2bit"])
        e = entries[0]
        self.assertEqual(e.source, "local")
        # Stub-less: the local copy explains the repo, so the stub is dropped.
        self.assertFalse(any(x.reason and "no snapshot" in x.reason for x in entries))

    def test_root_recovers_the_org_from_the_commit_sha(self):
        write(self.hub / "models--deepgrove--maple-2bit-mlx/refs/main", "sha999")
        local = self.root / "project"
        d = make_model_dir(local / "maple-2bit-mlx")
        write(d / ".cache/huggingface/trees/sha999.json", "{}")
        e = scan({"roots": [str(local)]})[0]
        self.assertEqual(e.repo, "deepgrove/maple-2bit-mlx")

    def test_root_without_sha_match_falls_back_to_directory_name(self):
        local = self.root / "project"
        make_model_dir(local / "some-model")
        e = scan({"roots": [str(local)]})[0]
        self.assertEqual(e.repo, "local/some-model")

    def test_root_ignores_the_directories_that_surround_a_model(self):
        local = self.root / "project"
        make_model_dir(local / "the-model")
        (local / ".venv/lib").mkdir(parents=True)
        (local / "tests").mkdir()
        (local / "__pycache__").mkdir()
        self.assertEqual(len(scan({"roots": [str(local)]})), 1)

    def test_split_gguf_only_exposes_the_first_shard(self):
        snap = make_hub_repo(self.hub, "org/Big-GGUF")
        for i in (1, 2, 3):
            write(snap / f"big-Q4_K_M-{i:05d}-of-00003.gguf", "x")
        entries = scan({"roots": []})
        self.assertEqual([e.handle for e in entries], ["gguf/big:Q4_K_M"])

    def test_mmproj_is_attached_not_listed(self):
        snap = make_hub_repo(self.hub, "org/Vis-GGUF")
        write(snap / "vis-Q4_K_M.gguf", "x")
        write(snap / "mmproj-vis-f16.gguf", "x")
        entries = scan({"roots": []})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].mmproj.name, "mmproj-vis-f16.gguf")

    def test_draft_models_pair_with_their_base(self):
        make_model_dir(make_hub_repo(self.hub, "org/Gem-4bit"))
        make_model_dir(make_hub_repo(self.hub, "org/Gem-assistant-4bit", "def456"),
                       dict(MLX_CONFIG, model_type="gemma4_assistant"))
        base = next(e for e in scan({"roots": []}) if e.role == "chat")
        self.assertIsNotNone(base.draft)


class TestSlug(unittest.TestCase):
    def test_redundant_format_suffix_is_dropped(self):
        self.assertEqual(make_slug("LFM2.5-2.6B-MLX"), "lfm2.5-2.6b")
        self.assertEqual(make_slug("Ornith-1.0-9B-GGUF"), "ornith-1.0-9b")

    def test_format_word_inside_a_name_survives(self):
        self.assertEqual(make_slug("mlx-swift-demo"), "mlx-swift-demo")


class TestRuntimeFlags(unittest.TestCase):
    def spec(self, **kw):
        return parse_flag("flash_head", dict(flag="--flash-head", **kw))

    def test_shorthand_string_is_a_value_flag(self):
        s = parse_flag("temp", "--temp")
        self.assertEqual((s.kind, s.flag), ("value", "--temp"))

    def test_bool_true_emits_the_bare_flag(self):
        rt = Runtime("fork", Path("/nowhere"), {"flash_head": self.spec(type="bool")})
        groups, dropped = settings_flags({"flash_head": (True, "model")},
                                         "mlx_lm.chat", rt)
        self.assertEqual(groups, [["--flash-head"]])
        self.assertEqual(dropped, [])

    def test_bool_false_emits_nothing_unless_negatable(self):
        plain = Runtime("f", Path("/x"), {"flash_head": self.spec(type="bool")})
        neg = Runtime("f", Path("/x"),
                      {"flash_head": self.spec(type="bool", negatable=True)})
        resolved = {"flash_head": (False, "model")}
        self.assertEqual(settings_flags(resolved, "mlx_lm.chat", plain)[0], [])
        self.assertEqual(settings_flags(resolved, "mlx_lm.chat", neg)[0],
                         [["--no-flash-head"]])

    def test_flag_restricted_to_named_tools_is_dropped_elsewhere(self):
        rt = Runtime("f", Path("/x"),
                     {"flash_head": self.spec(type="bool", tools=["mlx_lm.chat"])})
        _, dropped = settings_flags({"flash_head": (True, "model")},
                                    "mlx_lm.server", rt)
        self.assertEqual(dropped, ["flash_head"])

    def test_unknown_key_is_reported_not_forwarded(self):
        groups, dropped = settings_flags({"top_k": (20, "model")},
                                         "mlx_lm.chat", None)
        self.assertEqual((groups, dropped), ([], ["top_k"]))

    def test_runtime_is_never_translated_into_a_flag(self):
        groups, dropped = settings_flags({"runtime": ("fork", "model")},
                                         "mlx_lm.chat", None)
        self.assertEqual((groups, dropped), ([], []))

    def test_tool_specific_spelling(self):
        """mlx-vlm says --temperature where everyone else says --temp."""
        resolved = {"temp": (0.7, "defaults")}
        self.assertEqual(settings_flags(resolved, "mlx_lm.chat", None)[0],
                         [["--temp", "0.7"]])
        self.assertEqual(settings_flags(resolved, "mlx_vlm.chat", None)[0],
                         [["--temperature", "0.7"]])


class TestCascade(unittest.TestCase):
    def entry(self):
        return Entry("org/M", "mlx", "chat", Path("/x"), slug="m")

    def cfg(self, **layers):
        return {"settings": {"defaults": layers.get("defaults", {}),
                             "runner": layers.get("runner", {}),
                             "model": layers.get("model", {})}}

    def test_later_layers_win(self):
        cfg = self.cfg(defaults={"temp": 0.7}, runner={"mlx": {"temp": 0.5}},
                       model={"mlx/m": {"temp": 0.2}})
        resolved, _ = resolve_settings(self.entry(), cfg, "mlx")
        self.assertEqual(resolved["temp"], (0.2, "model"))

    def test_runner_layer_is_keyed_by_runtime_name(self):
        cfg = self.cfg(runner={"mlx": {"temp": 0.5}, "deepgrove": {"temp": 0.9}})
        resolved, _ = resolve_settings(self.entry(), cfg, "deepgrove")
        self.assertEqual(resolved["temp"], (0.9, "runner:deepgrove"))

    def test_raw_accumulates_rather_than_overriding(self):
        cfg = self.cfg(defaults={"raw": ["--a"]}, runner={"mlx": {"raw": ["--b"]}},
                       model={"mlx/m": {"raw": ["--c"]}})
        _, raw = resolve_settings(self.entry(), cfg, "mlx")
        self.assertEqual(raw, ["--a", "--b", "--c"])


class TestRuntimeSelection(unittest.TestCase):
    def entry(self, model_type="maple", custom=False, role="chat"):
        return Entry("org/M", "mlx", role, Path("/x"), slug="m",
                     model_type=model_type, custom_code=custom)

    def runtimes(self, fork_archs=("maple",), mlx_archs=("llama",)):
        mlx = Runtime("mlx", Path("/mlx"))
        mlx._archs = {"mlx_lm": set(mlx_archs), "mlx_vlm": set()}
        mlx._remap = {"mlx_lm": {}, "mlx_vlm": {}}
        fork = Runtime("deepgrove", Path("/fork"))
        fork._archs = {"mlx_lm": set(fork_archs), "mlx_vlm": set()}
        fork._remap = {"mlx_lm": {}, "mlx_vlm": {}}
        return {"mlx": mlx, "deepgrove": fork}

    def test_explicit_pin_wins(self):
        rt, why = resolve_runtime(self.entry("llama"),
                                  {"runtime": ("deepgrove", "model")},
                                  self.runtimes())
        self.assertEqual((rt.name, why), ("deepgrove", "pinned"))

    def test_default_is_used_when_it_can_build_the_model(self):
        rt, _ = resolve_runtime(self.entry("llama"), {}, self.runtimes())
        self.assertEqual(rt.name, "mlx")

    def test_falls_back_to_the_only_runtime_that_supports_it(self):
        rt, _ = resolve_runtime(self.entry("maple"), {}, self.runtimes())
        self.assertEqual(rt.name, "deepgrove")

    def test_custom_code_stays_on_the_default(self):
        """A bundled model_file runs anywhere, so don't drag it onto a fork."""
        rt, _ = resolve_runtime(self.entry("maple", custom=True), {},
                                self.runtimes())
        self.assertEqual(rt.name, "mlx")

    def test_uninspectable_runtime_is_assumed_capable(self):
        mlx = Runtime("mlx", Path("/nonexistent"))
        self.assertTrue(mlx.supports("anything-at-all"))


class FakeArgs:
    def __init__(self, **kw):
        self.__dict__.update(dict(port=8080, draft=False, vlm=False, lm=False), **kw)


class TestRunners(unittest.TestCase):
    def entry(self, fmt="litert", role="chat", **kw):
        return Entry("org/M", fmt, role, Path("/w/model.litertlm"), slug="m", **kw)

    def test_each_format_routes_to_exactly_one_runner(self):
        self.assertEqual(runners.for_format("mlx").NAME, "mlx")
        self.assertEqual(runners.for_format("hf").NAME, "mlx")
        self.assertEqual(runners.for_format("gguf").NAME, "llama")
        self.assertEqual(runners.for_format("litert").NAME, "litert")
        self.assertIsNone(runners.for_format("ct2"))

    def test_hf_falls_back_to_transformers_when_mlx_unavailable(self):
        real_avail = runners.mlx.available
        try:
            runners.mlx.available = lambda: "mlx only runs on macOS"
            self.assertEqual(runners.for_format("hf").NAME, "transformers")
        finally:
            runners.mlx.available = real_avail

    def test_flag_map_is_assembled_from_every_runner(self):
        merged = runners.flag_map()
        for runner in runners.RUNNERS:
            for tool in runner.FLAGS:
                self.assertIn(tool, merged)

    def test_unclaimed_format_is_not_runnable(self):
        e = self.entry(fmt="ct2")
        self.assertIn("no runner", runners.check(e, {}))

    def test_missing_program_demotes_with_its_reason(self):
        e = self.entry()
        real = litert.shutil
        try:
            litert.shutil = type("S", (), {"which": staticmethod(lambda n: None)})
            runners.annotate([e], {})
        finally:
            litert.shutil = real
        self.assertFalse(e.runnable)
        self.assertIn("litert-lm not on PATH", e.reason)

    def test_present_program_leaves_it_runnable(self):
        e = self.entry()
        real = litert.shutil
        try:
            litert.shutil = type("S", (), {"which": staticmethod(lambda n: "/bin/" + n)})
            runners.annotate([e], {})
        finally:
            litert.shutil = real
        self.assertTrue(e.runnable)


class TestLitertRunner(unittest.TestCase):
    """The runner whose shape a config template could not have expressed."""

    def entry(self):
        return Entry("org/M", "litert", "chat", Path("/w/m.litertlm"), slug="m")

    def test_model_is_positional_not_a_flag(self):
        cmd = litert.command(self.entry(), False, FakeArgs(), None)
        self.assertEqual(cmd, ["litert-lm", "run", "/w/m.litertlm"])

    def test_serve_takes_no_model_at_all(self):
        cmd = litert.command(self.entry(), True, FakeArgs(port=9379), None)
        self.assertEqual(cmd, ["litert-lm", "serve", "--port", "9379"])
        self.assertNotIn("/w/m.litertlm", cmd)

    def test_serve_expresses_no_sampling_settings(self):
        groups, dropped = settings_flags({"temp": (0.7, "defaults")},
                                         "litert-lm.serve", None)
        self.assertEqual(groups, [])
        self.assertEqual(dropped, ["temp"])

    def test_run_uses_its_own_spelling(self):
        groups, _ = settings_flags({"temp": (0.7, "defaults")},
                                   "litert-lm.run", None)
        self.assertEqual(groups, [["--temperature", "0.7"]])

    def test_max_num_tokens_is_kv_cache_not_a_reply_cap(self):
        """--max-num-tokens sizes the KV cache, so mapping max_tokens to it
        would silently mean something else entirely."""
        run = litert.FLAGS["litert-lm.run"]
        self.assertEqual(run["max_kv_size"], "--max-num-tokens")
        self.assertNotIn("max_tokens", run)


class TestScopes(unittest.TestCase):
    def test_singleton_runners_get_a_scope_each(self):
        found = scopes({"mlx": Runtime("mlx", Path("/x"))})
        self.assertEqual(found, ("@defaults", "@mlx", "@transformers", "@llama", "@litert"))

    def test_extra_runtimes_appear_alongside(self):
        found = scopes({"mlx": Runtime("mlx", Path("/x")),
                        "deepgrove": Runtime("deepgrove", Path("/y"))})
        self.assertIn("@deepgrove", found)

    def test_runner_name_is_the_scope_a_model_answers_to(self):
        gguf = Entry("org/M", "gguf", "chat", Path("/x"), slug="m")
        lit = Entry("org/M", "litert", "chat", Path("/x"), slug="m")
        self.assertEqual(runner_name(gguf, None), "llama")
        self.assertEqual(runner_name(lit, None), "litert")
        self.assertEqual(runner_name(gguf, Runtime("deepgrove", Path("/y"))),
                         "deepgrove")


class TestResolvePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve_directory_path(self):
        m = make_model_dir(self.d / "my-model")
        e = resolve(str(m), [], {"aliases": {}})
        self.assertEqual(e.source, "local")
        self.assertEqual(e.path, m)
        self.assertEqual(e.slug, "my-model")

    def test_resolve_gguf_file(self):
        f = write(self.d / "test-model.gguf", "fake gguf")
        e = resolve(str(f), [], {"aliases": {}})
        self.assertEqual(e.source, "local")
        self.assertEqual(e.path, f)
        self.assertEqual(e.fmt, "gguf")

    def test_resolve_remote_hf_repo(self):
        e = resolve("hf/LiquidAI/LFM2.5-350M", [], {"aliases": {}})
        self.assertEqual(e.fmt, "hf")
        self.assertEqual(e.repo, "LiquidAI/LFM2.5-350M")
        self.assertEqual(e.slug, "lfm2.5-350m")


class TestParseValue(unittest.TestCase):
    def test_types(self):
        self.assertEqual(parse_value("4096"), 4096)
        self.assertEqual(parse_value("-1"), -1)
        self.assertEqual(parse_value("0.7"), 0.7)
        self.assertIs(parse_value("true"), True)
        self.assertIs(parse_value("False"), False)
        self.assertEqual(parse_value("deepgrove"), "deepgrove")


if __name__ == "__main__":
    unittest.main(verbosity=2)
