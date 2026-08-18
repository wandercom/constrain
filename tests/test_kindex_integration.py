"""Tests for optional kindex integration — graceful degradation."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ============================================================================
# Availability and graceful degradation
# ============================================================================

class TestAvailability:
    def test_is_available_honors_explicit_disable(self, monkeypatch):
        """A bounded run can exclude ambient graph context and publishing."""
        from constrain import kindex_integration as ki

        monkeypatch.setenv("CONSTRAIN_KINDEX", "off")
        ki._store = MagicMock()
        ki._checked = True

        assert ki.is_available() is False

        ki.close()

    def test_is_available_when_kindex_missing(self):
        """is_available returns False when kindex is not installed."""
        from constrain import kindex_integration as ki
        ki.close()  # reset state
        with patch.dict("sys.modules", {"kindex": None, "kindex.config": None, "kindex.store": None}):
            ki._checked = False
            ki._store = None
            # Force re-import to fail
            with patch("constrain.kindex_integration._init_store", return_value=False):
                assert ki.is_available() is False

    def test_fetch_context_returns_empty_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "is_available", return_value=False):
            assert ki.fetch_context("test topic") == ""

    def test_search_returns_empty_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "is_available", return_value=False):
            assert ki.search("test") == []

    def test_publish_node_returns_none_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "is_available", return_value=False):
            assert ki.publish_node("title", "content") is None

    def test_publish_constraints_returns_zero_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "is_available", return_value=False):
            assert ki.publish_constraints("constraints: []") == 0

    def test_publish_components_returns_zero_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "is_available", return_value=False):
            assert ki.publish_components("components: []") == 0

    def test_learn_text_returns_zero_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "is_available", return_value=False):
            assert ki.learn_text("some text") == 0

    def test_index_codebase_returns_false_when_unavailable(self):
        from constrain import kindex_integration as ki
        ki.close()
        with patch.object(ki, "_cli_available", return_value=False):
            with patch.object(ki, "is_available", return_value=False):
                assert ki.index_codebase(Path(".")) is False

    def test_close_is_safe_when_not_initialized(self):
        from constrain import kindex_integration as ki
        ki._store = None
        ki._checked = False
        ki.close()  # should not raise


# ============================================================================
# .kin/config management
# ============================================================================

class TestKinConfig:
    def test_read_missing_config(self, tmp_path):
        from constrain.kindex_integration import read_kin_config
        assert read_kin_config(tmp_path) == {}

    def test_write_and_read_config(self, tmp_path):
        from constrain.kindex_integration import read_kin_config, write_kin_config
        write_kin_config(tmp_path, {"auto_index": True, "name": "test-project"})
        config = read_kin_config(tmp_path)
        assert config["auto_index"] is True
        assert config["name"] == "test-project"

    def test_write_merges_config(self, tmp_path):
        from constrain.kindex_integration import read_kin_config, write_kin_config
        write_kin_config(tmp_path, {"name": "test"})
        write_kin_config(tmp_path, {"auto_index": False})
        config = read_kin_config(tmp_path)
        assert config["name"] == "test"
        assert config["auto_index"] is False

    def test_should_auto_index_unset(self, tmp_path):
        from constrain.kindex_integration import should_auto_index
        assert should_auto_index(tmp_path) is None

    def test_should_auto_index_true(self, tmp_path):
        from constrain.kindex_integration import write_kin_config, should_auto_index
        write_kin_config(tmp_path, {"auto_index": True})
        assert should_auto_index(tmp_path) is True

    def test_should_auto_index_false(self, tmp_path):
        from constrain.kindex_integration import write_kin_config, should_auto_index
        write_kin_config(tmp_path, {"auto_index": False})
        assert should_auto_index(tmp_path) is False

    def test_read_legacy_kin_file(self, tmp_path):
        """Read .kin as a plain YAML file (legacy format)."""
        from constrain.kindex_integration import read_kin_config
        kin_file = tmp_path / ".kin"
        kin_file.write_text("name: legacy-project\nauto_index: true\n", encoding="utf-8")
        config = read_kin_config(tmp_path)
        assert config["name"] == "legacy-project"


# ============================================================================
# Constraint publishing (with mocked store)
# ============================================================================

class TestPublishConstraints:
    def test_parses_and_publishes(self):
        from constrain import kindex_integration as ki
        mock_store = MagicMock()
        mock_store.add_node.return_value = "node-123"
        ki._store = mock_store
        ki._checked = True

        yaml_content = (
            "constraints:\n"
            "  - id: C001\n"
            "    boundary: auth\n"
            "    condition: tokens expire in 24h\n"
            "    violation: stale session\n"
            "    severity: must\n"
            "    rationale: security\n"
            "    classification_tier: AUTH\n"
            "    affected_components: [auth-service]\n"
            "  - id: C002\n"
            "    boundary: api\n"
            "    condition: rate limit 100/min\n"
            "    violation: overload\n"
            "    severity: should\n"
            "    rationale: stability\n"
        )
        count = ki.publish_constraints(yaml_content, tags=["test"])
        assert count == 2
        assert mock_store.add_node.call_count == 2

        # Check first call
        call_args = mock_store.add_node.call_args_list[0]
        assert "C001" in call_args.kwargs.get("title", call_args[1].get("title", ""))

        ki.close()

    def test_empty_yaml(self):
        from constrain import kindex_integration as ki
        ki._store = MagicMock()
        ki._checked = True
        assert ki.publish_constraints("") == 0
        ki.close()

    def test_invalid_yaml(self):
        from constrain import kindex_integration as ki
        ki._store = MagicMock()
        ki._checked = True
        assert ki.publish_constraints("not: [valid: yaml: {{") == 0
        ki.close()


class TestPublishComponents:
    def test_parses_components(self):
        from constrain import kindex_integration as ki
        mock_store = MagicMock()
        mock_store.add_node.return_value = "node-456"
        ki._store = mock_store
        ki._checked = True

        yaml_content = (
            "components:\n"
            "  - name: auth-service\n"
            "    description: handles authentication\n"
            "    type: service\n"
            "    dependencies: [db-service]\n"
            "  - name: db-service\n"
            "    description: database layer\n"
            "    type: service\n"
            "    dependencies: []\n"
        )
        count = ki.publish_components(yaml_content, tags=["test"])
        assert count == 2
        ki.close()


# ============================================================================
# CLI integration (mocked)
# ============================================================================

class TestCLIKindexPrompt:
    def test_kindex_prompt_skips_when_unavailable(self):
        from constrain.cli import _kindex_prompt_and_index
        with patch("constrain.cli.kindex") as mock_ki:
            mock_ki.is_available.return_value = False
            _kindex_prompt_and_index(Path("."))
            mock_ki.index_codebase.assert_not_called()

    def test_kindex_prompt_auto_indexes(self, tmp_path):
        from constrain.cli import _kindex_prompt_and_index
        with patch("constrain.cli.kindex") as mock_ki:
            mock_ki.is_available.return_value = True
            mock_ki.should_auto_index.return_value = True
            _kindex_prompt_and_index(tmp_path)
            mock_ki.index_codebase.assert_called_once_with(tmp_path)

    def test_kindex_prompt_skips_when_never(self, tmp_path):
        from constrain.cli import _kindex_prompt_and_index
        with patch("constrain.cli.kindex") as mock_ki:
            mock_ki.is_available.return_value = True
            mock_ki.should_auto_index.return_value = False
            _kindex_prompt_and_index(tmp_path)
            mock_ki.index_codebase.assert_not_called()
