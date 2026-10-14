"""Regression tests for the exact Python guards embedded in release workflows."""
import contextlib
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def program(filename):
    text = (ROOT / ".github/workflows" / filename).read_text()
    match = re.search(r"python - <<'PY'\n(.*?)\n\s+PY(?:\n|$)", text, re.S)
    if not match:
        raise AssertionError("Missing release version guard")
    return textwrap.dedent(match.group(1))


class ReleasePolicyTests(unittest.TestCase):
    def run_guard(self, tag="v1.2.3", status=200, version="1.2.3", code=None):
        with tempfile.TemporaryDirectory(prefix="release-policy-") as tmp:
            directory = Path(tmp)
            (directory / "pyproject.toml").write_text("[project]\n")
            output = directory / "output"
            module = types.ModuleType("tomllib")
            module.load = lambda stream: {"project": {"name": "example-router", "version": version}}
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with mock.patch.dict(sys.modules, {"tomllib": module}), mock.patch.dict(
                    os.environ, {"RELEASE_TAG": tag, "GITHUB_OUTPUT": str(output)}
                ), mock.patch("urllib.request.urlopen") as get, contextlib.redirect_stdout(io.StringIO()):
                    if isinstance(status, Exception):
                        get.side_effect = status
                    elif status != 200:
                        get.side_effect = urllib.error.HTTPError("https://pypi.org", status, "test", {}, None)
                    exec(compile(code or program("publish-pypi.yml"), "release-version-guard", "exec"), {})
                    if get.called:
                        get.assert_called_once_with("https://pypi.org/pypi/example-router/" + version + "/json", timeout=30)
                return output.read_text() if output.exists() else ""
            finally:
                os.chdir(previous)

    def test_existing_version_is_skipped(self):
        self.assertEqual(self.run_guard(status=200), "publish_allowed=false\n")

    def test_missing_version_may_publish(self):
        self.assertEqual(self.run_guard(status=404), "publish_allowed=true\n")

    def test_registry_errors_fail_closed(self):
        for status in (401, 403, 429, 500, 503):
            with self.subTest(status=status), self.assertRaises(urllib.error.HTTPError):
                self.run_guard(status=status)

    def test_network_failure_fails_closed(self):
        with self.assertRaises(urllib.error.URLError):
            self.run_guard(status=urllib.error.URLError("network unavailable"))

    def test_mismatched_package_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.run_guard(tag="v9.9.9")

    def test_nonsoftware_tags_are_rejected(self):
        for tag in ("paper-videos-v1", "video-v1", "videos-v2", "v1", "v1.2", "v1.2.3/other", "main", ""):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.run_guard(tag=tag)

    def test_supported_prerelease_tag_matches_exact_version(self):
        self.assertEqual(self.run_guard(tag="v1.2.3rc1", version="1.2.3rc1", status=200), "publish_allowed=false\n")

    def test_workflow_filters_media_and_defaults_to_dry_run(self):
        text = (ROOT / ".github/workflows/publish-pypi.yml").read_text()
        self.assertIn("if: startsWith(github.event.release.tag_name || inputs.tag, 'v')", text)
        self.assertRegex(text, r"dry_run:\s+description:.*\n\s+required: true\n\s+default: true")
        self.assertIn("inputs.dry_run == false", text)
        self.assertIn("outputs.publish_allowed == 'true'", text)
        self.assertEqual(text.count("uses: actions/checkout@"), text.count("ref: ${{ github.event.release.tag_name || inputs.tag }}"))
        for tag in ("paper-videos-v1", "video-v1"):
            self.assertFalse(tag.startswith("v") and not tag.startswith("video"), tag)

    @unittest.skipUnless((ROOT / ".github/workflows/release.yml").exists(), "No separate release attachment workflow")
    def test_attachment_workflow_rejects_media_and_mismatched_versions(self):
        text = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertIn("if: startsWith(github.event.release.tag_name, 'v')", text)
        code = program("release.yml")
        self.assertEqual(self.run_guard(code=code), "")
        for tag in ("paper-videos-v1", "videos-v2", "v9.9.9"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.run_guard(tag=tag, code=code)


if __name__ == "__main__":
    unittest.main()
