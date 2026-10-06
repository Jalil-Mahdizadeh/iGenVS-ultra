"""Exit-status contracts for the mounted iGenVS and iGen3 entry points."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from igenvs_ultra import core_runtime


class CoreRuntimeTests(unittest.TestCase):
    def invoke(self, entry, *, result=None, error=None):
        module = types.ModuleType(f"{entry}.cli")
        module.main = mock.Mock(return_value=result, side_effect=error)
        arguments = [entry, "example-command"]
        with mock.patch.dict(sys.modules, {f"{entry}.cli": module}), \
                mock.patch.dict(os.environ), mock.patch.object(sys, "path", sys.path.copy()):
            status = core_runtime.main(arguments)
        module.main.assert_called_once_with(["example-command"])
        self.assertEqual(arguments, [entry, "example-command"])
        return status

    def test_success_accepts_none_and_zero(self):
        for entry in ("igenvs", "igen3"):
            for result in (None, 0):
                with self.subTest(entry=entry, result=result):
                    self.assertEqual(self.invoke(entry, result=result), 0)

    def test_nonzero_exit_codes_are_preserved(self):
        for entry in ("igenvs", "igen3"):
            for result in (1, 2, 7):
                with self.subTest(entry=entry, result=result):
                    self.assertEqual(self.invoke(entry, result=result), result)

    def test_errors_and_interrupts_are_not_swallowed(self):
        for entry in ("igenvs", "igen3"):
            for error in (SystemExit(0), SystemExit(3), SystemExit("generation failed"),
                          RuntimeError("generation failed"), KeyboardInterrupt()):
                with self.subTest(entry=entry, error=repr(error)):
                    with self.assertRaises(type(error)) as caught:
                        self.invoke(entry, error=error)
                    self.assertIs(caught.exception, error)

    def test_other_invalid_returns_are_not_treated_as_success(self):
        for entry in ("igenvs", "igen3"):
            for result in ("", [], {}):
                with self.subTest(entry=entry, result=result):
                    with self.assertRaises((TypeError, ValueError)):
                        self.invoke(entry, result=result)


if __name__ == "__main__":
    unittest.main()
