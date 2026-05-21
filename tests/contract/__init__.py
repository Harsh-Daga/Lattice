"""Public-surface contract tests for LATTICE.

These tests verify the v1.0.0 public contract (REFACTOR_PLAN.md §2):
  - CLI subcommands and their help output
  - HTTP endpoint shapes
  - Response headers
  - Python API imports

They protect the surface across the v1.0.0 refactor — any phase change
that breaks one of these breaks user-visible behavior. The fix is to revert
the change, not to relax the test.
"""
