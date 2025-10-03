#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tests.test_drift_diffusion import test_gradient_norm, test_eval_drift

print("=" * 60)
print("Running DriftDiffusion Tests")
print("=" * 60)

try:
    print("\n[TEST 1] test_gradient_norm...")
    test_gradient_norm()
    print("✓ test_gradient_norm PASSED")
except Exception as e:
    print(f"✗ test_gradient_norm FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[TEST 2] test_eval_drift...")
    test_eval_drift()
    print("✓ test_eval_drift PASSED")
except Exception as e:
    print(f"✗ test_eval_drift FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests PASSED!")
print("=" * 60)
