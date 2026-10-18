# Copyright (c) 2026 Kamal Mehta. MIT License.
import os

# Keep the test run light: 4 CPU threads, and no TensorFlow start-up info lines.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
