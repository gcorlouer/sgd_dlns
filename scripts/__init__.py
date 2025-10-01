"""
Deep Linear Networks experiments package.

This package contains modules for training and analyzing
Deep Linear Networks on teacher-student tasks.
"""

from scripts.models import DLN
from scripts.teacher import Teacher, TeacherDataset
from scripts.metrics import Observable

__all__ = ["DLN", "Teacher", "TeacherDataset", "Observable"]
