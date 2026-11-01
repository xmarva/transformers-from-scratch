"""The courses of this repository: where their lessons live and what their notebooks and posts are called.

A lesson folder is named <code>_<name>, and its code is unique across courses: 01_tokenization in the first course,
a01_bert in the second. Notebooks, figures, covers and executed runs are all named by that code, so the courses
share notebooks/, assets/ and build/ without clashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Course:
    key: str  # the folder of its posts' pictures on the site, assets/posts/<key>/
    folder: str  # where its lessons live
    title: str
    lessons: int  # lessons planned, shown as "lesson N of M"
    kaggle: str  # the start of its Kaggle notebooks' titles
    post: str  # the start of its posts' slugs, also their first tag


COURSES = [
    Course("tfs", "lessons", "Transformers: from scratch to production", 10, "Transformers from Scratch",
           "transformers-from-scratch"),
    Course("arch", "architectures", "Transformer architectures: from BERT to diffusion", 15,
           "Transformer Architectures", "transformer-architectures"),
]


def course_of(lesson_dir: Path) -> Course:
    for course in COURSES:
        if lesson_dir.resolve().parent == (ROOT / course.folder).resolve():
            return course
    return COURSES[0]  # a lesson outside the repository (a test fixture) is built like one of the first course


def code(lesson_dir: Path) -> str:
    return lesson_dir.name.split("_", 1)[0]


def lesson_dirs() -> list[Path]:
    found = []
    for course in COURSES:
        folder = ROOT / course.folder
        if folder.exists():
            found += sorted(d for d in folder.iterdir() if d.is_dir() and (d / "lesson.toml").exists())
    return found
