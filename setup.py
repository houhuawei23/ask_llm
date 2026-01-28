"""Setup script for ask-llm."""

from setuptools import setup, find_packages

setup(
    name="ask-llm",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
