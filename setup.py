from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="langdspy",
    version="0.0.1",
    description="Langchain implementation of Stanford's DSPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aelaguiz/langdspy",
    author="Amir Elaguizy",
    author_email="aelaguiz@gmail.com",
    packages=find_packages(include=["langdspy", "langdspy.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
)
