from setuptools import setup, find_packages

setup(
    name="cte",
    version="0.0.1",
    description="An enviornment for doing computational experiments within the context of my PhD.",
    author="Martin Sig Nørbjerg",
    author_email="msno@mp.aau.dk",
    packages=find_packages(where="source")
)