from setuptools import setup, find_packages

setup(
    name="megatron",
    version="0.1",
    description="Components of Megatron.",
    packages=find_packages(
        include=("megatron")
    )
)
