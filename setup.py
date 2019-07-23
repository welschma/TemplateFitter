from setuptools import setup
from pathlib import Path

this_dir = Path(__file__).resolve().parent
with (this_dir / "requirements.txt").open() as rf: 
    install_requires = [ 
        req.strip()
        for req in rf.readlines()
        if req.strip() and not req.startswith("#")
    ]   

setup(
    name="TemplateFitter",
    version="0.0.1",
    author="Maximilian Welsch",
    url="https://github.com/welschma/TemplateFitter",
    packages=[
        "templatefitter"
    ],
    description="Perform extended binnend log-likelhood fits using histogram templates as pdfs.",
    install_requires=install_requires
)
