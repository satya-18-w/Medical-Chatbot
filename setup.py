from setuptools import find_packages,setup
from typing import List





def get_requirements()->List[str]:
    requirements_list : List[str] = []
    return requirements_list




setup(
    name="Gen AI Project",
    version="0.0.1",
    author="Satyajit",
    author_email="satyajitsamal@198076@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
    
)