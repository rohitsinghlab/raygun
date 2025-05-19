from setuptools import setup, find_packages

setup(
    name="raygun",
    version="0.2.1",
    author="Kapil Devkota",
    author_email="kapil.devkota@duke.edu",
    description="Protein Redesign using Raygun",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rohitsinghlab/raygun",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "fair-esm",
        "tqdm",
        "biopython",
        "h5py",
        "einops",
        "pyyaml"
    ],
    entry_points={
        'console_scripts': [
            'raygun-train=raygun.commands.train:main',
            'raygun-sample-single=raygun.commands.generate_samples_single:main',
            'raygun-sample-multiple=raygun.commands.generate_samples_multiple:main'
        ],
    },
)