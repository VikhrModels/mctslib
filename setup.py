from setuptools import setup, find_packages

setup(
    name='mctslib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'run_mcts=mctslib.src.vllm_interface:main',
        ],
    },
)
