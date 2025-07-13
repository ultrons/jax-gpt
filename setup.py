from setuptools import setup, find_packages

setup(
    name='jax_gpt',
    version='0.1.0',
    packages=find_packages(include=['jax_gpt', 'jax_gpt.*']),
    install_requires=[
        # List your core dependencies here, matching requirements.txt
        'jax',
        'flax',
        'tiktoken',
        'torch',
        'transformers',
    ],
    # You can add more metadata here
    author='Your Name',
    author_email='your.email@example.com',
    description='A JAX implementation of GPT models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/jax-gpt', # Replace with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
