from setuptools import setup, find_packages

setup(
    name="ai_layer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "elevenlabs",
        "deepgram-sdk",
        "openai",
        "websockets",
        "pydantic",
        "python-multipart",
        "redis",
        "aiohttp",
        "numpy",
        "soundfile",
        "pydub",
        "python-jose",
        "passlib",
        "bcrypt"
    ],
) 