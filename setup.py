from setuptools import setup, find_packages

setup(
    name="ai_layer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-dotenv==1.0.0",
        "pydantic>=1.10,<2.0",
        "openai>=1.14.0",
        "aiohttp==3.9.1",
        "redis==5.0.1",
        "python-jose==3.3.0",
        "passlib==1.7.4",
        "bcrypt==4.0.1",
        "azure-cognitiveservices-speech==1.33.0",
        "azure-ai-textanalytics==5.3.0",
        "azure-identity==1.15.0"
    ],
)