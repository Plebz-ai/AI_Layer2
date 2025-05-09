#!/usr/bin/env python
import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load environment variables
load_dotenv()

def main():
    """Run the AI Layer server."""
    try:
        # Display startup info
        logging.info("Starting AI Layer service...")
        logging.info(f"Python version: {sys.version}")
        
        # Check environment variables
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 5000))
        logging.info(f"Server will run at http://{host}:{port}")
        
        # Check for critical environment variables
        azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not any([azure_openai_endpoint, openai_key]):
            logging.warning("⚠️ No LLM API keys found - using mock responses")
            logging.warning("Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY or OPENAI_API_KEY for real responses")
        
        # Start server
        uvicorn.run(
            "src.main:app",
            host=host,
            port=port,
            reload=True
        )
    except Exception as e:
        logging.error(f"Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 