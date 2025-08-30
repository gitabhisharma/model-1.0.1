# '''Important CMD to Run file ''''
## Install in development mode
### pip install -e .

## Run the API server
### python -m src.wrool_ai.api.server

## Or use the script
python scripts/serve_api.py

## Build Docker image
### docker build -t wrool-ai .

## Run with Docker Compose
### docker-compose up -d

## Or run directly
### docker run -p 8000:8000 -v $(pwd)/models:/app/models wrool-ai