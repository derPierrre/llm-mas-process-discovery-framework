# Process Discovery with LLM-based Multi-Agent Systems

This is the accompanying repository for the paper "Business Process Discovery through Agentic Generative AI" in ICSOC 2025.
This repository contains the implementation of the LLM-based agent choreography framework. 

## Quick Start

The project is designed to work seamlessly with **Visual Studio Code Dev Containers**, providing a complete development environment with MLflow integration out of the box.

### Prerequisites
- Visual Studio Code
- Docker Desktop
- Dev Containers extension for VS Code

### Setup
1. Clone this repository
2. Open in Visual Studio Code or any other IDE
3. When prompted, click "Reopen in Container"
4. The container will automatically set up the complete environment, including MLflow

## API Configuration

The project supports multiple LLM providers. Configure the required API keys as environment variables before starting the container:

### Supported Providers

| Provider | Environment Variable | Notes |
|----------|---------------------|--------|
| **DeepSeek** | `DEEPSEEK_API_KEY` | |
| **Mistral** | `MISTRAL_API_KEY` | ✅ Free tier available |
| **Gemini** | `GEMINI_API_KEY` | ✅ Free tier available |
| **Google Vertex AI** | `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON file |

## Detailed Information about the Framework

See **framework_info.pdf** for more information about the framework. 


