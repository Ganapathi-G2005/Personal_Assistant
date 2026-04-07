# Sidekick - Personal AI Assistant

[![CI](https://github.com/yourusername/sidekick/workflows/CI/badge.svg)](https://github.com/yourusername/sidekick/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Formatting: human-curated](https://img.shields.io/badge/formatting-human--curated-7c6efc.svg)](#-development)

A powerful personal AI assistant built with LangChain, featuring web search capabilities and deployable on Vercel. Sidekick uses a sophisticated agent architecture with evaluation loops to provide helpful, accurate responses.

## ✨ Features

- **Intelligent Agent Architecture**: Built with LangChain and LangGraph for sophisticated reasoning
- **Web Search Integration**: Powered by Tavily for real-time information retrieval
- **Multiple Interfaces**: Web UI, Gradio interface, and REST API
- **Vercel Ready**: Serverless deployment with FastAPI
- **Conversation Memory**: Maintains context across interactions
- **Success Criteria Evaluation**: Self-evaluating responses against user-defined criteria
- **Modern UI**: Clean, responsive web interface with dark/light theme support

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- (Optional) Tavily API key for web search

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/sidekick.git
   cd sidekick
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run locally**

   ```bash
   # Start the API server
   uvicorn api.chat:app --reload --port 8000

   # Open public/index.html in your browser
   ```

## 📖 Usage

### Web Interface

1. Start the API server: `uvicorn api.chat:app --reload --port 8000`
2. Open `public/index.html` in your browser
3. Enter your message and success criteria
4. Click "Go!" to get AI assistance

### API Usage

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Help me plan a vacation to Japan",
    "success_criteria": "Provide a detailed 7-day itinerary with recommendations"
  }'
```

### Gradio Interface (Optional)

```bash
python main.py
```

## 🏗️ Architecture

Sidekick uses a sophisticated agent architecture:

1. **Worker Agent**: Processes user requests and uses tools
2. **Tool Integration**: Web search via Tavily
3. **Evaluator**: Assesses responses against success criteria
4. **Memory**: Maintains conversation context
5. **Routing**: Intelligent flow control between components

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install frontend formatting/lint tooling
npm install

# Run tests
pytest

# Format frontend/docs
npm run format

# Check formatting (CI-friendly)
npm run format:check

# Frontend lint (correctness-focused)
npm run lint:js

# Type checking
mypy .
```

### Formatting Standards

- **Formatting guide:** See `FORMATTING.md` for project conventions.
- **Philosophy:** readable, practical formatting over rigid formatter-only output.
- **Markdown output:** AI responses are normalized to structured markdown (`##`, `**bold**`, `> warnings`, bullet lists).

### Project Structure

```
sidekick/
├── api/                    # FastAPI endpoints
│   ├── chat.py            # Main chat API
│   └── health.py          # Health check endpoint
├── public/                # Static web interface
│   ├── index.html         # Main UI
│   ├── app.js            # Frontend JavaScript
│   └── styles.css        # Styling
├── main.py               # Gradio interface
├── pyproject.toml        # Project configuration
├── requirements.txt      # Production dependencies
└── .github/workflows/    # CI/CD pipelines
```

## 🚀 Deployment

### Vercel Deployment

1. **Set up Vercel project**

   ```bash
   vercel
   ```

2. **Configure environment variables** in Vercel dashboard:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY` (optional)
   - `ENABLE_PLAYWRIGHT=false`

3. **Deploy**
   ```bash
   vercel --prod
   ```

### Environment Variables

| Variable            | Required | Description                               |
| ------------------- | -------- | ----------------------------------------- |
| `OPENAI_API_KEY`    | Yes      | OpenAI API key for GPT models             |
| `TAVILY_API_KEY`    | No       | Tavily API key for web search             |
| `ENABLE_PLAYWRIGHT` | No       | Enable Playwright tools (default: false)  |
| `ENVIRONMENT`       | No       | Environment mode (development/production) |
| `PORT`              | No       | API server port (default: 8000)           |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sidekick --cov-report=html

# Run specific test file
pytest tests/test_chat.py

# Run integration tests
pytest -m integration
```

## 📚 API Documentation

### Endpoints

#### `POST /api/chat`

Send a message to the AI assistant.

**Request Body:**

```json
{
  "message": "string",
  "success_criteria": "string",
  "thread_id": "string (optional)"
}
```

**Response:**

```json
{
  "thread_id": "string",
  "assistant": "string",
  "evaluator": "string"
}
```

#### `GET /api/health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [OpenAI](https://openai.com/) for the language models
- [Tavily](https://tavily.com/) for web search capabilities
- [Vercel](https://vercel.com/) for serverless deployment
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework

Made with ❤️ by GMS Ganapathi.
