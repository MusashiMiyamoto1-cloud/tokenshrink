"""Shared fixtures for TokenShrink test suite."""

import os
import json
import shutil
import tempfile
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer


# Session-scoped: load the embedding model ONCE for all tests
@pytest.fixture(scope="session")
def shared_model():
    """Load embedding model once per test session."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, clean up after."""
    d = tempfile.mkdtemp(prefix="tokenshrink_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_docs(tmp_dir):
    """Create sample documents for indexing."""
    docs_dir = tmp_dir / "docs"
    docs_dir.mkdir()

    # Auth documentation
    (docs_dir / "auth.md").write_text(
        "# Authentication\n\n"
        "All API requests require a Bearer token in the Authorization header. "
        "Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint. "
        "Rate limiting is enforced at 100 requests per minute per token. "
        "If you exceed the rate limit, you'll receive a 429 status code. "
        "OAuth2 flows are supported for third-party integrations. "
        "The client_id and client_secret must be stored securely. "
        "Never expose credentials in client-side code or version control. "
        "Use environment variables or a secrets manager for production deployments. "
        "Multi-factor authentication is required for admin endpoints. "
        "Session tokens are tied to IP address for security. "
        * 3
    )

    # Rate limiting documentation
    (docs_dir / "rate-limits.md").write_text(
        "# Rate Limits\n\n"
        "The API enforces the following rate limits:\n"
        "- Free tier: 10 requests per minute\n"
        "- Pro tier: 100 requests per minute\n"
        "- Enterprise: 1000 requests per minute\n\n"
        "Rate limit headers are included in every response:\n"
        "- X-RateLimit-Limit: Maximum requests allowed\n"
        "- X-RateLimit-Remaining: Requests remaining\n"
        "- X-RateLimit-Reset: Unix timestamp when limit resets\n\n"
        "When rate limited, the response includes a Retry-After header. "
        "Implement exponential backoff in your client. "
        "Batch endpoints have separate, higher limits. "
        "WebSocket connections have a message rate limit of 60 messages per minute. "
        "Exceeding limits temporarily blocks the API key for 5 minutes. "
        * 3
    )

    # Deployment guide
    (docs_dir / "deployment.md").write_text(
        "# Deployment Guide\n\n"
        "## Docker\n"
        "Build the image: `docker build -t myapp .`\n"
        "Run with: `docker run -p 8080:8080 myapp`\n\n"
        "## Kubernetes\n"
        "Apply manifests: `kubectl apply -f k8s/`\n"
        "The service uses a HorizontalPodAutoscaler with CPU target of 70%. "
        "Minimum 2 replicas, maximum 10 replicas. "
        "Persistent volumes are required for the database. "
        "Use ConfigMaps for environment-specific settings. "
        "Secrets should be managed via external-secrets-operator. "
        "Health checks are configured on /health and /ready endpoints. "
        "The readiness probe has an initial delay of 10 seconds. "
        "Rolling updates with maxUnavailable=1 and maxSurge=1. "
        * 3
    )

    # Near-duplicate of auth (for dedup testing)
    (docs_dir / "auth2.md").write_text(
        "# Authentication Guide\n\n"
        "All API requests require a Bearer token in the Authorization header. "
        "Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint. "
        "Rate limiting is enforced at 100 requests per minute per token. "
        "If you exceed the rate limit, you'll receive a 429 status code. "
        "OAuth2 flows are supported for third-party integrations. "
        "The client_id and client_secret must be stored securely. "
        "Never expose credentials in client-side code or version control. "
        "Use environment variables or a secrets manager for production deployments. "
        "Multi-factor authentication is required for admin endpoints. "
        "Session tokens are tied to IP address for extra security measures. "
        * 3
    )

    # Python code file
    (docs_dir / "client.py").write_text(
        '"""\nAPI Client for the service.\n"""\n\n'
        "import requests\n"
        "import time\n\n"
        "class APIClient:\n"
        '    """HTTP client with retry and rate limit handling."""\n\n'
        "    def __init__(self, base_url: str, token: str):\n"
        "        self.base_url = base_url\n"
        "        self.token = token\n"
        "        self.session = requests.Session()\n"
        '        self.session.headers["Authorization"] = f"Bearer {token}"\n\n'
        "    def get(self, path: str, **kwargs):\n"
        '        """GET request with retry."""\n'
        "        for attempt in range(3):\n"
        "            resp = self.session.get(f'{self.base_url}{path}', **kwargs)\n"
        "            if resp.status_code == 429:\n"
        "                wait = int(resp.headers.get('Retry-After', 5))\n"
        "                time.sleep(wait)\n"
        "                continue\n"
        "            return resp\n"
        "        raise Exception('Rate limited after 3 retries')\n\n"
        "    def post(self, path: str, data=None, **kwargs):\n"
        '        """POST request with retry."""\n'
        "        for attempt in range(3):\n"
        "            resp = self.session.post(f'{self.base_url}{path}', json=data, **kwargs)\n"
        "            if resp.status_code == 429:\n"
        "                wait = int(resp.headers.get('Retry-After', 5))\n"
        "                time.sleep(wait)\n"
        "                continue\n"
        "            return resp\n"
        "        raise Exception('Rate limited after 3 retries')\n"
    )

    return docs_dir


@pytest.fixture
def large_docs(tmp_dir):
    """Create a large document set for stress testing."""
    docs_dir = tmp_dir / "large_docs"
    docs_dir.mkdir()

    topics = [
        ("machine-learning", "Machine learning models use gradient descent to optimize loss functions. "
         "Neural networks consist of layers of interconnected nodes. "),
        ("databases", "PostgreSQL supports JSONB for semi-structured data storage. "
         "Indexes improve query performance on frequently accessed columns. "),
        ("networking", "TCP provides reliable ordered delivery of data between applications. "
         "DNS resolves domain names to IP addresses using a hierarchical system. "),
        ("security", "TLS encrypts data in transit between client and server. "
         "CORS headers control which origins can access API resources. "),
        ("devops", "CI/CD pipelines automate building, testing, and deploying code. "
         "Infrastructure as code tools like Terraform manage cloud resources. "),
        ("frontend", "React components re-render when state or props change. "
         "CSS Grid and Flexbox provide powerful layout capabilities. "),
        ("api-design", "REST APIs use HTTP methods to perform CRUD operations. "
         "GraphQL allows clients to request exactly the data they need. "),
        ("testing", "Unit tests verify individual functions in isolation. "
         "Integration tests check that components work together correctly. "),
        ("monitoring", "Prometheus collects metrics from instrumented applications. "
         "Grafana dashboards visualize time-series data for observability. "),
        ("caching", "Redis provides in-memory key-value storage with persistence options. "
         "CDN edge caching reduces latency for static assets. "),
    ]

    for i in range(50):
        topic_name, content = topics[i % len(topics)]
        filename = f"{topic_name}-{i:03d}.md"
        # Each file ~2000 words
        (docs_dir / filename).write_text(
            f"# {topic_name.replace('-', ' ').title()} - Part {i}\n\n"
            + (content * 80)
        )

    return docs_dir


@pytest.fixture
def indexed_ts(tmp_dir, sample_docs, shared_model):
    """Return a TokenShrink instance with sample docs already indexed."""
    from tokenshrink import TokenShrink

    ts = TokenShrink(
        index_dir=str(tmp_dir / ".tokenshrink"),
        compression=False,
        adaptive=True,
        dedup=True,
    )
    # Inject the shared model to avoid reloading
    ts._model = shared_model
    ts.index(str(sample_docs))
    return ts


@pytest.fixture
def make_ts(tmp_dir, shared_model):
    """Factory fixture: creates a TokenShrink with the shared model."""
    from tokenshrink import TokenShrink

    def _make(**kwargs):
        kwargs.setdefault("index_dir", str(tmp_dir / ".ts"))
        kwargs.setdefault("compression", False)
        ts = TokenShrink(**kwargs)
        ts._model = shared_model
        return ts

    return _make
