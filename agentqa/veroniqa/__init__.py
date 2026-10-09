"""VeroniQA: a conversational agent over AgentQA.

Projects are folders; each has its own knowledge base (uploaded documents and linked pages,
chunked, scanned for prompt injection and stored in a per-project Chroma collection). VeroniQA
answers questions from that knowledge with citations (retrieval-augmented generation) and drives
the repository's features: running test suites, listing runs and reading reports.
"""

from agentqa.veroniqa.agent import Reply, VeroniQA
from agentqa.veroniqa.projects import Project, create_demo_project, create_project, list_projects

__all__ = ["Project", "Reply", "VeroniQA", "create_demo_project", "create_project", "list_projects"]
