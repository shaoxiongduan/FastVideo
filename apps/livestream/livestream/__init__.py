"""Chat-driven FastH3 infinite livestream.

One process turns viewer prompts into a continuous broadcast: chat feeds a
director, the director rewrites prompts and queues clips, the engine builds
them on FastVideo and paces them into an HLS playlist the app serves itself.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
