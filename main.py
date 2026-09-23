"""Relationship Memory — terminal entry point.

The interface is one of three seams: Telegram or voice will call
`run_agent(text, owner_id)` directly and never import this file.
"""

from app.interfaces.cli import main

if __name__ == "__main__":
    main()
