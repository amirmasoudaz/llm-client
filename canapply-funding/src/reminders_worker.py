# src/reminders_worker.py
"""
Reminders Worker

A standalone background process that runs the periodic reply check and reminder cycle.
This does not expose an HTTP port; it operates purely as a background worker.

Usage:
    python src/reminders_worker.py
"""

import asyncio

from src.config import settings
from src.tools.logger import Logger
from src.outreach.logic import OutreachLogic


_LOG, _ = Logger().create(
    application="reminders_worker",
    file_name="reminders_worker",
    logger_name="reminders_worker",
)


async def main():
    reminders_enabled = settings.REMINDERS_ON
    
    if not reminders_enabled:
        _LOG.info("Reminders disabled (REMINDERS_ON is not True). Worker exiting.")
        print("Reminders disabled (REMINDERS_ON is not True). Worker exiting.")
        return

    _LOG.info("Starting Reminders Worker...")
    print("Starting Reminders Worker...")
    
    logic = OutreachLogic()
    await logic.start_reminder_cron()


if __name__ == "__main__":
    asyncio.run(main())
