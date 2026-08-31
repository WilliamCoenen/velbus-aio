"""Background EEPROM prefetch coordination for the config panel."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from velbusaio.controller import Velbus


class MemoryAccessCoordinator:
    """Pause background prefetch while interactive panel memory ops run."""

    def __init__(self, velbus: Velbus) -> None:
        """Initialize the coordinator."""
        self._velbus = velbus
        self._log = logging.getLogger("velbus-memory-prefetch")
        self._generation = 0
        self._interactive_count = 0
        self._prefetch_allowed = asyncio.Event()
        self._prefetch_allowed.set()
        self._task: asyncio.Task[None] | None = None
        self._in_progress = False

    @property
    def generation(self) -> int:
        """Return the current prefetch generation."""
        return self._generation

    @property
    def prefetch_in_progress(self) -> bool:
        """Return True while a background prefetch task is running."""
        return self._in_progress

    def begin_interactive(self) -> None:
        """Mark the start of a panel-driven memory operation."""
        self._interactive_count += 1
        if self._interactive_count == 1:
            self._prefetch_allowed.clear()

    def end_interactive(self) -> None:
        """Mark the end of a panel-driven memory operation."""
        if self._interactive_count <= 0:
            return
        self._interactive_count -= 1
        if self._interactive_count == 0:
            self._prefetch_allowed.set()

    async def wait_prefetch_allowed(self) -> None:
        """Wait until interactive memory ops are not holding the bus."""
        await self._prefetch_allowed.wait()

    def bump_generation(self) -> int:
        """Invalidate any in-flight prefetch and return the new generation."""
        self._generation += 1
        return self._generation

    def schedule_prefetch(self) -> None:
        """Start or restart background EEPROM prefetch for all modules."""
        if not self._velbus.get_modules():
            return
        self.cancel_prefetch()
        self._task = asyncio.create_task(self._run_prefetch())
        self._velbus.add_background_task(self._task)

    def cancel_prefetch(self) -> None:
        """Cancel any running prefetch task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run_prefetch(self) -> None:
        generation = self._generation
        self._in_progress = True
        modules = self._velbus.get_modules()
        self._log.debug("Starting EEPROM prefetch for %d module(s)", len(modules))
        try:
            for module in modules.values():
                if generation != self._generation:
                    return
                if not self._velbus.connected:
                    return
                try:
                    await module.prefetch_config_memory(self)
                except Exception:  # noqa: BLE001 - one module must not abort the rest
                    self._log.exception(
                        "EEPROM prefetch failed for module %s", module.get_addresses()
                    )
                await asyncio.sleep(0)
        finally:
            self._in_progress = False
            if generation == self._generation:
                self._log.debug("EEPROM prefetch finished")
