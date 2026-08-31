"""Tests for EEPROM prefetch and memory cache coordination."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from velbusaio.actions import ActionSlot, build_action_tables
from velbusaio.controller import Velbus
from velbusaio.memory import MemoryBackend
from velbusaio.memory_prefetch import MemoryAccessCoordinator
from velbusaio.messages.memory_data_block import MemoryDataBlockMessage
from velbusaio.messages.read_data_block_from_memory import (
    ReadDataBlockFromMemoryMessage,
)
from velbusaio.module import Module


class TestMemoryBackendVlpSeed:
    """Tests for seeding EEPROM cache from VLP hex dumps."""

    def test_seed_from_vlp_hex(self):
        """Test Seed from vlp hex."""
        backend = MemoryBackend(0x11, AsyncMock())
        backend.seed_from_vlp_hex("00FF12AB")
        assert backend.get_cached(0) == 0x00
        assert backend.get_cached(1) == 0xFF
        assert backend.get_cached(2) == 0x12
        assert backend.get_cached(3) == 0xAB


class TestModulePrefetch:
    """Tests for module-level EEPROM prefetch."""

    @pytest.mark.asyncio
    async def test_prefetch_fills_cache_for_action_table(self, tmp_path):
        """Test Prefetch fills cache for action table."""
        store = dict.fromkeys(range(0x0000, 0x0100), 0xFF)
        coordinator = MemoryAccessCoordinator(Velbus(""))
        writer = AsyncMock()
        backend = MemoryBackend(
            0x11, writer, access_coordinator=coordinator, timeout=1.0
        )

        async def respond(msg):
            if isinstance(msg, ReadDataBlockFromMemoryMessage):
                addr = (msg.high_address << 8) | msg.low_address
                data = bytes(store.get(addr + i, 0xFF) for i in range(4))
                reply = MemoryDataBlockMessage(0x11)
                reply.high_address = msg.high_address
                reply.low_address = msg.low_address
                reply.data = data
                backend.feed_message(reply)

        writer.side_effect = respond

        spec = {
            "actions": "relay_classic",
            "channels": {"01": {"bank": "0000"}},
            "slot_count": 4,
            "slot_size": 6,
        }
        tables = build_action_tables(backend, spec)
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        module._memory = backend
        module._action_tables = tables

        await module.prefetch_config_memory(coordinator)
        assert module.is_panel_metadata_ready()
        assert tables[1].loaded
        writer.reset_mock()
        slots = await module.get_channel_actions(1, include_empty=True)
        assert len(slots) == 4
        writer.assert_not_called()

    @pytest.mark.asyncio
    async def test_decode_from_cache_without_bus(self, tmp_path):
        """Test Decode from cache without bus."""
        writer = AsyncMock()
        backend = MemoryBackend(0x11, writer, timeout=1.0)
        spec = {
            "actions": "relay_classic",
            "channels": {"01": {"bank": "0000"}},
            "slot_count": 4,
            "slot_size": 6,
        }
        tables = build_action_tables(backend, spec)
        for addr in range(0x18):
            backend._cache[addr] = 0xFF
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        module._memory = backend
        module._action_tables = tables

        slots = await module.get_channel_actions(1, include_empty=True)
        assert len(slots) == 4
        writer.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalidate_config_memory_cache_clears_decoded_state(self, tmp_path):
        """Test Invalidate config memory cache clears decoded state."""
        backend = MemoryBackend(0x11, AsyncMock())
        spec = {
            "actions": "relay_classic",
            "channels": {"01": {"bank": "0000"}},
            "slot_count": 2,
            "slot_size": 6,
        }
        tables = build_action_tables(backend, spec)
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        module._memory = backend
        module._action_tables = tables
        module._panel_metadata_ready = True
        tables[1]._slots = []
        backend._cache[0] = 0x12

        module.invalidate_config_memory_cache()
        assert tables[1]._slots is None
        assert backend.get_cached(0) is None
        assert module.is_panel_metadata_ready() is False

    @pytest.mark.asyncio
    async def test_metadata_cache_does_not_load_action_tables(self, tmp_path):
        """Test Metadata cache does not load action tables."""
        writer = AsyncMock()
        backend = MemoryBackend(0x11, writer, timeout=1.0)
        spec = {
            "actions": "relay_classic",
            "channels": {"01": {"bank": "0000"}},
            "slot_count": 39,
            "slot_size": 6,
        }
        tables = build_action_tables(backend, spec)
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        module._memory = backend
        module._action_tables = tables

        await module.ensure_panel_metadata_cache()

        assert not tables[1].loaded
        writer.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_actions_uses_loaded_slots_without_bus(self, tmp_path):
        """Test Get actions uses decoded shadow slots without bus access."""
        writer = AsyncMock()
        backend = MemoryBackend(0x11, writer, timeout=1.0)
        spec = {
            "actions": "relay_classic",
            "channels": {"01": {"bank": "0000"}},
            "slot_count": 2,
            "slot_size": 6,
        }
        tables = build_action_tables(backend, spec)
        tables[1]._slots = [
            ActionSlot.empty_slot(0),
            ActionSlot.empty_slot(1),
        ]
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        module._memory = backend
        module._action_tables = tables
        module._panel_metadata_ready = True

        slots = await module.get_channel_actions(1, include_empty=True)
        assert len(slots) == 2
        writer.assert_not_called()


class TestMemoryAccessCoordinator:
    """Tests for interactive/prefetch coordination."""

    @pytest.mark.asyncio
    async def test_interactive_pauses_prefetch(self):
        """Test Interactive pauses prefetch."""
        velbus = Velbus("")
        coordinator = velbus.get_memory_access_coordinator()
        prefetch_started = asyncio.Event()
        prefetch_resumed = asyncio.Event()

        async def prefetch_waiter():
            prefetch_started.set()
            await coordinator.wait_prefetch_allowed()
            prefetch_resumed.set()

        coordinator.begin_interactive()
        task = asyncio.create_task(prefetch_waiter())
        await prefetch_started.wait()
        await asyncio.sleep(0)
        assert not prefetch_resumed.is_set()
        coordinator.end_interactive()
        await asyncio.wait_for(prefetch_resumed.wait(), timeout=1.0)
        await task

    @pytest.mark.asyncio
    async def test_schedule_prefetch_runs_in_background(self, tmp_path):
        """Test Schedule prefetch runs in background."""
        velbus = Velbus("", cache_dir=str(tmp_path))
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        module._memory = MemoryBackend(0x11, AsyncMock())
        module._action_tables = {}
        velbus._modules[0x11] = module
        velbus._is_connected = True

        prefetch_started = asyncio.Event()

        async def fake_prefetch(coordinator):
            prefetch_started.set()
            await asyncio.sleep(0.2)

        with patch.object(Module, "prefetch_config_memory", side_effect=fake_prefetch):
            velbus.get_memory_access_coordinator().schedule_prefetch()
            await asyncio.wait_for(prefetch_started.wait(), timeout=1.0)
            assert velbus.memory_prefetch_in_progress


class TestVelbusReconnectCache:
    """Tests for reconnect cache invalidation."""

    @pytest.mark.asyncio
    async def test_reconnect_invalidates_memory_cache(self, tmp_path):
        """Test Reconnect invalidates memory cache."""
        velbus = Velbus("", cache_dir=str(tmp_path))
        module = Module(0x11, 0x16, cache_dir=str(tmp_path))
        backend = MemoryBackend(0x11, AsyncMock())
        backend._cache[0] = 0x42
        module._memory = backend
        velbus._modules[0x11] = module
        velbus._had_modules_before_connect = True

        await velbus._on_connection_state(True)

        assert backend.get_cached(0) is None
