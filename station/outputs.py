"""Output drivers for lights and the alarm (implements contracts.OutputDriver).

Invariants, identical for every driver:
- show_good:   green only (alarm off)
- show_bad:    red and alarm on, never green
- show_review: amber only, never green, no alarm
- show_fault:  fault light only
- alarm_off:   turns only the alarm off
- clear:       everything off
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional

from station.contracts import StationError

LAMPS = ("green", "red", "alarm", "amber", "fault")

_GOOD = {"green": True, "red": False, "alarm": False, "amber": False, "fault": False}
_BAD = {"green": False, "red": True, "alarm": True, "amber": False, "fault": False}
_REVIEW = {"green": False, "red": False, "alarm": False, "amber": True, "fault": False}
_FAULT = {"green": False, "red": False, "alarm": False, "amber": False, "fault": True}
_OFF = {lamp: False for lamp in LAMPS}


class _OutputBase:
    """Holds the five lamp booleans and applies the invariant table above."""

    def __init__(self) -> None:
        self.green = False
        self.red = False
        self.alarm = False
        self.amber = False
        self.fault = False

    def show_good(self) -> None:
        self._apply("show_good", _GOOD)

    def show_bad(self) -> None:
        self._apply("show_bad", _BAD)

    def show_review(self) -> None:
        self._apply("show_review", _REVIEW)

    def show_fault(self) -> None:
        self._apply("show_fault", _FAULT)

    def alarm_off(self) -> None:
        self._apply("alarm_off", {"alarm": False})

    def clear(self) -> None:
        self._apply("clear", _OFF)

    def _apply(self, action: str, values: Mapping[str, bool]) -> None:
        for lamp, on in values.items():
            setattr(self, lamp, on)
        self._emit(action, tuple(values))

    def _emit(self, action: str, lamps: tuple[str, ...]) -> None:
        """Hook for subclasses; lamps are the outputs this action wrote, in LAMPS order."""


class RecordingOutputs(_OutputBase):
    """Test double: lamp booleans plus a history of the calls made."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[str] = []

    def _emit(self, action: str, lamps: tuple[str, ...]) -> None:
        self.history.append(action)


class ConsoleOutputs(_OutputBase):
    """Writes one line per output call, e.g. 'OUTPUTS show_bad: green=OFF red=ON ...'."""

    def __init__(self, write: Optional[Callable[[str], None]] = None) -> None:
        super().__init__()
        self._write = write if write is not None else print

    def _emit(self, action: str, lamps: tuple[str, ...]) -> None:
        states = " ".join(
            f"{lamp}={'ON' if getattr(self, lamp) else 'OFF'}" for lamp in LAMPS
        )
        self._write(f"OUTPUTS {action}: {states}")


DEFAULT_CHANNEL_MAP: dict[str, int] = {"green": 1, "red": 2, "alarm": 3, "amber": 4, "fault": 5}


class SerialRelayOutputs(_OutputBase):
    """Drives a USB relay board by writing b'ON<n>\\n' / b'OFF<n>\\n' bytes.

    show_* and clear write one line for each of the five channels, in LAMPS
    order. alarm_off writes only the alarm channel.
    """

    def __init__(
        self,
        write: Callable[[bytes], None],
        channel_map: Optional[Mapping[str, int]] = None,
    ) -> None:
        super().__init__()
        channels = dict(DEFAULT_CHANNEL_MAP if channel_map is None else channel_map)
        unknown = set(channels) - set(LAMPS)
        if unknown:
            raise StationError(f"unknown relay output(s): {sorted(unknown)}")
        missing = [lamp for lamp in LAMPS if lamp not in channels]
        if missing:
            raise StationError(f"channel_map is missing: {missing}")
        numbers = [channels[lamp] for lamp in LAMPS]
        if len(set(numbers)) != len(numbers):
            raise StationError("channel_map assigns one channel to two outputs")
        self._write = write
        self._channels = {lamp: channels[lamp] for lamp in LAMPS}

    def _emit(self, action: str, lamps: tuple[str, ...]) -> None:
        for lamp in lamps:
            state = "ON" if getattr(self, lamp) else "OFF"
            line = f"{state}{self._channels[lamp]}\n"
            self._write(line.encode("ascii"))
