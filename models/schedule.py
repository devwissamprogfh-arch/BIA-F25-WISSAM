"""Schedule: the candidate solution shared by GA, PSO, ACO and the evaluator.

A schedule is two integer lists aligned with ``problem.events``:

    slots[i] = global start slot of event i       (UNASSIGNED = -1 if not placed)
    rooms[i] = index of the room of event i       (UNASSIGNED = -1 if not placed)

This flat integer encoding was chosen so that every algorithm can use it:
    * GA  : the pair (slots[i], rooms[i]) is one gene of a chromosome.
    * PSO : the particle position is the flat vector from ``to_vector()``; real
            valued positions are rounded/clipped to integers by the PSO code.
    * ACO : an ant fills the lists one event at a time (partial schedules are
            allowed and are penalised by the evaluator until completed).
"""
from __future__ import annotations

import csv
from typing import Dict, List, Sequence, Tuple

UNASSIGNED = -1


class Schedule:
    """Mutable assignment of every event to a (start slot, room index) pair."""

    __slots__ = ("slots", "rooms")

    def __init__(self, slots: Sequence[int], rooms: Sequence[int]):
        if len(slots) != len(rooms):
            raise ValueError("slots and rooms must have the same length")
        self.slots: List[int] = list(slots)
        self.rooms: List[int] = list(rooms)

    # ------------------------------------------------------------------ factories
    @classmethod
    def empty(cls, n_events: int) -> "Schedule":
        """A schedule where no event is placed yet."""
        return cls([UNASSIGNED] * n_events, [UNASSIGNED] * n_events)

    @classmethod
    def from_vector(cls, vector: Sequence[int]) -> "Schedule":
        """Inverse of :meth:`to_vector` (layout: slot0, room0, slot1, room1, ...)."""
        if len(vector) % 2:
            raise ValueError("vector length must be even")
        return cls([int(v) for v in vector[0::2]], [int(v) for v in vector[1::2]])

    @classmethod
    def from_mapping(cls, event_ids: Sequence[str], room_ids: Sequence[str],
                     mapping: Dict[str, Tuple[int, str]]) -> "Schedule":
        """Build a schedule from ``{event_id: (start_slot, room_id)}``.

        Events missing from the mapping stay UNASSIGNED.
        """
        room_index = {rid: k for k, rid in enumerate(room_ids)}
        sched = cls.empty(len(event_ids))
        for i, eid in enumerate(event_ids):
            if eid in mapping:
                slot, room_id = mapping[eid]
                sched.slots[i] = int(slot)
                sched.rooms[i] = room_index[room_id]
        return sched

    # ------------------------------------------------------------------ accessors
    @property
    def n_events(self) -> int:
        return len(self.slots)

    def copy(self) -> "Schedule":
        return Schedule(self.slots, self.rooms)

    def assign(self, i: int, slot: int, room: int) -> None:
        """Place event ``i`` at ``slot`` in room index ``room``."""
        self.slots[i] = slot
        self.rooms[i] = room

    def is_complete(self) -> bool:
        """True when every event has been given a slot and a room."""
        return all(s != UNASSIGNED for s in self.slots) and all(r != UNASSIGNED for r in self.rooms)

    def to_vector(self) -> List[int]:
        """Flatten to [slot0, room0, slot1, room1, ...]."""
        flat: List[int] = []
        for s, r in zip(self.slots, self.rooms):
            flat.append(s)
            flat.append(r)
        return flat

    def to_mapping(self, event_ids: Sequence[str], room_ids: Sequence[str]) -> Dict[str, Tuple[int, str]]:
        """Return ``{event_id: (start_slot, room_id)}`` for the placed events."""
        out: Dict[str, Tuple[int, str]] = {}
        for i, eid in enumerate(event_ids):
            if self.slots[i] != UNASSIGNED and 0 <= self.rooms[i] < len(room_ids):
                out[eid] = (self.slots[i], room_ids[self.rooms[i]])
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Schedule) and self.slots == other.slots and self.rooms == other.rooms

    def __repr__(self) -> str:
        return f"Schedule(n_events={self.n_events}, complete={self.is_complete()})"


# ---------------------------------------------------------------------- CSV I/O
def save_schedule_csv(problem, schedule: Schedule, path: str) -> None:
    """Write a schedule as CSV (event_id, start_slot, room_id, time label).

    ``problem`` is only used for ids and time labels (duck typed to avoid a
    circular import between ``models`` and ``core``).
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["event_id", "start_slot", "room_id", "time"])
        for i, eid in enumerate(problem.event_ids):
            s, r = schedule.slots[i], schedule.rooms[i]
            if s == UNASSIGNED or r == UNASSIGNED:
                writer.writerow([eid, "", "", ""])
            else:
                writer.writerow([eid, s, problem.room_ids[r], problem.grid.label(s)])


def load_schedule_csv(problem, path: str) -> Schedule:
    """Read a schedule written by :func:`save_schedule_csv`."""
    mapping: Dict[str, Tuple[int, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["start_slot"] != "" and row["room_id"] != "":
                mapping[row["event_id"]] = (int(row["start_slot"]), row["room_id"])
    return Schedule.from_mapping(problem.event_ids, problem.room_ids, mapping)
