from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecapInfo:
    think: str
    subtasks: list[dict[str, Any]]
    # Optional: when a node declares completion (subtasks == []), it should provide a
    # parent-readable summary of its deliverable/conclusion. Kept as a string so it
    # can be injected into prompts without additional rendering logic.
    result: str = ""


class Node:
    def __init__(self, task_name: str, *, role: str, parent: "Node | None" = None) -> None:
        self.children: list[Node] = []
        self.parent: Node | None = parent
        self.task_name = task_name
        self.role = role
        self.info_list: list[RecapInfo] = []
        self.obs_list: list[str] = []
        # Strict acceptance: count how many times this node failed deliverable validation
        # (e.g., missing required mechanisms/roles). Used to cap repair loops.
        self.accept_failures: int = 0

    def add_child(self, child: "Node") -> None:
        self.children.append(child)
        child.parent = self

    def set_info(self, info: RecapInfo) -> None:
        self.info_list.append(info)

    def get_latest_info(self) -> RecapInfo:
        if not self.info_list:
            return RecapInfo(think="", subtasks=[])
        return self.info_list[-1]

    def set_obs(self, obs: str) -> None:
        self.obs_list.append(obs)

    def get_latest_obs(self) -> str | None:
        if not self.obs_list:
            return None
        return self.obs_list[-1]
