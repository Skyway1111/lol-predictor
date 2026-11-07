"""The five League roster positions — the one shared, ordered role vocabulary."""

from __future__ import annotations

from typing import Literal

import numpy as np

Role = Literal["top", "jng", "mid", "bot", "sup"]

#: THE canonical role tuple; every role-keyed vector/order in the tree reads this order.
ROLE_ORDER: tuple[Role, ...] = ("top", "jng", "mid", "bot", "sup")

N_ROLES: int = len(ROLE_ORDER)

#: THE role → position lookup for sorting role-keyed rows into ``ROLE_ORDER``.
ROLE_RANK: dict[str, int] = {role: rank for rank, role in enumerate(ROLE_ORDER)}

#: Picked cells on a full board, blue top..sup then red top..sup.
N_SLOTS: int = 2 * N_ROLES

#: THE half swap of those cells: the permutation every mirror in the tree applies,
#: to board columns, to slot-input rows and to the encoder's ten rows alike.
MIRROR_SLOTS: tuple[int, ...] = (*range(N_ROLES, N_SLOTS), *range(N_ROLES))

#: THE ten slots' role ids, the gather axis of every per-(champion, role) table: a board's
#: ids index the rows, these the columns. Read-only, because every reader shares this array.
SLOT_ROLES: np.ndarray = np.tile(np.arange(N_ROLES), 2)
SLOT_ROLES.flags.writeable = False
