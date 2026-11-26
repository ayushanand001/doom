"""Procedural UDMF WAD generator for curriculum & generalization training (Tiers 1-4).

Generates valid, manifold, sector-light (<200 sectors) ZDoom UDMF WADs using
the template's `doom_env.wad.write_wad`, supporting:
  - Tier 1: Combat & Survival (open 1-2 rooms, enemies, pickups, exit)
  - Tier 2: Multi-Room Exploration (4-9 rooms, hallways & unlocked doors, exit)
  - Tier 3: Long-Horizon Progression (branching rooms, keycards -> locked doors -> exit)
  - Tier 4: Full Gauntlet (larger branching maps, multiple locked/unlocked doors, combat + items)
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path

from doom_env.wad import Lump, write_wad

# DoomEdNums (standard Doom2 / Freedoom2)
THING_PLAYER1 = 1
THING_ZOMBIEMAN = 3004
THING_SHOTGUNGUY = 9
THING_IMP = 3001
THING_STIMPACK = 2011
THING_MEDIKIT = 2012
THING_CLIP = 2007
THING_SHELLS = 2008
THING_ARMOR_BONUS = 2015
THING_SHOTGUN = 2001

# Keycards -> (DoomEdNum, UDMF lock number, door texture)
KEY_TYPES = {
    "red": (13, 1, "DOORRED"),
    "blue": (5, 2, "DOORBLU"),
    "yellow": (6, 3, "DOORYEL"),
}

FLOOR_TEXTURES = ("FLOOR0_1", "FLOOR4_8", "FLAT1", "FLAT14", "MFLR8_1")
WALL_TEXTURES = ("STARTAN2", "STONE2", "STARGR1", "BROWN1", "METAL1")
CEIL_TEXTURES = ("CEIL1_1", "CEIL3_5", "FLAT19")


class UDMFBuilder:
    """Minimal rectilinear UDMF map builder."""

    def __init__(self) -> None:
        self.vertices: list[tuple[float, float]] = []
        self._vmap: dict[tuple[float, float], int] = {}
        self.sectors: list[dict] = []
        self.sidedefs: list[dict] = []
        self.linedefs: list[dict] = []
        self.things: list[dict] = []

    def add_vertex(self, x: float, y: float) -> int:
        key = (float(x), float(y))
        if key not in self._vmap:
            self._vmap[key] = len(self.vertices)
            self.vertices.append(key)
        return self._vmap[key]

    def add_sector(
        self,
        floor_tex: str = "FLOOR0_1",
        ceil_tex: str = "CEIL1_1",
        floor_h: int = 0,
        ceil_h: int = 128,
        light: int = 192,
    ) -> int:
        idx = len(self.sectors)
        self.sectors.append(
            {
                "texturefloor": floor_tex,
                "textureceiling": ceil_tex,
                "heightfloor": floor_h,
                "heightceiling": ceil_h,
                "lightlevel": light,
            }
        )
        return idx

    def add_sidedef(
        self,
        sector: int,
        top: str = "-",
        mid: str = "-",
        bot: str = "-",
    ) -> int:
        idx = len(self.sidedefs)
        self.sidedefs.append(
            {
                "sector": sector,
                "texturetop": top,
                "texturemiddle": mid,
                "texturebottom": bot,
            }
        )
        return idx

    def add_linedef(self, v1: int, v2: int, sidefront: int, **attrs) -> int:
        idx = len(self.linedefs)
        ld = {"v1": v1, "v2": v2, "sidefront": sidefront, **attrs}
        self.linedefs.append(ld)
        return idx

    def add_thing(self, x: float, y: float, thing_type: int, angle: int = 0) -> None:
        self.things.append(
            {
                "x": float(x),
                "y": float(y),
                "angle": int(angle),
                "type": int(thing_type),
                "skill1": True,
                "skill2": True,
                "skill3": True,
                "skill4": True,
                "skill5": True,
                "single": True,
            }
        )

    def to_textmap(self) -> str:
        lines = ['namespace = "zdoom";']
        for x, y in self.vertices:
            lines.append(f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}")
        for s in self.sectors:
            lines.append(
                f'sector {{ texturefloor = "{s["texturefloor"]}"; '
                f'textureceiling = "{s["textureceiling"]}"; '
                f'heightfloor = {s["heightfloor"]}; '
                f'heightceiling = {s["heightceiling"]}; '
                f'lightlevel = {s["lightlevel"]}; }}'
            )
        for sd in self.sidedefs:
            lines.append(
                f'sidedef {{ sector = {sd["sector"]}; '
                f'texturetop = "{sd["texturetop"]}"; '
                f'texturemiddle = "{sd["texturemiddle"]}"; '
                f'texturebottom = "{sd["texturebottom"]}"; }}'
            )
        for ld in self.linedefs:
            parts = []
            for k, v in ld.items():
                if isinstance(v, bool):
                    parts.append(f"{k} = {'true' if v else 'false'};")
                elif isinstance(v, str):
                    parts.append(f'{k} = "{v}";')
                else:
                    parts.append(f"{k} = {v};")
            lines.append("linedef { " + " ".join(parts) + " }")
        for th in self.things:
            parts = []
            for k, v in th.items():
                if isinstance(v, bool):
                    parts.append(f"{k} = {'true' if v else 'false'};")
                elif isinstance(v, float):
                    parts.append(f"{k} = {v:.1f};")
                else:
                    parts.append(f"{k} = {v};")
            lines.append("thing { " + " ".join(parts) + " }")
        return "\n".join(lines) + "\n"


def generate_map(seed: int, tier: int = 4) -> str:
    """Generate a procedural UDMF TEXTMAP string for the given seed and tier (1..4)."""
    rng = random.Random(seed)

    if tier == 0:
        cols, rows = (2, 2)
        target_rooms = 4
        door_prob = 0.0
        n_locks = 0
        enemy_range = (4, 6)
    elif tier == 1:
        cols, rows = rng.choice([(2, 1), (2, 2)])
        target_rooms = cols * rows
        door_prob = 0.0
        n_locks = 0
        enemy_range = (1, 3)
    elif tier == 2:
        cols, rows = rng.choice([(3, 2), (3, 3)])
        target_rooms = rng.randint(4, min(7, cols * rows))
        door_prob = 0.6
        n_locks = 0
        enemy_range = (2, 5)
    elif tier == 3:
        cols, rows = rng.choice([(3, 3), (4, 3)])
        target_rooms = rng.randint(5, min(9, cols * rows))
        door_prob = 0.5
        n_locks = 1
        enemy_range = (2, 5)
    else:  # tier 4
        cols, rows = rng.choice([(4, 3), (4, 4)])
        target_rooms = rng.randint(7, min(12, cols * rows))
        door_prob = 0.55
        n_locks = rng.choice([1, 2])
        enemy_range = (4, 8)

    # Carve a connected spanning tree of rooms on (cols x rows)
    start = (0, 0)
    visited = {start}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {start: []}
    frontier = [start]
    order = [start]

    while len(visited) < target_rooms and frontier:
        cur = rng.choice(frontier)
        cx, cy = cur
        nbrs = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nbrs = [n for n in nbrs if 0 <= n[0] < cols and 0 <= n[1] < rows and n not in visited]
        if not nbrs:
            frontier.remove(cur)
            continue
        nxt = rng.choice(nbrs)
        visited.add(nxt)
        adj[cur].append(nxt)
        adj[nxt] = [cur]
        frontier.append(nxt)
        order.append(nxt)

    # Optionally add 1-2 extra loop edges in Tier 2 or Tier 4 (avoid breaking lock gating in Tier 3)
    if tier in (1, 2) and len(visited) >= 4:
        for _ in range(2):
            a = rng.choice(order)
            cx, cy = a
            nbrs = [(cx + dx, cy + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            nbrs = [n for n in nbrs if n in visited and n not in adj[a]]
            if nbrs:
                b = rng.choice(nbrs)
                adj[a].append(b)
                adj[b].append(a)

    # BFS distances from start to place exit at farthest room and keys on side branches
    dist = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)

    exit_room = None if tier == 0 else max(dist, key=lambda k: dist[k])

    # Path from start to exit_room
    main_path = []
    if exit_room is not None:
        cur_p = exit_room
        while cur_p is not None:
            main_path.append(cur_p)
            cur_p = parent[cur_p]
        main_path.reverse()

    # Assign edge types: "hall", "door", or ("lock", key_color)
    edge_kind: dict[tuple[tuple[int, int], tuple[int, int]], tuple[str, str | None]] = {}

    def set_edge(a, b, kind, color=None):
        edge = (min(a, b), max(a, b))
        edge_kind[edge] = (kind, color)

    for u in adj:
        for v in adj[u]:
            edge = (min(u, v), max(u, v))
            if edge not in edge_kind:
                kind = "door" if rng.random() < door_prob else "hall"
                set_edge(u, v, kind)

    # Place locked doors on the main path & matching keycards in reachable rooms before the lock
    key_placements: list[tuple[tuple[int, int], str]] = []
    if n_locks > 0 and len(main_path) >= 3:
        avail_colors = rng.sample(list(KEY_TYPES.keys()), k=min(n_locks, 3))
        # Lock edge right before exit_room (guaranteed reachable from earlier rooms)
        lock_idx = len(main_path) - 2
        u, v = main_path[lock_idx], main_path[lock_idx + 1]
        color0 = avail_colors[0]
        set_edge(u, v, "lock", color0)

        # Find all rooms reachable from start without crossing any locked edge
        def reachable_without_locks() -> list[tuple[int, int]]:
            seen = {start}
            dq = deque([start])
            while dq:
                x = dq.popleft()
                for y in adj[x]:
                    ek = edge_kind[(min(x, y), max(x, y))][0]
                    if ek == "lock" or y in seen:
                        continue
                    seen.add(y)
                    dq.append(y)
            return list(seen)

        reach = [r for r in reachable_without_locks() if r != start]
        key_room = max(reach, key=lambda r: dist[r]) if reach else start
        key_placements.append((key_room, color0))

    # Build rectilinear grid coordinates:
    # Even grid coords = rooms (256 units), odd grid coords = connectors (64 units)
    ROOM_SIZE = 256.0
    CONN_SIZE = 64.0
    gw = 2 * cols - 1
    gh = 2 * rows - 1

    def col_bounds(gx: int) -> tuple[float, float]:
        x = 0.0
        for i in range(gx):
            x += ROOM_SIZE if i % 2 == 0 else CONN_SIZE
        w = ROOM_SIZE if gx % 2 == 0 else CONN_SIZE
        return x, x + w

    def row_bounds(gy: int) -> tuple[float, float]:
        y = 0.0
        for i in range(gy):
            y += ROOM_SIZE if i % 2 == 0 else CONN_SIZE
        h = ROOM_SIZE if gy % 2 == 0 else CONN_SIZE
        return y, y + h

    # Map walkable grid cells -> sector metadata
    floor_tex = rng.choice(FLOOR_TEXTURES)
    wall_tex = rng.choice(WALL_TEXTURES)
    ceil_tex = rng.choice(CEIL_TEXTURES)

    builder = UDMFBuilder()
    cell_sector: dict[tuple[int, int], int] = {}
    cell_props: dict[tuple[int, int], dict] = {}

    for rx, ry in visited:
        gx, gy = 2 * rx, 2 * ry
        is_exit = (rx, ry) == exit_room
        sec_idx = builder.add_sector(
            floor_tex="GATE3" if is_exit else floor_tex,
            ceil_tex=ceil_tex,
            floor_h=0,
            ceil_h=128,
            light=224 if is_exit else rng.choice([176, 192, 208]),
        )
        cell_sector[(gx, gy)] = sec_idx
        cell_props[(gx, gy)] = {"kind": "exit" if is_exit else "room"}

    for (a, b), (kind, color) in edge_kind.items():
        gx = a[0] + b[0]
        gy = a[1] + b[1]
        if kind == "hall":
            sec_idx = builder.add_sector(floor_tex=floor_tex, ceil_tex=ceil_tex, floor_h=0, ceil_h=128)
            cell_sector[(gx, gy)] = sec_idx
            cell_props[(gx, gy)] = {"kind": "hall"}
        else:
            sec_idx = builder.add_sector(floor_tex="FLAT5_4", ceil_tex=ceil_tex, floor_h=0, ceil_h=0)
            cell_sector[(gx, gy)] = sec_idx
            cell_props[(gx, gy)] = {"kind": kind, "color": color}

    # Emit linedefs for every cell boundary (avoiding duplicate shared edges)
    emitted_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    def room_center(rx: int, ry: int) -> tuple[float, float]:
        x0, x1 = col_bounds(2 * rx)
        y0, y1 = row_bounds(2 * ry)
        return (x0 + x1) * 0.5, (y0 + y1) * 0.5

    for (gx, gy), sec in cell_sector.items():
        x0, x1 = col_bounds(gx)
        y0, y1 = row_bounds(gy)
        # 4 edges in clockwise order around the cell interior
        # (dx_nbr, dy_nbr, p1, p2) where p1->p2 has interior on the RIGHT (Doom front-side convention)
        sides = [
            (0, 1, (x0, y1), (x1, y1)),   # top
            (1, 0, (x1, y1), (x1, y0)),   # right
            (0, -1, (x1, y0), (x0, y0)),  # bottom
            (-1, 0, (x0, y0), (x0, y1)),  # left
        ]
        for dx, dy, p1, p2 in sides:
            ekey = (min(p1, p2), max(p1, p2))
            if ekey in emitted_edges:
                continue
            emitted_edges.add(ekey)

            ngx, ngy = gx + dx, gy + dy
            nsec = cell_sector.get((ngx, ngy))
            v1 = builder.add_vertex(*p1)
            v2 = builder.add_vertex(*p2)

            if nsec is None:
                # One-sided solid wall
                is_door_track = cell_props[(gx, gy)]["kind"] in ("door", "lock")
                tex = "DOORTRAK" if is_door_track else wall_tex
                sd = builder.add_sidedef(sec, mid=tex)
                builder.add_linedef(v1, v2, sd, blocking=True, dontpegbottom=True)
            else:
                # Two-sided portal between `sec` and `nsec`
                p_cur = cell_props[(gx, gy)]
                p_nbr = cell_props[(ngx, ngy)]

                # Check if either side is a door/locked door
                door_prop = p_cur if p_cur["kind"] in ("door", "lock") else (p_nbr if p_nbr["kind"] in ("door", "lock") else None)
                if door_prop is not None:
                    # Front side faces into the room, back side is the door sector
                    if p_nbr["kind"] in ("door", "lock"):
                        front_sec, back_sec = sec, nsec
                        lv1, lv2 = v1, v2
                    else:
                        front_sec, back_sec = nsec, sec
                        lv1, lv2 = v2, v1

                    if door_prop["kind"] == "lock":
                        _, lock_num, dtex = KEY_TYPES[door_prop["color"]]
                        sd_f = builder.add_sidedef(front_sec, top=dtex)
                        sd_b = builder.add_sidedef(back_sec)
                        builder.add_linedef(
                            lv1, lv2, sd_f,
                            sideback=sd_b,
                            twosided=True,
                            special=13,
                            arg0=0,
                            arg1=16,
                            arg2=150,
                            arg3=lock_num,
                            playeruse=True,
                            repeatspecial=True,
                        )
                    else:
                        sd_f = builder.add_sidedef(front_sec, top="BIGDOOR2")
                        sd_b = builder.add_sidedef(back_sec)
                        builder.add_linedef(
                            lv1, lv2, sd_f,
                            sideback=sd_b,
                            twosided=True,
                            special=12,
                            arg0=0,
                            arg1=16,
                            arg2=150,
                            playeruse=True,
                            monsteruse=True,
                            repeatspecial=True,
                        )
                else:
                    sd_f = builder.add_sidedef(sec)
                    sd_b = builder.add_sidedef(nsec)
                    builder.add_linedef(v1, v2, sd_f, sideback=sd_b, twosided=True)

    # Add guaranteed Exit_Normal (243) trigger box in the center of exit_room (never blocked by doors)
    if exit_room is not None:
        ex_sec = cell_sector[(2 * exit_room[0], 2 * exit_room[1])]
        ex_cx, ex_cy = room_center(*exit_room)
        r_pad = 48.0
        ev0 = builder.add_vertex(ex_cx - r_pad, ex_cy - r_pad)
        ev1 = builder.add_vertex(ex_cx - r_pad, ex_cy + r_pad)
        ev2 = builder.add_vertex(ex_cx + r_pad, ex_cy + r_pad)
        ev3 = builder.add_vertex(ex_cx + r_pad, ex_cy - r_pad)
        esd = builder.add_sidedef(ex_sec)
        for va, vb in ((ev0, ev1), (ev1, ev2), (ev2, ev3), (ev3, ev0)):
            builder.add_linedef(va, vb, esd, sideback=esd, twosided=True, special=243, playercross=True)

    # Place player start in (0, 0)
    px, py = room_center(*start)
    builder.add_thing(px, py, THING_PLAYER1, angle=rng.choice([0, 90, 180, 270]))

    # Place keycards
    for kroom, color in key_placements:
        kx, ky = room_center(*kroom)
        builder.add_thing(kx + rng.uniform(-32, 32), ky + rng.uniform(-32, 32), KEY_TYPES[color][0])

    # Place enemies in non-start rooms
    non_start = [r for r in order if r != start]
    n_enemies = rng.randint(*enemy_range)
    enemy_pool = [THING_ZOMBIEMAN] if tier == 1 else [THING_ZOMBIEMAN, THING_SHOTGUNGUY, THING_IMP]
    for _ in range(n_enemies):
        if not non_start:
            break
        er = rng.choice(non_start)
        ex, ey = room_center(*er)
        builder.add_thing(
            ex + rng.uniform(-64, 64),
            ey + rng.uniform(-64, 64),
            rng.choice(enemy_pool),
            angle=rng.choice([0, 90, 180, 270]),
        )

    # Place helpful pickups (ammo, health, shotgun in tier 0 / tiers 2-4)
    if tier == 0:
        builder.add_thing(px + 24, py, THING_SHOTGUN)
        for dx, dy in ((-24, 0), (0, 24), (0, -24)):
            builder.add_thing(px + dx, py + dy, THING_SHELLS)
            builder.add_thing(px + dx, py + dy, THING_CLIP)
        builder.add_thing(px + 24, py + 24, THING_STIMPACK)
    elif tier >= 2 and len(order) >= 2:
        sr = rng.choice(order[:-1])
        sx, sy = room_center(*sr)
        builder.add_thing(sx + 48, sy - 48, THING_SHOTGUN)
    for _ in range(rng.randint(1, 3)):
        ir = rng.choice(order)
        ix, iy = room_center(*ir)
        builder.add_thing(
            ix + rng.uniform(-64, 64),
            iy + rng.uniform(-64, 64),
            rng.choice([THING_STIMPACK, THING_CLIP, THING_SHELLS, THING_ARMOR_BONUS]),
        )

    return builder.to_textmap()


def build_wad(out_path: str | Path, seeds_and_tiers: list[tuple[int, int]]) -> list[str]:
    """Write a multi-map PWAD (`MAP01`, `MAP02`, ...) and return the map names."""
    lumps: list[Lump] = []
    names: list[str] = []
    for idx, (seed, tier) in enumerate(seeds_and_tiers, start=1):
        mname = f"MAP{idx:02d}"
        names.append(mname)
        textmap = generate_map(seed=seed, tier=tier)
        lumps.append(Lump(mname, b""))
        lumps.append(Lump("TEXTMAP", textmap.encode("utf-8")))
        lumps.append(Lump("ENDMAP", b""))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(out_path, lumps)
    return names


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "generated_maps" / "train_pack.wad"
    # 500 maps across Tiers 1-4 (125 each) packed into a single fast-loading PWAD
    specs = [(s, (s % 4) + 1) for s in range(500)]
    # ViZDoom supports MAP01..MAP99 cleanly; split into 5 packs of 99 maps (~495 maps)
    for pack_idx in range(5):
        chunk = specs[pack_idx * 99 : (pack_idx + 1) * 99]
        pack_path = out.parent / f"pack_{pack_idx}.wad"
        mnames = build_wad(pack_path, chunk)
        print(f"Wrote {pack_path.name}: {len(mnames)} maps ({mnames[0]}..{mnames[-1]})")
    # Combat-only arenas (tier 0)
    combat_specs = [(9000 + s, 0) for s in range(32)]
    cnames = build_wad(out.parent / "combat_pack.wad", combat_specs)
    print(f"Wrote combat_pack.wad: {len(cnames)} maps ({cnames[0]}..{cnames[-1]})")
