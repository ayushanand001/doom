"""Play any generated Tier 1-4 map interactively as a human in ViZDoom."""

from __future__ import annotations

import argparse
from pathlib import Path

import vizdoom as vzd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", type=str, default="training/generated_maps/pack_0.wad")
    ap.add_argument("--map", type=str, default="MAP03", help="MAP01=T1, MAP02=T2, MAP03=T3, MAP04=T4")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    wad_path = Path(args.wad) if Path(args.wad).is_absolute() else root / args.wad

    g = vzd.DoomGame()
    g.set_doom_game_path(vzd.scenarios_path + "/freedoom2.wad")
    g.set_doom_scenario_path(str(wad_path))
    g.set_doom_map(args.map.upper())
    g.set_doom_skill(3)
    g.set_mode(vzd.Mode.SPECTATOR)
    g.set_window_visible(True)
    g.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    g.set_sound_enabled(True)
    g.set_render_hud(True)
    g.set_render_crosshair(True)
    g.set_available_buttons(
        [
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.USE,
            vzd.Button.SELECT_NEXT_WEAPON,
        ]
    )
    g.init()
    print(f"Playing {wad_path.name} ({args.map.upper()}) — close the window or press ESC to quit.")
    g.new_episode()
    while not g.is_episode_finished():
        g.advance_action()
    g.close()


if __name__ == "__main__":
    main()
