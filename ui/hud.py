"""HUD rendering helpers."""

from __future__ import annotations

import pygame

from game.camera import camera
from game.constants import SCREEN_HEIGHT, WHITE
from world.ship import PlayerShip


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    info_font: pygame.font.Font,
    ship: PlayerShip,
    sector: int,
    energy_goal: int,
) -> None:
    """Render the overlay with status text and control instructions."""
    hud_lines = [
        f"Sector: {sector}",
        f"Energie verzameld: {ship.sector_progress}/{energy_goal}",
        f"Totale energie: {ship.total_energy}",
        f"Modus: {'Rotatie' if ship.control_mode == 'rotation' else 'Translatie'}",
        f"Hoofdthruster: {int(ship.main_throttle * 100)}% (Shift/Ctrl)",
        "Q/E stijg/daal, X throttle uit",
        "Tab: wissel thrusters, F: focus camera",
        f"Camera focus: {'aan' if camera.focus_mode else 'uit'}",
    ]
    if ship.control_mode == "rotation":
        hud_lines.append("Rotatie: A/D yaw, pijltjes pitch")
    else:
        hud_lines.append("Translatie: WASD + pijltjes RCS")
    hud_lines.append("Vang planeten, ontwijk sterren")

    for i, text in enumerate(hud_lines):
        surface = font.render(text, True, WHITE)
        screen.blit(surface, (12, 12 + i * 20))

    info_surface = info_font.render(
        "Newtoniaanse zwaartekracht houdt alles in beweging", True, WHITE
    )
    screen.blit(info_surface, (12, SCREEN_HEIGHT - 30))
