import math
import random
from typing import List

import pygame

# Screen and simulation constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
G = 6.674e-3  # Gravitational constant scaled for the simulation

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 80, 60)
GREEN = (0, 255, 160)
BLUE = (100, 150, 255)
LIGHT_BLUE = (160, 200, 255)


class Planet:
    """A class that represents a planet orbiting a star."""

    def __init__(self, mass: float, x: float, y: float, vx: float, vy: float, color: tuple[int, int, int]):
        self.mass = mass
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.radius = max(2, int(self.mass ** (1.0 / 3.0)))
        self.orbit_radius: float | None = None

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def apply_force(self, fx: float, fy: float, dt: float) -> None:
        self.vx += (fx / self.mass) * dt
        self.vy += (fy / self.mass) * dt

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt

    def attract(self, other: "Planet") -> tuple[float, float]:
        dx = other.x - self.x
        dy = other.y - self.y
        r_sq = dx * dx + dy * dy
        if r_sq < 1e-4:
            return 0.0, 0.0
        r = math.sqrt(r_sq)
        force = G * self.mass * other.mass / r_sq
        fx = force * dx / r
        fy = force * dy / r
        return fx, fy


class Star(Planet):
    """A subclass of Planet that represents a stationary star."""

    def __init__(self, mass: float, x: float, y: float, color: tuple[int, int, int]):
        super().__init__(mass, x, y, 0.0, 0.0, color)
        self.radius = max(12, int(self.mass ** (1.0 / 3.0) * 1.4))

    def draw(self, screen: pygame.Surface) -> None:
        glow_radius = int(self.radius * 1.8)
        glow_color = (min(255, self.color[0] + 80), min(255, self.color[1] + 80), min(255, self.color[2] + 80))
        pygame.draw.circle(screen, glow_color, (int(self.x), int(self.y)), glow_radius)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), int(self.radius * 1.1))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self, dt: float) -> None:  # noqa: ARG002
        # Stars stay fixed in this simulation
        return

    def attract(self, other: Planet) -> tuple[float, float]:
        return super().attract(other)


class SolarSystem:
    """A solar system consisting of a star and a collection of planets."""

    def __init__(self, star: Star, planets: List[Planet] | None = None):
        self.star = star
        self.planets: List[Planet] = planets or []

    def add_planet(self, planet: Planet) -> None:
        if planet.orbit_radius is None:
            planet.orbit_radius = math.hypot(planet.x - self.star.x, planet.y - self.star.y)
        self.planets.append(planet)

    def update(self, dt: float) -> None:
        for planet in self.planets:
            fx, fy = self.star.attract(planet)
            planet.apply_force(fx, fy, dt)
            planet.update(dt)

    def draw(self, screen: pygame.Surface) -> None:
        for planet in self.planets:
            if planet.orbit_radius:
                pygame.draw.circle(screen, (40, 40, 70), (int(self.star.x), int(self.star.y)), int(planet.orbit_radius), 1)
        self.star.draw(screen)
        for planet in self.planets:
            planet.draw(screen)

    def is_empty(self) -> bool:
        return not self.planets


class Universe:
    """A universe containing multiple solar systems and decorative background stars."""

    def __init__(self) -> None:
        self.solar_systems: List[SolarSystem] = []
        self.background_color = BLACK
        self.backdrop: list[tuple[int, int, int, int, float]] = []
        self.refresh_space(initial=True)

    def _random_background(self) -> tuple[int, int, int]:
        base = random.randint(0, 30)
        return (
            base,
            base + random.randint(10, 40),
            base + random.randint(30, 70),
        )

    def _generate_backdrop(self) -> list[tuple[int, int, int, int, float]]:
        stars = []
        for _ in range(90):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            radius = random.choice([1, 1, 2])
            brightness = random.randint(140, 255)
            phase = random.random() * math.tau
            stars.append((x, y, radius, brightness, phase))
        return stars

    def _create_random_system(self) -> SolarSystem:
        star_mass = random.uniform(8_000, 24_000)
        star_x = random.uniform(120, SCREEN_WIDTH - 120)
        star_y = random.uniform(100, SCREEN_HEIGHT - 100)
        star_color = random.choice([YELLOW, RED, BLUE, (255, 220, 150)])
        star = Star(star_mass, star_x, star_y, star_color)

        solar_system = SolarSystem(star)
        for _ in range(random.randint(2, 6)):
            planet_mass = random.uniform(40, 160)
            distance = random.uniform(star.radius + 50, star.radius + 240)
            angle = random.uniform(0, math.tau)
            px = star.x + math.cos(angle) * distance
            py = star.y + math.sin(angle) * distance
            orbit_speed = math.sqrt(G * star.mass / max(distance, 1.0))
            direction = random.choice([-1, 1])
            vx = direction * -math.sin(angle) * orbit_speed
            vy = direction * math.cos(angle) * orbit_speed
            planet_color = random.choice([WHITE, GREEN, LIGHT_BLUE])
            planet = Planet(planet_mass, px, py, vx, vy, planet_color)
            planet.orbit_radius = distance
            solar_system.add_planet(planet)
        return solar_system

    def refresh_space(self, initial: bool = False) -> None:
        self.solar_systems = [self._create_random_system() for _ in range(random.randint(3, 6))]
        self.background_color = self._random_background()
        self.backdrop = self._generate_backdrop()
        if not initial:
            pygame.display.set_caption("Newtonian Universe - Nieuwe sector")

    def update(self, dt: float) -> None:
        for solar_system in self.solar_systems:
            solar_system.update(dt)
        self.solar_systems = [s for s in self.solar_systems if not s.is_empty()]

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(self.background_color)
        current_time = pygame.time.get_ticks() / 1000.0
        for x, y, radius, brightness, phase in self.backdrop:
            value = int(max(80, min(255, brightness + 40 * math.sin(current_time + phase))))
            color = (value, value, value)
            pygame.draw.circle(screen, color, (x, y), radius)
        for solar_system in self.solar_systems:
            solar_system.draw(screen)

    def get_stars(self) -> List[Star]:
        return [solar_system.star for solar_system in self.solar_systems]

    def is_empty(self) -> bool:
        return all(solar_system.is_empty() for solar_system in self.solar_systems)


class PlayerShip:
    """A controllable ship that gathers energy from planets."""

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -math.pi / 2
        self.mass = 120.0
        self.turn_speed = math.radians(180)
        self.thrust_force = 18000.0
        self.ship_radius = 10
        self.collection_radius = 18
        self.sector_progress = 0
        self.total_energy = 0

    def handle_input(self, keys: pygame.key.ScancodeWrapper, dt: float) -> None:
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angle -= self.turn_speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += self.turn_speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            thrust_ax = math.cos(self.angle) * (self.thrust_force / self.mass)
            thrust_ay = math.sin(self.angle) * (self.thrust_force / self.mass)
            self.vx += thrust_ax * dt
            self.vy += thrust_ay * dt

    def apply_gravity(self, universe: "Universe", dt: float) -> None:
        for solar_system in universe.solar_systems:
            bodies: list[Planet] = [solar_system.star, *solar_system.planets]
            for body in bodies:
                dx = body.x - self.x
                dy = body.y - self.y
                r_sq = dx * dx + dy * dy
                if r_sq < 1.0:
                    continue
                r = math.sqrt(r_sq)
                acceleration = G * body.mass / r_sq
                self.vx += (acceleration * dx / r) * dt
                self.vy += (acceleration * dy / r) * dt

    def update(self, dt: float) -> None:
        self.x = (self.x + self.vx * dt) % SCREEN_WIDTH
        self.y = (self.y + self.vy * dt) % SCREEN_HEIGHT
        self.vx *= 0.998
        self.vy *= 0.998

    def collect_energy(self, universe: Universe) -> int:
        collected = 0
        for solar_system in universe.solar_systems:
            remaining_planets = []
            for planet in solar_system.planets:
                distance = math.hypot(planet.x - self.x, planet.y - self.y)
                if distance <= planet.radius + self.collection_radius:
                    energy_value = max(1, int(planet.mass))
                    collected += energy_value
                else:
                    remaining_planets.append(planet)
            solar_system.planets = remaining_planets
        if collected:
            self.sector_progress += collected
            self.total_energy += collected
        return collected

    def check_star_collision(self, stars: List[Star]) -> bool:
        for star in stars:
            distance = math.hypot(star.x - self.x, star.y - self.y)
            if distance <= star.radius + self.ship_radius - 2:
                return True
        return False

    def respawn(self) -> None:
        self.x = SCREEN_WIDTH / 2
        self.y = SCREEN_HEIGHT / 2
        self.vx = 0.0
        self.vy = 0.0
        self.angle = -math.pi / 2
        self.sector_progress = 0

    def draw(self, screen: pygame.Surface) -> None:
        nose = (self.x + math.cos(self.angle) * self.ship_radius * 2.0, self.y + math.sin(self.angle) * self.ship_radius * 2.0)
        left = (self.x + math.cos(self.angle + 2.5) * self.ship_radius, self.y + math.sin(self.angle + 2.5) * self.ship_radius)
        right = (self.x + math.cos(self.angle - 2.5) * self.ship_radius, self.y + math.sin(self.angle - 2.5) * self.ship_radius)
        pygame.draw.polygon(screen, WHITE, [(int(nose[0]), int(nose[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))])
        pygame.draw.circle(screen, (90, 150, 255), (int(self.x), int(self.y)), self.collection_radius, 1)


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Newtonian Universe")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    info_font = pygame.font.SysFont("consolas", 20)

    universe = Universe()
    ship = PlayerShip(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    sector = 1
    energy_goal = 320

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        ship.handle_input(keys, dt)
        ship.apply_gravity(universe, dt)
        universe.update(dt)
        ship.update(dt)
        ship.collect_energy(universe)

        if ship.check_star_collision(universe.get_stars()):
            ship.respawn()

        if ship.sector_progress >= energy_goal or universe.is_empty():
            sector += 1
            energy_goal = int(energy_goal * 1.2)
            ship.sector_progress = 0
            universe.refresh_space()

        universe.draw(screen)
        ship.draw(screen)

        hud_lines = [
            f"Sector: {sector}",
            f"Energie verzameld: {ship.sector_progress}/{energy_goal}",
            f"Totale energie: {ship.total_energy}",
            "Besturing: pijltjes of WASD",
            "Vang planeten, ontwijk sterren",
        ]
        for i, text in enumerate(hud_lines):
            surface = font.render(text, True, WHITE)
            screen.blit(surface, (12, 12 + i * 20))

        info_surface = info_font.render("Newtoniaanse zwaartekracht houdt alles in beweging", True, WHITE)
        screen.blit(info_surface, (12, SCREEN_HEIGHT - 30))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
