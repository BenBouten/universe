# This is a python program which procedurally generates a 3D universe with different solar systems.

# Import the modules
import random
import math
import pygame

# Define some constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
G = 6.674e-11 # Gravitational constant

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define some classes
class Planet:
    # A class to represent a planet with mass, position, velocity and color
    def __init__(self, mass, x, y, vx, vy, color):
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color

    def draw(self, screen):
        # A method to draw the planet on the screen as a circle
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(math.sqrt(self.mass)))

    def update(self, dt):
        # A method to update the planet's position and velocity using Euler's method
        self.x += self.vx * dt
        self.y += self.vy * dt

    def attract(self, other):
        # A method to calculate the gravitational force between two planets and return it as a tuple
        dx = other.x - self.x
        dy = other.y - self.y
        r = math.sqrt(dx**2 + dy**2)
        f = G * self.mass * other.mass / r**2
        fx = f * dx / r
        fy = f * dy / r
        return (fx, fy)

class Star(Planet):
    # A subclass of Planet to represent a star with a fixed position and a higher mass and brightness
    def __init__(self, mass, x, y, color):
        super().__init__(mass, x, y, 0, 0, color)

    def draw(self, screen):
        # A method to draw the star on the screen as a bright circle with a halo effect
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), int(math.sqrt(self.mass) * 1.5))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(math.sqrt(self.mass)))

    def update(self, dt):
        # A method to do nothing since the star's position and velocity are fixed
        pass

class SolarSystem:
    # A class to represent a solar system with a star and a list of planets
    def __init__(self, star):
        self.star = star
        self.planets = []

    def add_planet(self, planet):
        # A method to add a planet to the solar system's list of planets
        self.planets.append(planet)

    def draw(self, screen):
        # A method to draw the solar system on the screen by drawing the star and the planets
        self.star.draw(screen)
        for planet in self.planets:
            planet.draw(screen)

    def update(self, dt):
        # A method to update the solar system by updating the planets and applying the gravitational forces between them and the star
        for planet in self.planets:
            planet.update(dt)
            fx, fy = self.star.attract(planet)
            planet.vx += fx / planet.mass * dt
            planet.vy += fy / planet.mass * dt

class Universe:
    # A class to represent a universe with a list of solar systems and a background color
    def __init__(self, color):
        self.solar_systems = []
        self.color = color

    def add_solar_system(self, solar_system):
        # A method to add a solar system to the universe's list of solar systems
        self.solar_systems.append(solar_system)

    def draw(self, screen):
        # A method to draw the universe on the screen by filling it with the background color and drawing the solar systems
        screen.fill(self.color)
        for solar_system in self.solar_systems:
            solar_system.draw(screen)

    def update(self, dt):
        # A method to update the universe by updating the solar systems
        for solar_system in self.solar_systems:
            solar_system.update(dt)  # Close the parenthesis

    def remove_solar_system(self, solar_system):
        # A method to remove a solar system from the universe's list of solar systems
        self.solar_systems.remove(solar_system)

# Initialize pygame and create a window
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Universe Generator")
clock = pygame.time.Clock()

# Create some random solar systems and add them to a universe
random.seed(42) # Set the random seed for reproducibility
solar_systems = [] # A list to store the solar systems
num_solar_systems = 10 # The number of solar systems to generate
for i in range(num_solar_systems): # For each solar system
    star_mass = random.randint(1000, 5000) # Generate a random mass for the star
    star_x = random.randint(0, SCREEN_WIDTH) # Generate a random x position for the star
    star_y = random.randint(0, SCREEN_HEIGHT) # Generate a random y position for the star
    star_color = random.choice([YELLOW, RED, BLUE]) # Generate a random color for the star
    star = Star(star_mass, star_x, star_y, star_color) # Create a star object

    planets = [] # A list to store the planets
    num_planets = random.randint(1, 5) # Generate a random number of planets for the solar system
    for j in range(num_planets): # For each planet
        planet_mass = random.randint(10, 100) # Generate a random mass for the planet
        planet_x = star_x + random.randint(-200, 200) # Generate a random x position for the planet near the star
        planet_y = star_y + random.randint(-200, 200) # Generate a random y position for the planet near the star
        planet_vx = random.randint(-10, 10) # Generate a random x velocity for the planet
        planet_vy = random.randint(-10, 10) # Generate a random y velocity for the planet
        planet_color = random.choice([WHITE, GREEN]) # Generate a random color for the planet
        planet = Planet(planet_mass, planet_x, planet_y, planet_vx, planet_vy, planet_color) # Create a planet object

        planets.append(planet) # Add the planet to the list

    solar_system = SolarSystem(star, planets) # Create a solar system object

    solar_systems.append(solar_system) # Add the solar system to the list

universe = Universe(solar_systems) # Create a universe object

# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT: # If user clicks on close button
            running = False # Stop the loop

    # Update logic
    dt = clock.tick(FPS) / 1000.0 # Get the time elapsed since the last frame in seconds
    universe.update(dt) # Update the universe

    # Draw graphics
    universe.draw(screen) # Draw the universe on the screen

    pygame.display.flip() # Update the display

# Quit pygame and exit the program
pygame.quit()