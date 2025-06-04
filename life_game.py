import math
import random
import pygame

pygame.init()

WIDTH, HEIGHT = 1200, 1200
SPEED = 2
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Particles

YELLOW = 0
RED = 1
BLUE = 2
GREEN = 3

particle_properties = {
    YELLOW: {"mass": 2, "color": (255, 255, 0)},  
    RED: {"mass": 3, "color": (255, 0, 0)},      
    BLUE: {"mass": 4, "color": (0, 0, 255)},     
    GREEN: {"mass": 5, "color": (0, 255, 0)}    
}

attraction_matrix = {
    YELLOW: {YELLOW: 0, RED: 0, BLUE: 0, GREEN: 0.2}, 
    RED: {YELLOW: 10, RED: 0, BLUE: 0, GREEN: 0},     
    BLUE: {YELLOW: 0.2, RED: 0, BLUE: 0, GREEN: 0},    
    GREEN: {YELLOW: 0, RED: 0.2, BLUE: 0, GREEN: 0}   
}

num_particles = {YELLOW: 700, RED: 100, BLUE: 500, GREEN: 600}


class Particle:
    total_particles = 0
    def __init__(self, x, y, mass, color, particle_type):
        self.x = x
        self.y = y
        self.fx = 0
        self.fy = 0
        self.vx = 0
        self.vy = 0
        self.mass = mass
        self.radius = 3.5  
        self.color = color
        self.type = particle_type

        Particle.total_particles += 1
        self.total_particles = Particle.total_particles
    
    def update(self):
        ax = self.fx / self.mass  # a = F/m
        ay = self.fy / self.mass
        
        self.vx += ax
        self.vy += ay

        self.x += self.vx
        self.y += self.vy

        # Reset forces for the next frame
        self.fx = 0
        self.fy = 0

        # Edges
        if (self.x - self.radius) < 0 or (self.x + self.radius) > WIDTH:
            self.vx *= -0.9
            self.x = max(self.radius, min(WIDTH - self.radius, self.x))  

        if (self.y - self.radius) < 0 or (self.y + self.radius) > HEIGHT:
            self.vy *= -0.9
            self.y = max(self.radius, min(HEIGHT - self.radius, self.y))
    
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


def gravity(particles):
    for a in particles:
        for b in particles:
            if a == b:
                continue
            dx = b.x - a.x
            dy = b.y - a.y
            d = math.sqrt(dx**2 + dy**2) + 1e-6 
            if d > 0:
                F = attraction_matrix[a.type][b.type]
                
                fx = F * (dx / d)
                fy = F * (dy / d)
                
                a.fx += fx
                a.fy += fy
    


particles = []
for particle_type, count in num_particles.items():
    properties = particle_properties[particle_type]
    for _ in range(count):
        mass = properties["mass"]
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        p = Particle(x, y, mass, properties["color"], particle_type)
        particles.append(p)


#---
running = True
while running:
    screen.fill((0, 0, 0))  # black wallpaper

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Reset forces for all particles
    for p in particles:
        p.fx = 0
        p.fy = 0

    gravity(particles)

    # Update and draw particles
    for p in particles:
        p.update()
        p.draw(screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()