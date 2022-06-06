import pygame as game

from RobotSimulation.App import *
from RobotSimulation.VerletPhysics import *
import random

# Constants defining tissue types
MUSCLE_A = -1
MUSCLE_B = 1
BONE = 0
LIGAMENT = 2
NONE = 3


class SoftRobot(App):

    # Parameters of SoftRobot stimulation
    strength = 0.9 # strength of muscles
    spring_map = {MUSCLE_A: 0.15, MUSCLE_B: 0.15, BONE: 1, LIGAMENT: 0.3} # stiffness of different tissue types
    mat = Material(1.0, 0.5, 0.8) # (friction, bounce, and mass) of all tissues
    gravity = Vector(0, 2) # The second number is the acceleration due to gravity
    muscle_switch_freq = 10 # number of timesteps between muscle expansion/contraction switching
    side = 30 # length of the sides of the squares of the robots
    grid = 8 # number of squares on a side
    color_map = {MUSCLE_A: (255, 0, 0), MUSCLE_B: (0, 255, 0), BONE: (255, 255, 255), LIGAMENT: (0, 0, 255)} # color of tissues (visual effect only)
    
    def __init__(self, genotype, t="SoftRobot", x=1200, y=500, f=60):
        """Create a new SoftRobot object with given genotype"""
        self.world_size_x, self.world_size_y = x, y
        self.world = World(Vector(x, y), self.gravity, 4)
        self.step = 0
        self.genotype = genotype
        super().__init__(t, x, y, f)


    def Initialize(self):
        """Build the SoftRobot, adding all particles and connections to the simulation"""
        self.joints = {}
        self.connections = []
        self.connection_types = []

        for loc,tissue in enumerate(self.genotype):
            if tissue == NONE:
                continue
            row = loc // self.grid
            col = loc % self.grid
            if (row, col) not in self.joints: 
                self.joints[(row,col)] = self.world.AddParticle(col*self.side, self.world_size_y - row*self.side, self.mat)
            if (row, col+1) not in self.joints:
                self.joints[(row, col+1)] = self.world.AddParticle((col+1)*self.side, self.world_size_y - row*self.side, self.mat)
            if (row+1, col) not in self.joints:
                self.joints[(row+1, col)] = self.world.AddParticle(col*self.side, self.world_size_y - (row+1)*self.side, self.mat)

            if (row+1, col+1) not in self.joints:
                self.joints[(row+1, col+1)] = self.world.AddParticle((col+1)*self.side, self.world_size_y - (row+1)*self.side, self.mat)
                self.connections.append(self.world.AddConstraint(self.joints[(row,col+1)], self.joints[(row+1, col+1)], self.spring_map[LIGAMENT]))
                self.connections.append(self.world.AddConstraint(self.joints[(row+1,col)], self.joints[(row+1, col+1)], self.spring_map[LIGAMENT]))
                self.connection_types.extend([LIGAMENT]*2)

            if loc-self.grid >= 0 and self.genotype[loc-self.grid] == NONE:
                self.connections.append(self.world.AddConstraint(self.joints[(row,col)], self.joints[(row, col+1)], self.spring_map[LIGAMENT]))
                self.connection_types.append(LIGAMENT)

            if (loc-1 >= 0 and self.genotype[loc-1] == NONE) or loc%self.grid==0:
                self.connections.append(self.world.AddConstraint(self.joints[(row,col)], self.joints[(row+1, col)], self.spring_map[LIGAMENT]))
                self.connection_types.append(LIGAMENT)

            self.connections.append(self.world.AddConstraint(self.joints[(row,col)], self.joints[(row+1, col+1)], self.spring_map[tissue]))
            self.connections.append(self.world.AddConstraint(self.joints[(row+1,col)], self.joints[(row, col+1)], self.spring_map[tissue]))
            self.connection_types.extend([tissue]*2)


    def Update(self):
        """Update the state of the SoftRobot with each timestep of the simulation"""
        if game.key.get_pressed()[game.K_ESCAPE]:
            self.Exit()

        for c,t in zip(self.connections, self.connection_types):
            if t != MUSCLE_A and t != MUSCLE_B:
                continue 
            j1, j2 = c.node1, c.node2
            if self.step % (self.muscle_switch_freq*2) < self.muscle_switch_freq: # expanding
                j1.ApplyForce(t * (j1.position-j2.position) * self.strength)
                j2.ApplyForce(t * (j2.position-j1.position) * self.strength)
            else: # contracting
                j1.ApplyForce(t * (j2.position-j1.position) * self.strength)
                j2.ApplyForce(t * (j1.position-j2.position) * self.strength)

        self.step += 1
        self.world.Simulate()


    def Render(self):
        """Draw the SoftRobot in the simulation"""
        self.screen.fill((24, 24, 24))
        for j in self.joints.values():
            pos = (int(j.position.x), int(j.position.y))
            game.draw.circle(self.screen, (255, 255, 255), pos, 4, 0)
        for c,t in zip(self.connections, self.connection_types):
            pos1 = (int(c.node1.position.x), int(c.node1.position.y))
            pos2 = (int(c.node2.position.x), int(c.node2.position.y))
            game.draw.line(self.screen, self.color_map[t], pos1, pos2, 2)
        game.display.update()


    def Reward(self):
        """Get the reward (fitness) of the SoftRobot (distance traveled along X axis)"""
        xs = []
        for j in self.joints.values():
            xs.append(j.position.x)
        return sum(xs)/len(xs) 


if __name__ == "__main__":
    genotype = random.choices([BONE, LIGAMENT], k=64)
    SoftRobot(genotype, f=60).Run(300)