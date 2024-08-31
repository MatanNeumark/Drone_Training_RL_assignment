import math
from typing import Optional
import numpy as np

class QuadModel():

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,

    }
    def __init__(self, mass, motor_distance,  max_steps: Optional[int] = 200, render_mode: Optional[str] = None):
        self.mass = mass                            # quadcopter all up mass
        self.g = 9.81                               # gravity
        self.motor_distance = motor_distance        # diagonal motor distance
        self.radius = motor_distance / 2            # radius
        self.I_yy = 2/5 * mass * self.radius ** 2   # moment of inertia assuming a spherical quadcopter
        self.state = None
        self.steps = 0
        self.max_steps = max_steps                  # maximum duration of an episode
        self.tau = 0.02                             # time step
        self.domain_bounds = [[-2, 2],                          # bounds in x direction
                         [-math.inf, math.inf],                 # bounds for x velocity
                         [0, 4],                                # bounds in z direction
                         [-math.inf, math.inf],                 # bounds for z velocity
                         [-60*math.pi/180, 60*math.pi/180],     # bounds for roll angle
                         [-math.inf, math.inf]]                 # bounds for angular velocity

        self.target_zone = [[-0.6, 0.6],                        # bounds in x direction
                       [-math.inf, math.inf],                   # bounds for x velocity
                       [2.2, 3.2],                              # bounds in z direction
                       [-math.inf, math.inf],                   # bounds for z velocity
                       [-45*math.pi/180, 45*math.pi/180],       # bounds for roll angle
                       [-math.inf, math.inf]]                   # bounds for angular velocity

        self.render_mode = render_mode
        self.scale = 100                                        # animation scale
        self.screen_width = (np.abs(self.domain_bounds[0][0]) + np.abs(self.domain_bounds[0][1]))* self.scale
        self.screen_height = self.domain_bounds[2][1] * self.scale
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

    # function to reset the environment to a stochastic state with in a limit bound
    def reset(self, mid_air: Optional[bool] = True, seed: Optional[int] = None):
        self.seed = seed
        np.random.seed(self.seed)
        self.steps = 0
        x_0_margin = 0.01               # meters
        x_d_0_margin = 0.01             # velocity m/s
        z_0_margin = 0.01               # meters
        z_d_0_margin = 0.01             # velocity m/s
        alpha_0_margin = 0.01           # angle in radians
        alpha_d_0_margin = 0.01         # angular velocity in rad/s
        if mid_air:
            height_0 = 0.5
            # initial position, altitude and angle
            x_0 = np.random.uniform(-x_0_margin, x_0_margin)
            z_0 = np.random.uniform(height_0 - z_0_margin, height_0 + z_0_margin)
            alpha_0 = np.random.uniform(-alpha_0_margin, alpha_0_margin)
            # initial horizontal, vertical and angular velocities
            x_d_0 = np.random.uniform(-x_d_0_margin, x_d_0_margin)
            z_d_0 = np.random.uniform(-z_d_0_margin, z_d_0_margin)
            alpha_d_0 = np.random.uniform(-alpha_d_0_margin, alpha_d_0_margin)
        else:
            x_0 = np.random.uniform(-x_0_margin, x_0_margin)
            alpha_0 = np.random.uniform(-alpha_0_margin, alpha_0_margin)
            x_d_0 = z_0 = z_d_0 = alpha_d_0 = 0

        self.state = (x_0, x_d_0, z_0, z_d_0, alpha_0, alpha_d_0)

        if self.render_mode == "human":
            self.render()

        return self.state

    def dynamics(self, action):
        assert len(action) == 2
        x, x_d, z, z_d, alpha, alpha_d = self.state
        T_l = action[0]                  # thrust of left motor
        T_r = action[1]                  # thrust of right motor
        # accelerations
        x_dd = (1 / self.mass) * math.sin(-alpha) * (T_l + T_r)             # acceleration in x
        z_dd = - self.g + (1/self.mass) * math.cos(alpha) * (T_l + T_r)     # acceleration in z
        alpha_dd = (self.radius / self.I_yy) * (T_r - T_l)                  # acceleration in alpha (roll)
        # velocities
        x_d = x_d + x_dd * self.tau                                         # velocity in x
        z_d = z_d + z_dd * self.tau                                         # velocity in z
        alpha_d = alpha_d + alpha_dd * self.tau                             # angular velocity
        # positions
        x = x + x_d * self.tau                                              # x position
        z = z + z_d * self.tau                                              # y position
        alpha = alpha + alpha_d * self.tau                                  # roll angle
        # state output
        self.state = (x, x_d, z, z_d, alpha, alpha_d)                       # resulting state after time step tau
        # termination conditions
        terminated = not bool(self.domain_bounds[0][0] < self.state[0] < self.domain_bounds[0][1] and
                              self.domain_bounds[1][0] < self.state[1] < self.domain_bounds[1][1] and
                              self.domain_bounds[2][0]-0.05 < self.state[2] < self.domain_bounds[2][1] and #offset so it doesn't terminate when taking off from the floor at z=0
                              self.domain_bounds[3][0] < self.state[3] < self.domain_bounds[3][1] and
                              self.domain_bounds[4][0] < self.state[4] < self.domain_bounds[4][1] and
                              self.domain_bounds[5][0] < self.state[5] < self.domain_bounds[5][1])
        # conditions for the target zone
        bulls_eye = bool(self.target_zone[0][0] < self.state[0] < self.target_zone[0][1] and
                         self.target_zone[2][0] < self.state[2] < self.target_zone[2][1])
        cause = None
        # reward function
        if not terminated:
            if bulls_eye:
                reward = 10
            else:
                reward = 1
        else:
            reward = 0
            cause = [self.domain_bounds[0][0] < self.state[0] < self.domain_bounds[0][1],
                     self.domain_bounds[1][0] < self.state[1] < self.domain_bounds[1][1],
                     self.domain_bounds[2][0]-0.05 < self.state[2] < self.domain_bounds[2][1], #offset so it doesn't terminate when taking off from the floor at z=0
                     self.domain_bounds[3][0] < self.state[3] < self.domain_bounds[3][1],
                     self.domain_bounds[4][0] < self.state[4] < self.domain_bounds[4][1],
                     self.domain_bounds[5][0] < self.state[5] < self.domain_bounds[5][1]]

        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
            cause = f"reward reached the limit of {self.max_steps}"
        else:
            truncated = False

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, cause

    # function to create the environment animations
    def render(self, render: Optional[bool] = None):
        if render:
            self.render_mode = "human"
        if self.render_mode is None:
            print(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. render=Ture or render=False)'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            print(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        size_scale = 50
        drone_width = math.sqrt(0.5*self.motor_distance)*size_scale
        drone_height = 0.2*drone_width

        x_pos = self.state[0]
        z_pos = self.state[2]
        alpha = self.state[4]
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # the drone:
        l, r, t, b = -drone_width / 2, drone_width / 2, drone_height/2, -drone_height/2
        drone_x = x_pos * self.scale + (self.screen_width / 2.0)  # MIDDLE OF DRONE
        drone_z = z_pos * self.scale
        drone_coords = [(l, b), (l, t), (r, t), (r, b)]
        drone_top = [(l, t-2), (l, t), (r, t), (r, t-2)] # black bar to show the top side of the drone
        drone_coords = [(c[0] + drone_x, c[1] + drone_z) for c in drone_coords]
        drone_top_coords = [(c[0] + drone_x, c[1] + drone_z) for c in drone_top]

        # rotation of the drone due to roll angle
        def rotate_point(point, pivot, angle):
            s = math.sin(angle)
            c = math.cos(angle)
            point = (point[0] - pivot[0], point[1] - pivot[1])
            x_new = point[0] * c - point[1] * s
            y_new = point[0] * s + point[1] * c
            point = (x_new + pivot[0], y_new + pivot[1])
            return point
        def rotate_polygon(vertices, pivot, angle):
            return [rotate_point(v, pivot, angle) for v in vertices]

        rotated_drone = rotate_polygon(drone_coords, (drone_x, drone_z), alpha)
        rotated_top = rotate_polygon(drone_top_coords, (drone_x, drone_z), alpha)

        # finally, the rotated drone
        gfxdraw.aapolygon(self.surf, rotated_drone, (0, 255, 0))
        gfxdraw.filled_polygon(self.surf, rotated_drone, (0, 255, 0))
        gfxdraw.filled_polygon(self.surf, rotated_top, (100, 0, 100))

        # square representing the target zone:
        l, r, t, b = self.target_zone[0][0], self.target_zone[0][1], self.target_zone[2][1], self.target_zone[2][0]
        target_zone_coords = [(l, b), (l, t), (r, t), (r, b)]
        target_zone_coords = [(c[0]*self.scale + (self.screen_width / 2.0), c[1]*self.scale) for c in target_zone_coords]
        gfxdraw.aapolygon(self.surf, target_zone_coords, (255, 0, 0))

        # horizontal line to represent the floor:
        gfxdraw.hline(self.surf, 0, self.screen_width, self.domain_bounds[2][0]+5, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    # function to close the environment
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

