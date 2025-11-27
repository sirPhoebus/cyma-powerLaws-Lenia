# Reaction Systems - Nonlinear local dynamics
# These define f(u,v) and g(u,v) in the reaction-diffusion equations

import numpy as np
from src.config import (
    DTYPE, FEED_RATE, KILL_RATE,
    FHN_EPSILON, FHN_GAMMA, FHN_BETA
)


class ReactionSystem:
    """
    Base class for reaction dynamics.
    
    A reaction system defines the local nonlinear interactions
    between field components (e.g., activator and inhibitor).
    """
    
    def __init__(self):
        self.name = "base"
    
    def react(self, u, v):
        """
        Compute reaction terms du_react and dv_react.
        
        Returns: (du, dv) - the reaction contributions
        """
        raise NotImplementedError


class GrayScott(ReactionSystem):
    """
    Gray-Scott reaction-diffusion system.
    
    Famous for producing spots, stripes, and complex patterns.
    
    du/dt = -u*v^2 + F*(1 - u)
    dv/dt = u*v^2 - (F + k)*v
    
    Where:
    - F is the feed rate (replenishment of u)
    - k is the kill rate (removal of v)
    """
    
    def __init__(self, feed_rate=FEED_RATE, kill_rate=KILL_RATE):
        super().__init__()
        self.name = "gray_scott"
        self.feed_rate = feed_rate
        self.kill_rate = kill_rate
    
    def react(self, u, v):
        """Compute Gray-Scott reaction terms."""
        uvv = u * v * v
        du = -uvv + self.feed_rate * (1.0 - u)
        dv = uvv - (self.feed_rate + self.kill_rate) * v
        return du, dv
    
    def get_equilibrium(self):
        """Return the trivial equilibrium state."""
        return (1.0, 0.0)


class FitzHughNagumo(ReactionSystem):
    """
    FitzHugh-Nagumo system for excitable media.
    
    Simplified model of nerve impulse propagation.
    Produces traveling waves and spiral patterns.
    
    du/dt = u - u^3 - v
    dv/dt = epsilon * (u - gamma*v + beta)
    """
    
    def __init__(self, epsilon=FHN_EPSILON, gamma=FHN_GAMMA, beta=FHN_BETA):
        super().__init__()
        self.name = "fitzhugh_nagumo"
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
    
    def react(self, u, v):
        """Compute FitzHugh-Nagumo reaction terms."""
        du = u - u**3 - v
        dv = self.epsilon * (u - self.gamma * v + self.beta)
        return du, dv
    
    def get_equilibrium(self):
        """Approximate equilibrium (depends on parameters)."""
        # Solve u - u^3 - v = 0 and u - gamma*v + beta = 0
        # This is parameter-dependent
        return (0.0, 0.0)


class Schnakenberg(ReactionSystem):
    """
    Schnakenberg reaction system.
    
    Classic Turing pattern generator with simpler analysis.
    
    du/dt = a - u + u^2*v
    dv/dt = b - u^2*v
    """
    
    def __init__(self, a=0.1, b=0.9):
        super().__init__()
        self.name = "schnakenberg"
        self.a = a
        self.b = b
    
    def react(self, u, v):
        """Compute Schnakenberg reaction terms."""
        u2v = u * u * v
        du = self.a - u + u2v
        dv = self.b - u2v
        return du, dv
    
    def get_equilibrium(self):
        """Compute the equilibrium state."""
        u_star = self.a + self.b
        v_star = self.b / (u_star * u_star)
        return (u_star, v_star)


class BrusselatorReaction(ReactionSystem):
    """
    Brusselator chemical oscillator.
    
    Models autocatalytic reactions with oscillatory behavior.
    
    du/dt = a + u^2*v - (b + 1)*u
    dv/dt = b*u - u^2*v
    """
    
    def __init__(self, a=1.0, b=3.0):
        super().__init__()
        self.name = "brusselator"
        self.a = a
        self.b = b
    
    def react(self, u, v):
        """Compute Brusselator reaction terms."""
        u2v = u * u * v
        du = self.a + u2v - (self.b + 1) * u
        dv = self.b * u - u2v
        return du, dv
    
    def get_equilibrium(self):
        """Compute the equilibrium state."""
        u_star = self.a
        v_star = self.b / self.a
        return (u_star, v_star)


class CustomReaction(ReactionSystem):
    """
    User-defined reaction system via callable functions.
    """
    
    def __init__(self, f_func, g_func, name="custom"):
        super().__init__()
        self.name = name
        self.f_func = f_func
        self.g_func = g_func
    
    def react(self, u, v):
        """Apply custom reaction functions."""
        du = self.f_func(u, v)
        dv = self.g_func(u, v)
        return du, dv
