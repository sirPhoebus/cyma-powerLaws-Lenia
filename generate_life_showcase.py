# Generate Artificial Life Showcase Images
# Creates time-lapse images of Lenia creatures and hybrid systems

import numpy as np
import matplotlib.pyplot as plt
import os

from src.lenia import (
    LeniaWorld, LeniaSimulation, 
    create_orbium_seed, create_blob_seed,
    CREATURE_PRESETS
)
from src.wave_equation import WaveSimulation
from src.chladni_plate import ChladniSimulation
from src.config import BOUNDARY_NEUMANN

OUTPUT_DIR = "output/life_showcase"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIZE = 256


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")


def showcase_lenia_creature():
    """Show Lenia creature evolution over time."""
    print("\n[1] Lenia Creature Evolution")
    
    sim = LeniaSimulation(SIZE, preset="orbium")
    sim.seed_creature()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    steps = [0, 50, 100, 200, 300, 500, 800, 1200]
    
    for i, target_step in enumerate(steps):
        while sim.world.step_count < target_step:
            sim.step()
        
        ax = axes.flatten()[i]
        ax.imshow(sim.get_field(), cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"Step {target_step}", color='white', fontsize=10)
        ax.axis('off')
        
        mass = sim.world.compute_mass()
        ax.text(5, SIZE-10, f"Mass: {mass:.0f}", color='#00ff88', fontsize=8)
    
    fig.suptitle("Lenia Creature: Self-Organizing Lifeform", color='white', fontsize=14)
    save_fig(fig, "01_lenia_evolution")


def showcase_multi_creature():
    """Show multiple interacting creatures."""
    print("\n[2] Multi-Creature Ecosystem")
    
    world = LeniaWorld(SIZE, dt=0.1)
    config = CREATURE_PRESETS["orbium"]
    world.set_kernel(0, **config["kernel"])
    world.set_growth(0, **config["growth"])
    
    # Add multiple creatures
    positions = [
        (SIZE//4, SIZE//4),
        (SIZE//4, 3*SIZE//4),
        (3*SIZE//4, SIZE//4),
        (3*SIZE//4, 3*SIZE//4),
        (SIZE//2, SIZE//2)
    ]
    
    for pos in positions:
        world.add_creature(pos, create_blob_seed(12))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    steps = [0, 100, 300, 500, 800, 1200, 1800, 2500]
    
    for i, target_step in enumerate(steps):
        while world.step_count < target_step:
            world.step()
        
        ax = axes.flatten()[i]
        ax.imshow(world.field[0], cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"Step {target_step}", color='white', fontsize=10)
        ax.axis('off')
    
    fig.suptitle("Multi-Creature Ecosystem: Interaction & Emergence", color='white', fontsize=14)
    save_fig(fig, "02_multi_creature")


def showcase_different_presets():
    """Show different creature presets."""
    print("\n[3] Creature Presets Comparison")
    
    presets = ["orbium", "geminium", "scutium", "wave_life", "pulsing"]
    
    fig, axes = plt.subplots(len(presets), 5, figsize=(15, 3*len(presets)))
    fig.patch.set_facecolor('#1a1a2e')
    
    for row, preset_name in enumerate(presets):
        sim = LeniaSimulation(SIZE, preset=preset_name)
        sim.seed_creature()
        
        steps_to_show = [0, 100, 300, 600, 1000]
        
        for col, target_step in enumerate(steps_to_show):
            while sim.world.step_count < target_step:
                sim.step()
            
            ax = axes[row, col]
            ax.imshow(sim.get_field(), cmap='inferno', vmin=0, vmax=1)
            
            if col == 0:
                ax.set_ylabel(preset_name, color='#00ff88', fontsize=10, rotation=0, ha='right')
            if row == 0:
                ax.set_title(f"Step {target_step}", color='white', fontsize=9)
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle("Different Lenia Creature Types", color='white', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, "03_creature_presets")


def showcase_wave_lenia_hybrid():
    """Show hybrid wave-Lenia coupling."""
    print("\n[4] Hybrid Wave-Lenia System")
    
    # Create coupled system
    lenia = LeniaWorld(SIZE, dt=0.1)
    config = CREATURE_PRESETS["wave_life"]
    lenia.set_kernel(0, **config["kernel"])
    lenia.set_growth(0, **config["growth"])
    
    waves = WaveSimulation(
        dimensions=(SIZE, SIZE),
        dt=0.05,
        wave_speed=0.5,
        damping=0.01
    )
    
    # Seed
    lenia.add_creature((SIZE//2, SIZE//2), create_orbium_seed(15))
    waves.inject_pulse((SIZE//3, SIZE//3), sigma=15, amplitude=1.0)
    
    # Coupling parameters
    wave_to_lenia = 0.02
    lenia_to_wave = 0.05
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.patch.set_facecolor('#1a1a2e')
    
    steps = [0, 100, 300, 600, 1000]
    step_count = 0
    
    for col, target_step in enumerate(steps):
        while step_count < target_step:
            # Coupled dynamics
            lenia.step()
            
            # Lenia creates wave sources
            lenia_field = lenia.field[0]
            source = lenia_to_wave * lenia_field * np.sin(step_count * 0.1)
            waves.state.amplitude[0] += 0.01 * source
            
            for _ in range(3):
                waves.step()
            
            step_count += 1
        
        # Lenia field
        axes[0, col].imshow(lenia.field[0], cmap='inferno', vmin=0, vmax=1)
        axes[0, col].set_title(f"Step {target_step}", color='white', fontsize=9)
        axes[0, col].axis('off')
        if col == 0:
            axes[0, col].set_ylabel("Lenia", color='#00ff88', fontsize=10)
        
        # Wave field
        wave_amp = waves.state.amplitude[0]
        vmax = max(0.1, np.max(np.abs(wave_amp)))
        axes[1, col].imshow(wave_amp, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1, col].axis('off')
        if col == 0:
            axes[1, col].set_ylabel("Waves", color='#00ff88', fontsize=10)
        
        # Combined
        waves_norm = (wave_amp / (np.max(np.abs(wave_amp)) + 1e-10) + 1) / 2
        combined = 0.6 * lenia.field[0] + 0.4 * waves_norm
        axes[2, col].imshow(combined, cmap='viridis', vmin=0, vmax=1)
        axes[2, col].axis('off')
        if col == 0:
            axes[2, col].set_ylabel("Combined", color='#00ff88', fontsize=10)
    
    fig.suptitle("Hybrid Wave-Lenia: Cymatics Meets Artificial Life", color='white', fontsize=14)
    plt.tight_layout()
    save_fig(fig, "04_hybrid_wave_lenia")


def showcase_emergence():
    """Show emergent behavior from random initial conditions."""
    print("\n[5] Emergence from Chaos")
    
    world = LeniaWorld(SIZE, dt=0.15)
    world.set_kernel(0, radius=10, peak=0.5, width=0.2)
    world.set_growth(0, mu=0.135, sigma=0.015)
    
    # Random initialization
    np.random.seed(42)
    world.field[0] = np.random.rand(SIZE, SIZE) * 0.3
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    steps = [0, 50, 150, 300, 500, 800, 1200, 1800, 2500, 3500]
    
    for i, target_step in enumerate(steps):
        while world.step_count < target_step:
            world.step()
        
        ax = axes.flatten()[i]
        ax.imshow(world.field[0], cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"Step {target_step}", color='white', fontsize=9)
        ax.axis('off')
    
    fig.suptitle("Emergence: Order from Random Initial Conditions", color='white', fontsize=14)
    save_fig(fig, "05_emergence")


def showcase_creature_tracking():
    """Track a single creature's movement."""
    print("\n[6] Creature Movement Tracking")
    
    sim = LeniaSimulation(SIZE, preset="orbium")
    # Offset initial position
    sim.lenia = LeniaSimulation(SIZE, preset="orbium")
    sim.lenia.world.add_creature((SIZE//4, SIZE//2), create_orbium_seed(15))
    
    # Track center of mass
    trajectory_y = []
    trajectory_x = []
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    
    # Run and collect trajectory
    for step in range(1500):
        sim.lenia.step()
        com = sim.lenia.world.compute_center_of_mass()
        trajectory_y.append(com[0])
        trajectory_x.append(com[1])
    
    # Final state with trajectory
    ax = axes[0]
    ax.imshow(sim.lenia.get_field(), cmap='inferno', vmin=0, vmax=1)
    ax.plot(trajectory_x, trajectory_y, 'g-', alpha=0.7, linewidth=1)
    ax.scatter(trajectory_x[0], trajectory_y[0], c='blue', s=50, label='Start')
    ax.scatter(trajectory_x[-1], trajectory_y[-1], c='red', s=50, label='End')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Creature with Movement Trail", color='white', fontsize=11)
    ax.axis('off')
    
    # Trajectory plot
    ax = axes[1]
    ax.set_facecolor('#0f0f23')
    ax.plot(trajectory_x, trajectory_y, 'lime', alpha=0.8)
    ax.scatter(trajectory_x[0], trajectory_y[0], c='blue', s=100, label='Start', zorder=5)
    ax.scatter(trajectory_x[-1], trajectory_y[-1], c='red', s=100, label='End', zorder=5)
    ax.set_xlabel("X Position", color='white')
    ax.set_ylabel("Y Position", color='white')
    ax.set_title("Center of Mass Trajectory", color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.legend(fontsize=8)
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.grid(True, alpha=0.2)
    
    fig.suptitle("Lenia Creature: Autonomous Movement", color='white', fontsize=14)
    plt.tight_layout()
    save_fig(fig, "06_creature_tracking")


def showcase_life_death():
    """Show creature lifecycle."""
    print("\n[7] Life and Death Dynamics")
    
    world = LeniaWorld(SIZE, dt=0.1)
    
    # Unstable parameters - some creatures will die
    world.set_kernel(0, radius=12, peak=0.55, width=0.12)
    world.set_growth(0, mu=0.16, sigma=0.012)
    
    # Add creatures
    for _ in range(8):
        pos = (np.random.randint(30, SIZE-30), np.random.randint(30, SIZE-30))
        world.add_creature(pos, create_blob_seed(10))
    
    # Track mass over time
    mass_history = []
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    steps = [0, 200, 500, 1000, 1500, 2000, 3000, 4000]
    
    for i, target_step in enumerate(steps):
        while world.step_count < target_step:
            world.step()
            if world.step_count % 10 == 0:
                mass_history.append(world.compute_mass())
        
        ax = axes.flatten()[i]
        ax.imshow(world.field[0], cmap='inferno', vmin=0, vmax=1)
        mass = world.compute_mass()
        ax.set_title(f"Step {target_step} (Mass: {mass:.0f})", color='white', fontsize=9)
        ax.axis('off')
    
    fig.suptitle("Life & Death: Some Survive, Some Perish", color='white', fontsize=14)
    save_fig(fig, "07_life_death")
    
    # Mass over time plot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#0f0f23')
    ax2.plot(np.arange(len(mass_history)) * 10, mass_history, 'lime', linewidth=1)
    ax2.set_xlabel("Step", color='white')
    ax2.set_ylabel("Total Mass", color='white')
    ax2.set_title("Population Dynamics", color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2)
    save_fig(fig2, "07b_mass_dynamics")


def showcase_cymatics_chladni():
    """Show Chladni-like patterns forming."""
    print("\n[8] Cymatics: Chladni Patterns")
    
    modes = [(2, 2), (3, 2), (3, 3), (4, 3)]
    
    fig, axes = plt.subplots(2, len(modes), figsize=(16, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    for col, mode in enumerate(modes):
        sim = ChladniSimulation(
            size=SIZE,
            thickness=0.5,
            youngs_modulus=100.0,
            poisson_ratio=0.3,
            density=1.0,
            damping=0.005,
            dt=0.01
        )
        
        freq = sim.drive_at_mode(*mode, amplitude=1.0)
        
        # Run to develop pattern
        for _ in range(5000):
            sim.step()
        
        amp = sim.get_amplitude()
        
        # Instantaneous
        axes[0, col].imshow(amp, cmap='RdBu', vmin=-np.max(np.abs(amp)), vmax=np.max(np.abs(amp)))
        axes[0, col].set_title(f"Mode ({mode[0]},{mode[1]})", color='white', fontsize=10)
        axes[0, col].axis('off')
        
        # Absolute (nodal pattern)
        axes[1, col].imshow(np.abs(amp), cmap='hot')
        axes[1, col].axis('off')
    
    axes[0, 0].set_ylabel("Wave", color='#00ff88', fontsize=10)
    axes[1, 0].set_ylabel("Pattern", color='#00ff88', fontsize=10)
    
    fig.suptitle("Cymatics: Chladni Plate Modal Patterns", color='white', fontsize=14)
    save_fig(fig, "08_chladni_cymatics")


if __name__ == "__main__":
    print("=" * 60)
    print("ARTIFICIAL LIFE SHOWCASE")
    print("Generating demonstration images...")
    print("=" * 60)
    
    showcase_lenia_creature()
    showcase_multi_creature()
    showcase_different_presets()
    showcase_wave_lenia_hybrid()
    showcase_emergence()
    showcase_creature_tracking()
    showcase_life_death()
    showcase_cymatics_chladni()
    
    print("\n" + "=" * 60)
    print(f"All images saved to: {OUTPUT_DIR}/")
    print("=" * 60)
