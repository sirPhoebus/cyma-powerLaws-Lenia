# Visualization Module - Real-time field rendering
# Displays the emergence of patterns from the substrate

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.config import COLORMAP, FPS_TARGET

OUTPUT_DIR = "output"


class FieldVisualizer:
    """
    Real-time visualization of ResonantField evolution.
    """
    
    def __init__(self, simulation, channel=1):
        self.simulation = simulation
        self.channel = channel  # Which channel to display (default: v/inhibitor)
        self.fig = None
        self.ax = None
        self.im = None
        self.text = None
        self.animation = None
    
    def setup_figure(self):
        """Initialize matplotlib figure."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_title(f"Resonant Field - {self.simulation.reaction.name}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        
        # Initial image
        data = self.simulation.field.get_channel(self.channel)
        self.im = self.ax.imshow(data, cmap=COLORMAP, vmin=0, vmax=1)
        self.fig.colorbar(self.im, ax=self.ax, label="Field Value")
        
        # Step counter text
        self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                  fontsize=10, verticalalignment='top',
                                  color='white', backgroundcolor='black')
    
    def update_frame(self, frame):
        """Update function for animation."""
        # Run simulation steps
        self.simulation.run(10)  # 10 steps per frame
        
        # Update image
        data = self.simulation.field.get_channel(self.channel)
        self.im.set_array(data)
        
        # Update text
        stats = self.simulation.get_statistics()
        self.text.set_text(f"Step: {stats['step']}\nEnergy: {stats['energy']:.2f}")
        
        return [self.im, self.text]
    
    def run_animation(self, frames=1000, interval=None):
        """Run live animation."""
        if interval is None:
            interval = 1000 // FPS_TARGET
        
        self.setup_figure()
        self.animation = FuncAnimation(
            self.fig, self.update_frame,
            frames=frames, interval=interval, blit=True
        )
        plt.show()
    
    def save_animation(self, filename, frames=500, interval=50):
        """Save animation to file."""
        self.setup_figure()
        self.animation = FuncAnimation(
            self.fig, self.update_frame,
            frames=frames, interval=interval, blit=True
        )
        self.animation.save(filename, writer='pillow', fps=FPS_TARGET)
        plt.close()
    
    def show_snapshot(self):
        """Display current field state as static image."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Channel 0 (u - activator)
        im0 = axes[0].imshow(self.simulation.field.get_channel(0), 
                             cmap=COLORMAP, vmin=0, vmax=1)
        axes[0].set_title("Activator (u)")
        fig.colorbar(im0, ax=axes[0])
        
        # Channel 1 (v - inhibitor)
        im1 = axes[1].imshow(self.simulation.field.get_channel(1), 
                             cmap=COLORMAP, vmin=0, vmax=1)
        axes[1].set_title("Inhibitor (v)")
        fig.colorbar(im1, ax=axes[1])
        
        plt.suptitle(f"{self.simulation.reaction.name} - Step {self.simulation.step_count}")
        plt.tight_layout()
        plt.show()
    
    def save_snapshot(self, filename):
        """Save current field state as image file."""
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        im0 = axes[0].imshow(self.simulation.field.get_channel(0), 
                             cmap=COLORMAP, vmin=0, vmax=1)
        axes[0].set_title("Activator (u)")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(self.simulation.field.get_channel(1), 
                             cmap=COLORMAP, vmin=0, vmax=1)
        axes[1].set_title("Inhibitor (v)")
        fig.colorbar(im1, ax=axes[1])
        
        plt.suptitle(f"{self.simulation.reaction.name} - Step {self.simulation.step_count}")
        plt.tight_layout()
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filepath}")
    
    def show_spectrum(self):
        """Display power spectrum of the field."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, name in enumerate(['Activator (u)', 'Inhibitor (v)']):
            spectrum = self.simulation.field.get_power_spectrum(i)
            spectrum_shifted = np.fft.fftshift(spectrum)
            spectrum_log = np.log10(spectrum_shifted + 1e-10)
            
            im = axes[i].imshow(spectrum_log, cmap='hot')
            axes[i].set_title(f"Power Spectrum - {name}")
            fig.colorbar(im, ax=axes[i], label="log10(power)")
        
        plt.tight_layout()
        plt.show()
    
    def save_spectrum(self, filename):
        """Save power spectrum as image file."""
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, name in enumerate(['Activator (u)', 'Inhibitor (v)']):
            spectrum = self.simulation.field.get_power_spectrum(i)
            spectrum_shifted = np.fft.fftshift(spectrum)
            spectrum_log = np.log10(spectrum_shifted + 1e-10)
            
            im = axes[i].imshow(spectrum_log, cmap='hot')
            axes[i].set_title(f"Power Spectrum - {name}")
            fig.colorbar(im, ax=axes[i], label="log10(power)")
        
        plt.tight_layout()
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filepath}")
    
    def plot_history(self):
        """Plot simulation history (energy, means, etc.)."""
        if not self.simulation.history:
            print("No history recorded yet.")
            return
        
        history = self.simulation.history
        steps = [h['step'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Energy
        axes[0, 0].plot(steps, [h['energy'] for h in history], 'b-')
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Total Energy")
        axes[0, 0].set_title("Field Energy Over Time")
        
        # Means
        axes[0, 1].plot(steps, [h['u_mean'] for h in history], 'r-', label='u (activator)')
        axes[0, 1].plot(steps, [h['v_mean'] for h in history], 'g-', label='v (inhibitor)')
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Mean Value")
        axes[0, 1].set_title("Field Means Over Time")
        axes[0, 1].legend()
        
        # Standard deviations (pattern strength indicator)
        axes[1, 0].plot(steps, [h['u_std'] for h in history], 'r-', label='u (activator)')
        axes[1, 0].plot(steps, [h['v_std'] for h in history], 'g-', label='v (inhibitor)')
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Std Dev")
        axes[1, 0].set_title("Pattern Strength (Spatial Variation)")
        axes[1, 0].legend()
        
        # Phase plot (u_mean vs v_mean)
        axes[1, 1].plot([h['u_mean'] for h in history], 
                        [h['v_mean'] for h in history], 'k-', alpha=0.5)
        axes[1, 1].scatter([history[0]['u_mean']], [history[0]['v_mean']], 
                           c='g', s=100, label='Start')
        axes[1, 1].scatter([history[-1]['u_mean']], [history[-1]['v_mean']], 
                           c='r', s=100, label='End')
        axes[1, 1].set_xlabel("u mean")
        axes[1, 1].set_ylabel("v mean")
        axes[1, 1].set_title("Phase Space Trajectory")
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_history(self, filename):
        """Save simulation history plot as image file."""
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if not self.simulation.history:
            print("No history recorded yet.")
            return
        
        history = self.simulation.history
        steps = [h['step'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(steps, [h['energy'] for h in history], 'b-')
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Total Energy")
        axes[0, 0].set_title("Field Energy Over Time")
        
        axes[0, 1].plot(steps, [h['u_mean'] for h in history], 'r-', label='u (activator)')
        axes[0, 1].plot(steps, [h['v_mean'] for h in history], 'g-', label='v (inhibitor)')
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Mean Value")
        axes[0, 1].set_title("Field Means Over Time")
        axes[0, 1].legend()
        
        axes[1, 0].plot(steps, [h['u_std'] for h in history], 'r-', label='u (activator)')
        axes[1, 0].plot(steps, [h['v_std'] for h in history], 'g-', label='v (inhibitor)')
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Std Dev")
        axes[1, 0].set_title("Pattern Strength (Spatial Variation)")
        axes[1, 0].legend()
        
        axes[1, 1].plot([h['u_mean'] for h in history], 
                        [h['v_mean'] for h in history], 'k-', alpha=0.5)
        axes[1, 1].scatter([history[0]['u_mean']], [history[0]['v_mean']], 
                           c='g', s=100, label='Start')
        axes[1, 1].scatter([history[-1]['u_mean']], [history[-1]['v_mean']], 
                           c='r', s=100, label='End')
        axes[1, 1].set_xlabel("u mean")
        axes[1, 1].set_ylabel("v mean")
        axes[1, 1].set_title("Phase Space Trajectory")
        axes[1, 1].legend()
        
        plt.tight_layout()
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Saved: {filepath}")
