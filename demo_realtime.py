# Real-Time Artificial Life Demo
#
# Interactive showcase of:
#   1. Lenia creatures (self-organizing lifeforms)
#   2. Wave-driven cymatics patterns
#   3. Hybrid wave-Lenia coupling
#   4. Pattern crystallization
#
# Controls:
#   SPACE - Pause/Resume
#   R - Reset/Restart
#   1-5 - Switch demo mode
#   +/- - Adjust speed
#   Click - Add creature/disturbance

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
import matplotlib.patches as mpatches

from src.lenia import LeniaWorld, LeniaSimulation, create_orbium_seed, create_blob_seed, CREATURE_PRESETS
from src.wave_equation import WaveSimulation
from src.chladni_plate import ChladniSimulation
from src.config import BOUNDARY_NEUMANN


# Demo modes
MODE_LENIA = 0
MODE_WAVES = 1
MODE_CHLADNI = 2
MODE_HYBRID = 3
MODE_MULTI_CREATURE = 4

MODE_NAMES = [
    "Lenia Life",
    "Wave Propagation", 
    "Chladni Cymatics",
    "Hybrid Wave-Life",
    "Multi-Creature Ecosystem"
]


class ArtificialLifeDemo:
    """
    Real-time interactive artificial life demonstration.
    """
    
    def __init__(self, size=256):
        self.size = size
        self.mode = MODE_LENIA
        self.paused = False
        self.frame_count = 0
        self.speed = 1
        
        # Initialize all simulation systems
        self._init_lenia()
        self._init_waves()
        self._init_chladni()
        self._init_hybrid()
        
        # Setup visualization
        self._setup_figure()
        
        # Statistics
        self.stats = {
            'mass': 0,
            'energy': 0,
            'velocity': (0, 0),
            'alive_creatures': 0
        }
    
    def _init_lenia(self):
        """Initialize Lenia simulation."""
        self.lenia = LeniaSimulation(self.size, preset="orbium")
        self.lenia.seed_creature()
    
    def _init_waves(self):
        """Initialize wave simulation."""
        self.waves = WaveSimulation(
            dimensions=(self.size, self.size),
            dt=0.1,
            wave_speed=1.0,
            damping=0.002
        )
        self.waves.state.boundary = BOUNDARY_NEUMANN
        
        # Add initial pulse
        center = (self.size // 2, self.size // 2)
        self.waves.inject_pulse(center, sigma=10.0, amplitude=1.0)
        
        self.wave_source_active = False
        self.wave_source_freq = 0.05
    
    def _init_chladni(self):
        """Initialize Chladni plate simulation."""
        self.chladni = ChladniSimulation(
            size=self.size,
            thickness=0.5,
            youngs_modulus=100.0,
            poisson_ratio=0.3,
            density=1.0,
            damping=0.01,
            dt=0.01
        )
        
        # Drive at a resonant mode
        self.chladni_mode = (2, 2)
        self.chladni.drive_at_mode(*self.chladni_mode, amplitude=1.0)
    
    def _init_hybrid(self):
        """Initialize hybrid wave-Lenia system."""
        # Create Lenia world
        self.hybrid_lenia = LeniaWorld(self.size, dt=0.1)
        config = CREATURE_PRESETS["wave_life"]
        self.hybrid_lenia.set_kernel(0, **config["kernel"])
        self.hybrid_lenia.set_growth(0, **config["growth"])
        
        # Create wave field
        self.hybrid_waves = WaveSimulation(
            dimensions=(self.size, self.size),
            dt=0.05,
            wave_speed=0.5,
            damping=0.01
        )
        
        # Coupling strength
        self.wave_to_lenia = 0.05
        self.lenia_to_wave = 0.1
        
        # Seed
        self.hybrid_lenia.add_creature(
            (self.size // 2, self.size // 2),
            create_orbium_seed(15)
        )
    
    def _setup_figure(self):
        """Setup matplotlib figure with controls."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Main display
        self.ax_main = self.fig.add_axes([0.05, 0.25, 0.6, 0.7])
        self.ax_main.set_facecolor('#0f0f23')
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        
        # Initial image
        self.im = self.ax_main.imshow(
            np.zeros((self.size, self.size)),
            cmap='inferno',
            vmin=0, vmax=1,
            interpolation='bilinear'
        )
        self.title = self.ax_main.set_title(
            MODE_NAMES[self.mode],
            color='white', fontsize=14, pad=10
        )
        
        # Stats panel
        self.ax_stats = self.fig.add_axes([0.68, 0.55, 0.28, 0.4])
        self.ax_stats.set_facecolor('#16213e')
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '', 
            transform=self.ax_stats.transAxes,
            fontsize=10, color='#00ff88',
            verticalalignment='top',
            fontfamily='monospace'
        )
        
        # Mode selection
        self.ax_mode = self.fig.add_axes([0.68, 0.25, 0.28, 0.25])
        self.ax_mode.set_facecolor('#16213e')
        self.radio = RadioButtons(
            self.ax_mode, 
            MODE_NAMES,
            activecolor='#00ff88'
        )
        for label in self.radio.labels:
            label.set_color('white')
            label.set_fontsize(9)
        self.radio.on_clicked(self._on_mode_change)
        
        # Control buttons
        self.ax_reset = self.fig.add_axes([0.05, 0.08, 0.12, 0.06])
        self.btn_reset = Button(self.ax_reset, 'Reset', color='#16213e', hovercolor='#1a1a4e')
        self.btn_reset.label.set_color('white')
        self.btn_reset.on_clicked(self._on_reset)
        
        self.ax_pause = self.fig.add_axes([0.19, 0.08, 0.12, 0.06])
        self.btn_pause = Button(self.ax_pause, 'Pause', color='#16213e', hovercolor='#1a1a4e')
        self.btn_pause.label.set_color('white')
        self.btn_pause.on_clicked(self._on_pause)
        
        self.ax_add = self.fig.add_axes([0.33, 0.08, 0.12, 0.06])
        self.btn_add = Button(self.ax_add, 'Add Life', color='#16213e', hovercolor='#1a1a4e')
        self.btn_add.label.set_color('white')
        self.btn_add.on_clicked(self._on_add_creature)
        
        # Speed slider
        self.ax_speed = self.fig.add_axes([0.05, 0.02, 0.4, 0.03])
        self.slider_speed = Slider(
            self.ax_speed, 'Speed', 0.1, 5.0,
            valinit=1.0, color='#00ff88'
        )
        self.slider_speed.label.set_color('white')
        self.slider_speed.valtext.set_color('white')
        self.slider_speed.on_changed(self._on_speed_change)
        
        # Instructions
        self.ax_info = self.fig.add_axes([0.5, 0.08, 0.15, 0.06])
        self.ax_info.set_facecolor('#16213e')
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])
        self.ax_info.text(
            0.5, 0.5, 'Click field to\nadd disturbance',
            transform=self.ax_info.transAxes,
            fontsize=8, color='#888888',
            ha='center', va='center'
        )
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _on_mode_change(self, label):
        """Handle mode change."""
        self.mode = MODE_NAMES.index(label)
        self.title.set_text(MODE_NAMES[self.mode])
        self._reset_current_mode()
    
    def _on_reset(self, event):
        """Handle reset button."""
        self._reset_current_mode()
    
    def _on_pause(self, event):
        """Handle pause button."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
    
    def _on_add_creature(self, event):
        """Add a creature at random position."""
        pos = (
            np.random.randint(50, self.size - 50),
            np.random.randint(50, self.size - 50)
        )
        
        if self.mode == MODE_LENIA:
            self.lenia.world.add_creature(pos, create_orbium_seed(12))
        elif self.mode == MODE_HYBRID:
            self.hybrid_lenia.add_creature(pos, create_orbium_seed(12))
        elif self.mode == MODE_MULTI_CREATURE:
            self.lenia.world.add_creature(pos, create_blob_seed(10))
    
    def _on_speed_change(self, val):
        """Handle speed change."""
        self.speed = val
    
    def _on_click(self, event):
        """Handle click on field."""
        if event.inaxes != self.ax_main:
            return
        
        x = int(event.xdata)
        y = int(event.ydata)
        
        if 0 <= x < self.size and 0 <= y < self.size:
            self._add_disturbance((y, x))
    
    def _add_disturbance(self, pos):
        """Add disturbance at position."""
        if self.mode == MODE_LENIA or self.mode == MODE_MULTI_CREATURE:
            self.lenia.world.add_creature(pos, create_blob_seed(8))
        
        elif self.mode == MODE_WAVES:
            self.waves.inject_pulse(pos, sigma=8.0, amplitude=0.5)
        
        elif self.mode == MODE_CHLADNI:
            # Disturb plate
            amp = self.chladni.get_amplitude()
            y, x = np.ogrid[-10:11, -10:11]
            r = np.sqrt(x**2 + y**2)
            pulse = np.exp(-r**2 / 25)
            
            py, px = pos
            y_start = max(0, py - 10)
            y_end = min(self.size, py + 11)
            x_start = max(0, px - 10)
            x_end = min(self.size, px + 11)
            
            amp[y_start:y_end, x_start:x_end] += pulse[:y_end-y_start, :x_end-x_start]
        
        elif self.mode == MODE_HYBRID:
            self.hybrid_waves.inject_pulse(pos, sigma=8.0, amplitude=0.5)
            self.hybrid_lenia.add_creature(pos, create_blob_seed(6))
    
    def _reset_current_mode(self):
        """Reset current simulation mode."""
        if self.mode == MODE_LENIA:
            self.lenia.world.clear()
            self.lenia.seed_creature()
            self.lenia.mass_history.clear()
            self.lenia.com_history.clear()
        
        elif self.mode == MODE_WAVES:
            self.waves.state.amplitude.fill(0)
            self.waves.state.velocity.fill(0)
            center = (self.size // 2, self.size // 2)
            self.waves.inject_pulse(center, sigma=10.0, amplitude=1.0)
        
        elif self.mode == MODE_CHLADNI:
            self.chladni = ChladniSimulation(
                size=self.size,
                thickness=0.5,
                youngs_modulus=100.0,
                poisson_ratio=0.3,
                density=1.0,
                damping=0.01,
                dt=0.01
            )
            self.chladni.drive_at_mode(*self.chladni_mode, amplitude=1.0)
        
        elif self.mode == MODE_HYBRID:
            self.hybrid_lenia.clear()
            self.hybrid_waves.state.amplitude.fill(0)
            self.hybrid_waves.state.velocity.fill(0)
            self.hybrid_lenia.add_creature(
                (self.size // 2, self.size // 2),
                create_orbium_seed(15)
            )
        
        elif self.mode == MODE_MULTI_CREATURE:
            self.lenia.world.clear()
            # Add multiple creatures
            for _ in range(5):
                pos = (
                    np.random.randint(40, self.size - 40),
                    np.random.randint(40, self.size - 40)
                )
                self.lenia.world.add_creature(pos, create_blob_seed(12))
        
        self.frame_count = 0
    
    def _step_simulation(self):
        """Advance current simulation."""
        steps = max(1, int(self.speed))
        
        if self.mode == MODE_LENIA or self.mode == MODE_MULTI_CREATURE:
            for _ in range(steps):
                self.lenia.step()
        
        elif self.mode == MODE_WAVES:
            # Add continuous wave source
            if self.frame_count % 20 == 0:
                center = (self.size // 2, self.size // 2)
                t = self.frame_count * 0.1
                amp = 0.3 * np.sin(2 * np.pi * self.wave_source_freq * t)
                self.waves.state.amplitude[0, center[0]-2:center[0]+3, center[1]-2:center[1]+3] += amp
            
            for _ in range(steps * 2):
                self.waves.step()
        
        elif self.mode == MODE_CHLADNI:
            for _ in range(steps * 5):
                self.chladni.step()
        
        elif self.mode == MODE_HYBRID:
            # Coupled dynamics
            for _ in range(steps):
                # Wave affects Lenia growth
                wave_amp = self.hybrid_waves.state.amplitude[0]
                wave_influence = np.clip(np.abs(wave_amp) * self.wave_to_lenia, 0, 0.1)
                
                # Step Lenia
                self.hybrid_lenia.step()
                
                # Lenia creates wave sources where it exists
                lenia_field = self.hybrid_lenia.field[0]
                source = self.lenia_to_wave * lenia_field * np.sin(self.frame_count * 0.1)
                self.hybrid_waves.state.amplitude[0] += 0.01 * source
                
                # Step waves
                for _ in range(3):
                    self.hybrid_waves.step()
    
    def _get_current_field(self):
        """Get current display field based on mode."""
        if self.mode == MODE_LENIA or self.mode == MODE_MULTI_CREATURE:
            return self.lenia.get_field()
        
        elif self.mode == MODE_WAVES:
            amp = self.waves.state.amplitude[0]
            # Normalize for display
            vmax = max(0.1, np.max(np.abs(amp)))
            return (amp / vmax + 1) / 2
        
        elif self.mode == MODE_CHLADNI:
            amp = np.abs(self.chladni.get_amplitude())
            return amp / (np.max(amp) + 1e-10)
        
        elif self.mode == MODE_HYBRID:
            # Blend Lenia and waves
            lenia = self.hybrid_lenia.field[0]
            waves = self.hybrid_waves.state.amplitude[0]
            waves_norm = (waves / (np.max(np.abs(waves)) + 1e-10) + 1) / 2
            return 0.7 * lenia + 0.3 * waves_norm
        
        return np.zeros((self.size, self.size))
    
    def _update_stats(self):
        """Update statistics display."""
        if self.mode == MODE_LENIA or self.mode == MODE_MULTI_CREATURE:
            mass = self.lenia.world.compute_mass()
            com = self.lenia.world.compute_center_of_mass()
            vel = self.lenia.get_velocity(window=20)
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            
            alive = "ALIVE" if self.lenia.is_alive() else "DEAD"
            status_color = "#00ff88" if alive == "ALIVE" else "#ff4444"
            
            text = f"""LENIA STATISTICS
================
Frame: {self.frame_count:6d}
Mass:  {mass:8.1f}
Center: ({com[0]:.1f}, {com[1]:.1f})
Speed: {speed:.4f}

Status: {alive}

Preset: {self.lenia.preset_name}
"""
        
        elif self.mode == MODE_WAVES:
            energy = self.waves.state.compute_total_energy()
            rms = np.sqrt(np.mean(self.waves.state.amplitude**2))
            
            text = f"""WAVE STATISTICS
===============
Frame: {self.frame_count:6d}
Energy: {energy:.4f}
RMS Amplitude: {rms:.6f}

Boundary: Reflective
Wave Speed: 1.0
"""
        
        elif self.mode == MODE_CHLADNI:
            amp = self.chladni.get_amplitude()
            rms = np.sqrt(np.mean(amp**2))
            max_amp = np.max(np.abs(amp))
            
            text = f"""CHLADNI PLATE
=============
Frame: {self.frame_count:6d}
Mode: ({self.chladni_mode[0]}, {self.chladni_mode[1]})
RMS: {rms:.6f}
Max Amplitude: {max_amp:.4f}

Forming nodal patterns...
"""
        
        elif self.mode == MODE_HYBRID:
            lenia_mass = np.sum(self.hybrid_lenia.field[0])
            wave_energy = self.hybrid_waves.state.compute_total_energy()
            
            text = f"""HYBRID SYSTEM
=============
Frame: {self.frame_count:6d}

Lenia Mass: {lenia_mass:.1f}
Wave Energy: {wave_energy:.4f}

Coupling:
  Wave->Life: {self.wave_to_lenia}
  Life->Wave: {self.lenia_to_wave}

Emergent behavior active
"""
        
        else:
            text = f"Frame: {self.frame_count}"
        
        self.stats_text.set_text(text)
    
    def update(self, frame):
        """Animation update function."""
        if not self.paused:
            self._step_simulation()
            self.frame_count += 1
        
        # Update display
        field = self._get_current_field()
        self.im.set_array(field)
        
        # Update colormap based on mode
        if self.mode == MODE_WAVES:
            self.im.set_cmap('RdBu')
            self.im.set_clim(-1, 1)
        elif self.mode == MODE_CHLADNI:
            self.im.set_cmap('hot')
            self.im.set_clim(0, 1)
        else:
            self.im.set_cmap('inferno')
            self.im.set_clim(0, 1)
        
        # Update stats
        self._update_stats()
        
        return [self.im, self.stats_text]
    
    def run(self):
        """Start the demo."""
        print("=" * 60)
        print("ARTIFICIAL LIFE DEMO")
        print("=" * 60)
        print("Starting real-time simulation...")
        print("Use the controls to interact with the simulation.")
        print("Click on the field to add disturbances.")
        print("=" * 60)
        
        self.anim = FuncAnimation(
            self.fig, self.update,
            interval=33,  # ~30 FPS
            blit=False,
            cache_frame_data=False
        )
        
        plt.show()


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  CYMATICS + POWER LAWS + LENIA")
    print("  Artificial Life Demonstration")
    print("=" * 60 + "\n")
    
    demo = ArtificialLifeDemo(size=256)
    demo.run()


if __name__ == "__main__":
    main()
