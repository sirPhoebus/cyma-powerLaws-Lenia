# Generate Phase 2 Visualization Images
# Creates images for analysis of all wave dynamics components

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists
OUTPUT_DIR = "output/phase2_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common parameters
GRID_SIZE = 128
DT = 0.05


def save_figure(fig, name):
    """Save figure to output directory."""
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_wave_propagation_images():
    """2.1-2.2: Wave State and Equation - Propagation from central pulse."""
    print("\n[1] Wave Propagation")
    
    from src.wave_equation import WaveSimulation
    
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.0
    )
    
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    sim.inject_pulse(center, sigma=5.0, amplitude=1.0)
    
    # Capture at different times
    times = [0, 50, 100, 200]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, t_steps in enumerate(times):
        if t_steps > 0:
            sim.run(t_steps - (times[i-1] if i > 0 else 0))
        
        amp = sim.state.amplitude[0]
        vmax = max(0.5, np.max(np.abs(amp)))
        im = axes[i].imshow(amp, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"t = {t_steps * DT:.1f}")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    
    fig.suptitle("Wave Propagation from Central Pulse", fontsize=14)
    save_figure(fig, "01_wave_propagation")


def generate_chladni_images():
    """2.3: Chladni Plate - Modal patterns."""
    print("\n[2] Chladni Plate Patterns")
    
    from src.chladni_plate import ChladniSimulation
    
    # Generate multiple mode patterns
    modes_to_show = [(1, 1), (2, 1), (2, 2), (3, 2)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (m, n) in enumerate(modes_to_show):
        sim = ChladniSimulation(
            size=GRID_SIZE,
            thickness=0.5,
            youngs_modulus=100.0,
            poisson_ratio=0.3,
            density=1.0,
            damping=0.005,
            dt=0.01
        )
        
        freq = sim.drive_at_mode(m, n, amplitude=1.0)
        
        # Run to build pattern
        for _ in range(3000):
            sim.step()
        
        amp = np.abs(sim.get_amplitude())
        axes[i].imshow(amp, cmap='hot')
        axes[i].set_title(f"Mode ({m},{n}), f={freq:.4f}")
        axes[i].axis('off')
    
    fig.suptitle("Chladni Plate Modal Patterns", fontsize=14)
    save_figure(fig, "02_chladni_modes")


def generate_3d_wave_slices():
    """2.4: 3D Volume Wave - Cross-section slices."""
    print("\n[3] 3D Volume Wave Slices")
    
    from src.volume_wave import VolumeWaveSimulation
    
    SIZE_3D = 48
    sim = VolumeWaveSimulation(
        resolution=SIZE_3D,
        dt=0.05,
        wave_speed=1.0,
        damping=0.01
    )
    
    center = (SIZE_3D // 2, SIZE_3D // 2, SIZE_3D // 2)
    sim.inject_point_source(center, sigma=4.0, amplitude=1.0)
    
    # Run simulation
    for _ in range(100):
        sim.step()
    
    slices = sim.get_center_slices()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, (name, slice_data) in zip(axes, slices.items()):
        vmax = np.max(np.abs(slice_data))
        im = ax.imshow(slice_data, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name.upper()} Slice")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle("3D Wave Propagation - Center Slices", fontsize=14)
    save_figure(fig, "03_3d_wave_slices")


def generate_interference_images():
    """2.5: Interference - Two-source pattern."""
    print("\n[4] Two-Source Interference")
    
    from src.interference import TwoSourceInterference
    
    source1 = (GRID_SIZE // 2, GRID_SIZE // 3)
    source2 = (GRID_SIZE // 2, 2 * GRID_SIZE // 3)
    
    interference = TwoSourceInterference(
        dimensions=(GRID_SIZE, GRID_SIZE),
        source1_pos=source1,
        source2_pos=source2,
        frequency=1.0 / 16
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Intensity pattern
    intensity = interference.compute_intensity_pattern()
    intensity = np.clip(intensity, 0, np.percentile(intensity, 99))
    axes[0, 0].imshow(intensity, cmap='hot')
    axes[0, 0].set_title("Intensity Pattern")
    axes[0, 0].axis('off')
    
    # Instantaneous wave at different times
    for i, t in enumerate([0, 0.25, 0.5]):
        ax = axes.flatten()[i + 1]
        pattern = interference.compute_pattern(t)
        vmax = np.percentile(np.abs(pattern), 99)
        ax.imshow(pattern, cmap='RdBu', vmin=-vmax, vmax=vmax)
        ax.set_title(f"Wave at t={t}")
        ax.axis('off')
    
    fig.suptitle("Two-Source Interference Pattern", fontsize=14)
    save_figure(fig, "04_interference")


def generate_additive_synthesis_images():
    """2.6: Additive Synthesis - Harmonic modes."""
    print("\n[5] Additive Synthesis")
    
    from src.additive_synthesis import AdditiveSynthesizer, ADSREnvelope, FrequencyBundle
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ADSR envelope
    env = ADSREnvelope(attack=0.1, decay=0.2, sustain=0.6, release=0.3)
    t, curve = env.generate_curve(1.0, sample_rate=100, gate_off_time=0.6)
    axes[0, 0].plot(t, curve, 'b-', linewidth=2)
    axes[0, 0].set_title("ADSR Envelope")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Different waveform spectra
    from src.additive_synthesis import HarmonicSeries
    
    waveforms = [
        ("Sawtooth", HarmonicSeries.sawtooth(1.0, 16)),
        ("Square", HarmonicSeries.square(1.0, 16)),
    ]
    
    for i, (name, series) in enumerate(waveforms):
        spectrum = series.get_spectrum()
        freqs = [s[0] for s in spectrum]
        amps = [abs(s[1]) for s in spectrum]
        axes[0, 1 + i].bar(range(len(freqs)), amps, alpha=0.7)
        axes[0, 1 + i].set_title(f"{name} Spectrum")
        axes[0, 1 + i].set_xlabel("Harmonic")
    
    # Synthesized spatial patterns
    synth = AdditiveSynthesizer((GRID_SIZE, GRID_SIZE), fundamental=0.1)
    for n in range(1, 5):
        env = ADSREnvelope(attack=0.1, decay=0.1, sustain=1.0, release=0.1)
        synth.harmonic_series.add_harmonic(n, amplitude=1.0/n, phase=0, envelope=env)
    synth.assign_modes_cartesian()
    synth.trigger(0.0)
    
    times_to_show = [0.2, 0.5, 1.0]
    for i, t in enumerate(times_to_show):
        field = synth.synthesize(t)
        vmax = np.max(np.abs(field))
        axes[1, i].imshow(field, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f"Synthesis at t={t}")
        axes[1, i].axis('off')
    
    fig.suptitle("Additive Synthesis Components", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "05_additive_synthesis")


def generate_resonance_images():
    """2.7: Resonance Detection - Spectral analysis."""
    print("\n[6] Resonance Detection")
    
    from src.resonance_detection import ResonanceAnalyzer
    from src.wave_equation import WaveSimulation
    from src.config import BOUNDARY_NEUMANN
    
    # Run driven simulation
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.005
    )
    sim.state.boundary = BOUNDARY_NEUMANN
    
    frequency = 0.08
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    
    def source(state, t):
        src = np.zeros(state.shape)
        omega = 2 * np.pi * frequency
        cx, cy = center
        src[0, cx-3:cx+4, cy-3:cy+4] = 2.0 * np.sin(omega * t)
        return src
    
    sim.equation.set_source(source)
    
    analyzer = ResonanceAnalyzer((GRID_SIZE, GRID_SIZE), sample_rate=1.0/DT)
    
    # Run and collect
    for i in range(300):
        sim.step()
        if i >= 100:
            analyzer.record_sample(sim.state.amplitude[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Current field
    amp = sim.state.amplitude[0]
    vmax = np.max(np.abs(amp))
    axes[0, 0].imshow(amp, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title("Driven Wave Field")
    axes[0, 0].axis('off')
    
    # Temporal spectrum
    freqs, power, _ = analyzer.temporal_fft()
    if freqs is not None:
        axes[0, 1].semilogy(freqs[:len(freqs)//2], power[:len(power)//2] + 1e-10)
        axes[0, 1].axvline(frequency, color='r', linestyle='--', label=f'Drive: {frequency}')
        axes[0, 1].set_title("Temporal Power Spectrum")
        axes[0, 1].set_xlabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Nodes
    nodes = analyzer.find_nodes(threshold=0.15)
    if nodes is not None:
        axes[1, 0].imshow(nodes, cmap='binary')
        axes[1, 0].set_title(f"Nodal Regions ({np.mean(nodes)*100:.1f}%)")
        axes[1, 0].axis('off')
    
    # Antinodes
    antinodes = analyzer.find_antinodes(threshold=0.4)
    if antinodes is not None:
        axes[1, 1].imshow(antinodes, cmap='hot')
        axes[1, 1].set_title(f"Antinode Regions ({np.mean(antinodes)*100:.1f}%)")
        axes[1, 1].axis('off')
    
    fig.suptitle("Resonance Analysis", fontsize=14)
    save_figure(fig, "06_resonance_detection")


def generate_boundary_images():
    """2.8: Boundary Conditions - PML absorption."""
    print("\n[7] Boundary Conditions")
    
    from src.boundary_conditions import PerfectlyMatchedLayer
    from src.wave_equation import WaveSimulation
    from src.config import BOUNDARY_ABSORBING
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # PML profile
    pml = PerfectlyMatchedLayer((GRID_SIZE, GRID_SIZE), thickness=15, sigma_max=1.0)
    sigma = pml.get_absorption_mask()
    axes[0, 0].imshow(sigma, cmap='hot')
    axes[0, 0].set_title("PML Absorption Profile")
    axes[0, 0].axis('off')
    
    # Interior mask
    interior = pml.get_interior_mask()
    axes[0, 1].imshow(interior, cmap='binary')
    axes[0, 1].set_title("Interior Region")
    axes[0, 1].axis('off')
    
    # Compare periodic vs absorbing
    for i, (bc_name, bc_type) in enumerate([("Periodic", "periodic"), ("Absorbing", "absorbing")]):
        sim = WaveSimulation(
            dimensions=(GRID_SIZE, GRID_SIZE),
            dt=DT,
            wave_speed=1.0,
            damping=0.0
        )
        sim.state.boundary = bc_type
        
        # Off-center pulse
        sim.inject_pulse((GRID_SIZE // 4, GRID_SIZE // 2), sigma=5.0, amplitude=1.0)
        
        # Run until pulse hits boundary
        for _ in range(80):
            sim.step()
        
        amp = sim.state.amplitude[0]
        vmax = 0.5
        axes[1, i].imshow(amp, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1, i].set_title(f"{bc_name} Boundary")
        axes[1, i].axis('off')
    
    # Energy comparison
    axes[0, 2].axis('off')
    axes[1, 2].axis('off')
    
    fig.suptitle("Boundary Condition Effects", fontsize=14)
    save_figure(fig, "07_boundary_conditions")


def generate_wave_sources_images():
    """2.10: Wave Sources - Different source types."""
    print("\n[8] Wave Source Types")
    
    from src.wave_sources import PointOscillator, LineDriver, RingSource, BoundaryBow
    
    dims = (GRID_SIZE, GRID_SIZE)
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    test_time = 2.5
    
    sources = [
        ("Point Oscillator", PointOscillator(center, amplitude=1.0, frequency=0.1, sigma=5.0)),
        ("Ring Source", RingSource(center, radius=20, amplitude=1.0, frequency=0.1, ring_width=3.0)),
        ("Line Driver", LineDriver((20, 20), (GRID_SIZE-20, GRID_SIZE-20), amplitude=1.0, frequency=0.1, width=3.0)),
        ("Boundary Bow", BoundaryBow('left', amplitude=1.0, frequency=0.1, depth=8))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (name, source) in enumerate(sources):
        field = source.generate(dims, t=test_time)
        vmax = np.max(np.abs(field))
        if vmax > 0:
            axes[i].imshow(field, cmap='RdBu', vmin=-vmax, vmax=vmax)
        else:
            axes[i].imshow(field, cmap='RdBu')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    fig.suptitle("Wave Source Primitives (t=2.5)", fontsize=14)
    save_figure(fig, "08_wave_sources")


def generate_phase_space_images():
    """2.11: Phase Space - Trajectory and modal analysis."""
    print("\n[9] Phase Space Analysis")
    
    from src.phase_space import PhaseSpaceTracker, ModalProjector
    from src.wave_equation import WaveSimulation
    
    # Run actual simulation
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.002
    )
    
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    sim.inject_pulse(center, sigma=8.0, amplitude=1.0)
    
    tracker = PhaseSpaceTracker(wave_speed=1.0)
    
    for i in range(200):
        sim.step()
        tracker.record(sim.state.amplitude[0], sim.state.velocity[0], i * DT)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Energy trace
    times, energies = tracker.get_energy_trace()
    axes[0, 0].plot(times, energies, 'b-')
    axes[0, 0].set_title("Energy vs Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Phase portrait at center
    u_vals, v_vals = tracker.compute_phase_portrait_2d(center)
    axes[0, 1].plot(u_vals, v_vals, 'b-', alpha=0.7)
    axes[0, 1].scatter(u_vals[0], v_vals[0], c='g', s=100, label='Start', zorder=5)
    axes[0, 1].scatter(u_vals[-1], v_vals[-1], c='r', s=100, label='End', zorder=5)
    axes[0, 1].set_title("Phase Portrait (center point)")
    axes[0, 1].set_xlabel("Amplitude u")
    axes[0, 1].set_ylabel("Velocity v")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Modal projections
    projector = ModalProjector((GRID_SIZE, GRID_SIZE))
    projector.add_modes_up_to(4)
    
    final_field = sim.state.amplitude[0]
    amplitudes = projector.project(final_field)
    
    mode_names = list(amplitudes.keys())[:12]
    mode_vals = [abs(amplitudes[n]) for n in mode_names]
    
    axes[1, 0].bar(range(len(mode_names)), mode_vals)
    axes[1, 0].set_xticks(range(len(mode_names)))
    axes[1, 0].set_xticklabels(mode_names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_title("Modal Amplitudes")
    axes[1, 0].set_ylabel("|Amplitude|")
    
    # Final field
    vmax = np.max(np.abs(final_field))
    axes[1, 1].imshow(final_field, cmap='RdBu', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title("Final Wave Field")
    axes[1, 1].axis('off')
    
    fig.suptitle("Phase Space Analysis", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "09_phase_space")


def generate_hybrid_system_images():
    """2.12: Hybrid System - Wave-RD coupling."""
    print("\n[10] Hybrid Wave-RD System")
    
    from src.hybrid_system import HybridSimulation, CouplingFunction
    
    coupling = CouplingFunction(
        wave_to_rd_strength=0.1,
        rd_to_wave_strength=0.05,
        coupling_type="additive"
    )
    
    sim = HybridSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt_wave=0.1,
        dt_rd=1.0,
        coupling=coupling
    )
    
    # Initialize
    sim.state.initialize_rd_uniform(1.0, 0.0)
    center = (GRID_SIZE // 2, GRID_SIZE // 2)
    sim.state.inject_wave_pulse(center, sigma=8.0, amplitude=1.0)
    sim.state.inject_rd_seed(center, radius=10, u_value=0.5, v_value=0.25)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    steps_to_show = [0, 20, 50, 100]
    
    for i, n_steps in enumerate(steps_to_show):
        if n_steps > 0:
            target = n_steps - (steps_to_show[i-1] if i > 0 else 0)
            for _ in range(target):
                sim.step(wave_steps=10, rd_steps=1)
        
        # Wave amplitude
        wave = sim.state.amplitude[0]
        vmax = max(0.1, np.max(np.abs(wave)))
        axes[0, i].imshow(wave, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f"Wave (step {n_steps})")
        axes[0, i].axis('off')
        
        # RD u field
        axes[1, i].imshow(sim.state.u, cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f"RD u (step {n_steps})")
        axes[1, i].axis('off')
        
        # RD v field
        axes[2, i].imshow(sim.state.v, cmap='magma', vmin=0, vmax=0.5)
        axes[2, i].set_title(f"RD v (step {n_steps})")
        axes[2, i].axis('off')
    
    fig.suptitle("Hybrid Wave-RD System Evolution", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "10_hybrid_system")


def generate_standing_wave_images():
    """Bonus: Standing wave formation."""
    print("\n[11] Standing Wave Formation")
    
    from src.wave_equation import WaveSimulation
    from src.config import BOUNDARY_NEUMANN
    
    sim = WaveSimulation(
        dimensions=(GRID_SIZE, GRID_SIZE),
        dt=DT,
        wave_speed=1.0,
        damping=0.002
    )
    sim.state.boundary = BOUNDARY_NEUMANN
    
    # Drive at resonant frequency for this domain
    # For Neumann BC, resonant freq ~ c*sqrt((n/Lx)^2 + (m/Ly)^2)/2
    n, m = 3, 2
    freq = 0.5 * np.sqrt((n / GRID_SIZE)**2 + (m / GRID_SIZE)**2)
    
    def source(state, t):
        src = np.zeros(state.shape)
        omega = 2 * np.pi * freq
        src[0, 5:10, GRID_SIZE//2-5:GRID_SIZE//2+5] = 0.5 * np.sin(omega * t)
        return src
    
    sim.equation.set_source(source)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    times = [0, 200, 500, 1000]
    
    for i, t_steps in enumerate(times):
        if t_steps > 0:
            target = t_steps - (times[i-1] if i > 0 else 0)
            sim.run(target)
        
        amp = sim.state.amplitude[0]
        vmax = max(0.1, np.max(np.abs(amp)))
        
        axes[0, i].imshow(amp, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[0, i].set_title(f"Step {t_steps}")
        axes[0, i].axis('off')
        
        # Show absolute amplitude for pattern
        axes[1, i].imshow(np.abs(amp), cmap='hot', vmin=0, vmax=vmax)
        axes[1, i].set_title(f"|Amplitude| at {t_steps}")
        axes[1, i].axis('off')
    
    fig.suptitle(f"Standing Wave Formation (driving at f={freq:.4f})", fontsize=14)
    save_figure(fig, "11_standing_waves")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Phase 2 Visualization Images")
    print("=" * 60)
    
    generate_wave_propagation_images()
    generate_chladni_images()
    generate_3d_wave_slices()
    generate_interference_images()
    generate_additive_synthesis_images()
    generate_resonance_images()
    generate_boundary_images()
    generate_wave_sources_images()
    generate_phase_space_images()
    generate_hybrid_system_images()
    generate_standing_wave_images()
    
    print("\n" + "=" * 60)
    print(f"All images saved to: {OUTPUT_DIR}/")
    print("=" * 60)
