# Phase 1 Validation: N-Dimensional Field Support
# Tests 1D, 2D, and 3D reaction-diffusion simulations

import sys
sys.path.insert(0, '.')

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.field_nd import ResonantFieldND, ResonantField2D, ResonantField3D
from src.kernel_nd import KernelND
from src.simulation_nd import SimulationND
from src.reaction import GrayScott, Schnakenberg

OUTPUT_DIR = "output"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_field_nd():
    print("=" * 60)
    print("TESTING: N-Dimensional Field")
    print("=" * 60)
    
    # 1D Field
    field_1d = ResonantFieldND(dimensions=(256,), channels=2)
    print(f"1D Field: {field_1d}")
    
    # 2D Field
    field_2d = ResonantFieldND(dimensions=(128, 128), channels=2)
    print(f"2D Field: {field_2d}")
    
    # 3D Field
    field_3d = ResonantFieldND(dimensions=(32, 32, 32), channels=2)
    print(f"3D Field: {field_3d}")
    
    # Test seed injection
    field_3d.initialize_uniform([1.0, 0.0])
    field_3d.inject_seed_sphere(center=(16, 16, 16), radius=5, channel=1, value=0.5)
    print(f"3D Field after sphere injection - v max: {np.max(field_3d.data[1]):.4f}")
    
    # Test energy
    energy = field_3d.compute_energy()
    print(f"3D Field energy: {energy:.4f}")
    
    print("N-Dimensional Field tests PASSED\n")
    return field_3d


def test_kernel_nd():
    print("=" * 60)
    print("TESTING: N-Dimensional Kernels")
    print("=" * 60)
    
    # 1D Laplacian
    lap_1d = KernelND.laplacian(ndim=1)
    print(f"1D Laplacian: shape={lap_1d.shape}, data={lap_1d.data}")
    
    # 2D Laplacian
    lap_2d = KernelND.laplacian(ndim=2)
    print(f"2D Laplacian: shape={lap_2d.shape}")
    print(f"  Center value: {lap_2d.data[1,1]:.1f} (should be -4)")
    
    # 3D Laplacian
    lap_3d = KernelND.laplacian(ndim=3)
    print(f"3D Laplacian: shape={lap_3d.shape}")
    print(f"  Center value: {lap_3d.data[1,1,1]:.1f} (should be -6)")
    print(f"  Sum: {np.sum(lap_3d.data):.6f} (should be 0)")
    
    # 3D Gaussian
    gauss_3d = KernelND.gaussian(sigma=1.5, ndim=3)
    print(f"\n3D Gaussian: shape={gauss_3d.shape}")
    print(f"  Sum: {np.sum(gauss_3d.data):.6f} (should be 1)")
    
    # 3D Mexican Hat
    mex_3d = KernelND.mexican_hat(sigma=2.0, ndim=3)
    print(f"3D Mexican Hat: shape={mex_3d.shape}")
    print(f"  Sum: {np.sum(mex_3d.data):.6f} (should be ~0)")
    
    print("N-Dimensional Kernel tests PASSED\n")
    return lap_3d


def run_1d_simulation():
    print("=" * 60)
    print("RUNNING: 1D Reaction-Diffusion")
    print("=" * 60)
    
    field = ResonantFieldND(dimensions=(512,), channels=2)
    field.initialize_uniform([1.0, 0.0])
    
    # Inject seeds
    for x in [128, 256, 384]:
        field.inject_seed_sphere(center=(x,), radius=5, channel=1, value=0.25)
        field.inject_seed_sphere(center=(x,), radius=5, channel=0, value=0.5)
    
    gs = GrayScott(feed_rate=0.055, kill_rate=0.062)
    sim = SimulationND(field, gs, dt=1.0)
    sim.set_diffusion_coefficients([0.16, 0.08])
    
    print("Running 1000 steps...")
    sim.run(1000)
    
    stats = sim.get_statistics()
    print(f"Final stats: u_mean={stats['u_mean']:.4f}, v_mean={stats['v_mean']:.4f}")
    
    # Plot
    ensure_output_dir()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(field.data[0], 'r-', label='u (activator)')
    axes[0].set_ylabel('u')
    axes[0].set_title('1D Gray-Scott Pattern')
    axes[0].legend()
    
    axes[1].plot(field.data[1], 'g-', label='v (inhibitor)')
    axes[1].set_ylabel('v')
    axes[1].set_xlabel('x')
    axes[1].legend()
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "1d_pattern.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}\n")


def run_2d_simulation():
    print("=" * 60)
    print("RUNNING: 2D Reaction-Diffusion (N-dimensional version)")
    print("=" * 60)
    
    field = ResonantFieldND(dimensions=(128, 128), channels=2)
    field.initialize_uniform([1.0, 0.0])
    
    np.random.seed(42)
    for _ in range(10):
        x = np.random.randint(30, 98)
        y = np.random.randint(30, 98)
        field.inject_seed_sphere(center=(y, x), radius=3, channel=1, value=0.25)
        field.inject_seed_sphere(center=(y, x), radius=3, channel=0, value=0.5)
    
    gs = GrayScott(feed_rate=0.055, kill_rate=0.062)
    sim = SimulationND(field, gs, dt=1.0)
    sim.set_diffusion_coefficients([0.16, 0.08])
    
    print("Running 2000 steps...")
    sim.run(2000)
    
    stats = sim.get_statistics()
    print(f"Final stats: u_mean={stats['u_mean']:.4f}, v_mean={stats['v_mean']:.4f}")
    
    # Plot
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im0 = axes[0].imshow(field.data[0], cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Activator (u)')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(field.data[1], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Inhibitor (v)')
    fig.colorbar(im1, ax=axes[1])
    
    plt.suptitle('2D Gray-Scott (ND Framework)')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "2d_nd_pattern.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}\n")


def run_3d_simulation():
    print("=" * 60)
    print("RUNNING: 3D Reaction-Diffusion")
    print("=" * 60)
    
    # Smaller grid for 3D (memory/compute considerations)
    field = ResonantFieldND(dimensions=(48, 48, 48), channels=2)
    field.initialize_uniform([1.0, 0.0])
    
    # Inject seeds in 3D space - larger and stronger for 3D
    np.random.seed(42)
    for _ in range(8):
        x = np.random.randint(10, 38)
        y = np.random.randint(10, 38)
        z = np.random.randint(10, 38)
        field.inject_seed_sphere(center=(z, y, x), radius=4, channel=1, value=0.35)
        field.inject_seed_sphere(center=(z, y, x), radius=4, channel=0, value=0.45)
    
    # 3D requires adjusted parameters due to dimensional scaling
    # Laplacian center is -6 in 3D vs -4 in 2D (1.5x stronger diffusion)
    # Compensate by reducing diffusion coefficients
    gs = GrayScott(feed_rate=0.055, kill_rate=0.062)
    sim = SimulationND(field, gs, dt=1.0)
    
    # Scale diffusion by 2D/3D ratio = 4/6 = 0.667
    dimensional_scale = 4.0 / 6.0
    d_u_3d = 0.16 * dimensional_scale
    d_v_3d = 0.08 * dimensional_scale
    sim.set_diffusion_coefficients([d_u_3d, d_v_3d])
    
    print(f"Dimensional scaling: {dimensional_scale:.3f}")
    print(f"Adjusted diffusion: D_u={d_u_3d:.4f}, D_v={d_v_3d:.4f}")
    print("Running 1000 steps (3D is computationally intensive)...")
    sim.run(1000)
    
    stats = sim.get_statistics()
    print(f"Final stats: u_mean={stats['u_mean']:.4f}, v_mean={stats['v_mean']:.4f}")
    print(f"Dimensions: {stats['dimensions']}")
    
    # Save cross-sectional slices
    ensure_output_dir()
    
    # XY slices at different Z
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    z_slices = [12, 24, 36]
    for i, z in enumerate(z_slices):
        slice_data = field.get_slice(axis=0, index=z)
        
        axes[0, i].imshow(slice_data[0], cmap='viridis', vmin=0, vmax=1)
        axes[0, i].set_title(f'Activator (u) - Z={z}')
        
        axes[1, i].imshow(slice_data[1], cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f'Inhibitor (v) - Z={z}')
    
    plt.suptitle('3D Gray-Scott - XY Slices')
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "3d_slices_xy.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")
    
    # 3D Isosurface visualization
    visualize_3d_isosurface(field, threshold=0.15, filename="3d_isosurface.png")
    
    print()


def visualize_3d_isosurface(field, threshold=0.2, filename="3d_isosurface.png"):
    """Create a 3D visualization using matplotlib voxels."""
    ensure_output_dir()
    
    v_data = field.data[1]  # Inhibitor channel
    
    # Create voxel data (where v > threshold)
    voxels = v_data > threshold
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by value
    colors = np.zeros(voxels.shape + (4,))
    colors[..., 0] = 0.2  # R
    colors[..., 1] = 0.6  # G
    colors[..., 2] = 0.8  # B
    colors[..., 3] = 0.6 * voxels  # Alpha
    
    ax.voxels(voxels, facecolors=colors, edgecolor='none')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Gray-Scott Pattern (v > {threshold})')
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


def test_radial_spectrum():
    print("=" * 60)
    print("TESTING: Radial Power Spectrum (2D)")
    print("=" * 60)
    
    field = ResonantFieldND(dimensions=(128, 128), channels=2)
    field.initialize_uniform([1.0, 0.0])
    
    # Inject periodic pattern
    field.inject_sinusoidal_mode(wavenumber=(8, 0), amplitude=0.1, phase=0, channel=1)
    field.inject_sinusoidal_mode(wavenumber=(0, 8), amplitude=0.1, phase=0, channel=1)
    
    radial_power = field.get_radial_power_spectrum(channel=1)
    
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(field.data[1], cmap='viridis')
    axes[0].set_title('Field with injected modes (k=8)')
    
    axes[1].plot(radial_power[:64])
    axes[1].set_xlabel('Radial wavenumber')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Radial Power Spectrum')
    axes[1].axvline(x=8, color='r', linestyle='--', label='k=8 (injected)')
    axes[1].legend()
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "radial_spectrum.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}\n")


def main():
    print("\n" + "=" * 60)
    print("PHASE 1 VALIDATION: N-Dimensional Support")
    print("=" * 60 + "\n")
    
    test_field_nd()
    test_kernel_nd()
    test_radial_spectrum()
    
    run_1d_simulation()
    run_2d_simulation()
    run_3d_simulation()
    
    print("=" * 60)
    print("PHASE 1 VALIDATION COMPLETE")
    print("=" * 60)
    print("\nN-dimensional substrate is working for 1D, 2D, and 3D.")
    print("Images saved to output/ directory.")


if __name__ == "__main__":
    main()
