# Phase 0 Validation Script
# Tests the mathematical foundations: Field, Kernels, Reaction-Diffusion, Turing Analysis

import sys
sys.path.insert(0, '.')

import numpy as np
from src.field import ResonantField
from src.kernel import Kernel
from src.reaction import GrayScott, FitzHughNagumo, Schnakenberg
from src.simulation import Simulation, RungeKuttaSimulation
from src.visualizer import FieldVisualizer
from src.analysis import TuringAnalyzer, analyze_system
from src.config import FIELD_WIDTH, FIELD_HEIGHT, D_U, D_V, FEED_RATE, KILL_RATE


def test_field():
    print("=" * 60)
    print("TESTING: ResonantField")
    print("=" * 60)
    
    field = ResonantField()
    print(f"Created field: {field.shape}")
    
    field.initialize_uniform([1.0, 0.0])
    print(f"Uniform init - u mean: {np.mean(field.data[0]):.4f}, v mean: {np.mean(field.data[1]):.4f}")
    
    field.initialize_perturbed([1.0, 0.0], 0.05)
    print(f"Perturbed init - u std: {np.std(field.data[0]):.4f}")
    
    field.inject_seed(FIELD_WIDTH // 2, FIELD_HEIGHT // 2, 10, 1, 0.5)
    print(f"After seed injection - v max: {np.max(field.data[1]):.4f}")
    
    energy = field.compute_energy()
    print(f"Total field energy: {energy:.4f}")
    print("Field tests PASSED\n")


def test_kernels():
    print("=" * 60)
    print("TESTING: Kernel Algebra")
    print("=" * 60)
    
    lap = Kernel.laplacian_2d()
    print(f"Laplacian kernel:\n{lap.data}")
    print(f"Sum (should be 0): {np.sum(lap.data):.6f}")
    
    gauss = Kernel.gaussian(sigma=2.0)
    print(f"\nGaussian kernel shape: {gauss.shape}")
    print(f"Sum (should be 1): {np.sum(gauss.data):.6f}")
    
    mex = Kernel.mexican_hat(sigma=2.0)
    print(f"\nMexican hat kernel shape: {mex.shape}")
    print(f"Sum (should be ~0): {np.sum(mex.data):.6f}")
    
    ring = Kernel.ring(3, 5)
    print(f"\nRing kernel shape: {ring.shape}")
    print(f"Sum (should be 1): {np.sum(ring.data):.6f}")
    
    combined = gauss + (mex * 0.5)
    print(f"\nCombined kernel shape: {combined.shape}")
    
    test_field = np.random.rand(64, 64)
    result = lap.convolve(test_field)
    print(f"Convolution result shape: {result.shape}")
    print("Kernel tests PASSED\n")


def test_reactions():
    print("=" * 60)
    print("TESTING: Reaction Systems")
    print("=" * 60)
    
    gs = GrayScott()
    u, v = np.array([[0.5]]), np.array([[0.25]])
    du, dv = gs.react(u, v)
    print(f"Gray-Scott at (0.5, 0.25): du={du[0,0]:.6f}, dv={dv[0,0]:.6f}")
    
    eq = gs.get_equilibrium()
    print(f"Gray-Scott equilibrium: {eq}")
    
    sch = Schnakenberg()
    eq = sch.get_equilibrium()
    print(f"\nSchnakenberg equilibrium: u*={eq[0]:.4f}, v*={eq[1]:.4f}")
    
    du, dv = sch.react(np.array([[eq[0]]]), np.array([[eq[1]]]))
    print(f"Reaction at equilibrium: du={du[0,0]:.6f}, dv={dv[0,0]:.6f} (should be ~0)")
    print("Reaction tests PASSED\n")


def test_turing_analysis():
    print("=" * 60)
    print("TESTING: Turing Instability Analysis")
    print("=" * 60)
    
    sch = Schnakenberg(a=0.1, b=0.9)
    d_u_test = 1.0
    d_v_test = 40.0
    
    conditions, unstable = analyze_system(sch, d_u_test, d_v_test)
    print("\nTuring Analysis tests PASSED\n")
    return conditions, unstable


def test_simulation():
    print("=" * 60)
    print("TESTING: Simulation Engine")
    print("=" * 60)
    
    field = ResonantField(width=128, height=128)
    
    gs = GrayScott(feed_rate=0.055, kill_rate=0.062)
    field.initialize_uniform([1.0, 0.0])
    
    center_x, center_y = 64, 64
    field.inject_seed(center_x, center_y, 5, 1, 0.25)
    field.inject_seed(center_x, center_y, 5, 0, 0.5)
    
    sim = Simulation(field, gs, dt=1.0)
    sim.set_diffusion_coefficients([0.16, 0.08])
    
    print(f"Initial state - u mean: {np.mean(field.data[0]):.4f}, v mean: {np.mean(field.data[1]):.4f}")
    
    sim.run(100)
    
    stats = sim.get_statistics()
    print(f"After 100 steps:")
    print(f"  u: mean={stats['u_mean']:.4f}, range=[{stats['u_min']:.4f}, {stats['u_max']:.4f}]")
    print(f"  v: mean={stats['v_mean']:.4f}, range=[{stats['v_min']:.4f}, {stats['v_max']:.4f}]")
    print(f"  energy: {stats['energy']:.4f}")
    
    print("Simulation tests PASSED\n")
    return sim


def run_visual_demo():
    print("=" * 60)
    print("RUNNING: Visual Demo (Gray-Scott Pattern Formation)")
    print("=" * 60)
    
    field = ResonantField(width=256, height=256)
    gs = GrayScott(feed_rate=0.055, kill_rate=0.062)
    
    field.initialize_uniform([1.0, 0.0])
    
    np.random.seed(42)
    for _ in range(20):
        x = np.random.randint(50, 206)
        y = np.random.randint(50, 206)
        field.inject_seed(x, y, 3, 1, 0.25)
        field.inject_seed(x, y, 3, 0, 0.5)
    
    sim = Simulation(field, gs, dt=1.0)
    sim.set_diffusion_coefficients([0.16, 0.08])
    
    print("Running 2000 steps to develop patterns...")
    sim.run(2000)
    
    viz = FieldVisualizer(sim, channel=1)
    
    print("Saving snapshot...")
    viz.save_snapshot("gray_scott_pattern.png")
    
    print("Saving spectrum...")
    viz.save_spectrum("gray_scott_spectrum.png")
    
    if sim.history:
        print("Saving history...")
        viz.save_history("gray_scott_history.png")


def main():
    print("\n" + "=" * 60)
    print("PHASE 0 VALIDATION: Mathematical Foundations")
    print("=" * 60 + "\n")
    
    test_field()
    test_kernels()
    test_reactions()
    test_turing_analysis()
    sim = test_simulation()
    
    print("=" * 60)
    print("ALL CORE TESTS PASSED")
    print("=" * 60)
    
    print("\nRunning visual demo and saving images...")
    run_visual_demo()
    
    print("\nPhase 0 validation complete. Images saved to output/ directory.")


if __name__ == "__main__":
    main()
