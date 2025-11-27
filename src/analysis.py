# Analysis Module - Stability analysis and Turing condition verification
# Validates the mathematical foundations from Phase 0

import numpy as np
from src.config import DTYPE


class TuringAnalyzer:
    """
    Analyze reaction-diffusion systems for Turing instability.
    
    Determines whether a system will produce spatial patterns
    based on linearized stability analysis.
    """
    
    def __init__(self, reaction_system, d_u, d_v):
        self.reaction = reaction_system
        self.d_u = d_u
        self.d_v = d_v
    
    def compute_jacobian(self, u_star, v_star, epsilon=1e-6):
        """
        Compute the Jacobian matrix at equilibrium (u*, v*).
        
        J = [[df/du, df/dv],
             [dg/du, dg/dv]]
        """
        # Numerical differentiation
        du_fu, du_fv = self.reaction.react(u_star + epsilon, v_star)
        du_fu_neg, _ = self.reaction.react(u_star - epsilon, v_star)
        _, dv_fv = self.reaction.react(u_star, v_star + epsilon)
        _, dv_fv_neg = self.reaction.react(u_star, v_star - epsilon)
        
        base_du, base_dv = self.reaction.react(u_star, v_star)
        
        # Scalar case - get single values
        if isinstance(du_fu, np.ndarray):
            du_fu = du_fu.flat[0]
            du_fu_neg = du_fu_neg.flat[0]
            dv_fv = dv_fv.flat[0]
            dv_fv_neg = dv_fv_neg.flat[0]
            base_du = base_du.flat[0]
            base_dv = base_dv.flat[0]
        
        df_du = (du_fu - du_fu_neg) / (2 * epsilon)
        
        du_at_v_plus, _ = self.reaction.react(u_star, v_star + epsilon)
        du_at_v_minus, _ = self.reaction.react(u_star, v_star - epsilon)
        if isinstance(du_at_v_plus, np.ndarray):
            du_at_v_plus = du_at_v_plus.flat[0]
            du_at_v_minus = du_at_v_minus.flat[0]
        df_dv = (du_at_v_plus - du_at_v_minus) / (2 * epsilon)
        
        _, dv_at_u_plus = self.reaction.react(u_star + epsilon, v_star)
        _, dv_at_u_minus = self.reaction.react(u_star - epsilon, v_star)
        if isinstance(dv_at_u_plus, np.ndarray):
            dv_at_u_plus = dv_at_u_plus.flat[0]
            dv_at_u_minus = dv_at_u_minus.flat[0]
        dg_du = (dv_at_u_plus - dv_at_u_minus) / (2 * epsilon)
        
        dg_dv = (dv_fv - dv_fv_neg) / (2 * epsilon)
        
        return np.array([[df_du, df_dv], 
                         [dg_du, dg_dv]], dtype=DTYPE)
    
    def check_turing_conditions(self, u_star, v_star):
        """
        Check the four Turing instability conditions.
        
        Returns dict with conditions and whether they are satisfied.
        """
        J = self.compute_jacobian(u_star, v_star)
        a, b = J[0, 0], J[0, 1]
        c, d = J[1, 0], J[1, 1]
        
        trace = a + d
        det = a * d - b * c
        
        # Turing conditions for diffusion-driven instability:
        # 1. Equilibrium is stable without diffusion: trace < 0 and det > 0
        # 2. Diffusion destabilizes: d*D_u + a*D_v < 0 (not required for all formulations)
        # 3. Pattern-forming instability: (d*D_u + a*D_v)^2 > 4*D_u*D_v*det
        # 4. There exists k^2 > 0 where Re(lambda(k)) > 0
        
        cond1_trace = trace < 0
        cond1_det = det > 0
        
        diffusion_sum = d * self.d_u + a * self.d_v
        cond2 = diffusion_sum > 0  # This needs to be positive for instability
        
        discriminant = diffusion_sum**2 - 4 * self.d_u * self.d_v * det
        cond3 = discriminant > 0
        
        # Critical wavenumber
        if cond3 and self.d_u * self.d_v > 0:
            k_c_squared = diffusion_sum / (2 * self.d_u * self.d_v)
            k_c = np.sqrt(max(0, k_c_squared))
        else:
            k_c = 0.0
        
        return {
            'jacobian': J,
            'trace': trace,
            'determinant': det,
            'stable_without_diffusion': cond1_trace and cond1_det,
            'diffusion_destabilizes': cond2,
            'pattern_possible': cond3,
            'critical_wavenumber': k_c,
            'turing_unstable': cond1_trace and cond1_det and cond2 and cond3
        }
    
    def dispersion_relation(self, k_squared, u_star, v_star):
        """
        Compute eigenvalues of the linearized system at wavenumber k.
        
        lambda(k) are roots of:
        lambda^2 - (trace - (D_u + D_v)*k^2)*lambda + det_k = 0
        
        where det_k includes diffusion terms.
        """
        J = self.compute_jacobian(u_star, v_star)
        a, b = J[0, 0], J[0, 1]
        c, d = J[1, 0], J[1, 1]
        
        # Modified trace and determinant with diffusion
        trace_k = (a - self.d_u * k_squared) + (d - self.d_v * k_squared)
        det_k = (a - self.d_u * k_squared) * (d - self.d_v * k_squared) - b * c
        
        # Eigenvalues from quadratic formula
        discriminant = trace_k**2 - 4 * det_k
        
        if discriminant >= 0:
            lambda1 = (trace_k + np.sqrt(discriminant)) / 2
            lambda2 = (trace_k - np.sqrt(discriminant)) / 2
        else:
            real_part = trace_k / 2
            imag_part = np.sqrt(-discriminant) / 2
            lambda1 = complex(real_part, imag_part)
            lambda2 = complex(real_part, -imag_part)
        
        return lambda1, lambda2
    
    def find_unstable_wavenumbers(self, u_star, v_star, k_max=10.0, num_points=1000):
        """
        Find range of wavenumbers with positive growth rate.
        """
        k_values = np.linspace(0, k_max, num_points)
        growth_rates = []
        
        for k in k_values:
            k_sq = k**2
            lambda1, lambda2 = self.dispersion_relation(k_sq, u_star, v_star)
            max_real = max(np.real(lambda1), np.real(lambda2))
            growth_rates.append(max_real)
        
        growth_rates = np.array(growth_rates)
        unstable_mask = growth_rates > 0
        
        if np.any(unstable_mask):
            unstable_k = k_values[unstable_mask]
            k_min_unstable = unstable_k.min()
            k_max_unstable = unstable_k.max()
            k_fastest = k_values[np.argmax(growth_rates)]
            max_growth = np.max(growth_rates)
        else:
            k_min_unstable = k_max_unstable = k_fastest = max_growth = 0.0
        
        return {
            'k_values': k_values,
            'growth_rates': growth_rates,
            'k_min_unstable': k_min_unstable,
            'k_max_unstable': k_max_unstable,
            'k_fastest_growing': k_fastest,
            'max_growth_rate': max_growth
        }


def analyze_system(reaction_system, d_u, d_v):
    """
    Comprehensive analysis of a reaction-diffusion system.
    """
    analyzer = TuringAnalyzer(reaction_system, d_u, d_v)
    
    # Get equilibrium
    u_star, v_star = reaction_system.get_equilibrium()
    
    print(f"Analyzing: {reaction_system.name}")
    print(f"Equilibrium: u* = {u_star:.4f}, v* = {v_star:.4f}")
    print(f"Diffusion: D_u = {d_u}, D_v = {d_v}")
    print("-" * 50)
    
    # Check Turing conditions
    conditions = analyzer.check_turing_conditions(u_star, v_star)
    
    print("Jacobian at equilibrium:")
    print(conditions['jacobian'])
    print(f"\nTrace: {conditions['trace']:.6f}")
    print(f"Determinant: {conditions['determinant']:.6f}")
    print(f"\nStable without diffusion: {conditions['stable_without_diffusion']}")
    print(f"Diffusion can destabilize: {conditions['diffusion_destabilizes']}")
    print(f"Pattern formation possible: {conditions['pattern_possible']}")
    print(f"Critical wavenumber: {conditions['critical_wavenumber']:.4f}")
    print(f"\nTURING UNSTABLE: {conditions['turing_unstable']}")
    
    # Find unstable wavenumbers
    unstable = analyzer.find_unstable_wavenumbers(u_star, v_star)
    
    if unstable['max_growth_rate'] > 0:
        print(f"\nUnstable wavenumber range: [{unstable['k_min_unstable']:.4f}, {unstable['k_max_unstable']:.4f}]")
        print(f"Fastest growing mode: k = {unstable['k_fastest_growing']:.4f}")
        print(f"Characteristic wavelength: lambda = {2*np.pi/unstable['k_fastest_growing']:.2f}")
    else:
        print("\nNo unstable wavenumbers found - system will not form patterns.")
    
    return conditions, unstable
