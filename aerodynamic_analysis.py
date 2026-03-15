"""
BIPOL Regional Jet - Aerodynamic Analysis
==========================================
MEiL PW Warsaw

Generates aerodynamic analyses using SUAVE Fidelity_Zero methods:
  1. Drag polar (CD vs CL) at cruise conditions
  2. L/D vs CL  
  3. CL vs alpha
  4. Drag breakdown by component (parasite, induced, compressible)
  5. Drag buildup table
  6. Comparison with BIPOL Excel values

Usage:
    python aerodynamic_analysis.py

All results saved to results/ directory.
"""

import sys
sys.path.insert(0, '/tmp/SUAVE/trunk')
sys.path.insert(0, '/tmp')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import SUAVE
from SUAVE.Core import Units, Data

from vehicle_definition import vehicle_setup, configs_setup, PARAMS


def run_aero_sweep(vehicle, mach, altitude_m, CL_range=None, n_points=40):
    """
    Run a sweep of angle of attack to build drag polar at given flight conditions.
    
    Uses a single cruise segment with varying AoA to get SUAVE's aerodynamic predictions.
    
    Args:
        vehicle: SUAVE vehicle object
        mach: Mach number
        altitude_m: Altitude in meters
        CL_range: Tuple of (CL_min, CL_max) or None for auto
        n_points: Number of sweep points
        
    Returns:
        dict with arrays: alpha, CL, CD, CD_parasit, CD_induced, CD_compress, CD_misc
    """
    # Build analysis
    analyses = SUAVE.Analyses.Vehicle()
    
    # Sizing
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)
    
    # Aerodynamics
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0
    aerodynamics.settings.aircraft_span_efficiency_factor = 1.0
    analyses.append(aerodynamics)
    
    analyses.finalize()
    
    # Atmosphere conditions at altitude
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmo.compute_values(altitude_m)
    
    T   = float(atmo_data.temperature[0, 0])
    p   = float(atmo_data.pressure[0, 0])
    rho = float(atmo_data.density[0, 0])
    a   = float(atmo_data.speed_of_sound[0, 0])
    mu  = float(atmo_data.dynamic_viscosity[0, 0])
    
    V = mach * a
    q = 0.5 * rho * V**2
    
    # Alpha sweep
    if CL_range is None:
        alpha_range = np.linspace(-2, 10, n_points) * Units.deg
    else:
        # Estimate alpha range from CL range
        CL_alpha_approx = 2 * np.pi * PARAMS.wing_AR / (2 + np.sqrt(4 + PARAMS.wing_AR**2))
        alpha_min = CL_range[0] / CL_alpha_approx
        alpha_max = CL_range[1] / CL_alpha_approx
        alpha_range = np.linspace(alpha_min, alpha_max, n_points)
    
    # Build conditions manually for the aero analysis
    state = Data()
    state.conditions = SUAVE.Analyses.Mission.Segments.Conditions.Aerodynamics()
    
    ones = np.ones([n_points, 1])
    
    state.conditions.freestream.velocity           = V * ones
    state.conditions.freestream.mach_number        = mach * ones
    state.conditions.freestream.density             = rho * ones
    state.conditions.freestream.dynamic_viscosity   = mu * ones
    state.conditions.freestream.temperature         = T * ones
    state.conditions.freestream.pressure            = p * ones
    state.conditions.freestream.speed_of_sound      = a * ones
    state.conditions.freestream.dynamic_pressure    = q * ones
    state.conditions.freestream.reynolds_number     = rho * V * vehicle.wings.main_wing.chords.mean_aerodynamic / mu * ones
    
    state.conditions.aerodynamics.angle_of_attack = alpha_range.reshape(-1, 1)
    state.conditions.aerodynamics.side_slip_angle = 0.0 * ones
    state.conditions.aerodynamics.roll_angle      = 0.0 * ones
    
    # Evaluate aerodynamics
    results = aerodynamics.evaluate(state)
    
    CL = results.lift.total[:, 0]
    CD = results.drag.total[:, 0]
    
    # Drag breakdown
    try:
        CD_p = results.drag.parasite.total[:, 0]
        CD_i = results.drag.induced.total[:, 0]
        CD_c = results.drag.compressible.total[:, 0]
        CD_m = results.drag.miscellaneous.total[:, 0]
    except:
        CD_p = np.zeros(n_points)
        CD_i = np.zeros(n_points)
        CD_c = np.zeros(n_points)
        CD_m = np.zeros(n_points)
    
    alpha_deg = alpha_range / Units.deg
    
    return {
        'alpha': alpha_deg,
        'CL': CL,
        'CD': CD,
        'CD_parasite': CD_p,
        'CD_induced': CD_i,
        'CD_compressible': CD_c,
        'CD_miscellaneous': CD_m,
        'L_D': CL / CD,
        'mach': mach,
        'altitude': altitude_m,
        'q': q,
        'rho': rho,
        'V': V,
    }


def plot_aero_results(results_dict, save_dir='results'):
    """Generate all aerodynamic plots."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    r = results_dict
    
    # ---- Plot 1: Drag Polar (CD vs CL) ----
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(r['CD'], r['CL'], 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('CD [-]', fontsize=12)
    ax.set_ylabel('CL [-]', fontsize=12)
    ax.set_title(f'BIPOL Regional Jet - Drag Polar\n'
                 f'M={r["mach"]:.2f}, h={r["altitude"]/1000:.0f} km', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    # Mark CLmax
    ax.axhline(y=PARAMS.CLmax_clean, color='r', linestyle='--', alpha=0.5, label=f'CLmax clean = {PARAMS.CLmax_clean}')
    
    # Mark max L/D point
    idx_max_ld = np.argmax(r['L_D'])
    ax.plot(r['CD'][idx_max_ld], r['CL'][idx_max_ld], 'r*', markersize=15,
            label=f'Max L/D = {r["L_D"][idx_max_ld]:.1f} at CL = {r["CL"][idx_max_ld]:.3f}')
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/11_drag_polar.png', dpi=150)
    plt.close(fig)
    
    # ---- Plot 2: L/D vs CL ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r['CL'], r['L_D'], 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('CL [-]', fontsize=12)
    ax.set_ylabel('L/D [-]', fontsize=12)
    ax.set_title(f'BIPOL Regional Jet - Lift-to-Drag Ratio\n'
                 f'M={r["mach"]:.2f}, h={r["altitude"]/1000:.0f} km', fontsize=13)
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=r['L_D'][idx_max_ld], color='r', linestyle='--', alpha=0.5)
    ax.plot(r['CL'][idx_max_ld], r['L_D'][idx_max_ld], 'r*', markersize=15,
            label=f'(L/D)max = {r["L_D"][idx_max_ld]:.1f} at CL = {r["CL"][idx_max_ld]:.3f}')
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/12_LD_vs_CL.png', dpi=150)
    plt.close(fig)
    
    # ---- Plot 3: CL vs alpha ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r['alpha'], r['CL'], 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Angle of Attack [deg]', fontsize=12)
    ax.set_ylabel('CL [-]', fontsize=12)
    ax.set_title(f'BIPOL Regional Jet - Lift Curve\n'
                 f'M={r["mach"]:.2f}, h={r["altitude"]/1000:.0f} km', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=PARAMS.CLmax_clean, color='r', linestyle='--', alpha=0.5, 
               label=f'CLmax clean = {PARAMS.CLmax_clean}')
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/13_CL_vs_alpha.png', dpi=150)
    plt.close(fig)
    
    # ---- Plot 4: Drag Breakdown ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r['CL'], r['CD_parasite'], 'b-', linewidth=1.5, label='Parasite (friction + form)')
    ax.plot(r['CL'], r['CD_induced'], 'r-', linewidth=1.5, label='Induced')
    ax.plot(r['CL'], r['CD_compressible'], 'g-', linewidth=1.5, label='Compressible (wave)')
    ax.plot(r['CL'], r['CD_miscellaneous'], 'm-', linewidth=1.5, label='Miscellaneous')
    ax.plot(r['CL'], r['CD'], 'k-', linewidth=2.0, label='Total CD')
    ax.set_xlabel('CL [-]', fontsize=12)
    ax.set_ylabel('CD [-]', fontsize=12)
    ax.set_title(f'BIPOL Regional Jet - Drag Buildup\n'
                 f'M={r["mach"]:.2f}, h={r["altitude"]/1000:.0f} km', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/14_drag_buildup.png', dpi=150)
    plt.close(fig)
    
    # ---- Plot 5: Drag Polar at multiple Mach numbers ----
    # (This will be generated separately)
    
    print(f"Aerodynamic plots saved to {save_dir}/")


def print_aero_summary(r):
    """Print aerodynamic analysis summary with comparison to BIPOL values."""
    print("\n" + "=" * 70)
    print("BIPOL REGIONAL JET - AERODYNAMIC SUMMARY")
    print(f"Conditions: M={r['mach']:.2f}, h={r['altitude']/1000:.0f} km, "
          f"q={r['q']:.1f} Pa, V={r['V']:.1f} m/s")
    print("=" * 70)
    
    idx_max_ld = np.argmax(r['L_D'])
    
    # Find CD0 (CD at CL≈0)
    idx_cl0 = np.argmin(np.abs(r['CL']))
    CD0_suave = r['CD'][idx_cl0]
    
    # Find cruise CL (weight balance at cruise)
    W_cruise = PARAMS.MTOW * 9.81 * 0.95  # Approx mid-cruise weight (95% MTOW)
    CL_cruise = W_cruise / (r['q'] * PARAMS.wing_area)
    idx_cruise = np.argmin(np.abs(r['CL'] - CL_cruise))
    
    print(f"\n  Zero-lift drag (CD0 from SUAVE):  {CD0_suave:.5f}")
    print(f"  CD0 from BIPOL Excel (Cx0):       {0.02897:.5f}")
    print(f"  Difference:                        {(CD0_suave - 0.02897)/0.02897*100:+.1f}%")
    
    print(f"\n  Max L/D:  {r['L_D'][idx_max_ld]:.2f}  at CL = {r['CL'][idx_max_ld]:.4f}")
    print(f"  BIPOL Le_eff:  {8.095:.2f},  Kmax: {15.29:.2f}")
    print(f"  (Note: Le_eff = L/D at cruise CL, Kmax = max L/D from parabolic polar)")
    
    print(f"\n  Estimated cruise CL:  {CL_cruise:.4f}")
    if idx_cruise < len(r['CD']):
        print(f"  CD at cruise CL:      {r['CD'][idx_cruise]:.5f}")
        print(f"  L/D at cruise CL:     {r['L_D'][idx_cruise]:.2f}")
    
    print(f"\n  Drag breakdown at cruise CL:")
    print(f"    Parasite:      {r['CD_parasite'][idx_cruise]:.5f}")
    print(f"    Induced:       {r['CD_induced'][idx_cruise]:.5f}")
    print(f"    Compressible:  {r['CD_compressible'][idx_cruise]:.5f}")
    print(f"    Miscellaneous: {r['CD_miscellaneous'][idx_cruise]:.5f}")
    print(f"    Total:         {r['CD'][idx_cruise]:.5f}")
    
    # Oswald factor estimation from SUAVE data
    # CD = CD0 + CL^2 / (pi * e * AR)
    if r['CD_induced'][idx_cruise] > 0:
        e_oswald = r['CL'][idx_cruise]**2 / (np.pi * PARAMS.wing_AR * r['CD_induced'][idx_cruise])
        print(f"\n  Oswald factor (from induced drag): e = {e_oswald:.4f}")
    
    print("=" * 70)


def run_multi_mach_polar(vehicle, altitudes_km=[0, 6, 12], machs=[0.3, 0.5, 0.78]):
    """Run drag polar at multiple Mach/altitude combinations."""
    results_all = {}
    for alt, mach in zip(altitudes_km, machs):
        key = f'M{mach:.2f}_h{alt}km'
        print(f"  Computing polar: {key}...")
        r = run_aero_sweep(vehicle, mach, alt * 1000.0, n_points=30)
        results_all[key] = r
    return results_all


def plot_multi_mach_polar(results_all, save_dir='results'):
    """Plot drag polars at multiple flight conditions on one chart."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (key, r) in enumerate(results_all.items()):
        ax.plot(r['CD'], r['CL'], '-o', color=colors[i % len(colors)],
                markersize=3, linewidth=1.5, label=key)
    
    ax.set_xlabel('CD [-]', fontsize=12)
    ax.set_ylabel('CL [-]', fontsize=12)
    ax.set_title('BIPOL Regional Jet - Drag Polars at Multiple Conditions', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/15_multi_mach_polars.png', dpi=150)
    plt.close(fig)
    
    print(f"Multi-Mach polar plot saved to {save_dir}/")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    print("BIPOL Regional Jet - Aerodynamic Analysis")
    print("-" * 50)
    
    vehicle = vehicle_setup()
    
    # Main polar at cruise conditions
    print("Computing drag polar at cruise (M=0.78, h=12 km)...")
    results = run_aero_sweep(vehicle, PARAMS.cruise_Mach, PARAMS.cruise_alt, n_points=40)
    
    print_aero_summary(results)
    plot_aero_results(results, save_dir='results')
    
    # Multi-Mach polar
    print("\nComputing drag polars at multiple conditions...")
    multi_results = run_multi_mach_polar(vehicle,
                                          altitudes_km=[0, 6, 12],
                                          machs=[0.3, 0.5, 0.78])
    plot_multi_mach_polar(multi_results, save_dir='results')
    
    # Save data to CSV
    import csv
    with open('results/aero_polar_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alpha_deg', 'CL', 'CD', 'CD_parasite', 'CD_induced',
                         'CD_compressible', 'CD_misc', 'L_D'])
        for j in range(len(results['alpha'])):
            writer.writerow([
                f"{results['alpha'][j]:.3f}",
                f"{results['CL'][j]:.5f}",
                f"{results['CD'][j]:.6f}",
                f"{results['CD_parasite'][j]:.6f}",
                f"{results['CD_induced'][j]:.6f}",
                f"{results['CD_compressible'][j]:.6f}",
                f"{results['CD_miscellaneous'][j]:.6f}",
                f"{results['L_D'][j]:.2f}",
            ])
    print("Polar data saved to results/aero_polar_data.csv")
    
    print("\nAerodynamic analysis complete!")
