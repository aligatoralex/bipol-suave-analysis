"""
BIPOL Regional Jet - Complete SUAVE Analysis Suite
===================================================
MEiL PW Warsaw

Master script that runs all analyses in sequence:
  1. Vehicle definition & validation
  2. Mission analysis (climb, cruise, descent)
  3. Aerodynamic analysis (drag polars, L/D, drag breakdown)
  4. Performance analysis (V-n, payload-range, climb, specific range)

Usage:
    python run_all.py

All results saved to results/ directory.
All parameters can be modified in vehicle_definition.py (PARAMS section).

Files in this package:
    vehicle_definition.py  - Aircraft geometry, propulsion, weights
    mission_analysis.py    - Mission profile definition & evaluation
    aerodynamic_analysis.py - Drag polars, lift curves, drag buildup
    performance_analysis.py - V-n, payload-range, climb, specific range
    run_all.py             - This file: runs everything

Results:
    results/01_altitude_profile.png     - Mission altitude vs time
    results/02_velocity_profile.png     - TAS and Mach vs time
    results/03_throttle.png             - Throttle setting vs time
    results/04_angle_of_attack.png      - AoA vs time
    results/05_fuel_burn_rate.png       - Fuel flow vs time
    results/06_vehicle_mass.png         - Mass depletion vs time
    results/07_aero_coefficients.png    - CL, CD vs time
    results/08_lift_to_drag.png         - L/D vs time
    results/09_drag_breakdown.png       - Drag components vs time
    results/10_thrust_vs_drag.png       - Thrust and drag vs time
    results/11_drag_polar.png           - CD vs CL (cruise)
    results/12_LD_vs_CL.png            - L/D vs CL
    results/13_CL_vs_alpha.png         - Lift curve
    results/14_drag_buildup.png         - Drag components vs CL
    results/15_multi_mach_polars.png    - Polars at multiple Mach
    results/16_vn_diagram.png           - V-n envelope (CS-25)
    results/17_payload_range.png        - Payload-range diagram
    results/18_specific_range.png       - Specific range + L/D vs weight
    results/19_climb_performance.png    - ROC and climb speed vs altitude
    results/mission_data.csv            - Full mission trajectory data
    results/aero_polar_data.csv         - Drag polar numerical data
"""

import sys
sys.path.insert(0, '/tmp/SUAVE/trunk')
sys.path.insert(0, '/tmp')

import warnings
warnings.filterwarnings('ignore')

import time
import os

# =============================================================================
# 1. Vehicle Definition
# =============================================================================
print("=" * 70)
print("BIPOL REGIONAL JET - SUAVE ANALYSIS SUITE")
print("=" * 70)
print()

t0 = time.time()

print("PHASE 1: Vehicle Definition")
print("-" * 40)
from vehicle_definition import vehicle_setup, configs_setup, PARAMS

vehicle = vehicle_setup()
configs = configs_setup(vehicle)

print(f"  Vehicle: {vehicle.tag}")
print(f"  MTOW:    {vehicle.mass_properties.max_takeoff:.0f} kg")
print(f"  Sref:    {vehicle.reference_area:.2f} m²")
eng = vehicle.networks['turbofan']
print(f"  Engines: {eng.number_of_engines}x, BPR={eng.bypass_ratio}, "
      f"T_design={eng.thrust.total_design:.0f} N")
print(f"  Configs: {[c.tag for c in configs]}")
print(f"  Phase 1 done in {time.time()-t0:.1f}s")

# =============================================================================
# 2. Mission Analysis
# =============================================================================
print()
print("PHASE 2: Mission Analysis")
print("-" * 40)
t1 = time.time()

from mission_analysis import full_setup, plot_mission, print_mission_summary

configs_full, analyses = full_setup()
configs_full.finalize()
analyses.finalize()

print("  Running weight analysis...")
weights = analyses.configs.base.weights
breakdown = weights.evaluate()

print("  Running mission simulation...")
mission = analyses.missions
results = mission.evaluate()

print_mission_summary(results)
plot_mission(results, save_dir='results')

# Save mission CSV
import csv
with open('results/mission_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['segment', 'time_min', 'altitude_km', 'mach', 'TAS_ms',
                     'CL', 'CD', 'LD', 'mass_kg', 'thrust_N', 'drag_N',
                     'throttle', 'aoa_deg', 'fuel_rate_kgs'])
    for seg in results.segments.values():
        c = seg.conditions
        n = len(c.frames.inertial.time[:, 0])
        for j in range(n):
            writer.writerow([
                seg.tag,
                f"{c.frames.inertial.time[j,0]/60.0:.2f}",
                f"{c.freestream.altitude[j,0]/1000.0:.3f}",
                f"{c.freestream.mach_number[j,0]:.4f}",
                f"{c.freestream.velocity[j,0]:.2f}",
                f"{c.aerodynamics.lift_coefficient[j,0]:.5f}",
                f"{c.aerodynamics.drag_coefficient[j,0]:.6f}",
                f"{c.aerodynamics.lift_coefficient[j,0]/max(c.aerodynamics.drag_coefficient[j,0],1e-10):.2f}",
                f"{c.weights.total_mass[j,0]:.1f}",
                f"{c.frames.body.thrust_force_vector[j,0]:.1f}",
                f"{-c.frames.wind.drag_force_vector[j,0]:.1f}",
                f"{c.propulsion.throttle[j,0]:.4f}",
                f"{c.aerodynamics.angle_of_attack[j,0]*57.2958:.3f}",
                f"{-c.weights.vehicle_mass_rate[j,0]:.4f}",
            ])
print(f"  Phase 2 done in {time.time()-t1:.1f}s")

# =============================================================================
# 3. Aerodynamic Analysis
# =============================================================================
print()
print("PHASE 3: Aerodynamic Analysis")
print("-" * 40)
t2 = time.time()

from aerodynamic_analysis import run_aero_sweep, plot_aero_results, print_aero_summary
from aerodynamic_analysis import run_multi_mach_polar, plot_multi_mach_polar

vehicle_aero = vehicle_setup()

print("  Computing cruise drag polar...")
aero_results = run_aero_sweep(vehicle_aero, PARAMS.cruise_Mach, PARAMS.cruise_alt, n_points=40)
print_aero_summary(aero_results)
plot_aero_results(aero_results, save_dir='results')

print("  Computing multi-Mach polars...")
multi = run_multi_mach_polar(vehicle_aero, altitudes_km=[0, 6, 12], machs=[0.3, 0.5, 0.78])
plot_multi_mach_polar(multi, save_dir='results')

# Save aero CSV
with open('results/aero_polar_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['alpha_deg', 'CL', 'CD', 'CD_parasite', 'CD_induced',
                     'CD_compressible', 'CD_misc', 'L_D'])
    for j in range(len(aero_results['alpha'])):
        writer.writerow([
            f"{aero_results['alpha'][j]:.3f}",
            f"{aero_results['CL'][j]:.5f}",
            f"{aero_results['CD'][j]:.6f}",
            f"{aero_results['CD_parasite'][j]:.6f}",
            f"{aero_results['CD_induced'][j]:.6f}",
            f"{aero_results['CD_compressible'][j]:.6f}",
            f"{aero_results['CD_miscellaneous'][j]:.6f}",
            f"{aero_results['L_D'][j]:.2f}",
        ])
print(f"  Phase 3 done in {time.time()-t2:.1f}s")

# =============================================================================
# 4. Performance Analysis
# =============================================================================
print()
print("PHASE 4: Performance Analysis")
print("-" * 40)
t3 = time.time()

from performance_analysis import compute_vn_diagram, compute_payload_range
from performance_analysis import compute_specific_range, compute_climb_performance

print("  Computing V-n diagram...")
vn_data = compute_vn_diagram(save_dir='results')

print("  Computing payload-range...")
pr_data = compute_payload_range(save_dir='results')

print("  Computing specific range...")
compute_specific_range(save_dir='results')

print("  Computing climb performance...")
climb_data = compute_climb_performance(save_dir='results')

print(f"  Phase 4 done in {time.time()-t3:.1f}s")

# =============================================================================
# Summary
# =============================================================================
total_time = time.time() - t0

print()
print("=" * 70)
print("ALL ANALYSES COMPLETE")
print("=" * 70)
print(f"  Total runtime: {total_time:.1f} seconds")
print(f"  Results directory: results/")
print(f"  Plots: {len([f for f in os.listdir('results') if f.endswith('.png')])} PNG files")
print(f"  Data: {len([f for f in os.listdir('results') if f.endswith('.csv')])} CSV files")
print()
print("  To modify parameters and re-run:")
print("    1. Edit PARAMS in vehicle_definition.py")
print("    2. Run: python run_all.py")
print("=" * 70)
