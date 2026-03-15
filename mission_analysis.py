"""
BIPOL Regional Jet - Mission Analysis
======================================
MEiL PW Warsaw

Defines and runs a complete mission profile:
  1. Climb: 0 → 3 km (250 KCAS, full throttle)
  2. Climb: 3 → 10 km (300 KCAS, full throttle)  
  3. Climb: 10 → 12 km (M0.78, full throttle)
  4. Cruise: 12 km, M0.78, ~2800 km range
  5. Descent: 12 → 8 km (M0.72, 2000 ft/min)
  6. Descent: 8 → 3 km (300 KCAS, 1800 ft/min)
  7. Descent: 3 → 0 km (250 KCAS, 1200 ft/min)

Usage:
    python mission_analysis.py
    
    # Or import from another script:
    from mission_analysis import full_setup, plot_mission

Modifiable parameters: edit PARAMS in vehicle_definition.py and
cruise distance below.
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

# =============================================================================
# USER-MODIFIABLE MISSION PARAMETERS
# =============================================================================
CRUISE_DISTANCE = 2800.0  # [km] - design range (change for parametric studies)
CRUISE_SPEED    = 450.0    # [knots] - cruise speed (~M0.78 at FL390)


def analyses_setup(configs):
    """
    Build SUAVE analysis stack for each configuration.
    Includes: sizing, weights, aerodynamics, stability, energy, planet, atmosphere.
    """
    analyses = SUAVE.Analyses.Analysis.Container()

    for tag, config in list(configs.items()):
        analysis = base_analysis(config)
        analyses[tag] = analysis

    # Takeoff drag increment (gear, slats)
    analyses.takeoff.aerodynamics.settings.drag_coefficient_increment = 0.0100

    return analyses


def base_analysis(vehicle):
    """Build the standard analysis set for one configuration."""
    analyses = SUAVE.Analyses.Vehicle()

    # Sizing
    sizing = SUAVE.Analyses.Sizing.Sizing()
    sizing.features.vehicle = vehicle
    analyses.append(sizing)

    # Weights (transport category)
    weights = SUAVE.Analyses.Weights.Weights_Transport()
    weights.vehicle = vehicle
    analyses.append(weights)

    # Aerodynamics (Fidelity Zero - vortex lattice + form factor)
    aerodynamics = SUAVE.Analyses.Aerodynamics.Fidelity_Zero()
    aerodynamics.geometry = vehicle
    aerodynamics.settings.drag_coefficient_increment = 0.0000
    aerodynamics.settings.aircraft_span_efficiency_factor = 1.0
    analyses.append(aerodynamics)

    # Stability
    stability = SUAVE.Analyses.Stability.Fidelity_Zero()
    stability.geometry = vehicle
    analyses.append(stability)

    # Energy
    energy = SUAVE.Analyses.Energy.Energy()
    energy.network = vehicle.networks
    analyses.append(energy)

    # Planet
    planet = SUAVE.Analyses.Planets.Planet()
    analyses.append(planet)

    # Atmosphere (US Standard 1976)
    atmosphere = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmosphere.features.planet = planet.features
    analyses.append(atmosphere)

    return analyses


def mission_setup(analyses):
    """
    Define the complete mission profile.
    
    The mission has 7 segments following standard airline operations:
    - 3 climb segments (below 10kft, transition zone, to cruise alt)
    - 1 cruise segment at constant speed and altitude
    - 3 descent segments (mirror of climb)
    """
    mission = SUAVE.Analyses.Mission.Sequential_Segments()
    mission.tag = 'BIPOL_design_mission'

    # Atmospheric model
    atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    planet     = SUAVE.Attributes.Planets.Earth()

    # Airport
    airport = SUAVE.Attributes.Airports.Airport()
    airport.altitude   = 0.0 * Units.ft
    airport.delta_isa  = 0.0
    airport.atmosphere = SUAVE.Attributes.Atmospheres.Earth.US_Standard_1976()
    mission.airport    = airport

    Segments = SUAVE.Analyses.Mission.Segments

    # ------------------------------------------------------------------
    # Segment 1: Initial climb 0 → 3 km (250 KCAS)
    # ------------------------------------------------------------------
    segment = Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = 'climb_1_250kcas'
    segment.analyses.extend(analyses.takeoff)
    segment.atmosphere     = atmosphere
    segment.planet         = planet
    segment.altitude_start = 0.0 * Units.km
    segment.altitude_end   = 3.0 * Units.km
    segment.air_speed      = 250.0 * Units.knots
    segment.throttle       = 1.0
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    # Segment 2: Accelerating climb 3 → 10 km (300 KCAS)
    # ------------------------------------------------------------------
    segment = Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = 'climb_2_300kcas'
    segment.analyses.extend(analyses.cruise)
    segment.atmosphere   = atmosphere
    segment.planet       = planet
    segment.altitude_end = 10.0 * Units.km
    segment.air_speed    = 300.0 * Units.knots
    segment.throttle     = 1.0
    segment.climb_rate   = 0.1  # Seed for solver
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    # Segment 3: Final climb 10 → 12 km (M0.78)
    # ------------------------------------------------------------------
    segment = Segments.Climb.Constant_Throttle_Constant_Speed()
    segment.tag = 'climb_3_M078'
    segment.analyses.extend(analyses.cruise)
    segment.atmosphere   = atmosphere
    segment.planet       = planet
    segment.altitude_end = PARAMS.cruise_alt * Units.m  # 12,000 m
    segment.air_speed    = 380.0 * Units.knots
    segment.throttle     = 1.0
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    # Segment 4: Cruise at constant speed, constant altitude
    # ------------------------------------------------------------------
    segment = Segments.Cruise.Constant_Speed_Constant_Altitude()
    segment.tag = 'cruise'
    segment.analyses.extend(analyses.cruise)
    segment.atmosphere = atmosphere
    segment.planet     = planet
    segment.air_speed  = CRUISE_SPEED * Units.knots  # ~M0.78 at FL390
    segment.distance   = CRUISE_DISTANCE * Units.km
    segment.state.numerics.number_control_points = 8
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    # Segment 5: First descent 12 → 8 km
    # ------------------------------------------------------------------
    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = 'descent_1_M072'
    segment.analyses.extend(analyses.cruise)
    segment.atmosphere   = atmosphere
    segment.planet       = planet
    segment.altitude_end = 8.0 * Units.km
    segment.air_speed    = 400.0 * Units.knots
    segment.descent_rate = 2000.0 * Units['ft/min']
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    # Segment 6: Second descent 8 → 3 km
    # ------------------------------------------------------------------
    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = 'descent_2_300kcas'
    segment.analyses.extend(analyses.cruise)
    segment.atmosphere   = atmosphere
    segment.planet       = planet
    segment.altitude_end = 3.0 * Units.km
    segment.air_speed    = 300.0 * Units.knots
    segment.descent_rate = 1800.0 * Units['ft/min']
    mission.append_segment(segment)

    # ------------------------------------------------------------------
    # Segment 7: Final descent 3 → 0 km
    # ------------------------------------------------------------------
    segment = Segments.Descent.Constant_Speed_Constant_Rate()
    segment.tag = 'descent_3_250kcas'
    segment.analyses.extend(analyses.cruise)
    segment.atmosphere   = atmosphere
    segment.planet       = planet
    segment.altitude_end = 0.0 * Units.km
    segment.air_speed    = 250.0 * Units.knots
    segment.descent_rate = 1200.0 * Units['ft/min']
    mission.append_segment(segment)

    return mission


def full_setup():
    """
    Complete analysis pipeline: vehicle → configs → analyses → mission.
    Returns (configs, analyses) for downstream use.
    """
    vehicle  = vehicle_setup()
    configs  = configs_setup(vehicle)
    analyses = analyses_setup(configs)
    mission  = mission_setup(analyses)

    container = SUAVE.Analyses.Analysis.Container()
    container.configs  = analyses
    container.missions = mission

    return configs, container


def plot_mission(results, save_dir='results'):
    """
    Generate all mission profile plots and save as PNG files.
    
    Plots generated:
        1. Altitude vs Time
        2. Velocity vs Time 
        3. Throttle vs Time
        4. Angle of Attack vs Time
        5. Fuel Burn Rate vs Time
        6. Vehicle Mass vs Time
        7. CL, CD vs Time
        8. L/D vs Time
        9. Drag Components vs Time
        10. Thrust vs Drag vs Time
    
    Args:
        results: SUAVE mission results object
        save_dir: Directory to save plot PNGs
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Color scheme for segments
    colors = plt.cm.tab10(np.linspace(0, 1, len(results.segments)))

    # ---- Plot 1: Altitude Profile ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        alt  = seg.conditions.freestream.altitude[:, 0] / 1000.0  # km
        ax.plot(time, alt, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('BIPOL Regional Jet - Mission Altitude Profile')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/01_altitude_profile.png', dpi=150)
    plt.close(fig)

    # ---- Plot 2: Velocity Profile ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        vel  = seg.conditions.freestream.velocity[:, 0]
        mach = seg.conditions.freestream.mach_number[:, 0]
        ax1.plot(time, vel, '-o', color=colors[i], label=seg.tag, markersize=3)
        ax2.plot(time, mach, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax1.set_ylabel('TAS [m/s]')
    ax1.set_title('BIPOL Regional Jet - Velocity Profile')
    ax1.legend(loc='best', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Mach Number')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/02_velocity_profile.png', dpi=150)
    plt.close(fig)

    # ---- Plot 3: Throttle ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        eta  = seg.conditions.propulsion.throttle[:, 0]
        ax.plot(time, eta, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Throttle [-]')
    ax.set_title('BIPOL Regional Jet - Throttle Setting')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/03_throttle.png', dpi=150)
    plt.close(fig)

    # ---- Plot 4: Angle of Attack ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        aoa  = seg.conditions.aerodynamics.angle_of_attack[:, 0] / Units.deg
        ax.plot(time, aoa, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Angle of Attack [deg]')
    ax.set_title('BIPOL Regional Jet - Angle of Attack')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/04_angle_of_attack.png', dpi=150)
    plt.close(fig)

    # ---- Plot 5: Fuel Burn Rate ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        mdot = -seg.conditions.weights.vehicle_mass_rate[:, 0]  # positive = fuel burn
        ax.plot(time, mdot, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Fuel Burn Rate [kg/s]')
    ax.set_title('BIPOL Regional Jet - Fuel Burn Rate')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/05_fuel_burn_rate.png', dpi=150)
    plt.close(fig)

    # ---- Plot 6: Vehicle Mass ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        mass = seg.conditions.weights.total_mass[:, 0]
        ax.plot(time, mass, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Vehicle Mass [kg]')
    ax.set_title('BIPOL Regional Jet - Mass Variation')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/06_vehicle_mass.png', dpi=150)
    plt.close(fig)

    # ---- Plot 7: CL and CD ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        CL   = seg.conditions.aerodynamics.lift_coefficient[:, 0]
        CD   = seg.conditions.aerodynamics.drag_coefficient[:, 0]
        ax1.plot(time, CL, '-o', color=colors[i], label=seg.tag, markersize=3)
        ax2.plot(time, CD, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax1.set_ylabel('CL [-]')
    ax1.set_title('BIPOL Regional Jet - Aerodynamic Coefficients')
    ax1.legend(loc='best', fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('CD [-]')
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/07_aero_coefficients.png', dpi=150)
    plt.close(fig)

    # ---- Plot 8: L/D Ratio ----
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        CL   = seg.conditions.aerodynamics.lift_coefficient[:, 0]
        CD   = seg.conditions.aerodynamics.drag_coefficient[:, 0]
        LD   = CL / CD
        ax.plot(time, LD, '-o', color=colors[i], label=seg.tag, markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('L/D [-]')
    ax.set_title('BIPOL Regional Jet - Lift-to-Drag Ratio')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/08_lift_to_drag.png', dpi=150)
    plt.close(fig)

    # ---- Plot 9: Drag Breakdown ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, seg in enumerate(results.segments.values()):
        time = seg.conditions.frames.inertial.time[:, 0] / Units.min
        try:
            db = seg.conditions.aerodynamics.drag_breakdown
            cdp = db.parasite.total[:, 0]
            cdi = db.induced.total[:, 0]
            cdc = db.compressible.total[:, 0]
            cdm = db.miscellaneous.total[:, 0]
            cd  = db.total[:, 0]
            if i == 0:
                ax.plot(time, cdp, 'b-', linewidth=1.5, label='Parasite')
                ax.plot(time, cdi, 'r-', linewidth=1.5, label='Induced')
                ax.plot(time, cdc, 'g-', linewidth=1.5, label='Compressible')
                ax.plot(time, cdm, 'm-', linewidth=1.5, label='Miscellaneous')
                ax.plot(time, cd,  'k-', linewidth=2.0, label='Total')
            else:
                ax.plot(time, cdp, 'b-', linewidth=1.5)
                ax.plot(time, cdi, 'r-', linewidth=1.5)
                ax.plot(time, cdc, 'g-', linewidth=1.5)
                ax.plot(time, cdm, 'm-', linewidth=1.5)
                ax.plot(time, cd,  'k-', linewidth=2.0)
        except:
            pass
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('CD [-]')
    ax.set_title('BIPOL Regional Jet - Drag Component Breakdown')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/09_drag_breakdown.png', dpi=150)
    plt.close(fig)

    # ---- Plot 10: Thrust vs Drag ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, seg in enumerate(results.segments.values()):
        time   = seg.conditions.frames.inertial.time[:, 0] / Units.min
        Thrust = seg.conditions.frames.body.thrust_force_vector[:, 0] / 1000.0
        Drag   = -seg.conditions.frames.wind.drag_force_vector[:, 0] / 1000.0
        if i == 0:
            ax.plot(time, Thrust, 'b-o', markersize=3, label='Thrust')
            ax.plot(time, Drag,   'r-s', markersize=3, label='Drag')
        else:
            ax.plot(time, Thrust, 'b-o', markersize=3)
            ax.plot(time, Drag,   'r-s', markersize=3)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Force [kN]')
    ax.set_title('BIPOL Regional Jet - Thrust vs Drag')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/10_thrust_vs_drag.png', dpi=150)
    plt.close(fig)

    print(f"All mission plots saved to {save_dir}/")


def print_mission_summary(results):
    """Print a text summary of the mission results."""
    print("\n" + "=" * 70)
    print("BIPOL REGIONAL JET - MISSION SUMMARY")
    print("=" * 70)

    total_time = 0.0
    total_fuel = 0.0

    for seg in results.segments.values():
        t = seg.conditions.frames.inertial.time[:, 0]
        m = seg.conditions.weights.total_mass[:, 0]
        alt = seg.conditions.freestream.altitude[:, 0]
        mach = seg.conditions.freestream.mach_number[:, 0]

        dt = (t[-1] - t[0]) / 60.0  # minutes
        dm = m[0] - m[-1]            # kg fuel burned
        total_time += dt
        total_fuel += dm

        print(f"\n  {seg.tag}:")
        print(f"    Duration:  {float(dt):>7.1f} min")
        print(f"    Alt range: {float(alt[0]/1000):>7.2f} → {float(alt[-1]/1000):.2f} km")
        print(f"    Mach:      {float(mach[0]):>7.3f} → {float(mach[-1]):.3f}")
        print(f"    Mass:      {float(m[0]):>7.0f} → {float(m[-1]):.0f} kg")
        print(f"    Fuel burn: {float(dm):>7.1f} kg")

    print("\n" + "-" * 70)
    print(f"  TOTAL flight time:  {float(total_time):>7.1f} min ({float(total_time/60):.2f} h)")
    print(f"  TOTAL fuel burned:  {float(total_fuel):>7.1f} kg")
    print(f"  Fuel available:     {PARAMS.fuel_mass:>7.1f} kg")
    print(f"  Fuel remaining:     {PARAMS.fuel_mass - float(total_fuel):>7.1f} kg "
          f"({(PARAMS.fuel_mass - float(total_fuel))/PARAMS.fuel_mass*100:.1f}%)")
    print("=" * 70)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    print("Setting up BIPOL Regional Jet mission analysis...")

    configs, analyses = full_setup()

    print("Finalizing configurations...")
    configs.finalize()
    analyses.finalize()

    # Run weight analysis
    print("Running weight analysis...")
    weights = analyses.configs.base.weights
    breakdown = weights.evaluate()

    # Run mission
    print("Running mission analysis...")
    mission = analyses.missions
    results = mission.evaluate()

    # Print summary
    print_mission_summary(results)

    # Generate plots
    plot_mission(results, save_dir='results')

    # Save results data to CSV for inspection
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
                    f"{c.aerodynamics.lift_coefficient[j,0]/c.aerodynamics.drag_coefficient[j,0]:.2f}",
                    f"{c.weights.total_mass[j,0]:.1f}",
                    f"{c.frames.body.thrust_force_vector[j,0]:.1f}",
                    f"{-c.frames.wind.drag_force_vector[j,0]:.1f}",
                    f"{c.propulsion.throttle[j,0]:.4f}",
                    f"{c.aerodynamics.angle_of_attack[j,0]/Units.deg:.3f}",
                    f"{-c.weights.vehicle_mass_rate[j,0]:.4f}",
                ])

    print("Mission data saved to results/mission_data.csv")
    print("\nMission analysis complete!")
