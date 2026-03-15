"""
BIPOL Regional Jet - Vehicle Definition for SUAVE
==================================================
MEiL PW - Grzegorz Wasilczuk
Aircraft: ~40 pax regional jet, 2 turbofan engines, T-tail
Certification: CS-25

This module defines the complete vehicle geometry, propulsion, and weights
for the BIPOL regional jet project. All parameters come directly from the
BIPOL Excel files (BI1-BI6).

Usage:
    from vehicle_definition import vehicle_setup, configs_setup
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)

To modify parameters, edit the clearly marked PARAMS section below.
"""

import sys
sys.path.insert(0, '/tmp/SUAVE/trunk')
sys.path.insert(0, '/tmp')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import SUAVE
from SUAVE.Core import Data, Units
from SUAVE.Components.Energy.Networks.Turbofan import Turbofan
from SUAVE.Components.Energy.Converters import Fan, Compressor, Turbine, Combustor
from SUAVE.Components.Energy.Converters.Ram import Ram
from SUAVE.Components.Energy.Converters.Compression_Nozzle import Compression_Nozzle
from SUAVE.Components.Energy.Converters.Expansion_Nozzle import Expansion_Nozzle
from SUAVE.Components.Energy.Processes.Thrust import Thrust
from SUAVE.Methods.Propulsion.turbofan_sizing import turbofan_sizing
from SUAVE.Methods.Geometry.Two_Dimensional.Planform import wing_planform

from copy import deepcopy


# =============================================================================
# USER-MODIFIABLE AIRCRAFT PARAMETERS
# =============================================================================
# Change these values to explore design variations

PARAMS = Data()

# --- Mass parameters (from BI2 mass iteration) ---
PARAMS.MTOW      = 24035.0   # [kg] Maximum takeoff weight
PARAMS.OEW       = 14360.0   # [kg] Operating empty weight
PARAMS.payload   = 4000.0    # [kg] Design payload
PARAMS.fuel_mass = 5335.0    # [kg] Design fuel
PARAMS.crew_mass = 340.0     # [kg] Crew (2 pilots + cabin crew)
PARAMS.pax       = 40        # Number of passengers

# --- Wing parameters (from BI2 matching chart + geometry) ---
PARAMS.wing_area      = 70.0265  # [m²] Wing reference area
PARAMS.wing_span      = 27.5     # [m] Wingspan
PARAMS.wing_AR        = 10.80    # Aspect ratio
PARAMS.wing_taper     = 0.36     # Taper ratio
PARAMS.wing_sweep_qc  = 25.0     # [deg] Quarter-chord sweep
PARAMS.wing_twist     = -3.0     # [deg] Twist at tip (washout)
PARAMS.wing_tc        = 0.14     # Thickness-to-chord ratio
PARAMS.wing_dihedral  = 3.0      # [deg] Dihedral angle
PARAMS.wing_incidence = 2.0      # [deg] Wing incidence angle
PARAMS.wing_pos_z     = -0.5     # [m] Wing vertical position (low wing)

# --- Airfoil (NASA SC(2)-0614) ---
PARAMS.CLmax_clean = 1.705   # Max CL clean
PARAMS.CLmax_TO    = 1.845   # Max CL takeoff flaps
PARAMS.CLmax_land  = 2.065   # Max CL landing flaps

# --- Fuselage parameters (from BI3 + hand drawings) ---
PARAMS.fus_length   = 26.6   # [m] Fuselage length
PARAMS.fus_width    = 3.0    # [m] Fuselage width (estimated from pax layout)
PARAMS.fus_height   = 3.2    # [m] Fuselage height
PARAMS.fus_fineness = PARAMS.fus_length / PARAMS.fus_width  # Fineness ratio

# --- Horizontal tail (from BI3) ---
PARAMS.HT_area    = 11.165  # [m²]
PARAMS.HT_span    = 6.5     # [m] (estimated from AR ≈ 3.3)
PARAMS.HT_AR      = 3.29    # Aspect ratio
PARAMS.HT_sweep   = 30.0    # [deg] Quarter-chord sweep
PARAMS.HT_taper   = 0.40    # Taper ratio
PARAMS.HT_tc      = 0.10    # Thickness-to-chord
PARAMS.HT_arm     = 12.4    # [m] Moment arm from wing AC to HT AC

# --- Vertical tail (from BI3) ---
PARAMS.VT_area    = 11.07   # [m²]
PARAMS.VT_span    = 3.2     # [m]
PARAMS.VT_AR      = 0.92    # Aspect ratio
PARAMS.VT_sweep   = 35.0    # [deg] Quarter-chord sweep
PARAMS.VT_taper   = 0.40    # Taper ratio
PARAMS.VT_tc      = 0.10    # Thickness-to-chord
PARAMS.VT_arm     = 11.3    # [m] Moment arm

# --- Engine parameters (2x turbofan) ---
PARAMS.n_engines       = 2
PARAMS.total_thrust    = 62275.0  # [N] Total static thrust (2 engines)
PARAMS.single_thrust   = 31137.5  # [N] Per-engine thrust
PARAMS.SFC_cruise      = 1.84e-5  # [kg/(s·N)] Specific fuel consumption
PARAMS.bypass_ratio    = 5.0      # Bypass ratio (typical for regional jet)
PARAMS.engine_length   = 2.5      # [m] Engine nacelle length
PARAMS.engine_diameter = 1.2      # [m] Fan diameter

# --- Performance targets ---
PARAMS.cruise_Mach  = 0.78     # Cruise Mach number
PARAMS.cruise_alt   = 12000.0  # [m] Cruise altitude (~39,370 ft)
PARAMS.design_range = 2800.0   # [km] Design range


def vehicle_setup():
    """
    Build the complete SUAVE vehicle object for the BIPOL regional jet.
    
    Based on the Embraer E190 reference pattern from SUAVE regression tests.
    All parameters from BIPOL BI1-BI6 Excel spreadsheets.

    Returns:
        SUAVE.Vehicle: Fully configured vehicle
    """
    vehicle = SUAVE.Vehicle()
    vehicle.tag = 'BIPOL_Regional_Jet'

    # ------------------------------------------------------------------
    # Vehicle-level properties
    # ------------------------------------------------------------------
    vehicle.mass_properties.max_takeoff     = PARAMS.MTOW * Units.kg
    vehicle.mass_properties.takeoff         = PARAMS.MTOW * Units.kg
    vehicle.mass_properties.operating_empty = PARAMS.OEW * Units.kg
    vehicle.mass_properties.max_zero_fuel   = (PARAMS.OEW + PARAMS.payload) * Units.kg
    vehicle.mass_properties.max_fuel        = PARAMS.fuel_mass * Units.kg
    vehicle.mass_properties.max_payload     = PARAMS.payload * Units.kg
    vehicle.mass_properties.cargo           = 0.0

    vehicle.passengers = PARAMS.pax

    vehicle.reference_area = PARAMS.wing_area

    vehicle.envelope.ultimate_load = 3.75  # 2.5 × 1.5 safety factor
    vehicle.envelope.limit_load    = 2.5   # CS-25

    vehicle.systems.control    = "fully powered"
    vehicle.systems.accessories = "medium range"

    # ------------------------------------------------------------------
    # Main Wing
    # ------------------------------------------------------------------
    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'

    wing.areas.reference   = PARAMS.wing_area
    wing.spans.projected   = PARAMS.wing_span
    wing.aspect_ratio      = PARAMS.wing_AR
    wing.sweeps.quarter_chord = PARAMS.wing_sweep_qc * Units.deg
    wing.taper             = PARAMS.wing_taper
    wing.thickness_to_chord = PARAMS.wing_tc
    wing.dihedral          = PARAMS.wing_dihedral * Units.deg
    wing.twists.root       = PARAMS.wing_incidence * Units.deg
    wing.twists.tip        = PARAMS.wing_twist * Units.deg

    wing.chords.root       = 2 * PARAMS.wing_area / (PARAMS.wing_span * (1 + PARAMS.wing_taper))
    wing.chords.tip        = wing.chords.root * PARAMS.wing_taper
    wing.chords.mean_aerodynamic = (2/3) * wing.chords.root * (
        (1 + PARAMS.wing_taper + PARAMS.wing_taper**2) / (1 + PARAMS.wing_taper)
    )

    wing.origin            = [[10.5, 0.0, PARAMS.wing_pos_z]]
    wing.aerodynamic_center = [0.25 * wing.chords.mean_aerodynamic, 0.0, 0.0]

    wing.vertical          = False
    wing.symmetric         = True
    wing.high_lift         = True
    wing.dynamic_pressure_ratio = 1.0

    # Flap (control surface)
    flap = SUAVE.Components.Wings.Control_Surfaces.Flap()
    flap.tag = 'flap'
    flap.span_fraction_start = 0.10
    flap.span_fraction_end   = 0.75
    flap.chord_fraction      = 0.30   # 30% chord flap
    flap.deflection          = 0.0 * Units.deg
    flap.configuration_type  = 'double_slotted'
    wing.append_control_surface(flap)

    # Apply wing_planform to calculate wetted area etc.
    wing = wing_planform(wing)
    wing.areas.exposed = 0.80 * wing.areas.wetted
    wing.twists.root   = PARAMS.wing_incidence * Units.deg
    wing.twists.tip    = PARAMS.wing_twist * Units.deg
    wing.dynamic_pressure_ratio = 1.0

    vehicle.append_component(wing)

    # ------------------------------------------------------------------
    # Horizontal Tail (T-tail)
    # ------------------------------------------------------------------
    h_tail = SUAVE.Components.Wings.Horizontal_Tail()
    h_tail.tag = 'horizontal_stabilizer'

    h_tail.areas.reference   = PARAMS.HT_area
    h_tail.spans.projected   = PARAMS.HT_span
    h_tail.aspect_ratio      = PARAMS.HT_AR
    h_tail.sweeps.quarter_chord = PARAMS.HT_sweep * Units.deg
    h_tail.taper             = PARAMS.HT_taper
    h_tail.thickness_to_chord = PARAMS.HT_tc
    h_tail.dihedral          = 0.0 * Units.deg
    h_tail.twists.root       = 0.0 * Units.deg
    h_tail.twists.tip        = 0.0 * Units.deg

    h_tail.chords.root = 2 * PARAMS.HT_area / (PARAMS.HT_span * (1 + PARAMS.HT_taper))
    h_tail.chords.tip  = h_tail.chords.root * PARAMS.HT_taper

    # T-tail: HT mounted on top of VT
    h_tail.origin = [[24.7, 0.0, 4.5]]

    h_tail.vertical   = False
    h_tail.symmetric  = True
    h_tail.high_lift  = False
    h_tail.dynamic_pressure_ratio = 0.90

    h_tail = wing_planform(h_tail)
    h_tail.areas.exposed = 0.9 * h_tail.areas.wetted
    h_tail.twists.root   = 0.0 * Units.deg
    h_tail.twists.tip    = 0.0 * Units.deg
    h_tail.dynamic_pressure_ratio = 0.90

    vehicle.append_component(h_tail)

    # ------------------------------------------------------------------
    # Vertical Tail
    # ------------------------------------------------------------------
    v_tail = SUAVE.Components.Wings.Vertical_Tail()
    v_tail.tag = 'vertical_stabilizer'

    v_tail.areas.reference   = PARAMS.VT_area
    v_tail.spans.projected   = PARAMS.VT_span
    v_tail.aspect_ratio      = PARAMS.VT_AR
    v_tail.sweeps.quarter_chord = PARAMS.VT_sweep * Units.deg
    v_tail.taper             = PARAMS.VT_taper
    v_tail.thickness_to_chord = PARAMS.VT_tc

    v_tail.chords.root = 2 * PARAMS.VT_area / (PARAMS.VT_span * (1 + PARAMS.VT_taper))
    v_tail.chords.tip  = v_tail.chords.root * PARAMS.VT_taper

    v_tail.origin = [[25.8, 0.0, 1.5]]

    v_tail.vertical   = True
    v_tail.symmetric  = False
    v_tail.high_lift  = False
    v_tail.dynamic_pressure_ratio = 1.0

    v_tail = wing_planform(v_tail)
    v_tail.areas.exposed = 0.9 * v_tail.areas.wetted
    v_tail.twists.root   = 0.0 * Units.deg
    v_tail.twists.tip    = 0.0 * Units.deg
    v_tail.dynamic_pressure_ratio = 1.0

    vehicle.append_component(v_tail)

    # ------------------------------------------------------------------
    # Fuselage
    # ------------------------------------------------------------------
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    fuselage.origin = [[0, 0, 0]]

    fuselage.number_coach_seats = PARAMS.pax
    fuselage.seats_abreast      = 4
    fuselage.seat_pitch         = 0.81  # [m] ~32 inch pitch
    fuselage.fineness.nose      = 1.5
    fuselage.fineness.tail      = 2.5

    fuselage.lengths.nose       = 4.5   # [m]
    fuselage.lengths.tail       = 7.5   # [m]
    fuselage.lengths.cabin      = PARAMS.fus_length - 4.5 - 7.5  # [m]
    fuselage.lengths.fore_space = 0.0
    fuselage.lengths.aft_space  = 0.0
    fuselage.lengths.total      = PARAMS.fus_length

    fuselage.width              = PARAMS.fus_width

    fuselage.heights.maximum    = PARAMS.fus_height
    fuselage.heights.at_quarter_length          = PARAMS.fus_height
    fuselage.heights.at_three_quarters_length   = PARAMS.fus_height * 0.85
    fuselage.heights.at_wing_root_quarter_chord = PARAMS.fus_height

    fuselage.areas.side_projected = PARAMS.fus_length * PARAMS.fus_height * 0.65
    fuselage.areas.wetted = np.pi * (PARAMS.fus_width/2) * PARAMS.fus_length * 0.85
    fuselage.areas.front_projected = np.pi * (PARAMS.fus_width/2) * (PARAMS.fus_height/2)

    fuselage.effective_diameter = (PARAMS.fus_width + PARAMS.fus_height) / 2

    fuselage.differential_pressure = 8.0 * Units.psi  # Cabin pressurization

    vehicle.append_component(fuselage)

    # ------------------------------------------------------------------
    # Nacelle
    # ------------------------------------------------------------------
    nacelle                            = SUAVE.Components.Nacelles.Nacelle()
    nacelle.diameter                   = PARAMS.engine_diameter
    nacelle.length                     = PARAMS.engine_length
    nacelle.tag                        = 'nacelle_1'
    nacelle.inlet_diameter             = PARAMS.engine_diameter * 0.95
    nacelle.origin                     = [[9.5, 3.0, -0.5]]
    nacelle.areas.wetted               = 1.1 * np.pi * nacelle.diameter * nacelle.length
    nacelle_2                          = deepcopy(nacelle)
    nacelle_2.tag                      = 'nacelle_2'
    nacelle_2.origin                   = [[9.5, -3.0, -0.5]]

    vehicle.append_component(nacelle)
    vehicle.append_component(nacelle_2)

    # ------------------------------------------------------------------
    # Turbofan Engine Network
    # ------------------------------------------------------------------
    # Following the SUAVE Embraer E190 reference pattern exactly:
    # All 11 components required by turbofan_sizing()

    gt_engine                   = Turbofan()
    gt_engine.tag               = 'turbofan'
    gt_engine.origin            = [[9.5, 3.0, -0.5], [9.5, -3.0, -0.5]]
    gt_engine.engine_length     = PARAMS.engine_length
    gt_engine.number_of_engines = PARAMS.n_engines
    gt_engine.bypass_ratio      = PARAMS.bypass_ratio

    # Working fluid
    gt_engine.working_fluid = SUAVE.Attributes.Gases.Air()

    # Component 1: Ram - converts freestream static to stagnation quantities
    ram       = Ram()
    ram.tag   = 'ram'
    gt_engine.ram = ram

    # Component 2: Inlet nozzle (compression nozzle)
    inlet_nozzle                       = Compression_Nozzle()
    inlet_nozzle.tag                   = 'inlet nozzle'
    inlet_nozzle.polytropic_efficiency = 0.98
    inlet_nozzle.pressure_ratio        = 0.98
    gt_engine.inlet_nozzle             = inlet_nozzle

    # Component 3: Low pressure compressor
    low_pressure_compressor                       = Compressor()
    low_pressure_compressor.tag                   = 'lpc'
    low_pressure_compressor.polytropic_efficiency = 0.91
    low_pressure_compressor.pressure_ratio        = 2.0
    gt_engine.low_pressure_compressor             = low_pressure_compressor

    # Component 4: High pressure compressor
    high_pressure_compressor                       = Compressor()
    high_pressure_compressor.tag                   = 'hpc'
    high_pressure_compressor.polytropic_efficiency = 0.91
    high_pressure_compressor.pressure_ratio        = 14.0
    gt_engine.high_pressure_compressor             = high_pressure_compressor

    # Component 5: Low pressure turbine
    low_pressure_turbine                        = Turbine()
    low_pressure_turbine.tag                    = 'lpt'
    low_pressure_turbine.mechanical_efficiency  = 0.99
    low_pressure_turbine.polytropic_efficiency  = 0.93
    gt_engine.low_pressure_turbine              = low_pressure_turbine

    # Component 6: High pressure turbine
    high_pressure_turbine                       = Turbine()
    high_pressure_turbine.tag                   = 'hpt'
    high_pressure_turbine.mechanical_efficiency = 0.99
    high_pressure_turbine.polytropic_efficiency = 0.93
    gt_engine.high_pressure_turbine             = high_pressure_turbine

    # Component 7: Combustor
    combustor                           = Combustor()
    combustor.tag                       = 'Comb'
    combustor.efficiency                = 0.99
    combustor.alphac                    = 1.0
    combustor.turbine_inlet_temperature = 1450.0  # [K]
    combustor.pressure_ratio            = 0.95
    combustor.fuel_data                 = SUAVE.Attributes.Propellants.Jet_A()
    gt_engine.combustor                 = combustor

    # Component 8: Core nozzle (expansion)
    core_nozzle                       = Expansion_Nozzle()
    core_nozzle.tag                   = 'core nozzle'
    core_nozzle.polytropic_efficiency = 0.95
    core_nozzle.pressure_ratio        = 0.99
    gt_engine.core_nozzle             = core_nozzle

    # Component 9: Fan nozzle (expansion)
    fan_nozzle                       = Expansion_Nozzle()
    fan_nozzle.tag                   = 'fan nozzle'
    fan_nozzle.polytropic_efficiency = 0.95
    fan_nozzle.pressure_ratio        = 0.99
    gt_engine.fan_nozzle             = fan_nozzle

    # Component 10: Fan
    fan                       = Fan()
    fan.tag                   = 'fan'
    fan.polytropic_efficiency = 0.93
    fan.pressure_ratio        = 1.7
    gt_engine.fan             = fan

    # Component 11: Thrust (to compute thrust)
    thrust     = Thrust()
    thrust.tag = 'compute_thrust'
    # total_design = total thrust from ALL engines at design point
    thrust.total_design = PARAMS.total_thrust * Units.N
    gt_engine.thrust    = thrust

    # Size the turbofan at cruise conditions (like E190 reference)
    design_altitude    = PARAMS.cruise_alt * Units.m   # 12,000 m
    design_mach_number = PARAMS.cruise_Mach            # 0.78

    turbofan_sizing(gt_engine, design_mach_number, design_altitude)

    # Add engine to vehicle
    vehicle.append_component(gt_engine)

    # Fuel
    fuel                                    = SUAVE.Components.Physical_Component()
    fuel.tag                                = 'fuel'
    fuel.mass_properties.mass               = PARAMS.fuel_mass
    fuel.origin                             = vehicle.wings.main_wing.mass_properties.center_of_gravity
    fuel.mass_properties.center_of_gravity  = vehicle.wings.main_wing.aerodynamic_center
    vehicle.fuel = fuel

    # ------------------------------------------------------------------
    # Vehicle-level aerodynamic properties
    # ------------------------------------------------------------------
    vehicle.maximum_lift_coefficient = PARAMS.CLmax_clean

    return vehicle


def configs_setup(vehicle):
    """
    Set up vehicle configurations for different flight phases.
    
    Configurations:
        - base: clean configuration
        - cruise: clean, same as base
        - takeoff: flaps 20°, CLmax_TO
        - landing: flaps 40°, CLmax_land

    Args:
        vehicle: Base SUAVE vehicle

    Returns:
        SUAVE.Components.Configs.Config.Container: Configuration container
    """
    configs = SUAVE.Components.Configs.Config.Container()

    # Base configuration
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)

    # Cruise configuration (clean)
    cruise_config = SUAVE.Components.Configs.Config(base_config)
    cruise_config.tag = 'cruise'
    configs.append(cruise_config)

    # Takeoff configuration
    takeoff_config = SUAVE.Components.Configs.Config(base_config)
    takeoff_config.tag = 'takeoff'
    takeoff_config.wings['main_wing'].control_surfaces.flap.deflection = 20.0 * Units.deg
    takeoff_config.wings['main_wing'].maximum_lift_coefficient = PARAMS.CLmax_TO
    takeoff_config.maximum_lift_coefficient = PARAMS.CLmax_TO
    takeoff_config.V2_VS_ratio = 1.20
    configs.append(takeoff_config)

    # Landing configuration
    landing_config = SUAVE.Components.Configs.Config(base_config)
    landing_config.tag = 'landing'
    landing_config.wings['main_wing'].control_surfaces.flap.deflection = 40.0 * Units.deg
    landing_config.wings['main_wing'].maximum_lift_coefficient = PARAMS.CLmax_land
    landing_config.maximum_lift_coefficient = PARAMS.CLmax_land
    landing_config.Vref_VS_ratio = 1.23
    configs.append(landing_config)

    return configs


# =============================================================================
# Test the vehicle definition
# =============================================================================
if __name__ == '__main__':
    vehicle = vehicle_setup()
    configs = configs_setup(vehicle)

    print("=" * 60)
    print("BIPOL Regional Jet - Vehicle Summary")
    print("=" * 60)
    print(f"Tag: {vehicle.tag}")
    print(f"MTOW: {vehicle.mass_properties.max_takeoff:.0f} kg")
    print(f"OEW:  {vehicle.mass_properties.operating_empty:.0f} kg")
    print(f"MZF:  {vehicle.mass_properties.max_zero_fuel:.0f} kg")
    print(f"Fuel: {vehicle.mass_properties.max_fuel:.0f} kg")
    print(f"Passengers: {vehicle.passengers}")
    print(f"Reference area: {vehicle.reference_area:.2f} m²")
    print()
    print("Wings:")
    for w in vehicle.wings:
        print(f"  {w.tag}: S={w.areas.reference:.2f} m², b={w.spans.projected:.2f} m, "
              f"AR={w.aspect_ratio:.2f}")
    print()
    print("Fuselage:")
    fus = vehicle.fuselages['fuselage']
    print(f"  Length: {fus.lengths.total:.1f} m, Width: {fus.width:.1f} m")
    print()
    print("Engine:")
    eng = vehicle.networks['turbofan']
    print(f"  N_eng: {eng.number_of_engines}, BPR: {eng.bypass_ratio}")
    print(f"  Design thrust (total): {eng.thrust.total_design:.0f} N")
    print(f"  Sized mass flow: {float(eng.thrust.mass_flow_rate_design):.4f} kg/s")
    print(f"  SLS thrust: {float(eng.sealevel_static_thrust):.0f} N")
    print()
    print("Configurations:", [c.tag for c in configs])
    print("Vehicle definition OK!")
