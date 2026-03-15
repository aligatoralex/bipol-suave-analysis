"""
BIPOL Regional Jet - Performance Analysis
==========================================
MEiL PW Warsaw

Generates performance analyses:
  1. Payload-Range diagram (from SUAVE)
  2. V-n diagram (gust + maneuver envelope per CS-25)
  3. Specific range vs weight
  4. Rate of climb vs altitude
  5. Comparison with BIPOL Excel V-n values

Usage:
    python performance_analysis.py

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


def compute_vn_diagram(save_dir='results'):
    """
    Compute and plot V-n diagram per CS-25 regulations.
    
    Maneuver envelope + gust loads at sea level (rho=1.225 kg/m³).
    Compares with BIPOL Excel BI6 values.
    
    Returns:
        dict with V-n envelope data
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # ---- Constants from BIPOL BI6 ----
    rho_0 = 1.225          # [kg/m³] sea level
    g     = 9.81           # [m/s²]
    W     = PARAMS.MTOW * g  # [N]
    S     = PARAMS.wing_area # [m²]
    
    CLmax_clean = PARAMS.CLmax_clean  # 1.705
    CLmax_neg   = -1.0               # Negative CLmax (symmetric airfoil approx)
    
    # Load factors (CS-25)
    n_max = 2.5    # Positive limit
    n_min = -1.0   # Negative limit
    
    # Wing loading
    WS = W / S  # [N/m²] = [Pa]
    
    # ---- Characteristic speeds ----
    # VS1 (1g stall, clean)
    VS1 = np.sqrt(2 * WS / (rho_0 * CLmax_clean))
    
    # VA (maneuver speed = VS1 * sqrt(n_max))
    VA = VS1 * np.sqrt(n_max)
    
    # VC (cruise speed) - from BIPOL: 231 m/s EAS
    VC = 231.0  # [m/s] EAS from BIPOL BI6
    
    # VD (dive speed) - from BIPOL: 1.5 × VC or from BI6
    VD = 346.5  # [m/s] EAS from BIPOL BI6 (1.5 × VC)
    
    # VB (design speed for max gust intensity)
    # From BIPOL BI6: 114.24 m/s EAS
    VB = 114.24  # [m/s] EAS
    
    # VS_neg (negative stall speed)
    VS_neg = np.sqrt(2 * WS / (rho_0 * abs(CLmax_neg)))
    
    # ---- Maneuver Envelope ----
    # Positive stall boundary: n = 0.5 * rho * V^2 * CLmax_clean * S / W
    V_pos_stall = np.linspace(0, VA, 100)
    n_pos_stall = 0.5 * rho_0 * V_pos_stall**2 * CLmax_clean / WS
    
    # Negative stall boundary
    V_neg_stall = np.linspace(0, VS_neg * np.sqrt(abs(n_min)), 100)
    n_neg_stall = -0.5 * rho_0 * V_neg_stall**2 * abs(CLmax_neg) / WS
    
    # ---- Gust Envelope (CS-25.341) ----
    # Gust velocities (EAS)
    Ude_B = 20.12  # [m/s] at VB (66 fps)
    Ude_C = 15.24  # [m/s] at VC (50 fps)  
    Ude_D = 7.62   # [m/s] at VD (25 fps)
    
    # Lift curve slope (per radian)
    # Using compressible correction: CLa = 2*pi*AR/(2 + sqrt(4 + AR^2*(1 + tan^2(sweep) - M^2)))
    M_sl = 0.0  # sea level
    sweep_rad = PARAMS.wing_sweep_qc * np.pi / 180.0
    AR = PARAMS.wing_AR
    CLa = 2 * np.pi * AR / (2 + np.sqrt(4 + AR**2 * (1 - M_sl**2)))  # [1/rad]
    
    # Aircraft mass ratio (mu_g)
    c_bar = PARAMS.wing_area / PARAMS.wing_span  # Mean geometric chord
    mu_g = 2 * WS / (rho_0 * c_bar * CLa * g)
    
    # Gust alleviation factor (CS-25.341)
    Kg = 0.88 * mu_g / (5.3 + mu_g)
    
    # Gust load increment: delta_n = Kg * rho_0 * Ude * V * CLa / (2 * WS)
    def gust_n(V_eas, Ude):
        delta_n = Kg * rho_0 * Ude * V_eas * CLa / (2 * WS)
        return 1.0 + delta_n, 1.0 - delta_n  # pos, neg
    
    n_gust_B_pos, n_gust_B_neg = gust_n(VB, Ude_B)
    n_gust_C_pos, n_gust_C_neg = gust_n(VC, Ude_C)
    n_gust_D_pos, n_gust_D_neg = gust_n(VD, Ude_D)
    
    # ---- BIPOL Reference Values for comparison ----
    bipol_ref = {
        'VS1': 57.39, 'VA': 90.75, 'VB': 114.24, 'VC': 231.0, 'VD': 346.5,
        'n_max': 2.5, 'n_min': -1.0
    }
    
    # ---- Plot V-n Diagram ----
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Positive maneuver boundary (stall curve to VA)
    ax.plot(V_pos_stall, n_pos_stall, 'b-', linewidth=2)
    
    # Positive limit: VA → VC → VD
    ax.plot([VA, VC, VD], [n_max, n_max, n_max], 'b-', linewidth=2)
    
    # Dive return to n=0
    ax.plot([VD, VD], [n_max, 0], 'b-', linewidth=2)
    
    # Negative maneuver boundary
    V_neg_limit = VS_neg * np.sqrt(abs(n_min))
    V_neg_curve = np.linspace(0, V_neg_limit, 100)
    n_neg_curve = -0.5 * rho_0 * V_neg_curve**2 * abs(CLmax_neg) / WS
    n_neg_curve = np.maximum(n_neg_curve, n_min)
    ax.plot(V_neg_curve, n_neg_curve, 'b-', linewidth=2)
    
    # Negative limit: from stall to VC
    ax.plot([V_neg_limit, VC], [n_min, n_min], 'b-', linewidth=2)
    
    # VC → VD negative ramp to 0
    ax.plot([VC, VD], [n_min, 0], 'b-', linewidth=2)
    
    # Gust lines (positive)
    ax.plot([0, VB], [1, n_gust_B_pos], 'r--', linewidth=1.5, label=f'Gust VB ({Ude_B:.1f} m/s)')
    ax.plot([0, VC], [1, n_gust_C_pos], 'g--', linewidth=1.5, label=f'Gust VC ({Ude_C:.1f} m/s)')
    ax.plot([0, VD], [1, n_gust_D_pos], 'm--', linewidth=1.5, label=f'Gust VD ({Ude_D:.1f} m/s)')
    
    # Gust lines (negative)
    ax.plot([0, VB], [1, n_gust_B_neg], 'r--', linewidth=1.5)
    ax.plot([0, VC], [1, n_gust_C_neg], 'g--', linewidth=1.5)
    ax.plot([0, VD], [1, n_gust_D_neg], 'm--', linewidth=1.5)
    
    # Gust envelope (thick orange)
    gust_V = [0, VB, VC, VD]
    gust_n_pos = [1, n_gust_B_pos, n_gust_C_pos, n_gust_D_pos]
    gust_n_neg = [1, n_gust_B_neg, n_gust_C_neg, n_gust_D_neg]
    ax.plot(gust_V, gust_n_pos, 'r-', linewidth=2.5, alpha=0.7, label='Gust envelope')
    ax.plot(gust_V, gust_n_neg, 'r-', linewidth=2.5, alpha=0.7)
    
    # Reference lines
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=1, color='grey', linewidth=0.5, linestyle=':')
    
    # Annotate key speeds
    for name, V_val in [('VS1', VS1), ('VA', VA), ('VB', VB), ('VC', VC), ('VD', VD)]:
        ax.axvline(x=V_val, color='grey', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.annotate(f'{name}\n{V_val:.1f}', xy=(V_val, -1.5), fontsize=9,
                    ha='center', color='darkblue')
    
    ax.set_xlabel('Equivalent Airspeed V_EAS [m/s]', fontsize=12)
    ax.set_ylabel('Load Factor n [-]', fontsize=12)
    ax.set_title('BIPOL Regional Jet - V-n Diagram (CS-25, Sea Level, MTOW)', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, VD * 1.1])
    ax.set_ylim([-2.0, 4.0])
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/16_vn_diagram.png', dpi=150)
    plt.close(fig)
    
    # ---- Print comparison ----
    print("\n" + "=" * 70)
    print("V-n DIAGRAM COMPARISON: SUAVE vs BIPOL BI6")
    print("=" * 70)
    print(f"  {'Parameter':<20} {'SUAVE':>10} {'BIPOL':>10} {'Diff':>10}")
    print(f"  {'-'*50}")
    print(f"  {'VS1 [m/s]':<20} {VS1:>10.2f} {bipol_ref['VS1']:>10.2f} "
          f"{(VS1-bipol_ref['VS1'])/bipol_ref['VS1']*100:>+10.1f}%")
    print(f"  {'VA [m/s]':<20} {VA:>10.2f} {bipol_ref['VA']:>10.2f} "
          f"{(VA-bipol_ref['VA'])/bipol_ref['VA']*100:>+10.1f}%")
    print(f"  {'VB [m/s]':<20} {VB:>10.2f} {bipol_ref['VB']:>10.2f} {'(input)':>10}")
    print(f"  {'VC [m/s]':<20} {VC:>10.2f} {bipol_ref['VC']:>10.2f} {'(input)':>10}")
    print(f"  {'VD [m/s]':<20} {VD:>10.2f} {bipol_ref['VD']:>10.2f} {'(input)':>10}")
    print(f"  {'n_max':<20} {n_max:>10.1f} {bipol_ref['n_max']:>10.1f}")
    print(f"  {'n_min':<20} {n_min:>10.1f} {bipol_ref['n_min']:>10.1f}")
    print(f"\n  Gust alleviation factor Kg = {Kg:.4f}")
    print(f"  Mass ratio mu_g = {mu_g:.2f}")
    print(f"  Gust n at VB: +{n_gust_B_pos:.3f} / {n_gust_B_neg:.3f}")
    print(f"  Gust n at VC: +{n_gust_C_pos:.3f} / {n_gust_C_neg:.3f}")
    print(f"  Gust n at VD: +{n_gust_D_pos:.3f} / {n_gust_D_neg:.3f}")
    print("=" * 70)
    
    return {
        'VS1': VS1, 'VA': VA, 'VB': VB, 'VC': VC, 'VD': VD,
        'n_max': n_max, 'n_min': n_min, 'Kg': Kg, 'mu_g': mu_g,
        'gust_B': (n_gust_B_pos, n_gust_B_neg),
        'gust_C': (n_gust_C_pos, n_gust_C_neg),
        'gust_D': (n_gust_D_pos, n_gust_D_neg),
    }


def compute_payload_range(save_dir='results'):
    """
    Compute and plot payload-range diagram.
    
    Four key points:
      A: Max payload, min fuel (ferry from 0)
      B: Max payload, max range with max payload
      C: Reduced payload, max fuel, extended range
      D: Zero payload, max fuel, max ferry range
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Key weights
    MTOW = PARAMS.MTOW      # 24,035 kg
    OEW  = PARAMS.OEW       # 14,360 kg
    max_payload = PARAMS.payload  # 4,000 kg
    max_fuel = PARAMS.fuel_mass   # 5,335 kg
    
    # Breguet range equation:
    # R = (V / SFC / g) * (L/D) * ln(Wi / Wf)
    
    # Cruise parameters
    V_cruise = PARAMS.cruise_Mach * 295.07  # approx speed of sound at 12 km ≈ 295 m/s
    V = PARAMS.cruise_Mach * 295.07  # m/s
    SFC = PARAMS.SFC_cruise          # kg/(s·N)
    g = 9.81
    L_D = 15.0  # Estimated from BIPOL (will be refined by SUAVE)
    
    def breguet_range(W_start, W_fuel):
        """Range [km] from Breguet equation."""
        W_end = W_start - W_fuel
        if W_end <= 0 or W_start <= 0:
            return 0.0
        R = (V / (SFC * g)) * (L_D / 1.0) * np.log(W_start / W_end)
        return R / 1000.0  # km
    
    # Point A: Max payload, zero range
    payload_A = max_payload
    range_A   = 0.0
    
    # Point B: Max payload, design fuel
    W_start_B = OEW + max_payload + max_fuel
    if W_start_B > MTOW:
        # Fuel limited by MTOW
        fuel_B = MTOW - OEW - max_payload
    else:
        fuel_B = max_fuel
    W_start_B = OEW + max_payload + fuel_B
    payload_B = max_payload
    range_B = breguet_range(W_start_B, fuel_B * 0.95)  # 95% usable fuel
    
    # Point C: Reduced payload, max fuel, MTOW limited
    fuel_C = max_fuel
    payload_C = MTOW - OEW - fuel_C
    if payload_C < 0:
        payload_C = 0
        fuel_C = MTOW - OEW
    W_start_C = OEW + payload_C + fuel_C
    range_C = breguet_range(W_start_C, fuel_C * 0.95)
    
    # Point D: Zero payload, max fuel
    payload_D = 0.0
    fuel_D = max_fuel
    W_start_D = OEW + fuel_D
    range_D = breguet_range(W_start_D, fuel_D * 0.95)
    
    # Build full curve
    ranges   = [range_A, range_B, range_C, range_D]
    payloads = [payload_A, payload_B, payload_C, payload_D]
    
    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(ranges, payloads, 'b-o', linewidth=2.5, markersize=8, zorder=5)
    
    # Annotate points
    labels = ['A: Max Payload\nZero Range', 
              f'B: Max Payload\nDesign Fuel\nR={range_B:.0f} km',
              f'C: Max Fuel\nReduced Payload\nR={range_C:.0f} km',
              f'D: Ferry\nZero Payload\nR={range_D:.0f} km']
    offsets = [(50, 200), (50, 200), (50, -200), (50, 200)]
    
    for i, (r, p, lab, off) in enumerate(zip(ranges, payloads, labels, offsets)):
        ax.annotate(lab, xy=(r, p), xytext=(r+off[0], p+off[1]),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    # Design range line
    ax.axvline(x=PARAMS.design_range, color='green', linestyle='--', alpha=0.5, 
               label=f'Design Range = {PARAMS.design_range:.0f} km')
    
    # Fill
    ax.fill_between(ranges, payloads, alpha=0.1, color='blue')
    
    ax.set_xlabel('Range [km]', fontsize=12)
    ax.set_ylabel('Payload [kg]', fontsize=12)
    ax.set_title('BIPOL Regional Jet - Payload-Range Diagram\n'
                 f'(Breguet, M={PARAMS.cruise_Mach}, h={PARAMS.cruise_alt/1000:.0f} km, '
                 f'L/D={L_D:.0f})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    fig.savefig(f'{save_dir}/17_payload_range.png', dpi=150)
    plt.close(fig)
    
    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("PAYLOAD-RANGE DIAGRAM SUMMARY")
    print("=" * 70)
    print(f"  {'Point':<10} {'Payload [kg]':>15} {'Fuel [kg]':>12} {'Range [km]':>12}")
    print(f"  {'-'*49}")
    print(f"  {'A':<10} {payload_A:>15.0f} {0:>12.0f} {range_A:>12.0f}")
    print(f"  {'B':<10} {payload_B:>15.0f} {fuel_B:>12.0f} {range_B:>12.0f}")
    print(f"  {'C':<10} {payload_C:>15.0f} {fuel_C:>12.0f} {range_C:>12.0f}")
    print(f"  {'D':<10} {payload_D:>15.0f} {fuel_D:>12.0f} {range_D:>12.0f}")
    print(f"\n  BIPOL design range: {PARAMS.design_range:.0f} km")
    print(f"  L/D assumed: {L_D:.1f}")
    print(f"  SFC: {PARAMS.SFC_cruise:.2e} kg/(s·N)")
    print("=" * 70)
    
    return {'ranges': ranges, 'payloads': payloads,
            'points': {'A': (range_A, payload_A), 'B': (range_B, payload_B),
                       'C': (range_C, payload_C), 'D': (range_D, payload_D)}}


def compute_specific_range(save_dir='results'):
    """
    Compute and plot specific range (SR) vs aircraft weight.
    SR = V / (SFC * D) = V * L/D / (SFC * W)
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    V = PARAMS.cruise_Mach * 295.07  # m/s at 12 km
    SFC = PARAMS.SFC_cruise          # kg/(s·N)
    g = 9.81
    
    # Weight range from OEW to MTOW
    W_range = np.linspace(PARAMS.OEW, PARAMS.MTOW, 50) * g  # [N]
    
    # CL at cruise for each weight
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    atmo_data = atmo.compute_values(PARAMS.cruise_alt)
    rho = float(atmo_data.density[0, 0])
    q = 0.5 * rho * V**2
    
    CL_range = W_range / (q * PARAMS.wing_area)
    
    # Drag polar (parabolic approximation from BIPOL)
    CD0 = 0.02897
    e = 0.80  # Oswald factor (estimated)
    K = 1.0 / (np.pi * e * PARAMS.wing_AR)
    CD_range = CD0 + K * CL_range**2
    L_D_range = CL_range / CD_range
    
    # Specific range: SR = V / (SFC * g * W * (CD/CL)) = V * L/D / (SFC * W)
    # In km per kg fuel
    SR = V * L_D_range / (SFC * g * (W_range/g))  # m per kg fuel
    SR_km = SR / 1000.0  # km per kg fuel
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # SR vs Weight
    ax1.plot(W_range / (g * 1000), SR_km, 'b-', linewidth=2)
    ax1.set_xlabel('Aircraft Weight [tonnes]', fontsize=12)
    ax1.set_ylabel('Specific Range [km/kg fuel]', fontsize=12)
    ax1.set_title('Specific Range vs Weight', fontsize=13)
    ax1.axvline(x=PARAMS.MTOW/1000, color='r', linestyle='--', alpha=0.5, label='MTOW')
    ax1.axvline(x=PARAMS.OEW/1000, color='g', linestyle='--', alpha=0.5, label='OEW')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # L/D vs Weight
    ax2.plot(W_range / (g * 1000), L_D_range, 'r-', linewidth=2)
    ax2.set_xlabel('Aircraft Weight [tonnes]', fontsize=12)
    ax2.set_ylabel('L/D [-]', fontsize=12)
    ax2.set_title('Cruise L/D vs Weight', fontsize=13)
    ax2.axvline(x=PARAMS.MTOW/1000, color='r', linestyle='--', alpha=0.5, label='MTOW')
    ax2.axvline(x=PARAMS.OEW/1000, color='g', linestyle='--', alpha=0.5, label='OEW')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('BIPOL Regional Jet - Cruise Performance', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/18_specific_range.png', dpi=150)
    plt.close(fig)
    
    print(f"Specific range plot saved to {save_dir}/")


def compute_climb_performance(save_dir='results'):
    """
    Compute and plot rate of climb vs altitude.
    Uses excess thrust method: ROC = (T - D) * V / W
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    g = 9.81
    W = PARAMS.MTOW * g  # [N]
    S = PARAMS.wing_area
    T_total_SL = PARAMS.total_thrust  # [N] total sea level static
    
    # Drag polar
    CD0 = 0.02897
    e = 0.80
    K = 1.0 / (np.pi * e * PARAMS.wing_AR)
    
    altitudes = np.linspace(0, 14000, 50)  # [m]
    
    atmo = SUAVE.Analyses.Atmospheric.US_Standard_1976()
    
    ROC_max = []
    V_best_climb = []
    
    for h in altitudes:
        atmo_data = atmo.compute_values(h)
        rho = float(atmo_data.density[0, 0])
        a = float(atmo_data.speed_of_sound[0, 0])
        T = float(atmo_data.temperature[0, 0])
        
        # Thrust lapse with altitude (simple model)
        sigma = rho / 1.225
        T_avail = T_total_SL * sigma**0.7  # Typical lapse for turbofan
        
        # Best climb speed: V for minimum drag
        V_md = np.sqrt(2 * W / (rho * S) * np.sqrt(K / CD0))
        
        # Limit to reasonable speeds
        V_md = min(V_md, 0.78 * a)  # Don't exceed M0.78
        V_md = max(V_md, 60.0)      # Don't go below ~120 kts
        
        # Drag at best climb speed
        CL = 2 * W / (rho * V_md**2 * S)
        CD = CD0 + K * CL**2
        D = 0.5 * rho * V_md**2 * S * CD
        
        # Rate of climb
        roc = (T_avail - D) * V_md / W  # [m/s]
        roc = max(roc, 0.0)
        
        ROC_max.append(roc)
        V_best_climb.append(V_md)
    
    ROC_max = np.array(ROC_max)
    V_best_climb = np.array(V_best_climb)
    
    # Service ceiling (ROC = 0.5 m/s ~ 100 fpm)
    idx_ceiling = np.where(ROC_max < 0.5)[0]
    if len(idx_ceiling) > 0:
        ceiling = altitudes[idx_ceiling[0]] / 1000.0
    else:
        ceiling = 14.0
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    ax1.plot(ROC_max * 60.0 / 0.3048, altitudes / 1000.0, 'b-', linewidth=2)
    ax1.set_xlabel('Rate of Climb [ft/min]', fontsize=12)
    ax1.set_ylabel('Altitude [km]', fontsize=12)
    ax1.set_title('Rate of Climb vs Altitude', fontsize=13)
    ax1.axhline(y=PARAMS.cruise_alt/1000, color='r', linestyle='--', alpha=0.5,
                label=f'Cruise alt = {PARAMS.cruise_alt/1000:.0f} km')
    ax1.axhline(y=ceiling, color='g', linestyle='--', alpha=0.5,
                label=f'Service ceiling ≈ {ceiling:.1f} km')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(V_best_climb, altitudes / 1000.0, 'r-', linewidth=2)
    ax2.set_xlabel('Best Climb Speed [m/s]', fontsize=12)
    ax2.set_ylabel('Altitude [km]', fontsize=12)
    ax2.set_title('Best Climb Speed vs Altitude', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'BIPOL Regional Jet - Climb Performance (MTOW={PARAMS.MTOW:.0f} kg)', 
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f'{save_dir}/19_climb_performance.png', dpi=150)
    plt.close(fig)
    
    print(f"\n  Service ceiling: ~{ceiling:.1f} km ({ceiling*3280.84:.0f} ft)")
    print(f"  Max ROC at SL: {ROC_max[0]*60/0.3048:.0f} ft/min ({ROC_max[0]:.1f} m/s)")
    print(f"  ROC at cruise alt: {ROC_max[np.argmin(np.abs(altitudes-PARAMS.cruise_alt))]*60/0.3048:.0f} ft/min")
    
    return {'altitudes': altitudes, 'ROC': ROC_max, 'V_climb': V_best_climb,
            'ceiling_km': ceiling}


# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    print("BIPOL Regional Jet - Performance Analysis")
    print("=" * 50)
    
    print("\n1. V-n Diagram...")
    vn_data = compute_vn_diagram(save_dir='results')
    
    print("\n2. Payload-Range Diagram...")
    pr_data = compute_payload_range(save_dir='results')
    
    print("\n3. Specific Range...")
    compute_specific_range(save_dir='results')
    
    print("\n4. Climb Performance...")
    climb_data = compute_climb_performance(save_dir='results')
    
    print("\nPerformance analysis complete!")
