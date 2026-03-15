#!/bin/bash
# ============================================================
# BIPOL SUAVE Analysis - Setup Script
# ============================================================
# This script installs SUAVE and applies all necessary patches
# for Python 3.13+ / NumPy 2.x compatibility.
#
# Usage:
#   chmod +x setup_suave.sh
#   ./setup_suave.sh
# ============================================================

set -e

echo "=== BIPOL SUAVE Setup ==="

# 1. Install dependencies
echo "Installing Python dependencies..."
pip install numpy scipy matplotlib "setuptools<80" --quiet

# 2. Clone SUAVE if not present
SUAVE_DIR="/tmp/SUAVE"
if [ ! -d "$SUAVE_DIR" ]; then
    echo "Cloning SUAVE from GitHub..."
    git clone https://github.com/suavecode/SUAVE.git "$SUAVE_DIR"
else
    echo "SUAVE already present at $SUAVE_DIR"
fi

# 3. Apply Python 3.13 / NumPy 2.x compatibility patches
echo "Applying compatibility patches..."

# Patch: cumtrapz -> cumulative_trapezoid
find "$SUAVE_DIR/trunk" -name "*.py" -exec sed -i \
    's/from scipy.integrate import cumtrapz/from scipy.integrate import cumulative_trapezoid as cumtrapz/g' {} +

# Patch: np.float( -> float(
find "$SUAVE_DIR/trunk" -name "*.py" -exec sed -i 's/np\.float(/float(/g' {} +
find "$SUAVE_DIR/trunk" -name "*.py" -exec sed -i 's/np\.int(/int(/g' {} +

# Patch: VLM numpy 2.x fix (np.unique return_inverse shape + np.linalg.solve batching)
VLM_FILE="$SUAVE_DIR/trunk/SUAVE/Methods/Aerodynamics/Common/Fidelity_Zero/Lift/VLM.py"
if [ -f "$VLM_FILE" ]; then
    # Add inv.ravel() after np.unique
    sed -i '/m_unique, inv = np.unique(mach,return_inverse=True)/a\    inv           = inv.ravel()  # numpy 2.x compat: ensure 1D index array' "$VLM_FILE"
    
    # Fix np.linalg.solve for batched 3D arrays
    sed -i 's/    GAMMA  = np.linalg.solve(A,RHS)/    # numpy 2.x compat: expand RHS for batched solve\n    if A.ndim == 3 and RHS.ndim == 2:\n        GAMMA = np.linalg.solve(A, RHS[:,:,np.newaxis])[:,:,0]\n    else:\n        GAMMA = np.linalg.solve(A, RHS)/' "$VLM_FILE"
    
    echo "  VLM.py patched for numpy 2.x"
fi

# 4. Create scipy compatibility shim
cat > /tmp/scipy_compat.py << 'EOF'
"""Scipy compatibility shim for SUAVE on Python 3.13+"""
import scipy.integrate
if not hasattr(scipy.integrate, 'cumtrapz'):
    try:
        from scipy.integrate import cumulative_trapezoid as cumtrapz
        scipy.integrate.cumtrapz = cumtrapz
    except ImportError:
        pass
EOF
echo "  scipy_compat.py created"

# 5. Verify installation
echo ""
echo "Verifying SUAVE installation..."
python -c "
import sys
sys.path.insert(0, '$SUAVE_DIR/trunk')
sys.path.insert(0, '/tmp')
import warnings; warnings.filterwarnings('ignore')
import SUAVE
print(f'  SUAVE imported successfully')
print(f'  Python: {sys.version.split()[0]}')
import numpy; print(f'  NumPy: {numpy.__version__}')
import scipy; print(f'  SciPy: {scipy.__version__}')
import matplotlib; print(f'  Matplotlib: {matplotlib.__version__}')
"

echo ""
echo "=== Setup complete! ==="
echo "Run: cd bipol_suave && python run_all.py"
