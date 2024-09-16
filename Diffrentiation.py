# This module numerically computes 1st and 2nd derivaties
# 2nd order accuracy for internal points
# Q is the quantity to take the derivative of
# x is the grid
import numpy as np
# First derivative
def FirstDerivative(x, Q):
    if Q.ndim == 2 and x.ndim == 2:  
        Q1 = Q.shape[0]
        Q2 = Q.shape[1]
        dQdx = np.zeros((Q1, Q2))

        # Calculate interior points (2nd-order)
        hn = x[1:-1, :] - x[:-2, :]
        hp = x[2:, :] - x[1:-1, :]
        Qp = Q[2:, :]
        Qn = Q[0:-2, :]
        Q0 = Q[1:-1, :]
        dQdx[1:-1, :] = (hn/(hp+hn))*(Qp-Q0)/hp + (hp/(hp+hn))*(Q0-Qn)/hn
        # dQdx[1:-1, :] = (Q[2:, :]-Q[:-2, :]) / (x[2:, :]-x[:-2, :])
        # Calculate first point (1st-order Forward difference)
        dQdx[0, :] = (Q[1, :]-Q[0, :]) / (x[1, :]-x[0, :])
        # Calculate end point (1st-order Backward difference)
        dQdx[-1, :] = (Q[-1, :]-Q[-2, :]) / (x[-1, :]-x[-2, :])
    elif Q.ndim == 2 and x.ndim == 1:
        Q1 = Q.shape[0]
        Q2 = Q.shape[1]
        dQdx = np.zeros((Q1, Q2))

        # Calculate interior points (2nd-order)
        hn = x[1:-1] - x[:-2]
        hp = x[2:] - x[1:-1]
        Qp = Q[2:, :]
        Qn = Q[0:-2, :]
        Q0 = Q[1:-1, :]
        dQdx[1:-1, :] = (hn/(hp+hn))*(Qp-Q0)/hp + (hp/(hp+hn))*(Q0-Qn)/hn
        # dQdx[1:-1, :] = (Q[2:, :]-Q[:-2, :]) / (x[2:, :]-x[:-2, :])
        # Calculate first point (1st-order Forward difference)
        dQdx[0, :] = (Q[1, :]-Q[0, :]) / (x[1]-x[0])
        # Calculate end point (1st-order Backward difference)
        dQdx[-1, :] = (Q[-1, :]-Q[-2, :]) / (x[-1]-x[-2])
    elif Q.ndim == 1 and x.ndim == 1:
        Q1 = Q.shape[0]
        dQdx = np.zeros(Q1)

        # Calculate interior points (2nd-order)
        hn = x[1:-1] - x[:-2]
        hp = x[2:] - x[1:-1]
        Qp = Q[2:]
        Qn = Q[0:-2]
        Q0 = Q[1:-1]
        dQdx[1:-1] = (hn/(hp+hn))*(Qp-Q0)/hp + (hp/(hp+hn))*(Q0-Qn)/hn
        # dQdx[1:-1, :] = (Q[2:, :]-Q[:-2, :]) / (x[2:, :]-x[:-2, :])
        # Calculate first point (1st-order Forward difference)
        dQdx[0] = (Q[1]-Q[0]) / (x[1]-x[0])
        # Calculate end point (1st-order Backward difference)
        dQdx[-1] = (Q[-1]-Q[-2]) / (x[-1]-x[-2])
    elif Q.ndim == 1 and x.ndim == 2:
        Q1 = Q.shape[0]
        dQdx = np.zeros(Q1)
        xw = np.zeros(Q1)
        xw = x[:, 0]

        # Calculate interior points (2nd-order)
        hn = xw[1:-1] - xw[:-2]
        hp = xw[2:] - xw[1:-1]
        Qp = Q[2:]
        Qn = Q[0:-2]
        Q0 = Q[1:-1]
        dQdx[1:-1] = (hn/(hp+hn))*(Qp-Q0)/hp + (hp/(hp+hn))*(Q0-Qn)/hn
        # dQdx[1:-1, :] = (Q[2:, :]-Q[:-2, :]) / (x[2:, :]-x[:-2, :])
        # Calculate first point (1st-order Forward difference)
        dQdx[0] = (Q[1]-Q[0]) / (xw[1]-xw[0])
        # Calculate end point (1st-order Backward difference)
        dQdx[-1] = (Q[-1]-Q[-2]) / (xw[-1]-xw[-2])
    
    return dQdx

# Second derivative
def SecondDerivative(x, Q):
    if Q.ndim == 2 and x.ndim == 2:
        Q1 = Q.shape[0]
        Q2 = Q.shape[1]
        d2Qdx2 = np.zeros((Q1, Q2))
        
        # Calculate interior points (2nd-order ??)
        hn = x[1:-1, :] - x[:-2, :]
        hp = x[2:, :] - x[1:-1, :]
        Qp = Q[2:, :]
        Qn = Q[0:-2, :]
        Q0 = Q[1:-1, :]
        d2Qdx2[1:-1, :] = (hn*Qp - (hn+hp)*Q0 + hp*Qn) / (0.5*hp*hn*(hp+hn))
        # Calculate first point (not accurate!)
        d2Qdx2[0, :] = (Q[2, :]-2*Q[1, :]+Q[0, :]) / (x[1, :]-x[0, :])**2
        # Calculate end point (not accurate!)
        d2Qdx2[-1, :] = (Q[-1, :]-2*Q[-2, :]+Q[-3, :]) / (x[-1, :]-x[-2, :])**2
    elif Q.ndim == 2 and x.ndim == 1:
        Q1 = Q.shape[0]
        Q2 = Q.shape[1]
        d2Qdx2 = np.zeros((Q1, Q2))
        
        # Calculate interior points (2nd-order ??)
        hn = x[1:-1] - x[:-2]
        hp = x[2:] - x[1:-1]
        Qp = Q[2:, :]
        Qn = Q[0:-2, :]
        Q0 = Q[1:-1, :]
        d2Qdx2[1:-1, :] = (hn*Qp - (hn+hp)*Q0 + hp*Qn) / (0.5*hp*hn*(hp+hn))
        # Calculate first point (not accurate!)
        d2Qdx2[0, :] = (Q[2, :]-2*Q[1, :]+Q[0, :]) / (x[1]-x[0])**2
        # Calculate end point (not accurate!)
        d2Qdx2[-1, :] = (Q[-1, :]-2*Q[-2, :]+Q[-3, :]) / (x[-1]-x[-2])**2
    elif Q.ndim == 1 and x.ndim == 1:
        Q1 = Q.shape[0]
        d2Qdx2 = np.zeros(Q1)
        
        # Calculate interior points (2nd-order ??)
        hn = x[1:-1] - x[:-2]
        hp = x[2:] - x[1:-1]
        Qp = Q[2:]
        Qn = Q[0:-2]
        Q0 = Q[1:-1]
        d2Qdx2[1:-1] = (hn*Qp - (hn+hp)*Q0 + hp*Qn) / (0.5*hp*hn*(hp+hn))
        # Calculate first point (not accurate!)
        d2Qdx2[0] = (Q[2]-2*Q[1]+Q[0]) / (x[1]-x[0])**2
        # Calculate end point (not accurate!)
        d2Qdx2[-1] = (Q[-1]-2*Q[-2]+Q[-3]) / (x[-1]-x[-2])**2
    elif Q.ndim == 1 and x.ndim == 2:
        Q1 = Q.shape[0]
        dQdx = np.zeros(Q1)
        xw = np.zeros(Q1)
        xw = x[:, 0]
        
        # Calculate interior points (2nd-order ??)
        hn = xw[1:-1] - xw[:-2]
        hp = xw[2:] - xw[1:-1]
        Qp = Q[2:]
        Qn = Q[0:-2]
        Q0 = Q[1:-1]
        d2Qdx2[1:-1] = (hn*Qp - (hn+hp)*Q0 + hp*Qn) / (0.5*hp*hn*(hp+hn))
        # Calculate first point (not accurate!)
        d2Qdx2[0] = (Q[2]-2*Q[1]+Q[0]) / (xw[1]-xw[0])**2
        # Calculate end point (not accurate!)
        d2Qdx2[-1] = (Q[-1]-2*Q[-2]+Q[-3]) / (xw[-1]-xw[-2])**2

    return d2Qdx2