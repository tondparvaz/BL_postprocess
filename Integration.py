# This module numerically integrate a field
# The integration can be truncated using windowing
import numpy as np

# Armin embed integration and windowing together!!!!
# Windowing the field to truncate integration at c*delta_99
def MP_Integral(y, upend, Q, c):
    upendc = upend*c
    n = np.shape(Q)[0]
    m = np.shape(Q)[1]
    IQ = np.zeros(n)
    for i in range(0, n):
        # Integration upper limit
        y_ind = np.argmax(y > upendc[i])
        yInt = np.concatenate((y[:y_ind], [upendc[i]]))
        my = len(yInt)
        
        # Windowing
        Qdlta = np.interp(upendc[i], y, Q[i, :])
        Q[i, :] = np.concatenate([Q[i, :y_ind], [Qdlta], np.zeros(m - len(yInt))])
        
        # Mid-point integration
        IQ[i] = np.sum(((Q[i, 1:my] + Q[i, :my-1])/2) * np.diff(yInt))
        
    return IQ     

def TZ_Integral(y, lowend, upend, Q, c):
    upendc = upend*c
    m = len(Q)
    # Integration upper limit
    y_ind_up = np.argmax(y >= upendc)
    # Integration lower limit
    y_ind_low = np.argmax(y >= lowend)
    
    yInt = np.concatenate(([lowend], y[y_ind_low:y_ind_up], [upendc]))
    my = len(yInt)
    
    # Windowing
    Qup = np.interp(upendc, y, Q)
    Qlow = np.interp(lowend, y, Q)
    Q = np.concatenate([[Qlow], Q[y_ind_low:y_ind_up], [Qup]])
    # Q = np.concatenate([np.zeros(y_ind_low), [Qlow], Q[y_ind_low:y_ind_up], [Qup], np.zeros(m-y_ind_up-1)])
    
    # Mid-point integration
    # IQ = np.sum(((Q[1:my] + Q[:my-1])/2) * np.diff(yInt))
    # Trapezoidal integration
    IQ = np.trapz(Q, yInt)
        
    return IQ  

