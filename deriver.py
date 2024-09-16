import numpy as np
import matplotlib.pyplot as plt
from PPAnalysis import *

# directory = 'FSI_v5'
nu = 5.547542679944773e-06 

# t1 = BL_PPAnalysis(directory, nu)
# t1.LoadGrids()
# t1.LoadStats()
# t1.BL_edge()
# t1.BL_Thickness()
# t1.BL_Reynolds(t1.Ue)
# t1.BL_TKE()
# t1.BL_Cf()

# directory = 'v5'
# t2 = BL_PPAnalysis(directory, nu)
# t2.LoadGrids()
# t2.LoadStats()
# t2.BL_edge()
# t2.BL_Thickness()
# t2.BL_Reynolds(t2.Ue)
# t2.BL_TKE()
# t2.BL_Cf()

directory = 'v7'
t3 = BL_PPAnalysis(directory, nu)
t3.LoadGrids()
t3.LoadStats()
t3.BL_edge()
t3.BL_Thickness()
t3.BL_Reynolds(1.)
t3.BL_TKE()
t3.BL_Cf()

# ind = 900
# t3.Plot_U_plus(ind)
# t3.Plot_ReynoldsStress_plus(ind)
# t3.Plot_CfRed2()
# t3.Plot_CfRex()
t3.Plot_TKE(0)
# t3.Plot_BL()
plt.show()

