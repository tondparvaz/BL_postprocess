# __________________________________________________________________________________________________ 
# Boundary layer flows - post processing 
# Armin Kianfar - 09/2024
# University of Colorado Boulder
# __________________________________________________________________________________________________   
import numpy as np
import matplotlib.pyplot as plt
import os
import Integration as integ
import Diffrentiation as diff

class BL_PPAnalysis:
    def __init__(self, dat, nu):
        self.dat = dat # directory of the data
        self.nu = nu # kinematic viscosity of the flow
        # self.FSI = FSI # FSI coupling flag: 0>uncoupled; 1>coupled
        # self.xs = xs # leading edge of the control region
        # self.xe = xe # trailing edge of the control region
#___________________________________________________________________________________________________
    # > This method load the data grid points in the streamwise, x, and wall-normal, y, directions.
    def LoadGrids(self):
        griddir = self.dat + '/grids.txt'
        grid = np.loadtxt(griddir)
        self.x = grid[:1266]
        self.nx = len(self.x)
        self.y = grid[1266:]
        self.ny = len(self.y)
        print(' > Grid points, x and y, are loaded.')
#___________________________________________________________________________________________________
    # > This method load the required statistics from the file
    # outputs: (U, V, W) mean velocity
    #          (uu, vv, ww, uv) Reynolds stress
    #          P mean pressure
    def LoadStats(self):
        statdir = self.dat + '/stats.txt'
        stats = np.loadtxt(statdir)
        # Mean velocity
        self.U = stats[0:self.nx, :]
        self.V = stats[self.nx:self.nx*2, :]
        self.W = stats[self.nx*2:self.nx*3, :]
        # Reynolds stress
        self.u2 = stats[self.nx*3:self.nx*4, :] - self.U**2
        self.v2 = stats[self.nx*4:self.nx*5, :] - self.V**2
        self.w2 = stats[self.nx*5:self.nx*6, :] - self.W**2
        self.uv = stats[self.nx*6:self.nx*7, :] - self.U*self.V
        # Mean pressure
        self.P = stats[self.nx*7:self.nx*8, :]
        print(' > Flow statistics are loaded.')
    
    def LoadStats2(self):
        statdir = self.dat + '/stats.txt'
        stats = np.loadtxt(statdir)
        # Mean velocity
        self.U = stats[0:self.nx, :]
        self.V = stats[self.nx:self.nx*2, :]
        self.W = stats[self.nx*2:self.nx*3, :]
        # Reynolds stress
        self.u2 = stats[self.nx*2:self.nx*3, :] - self.U**2
        self.v2 = stats[self.nx*3:self.nx*4, :] - self.V**2
        self.w2 = stats[self.nx*5:self.nx*6, :] - self.W**2
        self.uv = stats[self.nx*4:self.nx*5, :] - self.U*self.V
        # Mean pressure
        # self.P = stats[self.nx*7:self.nx*8, :]
        # print(' > Flow statistics are loaded.')
#___________________________________________________________________________________________________
# > This method detect the edge of the boundary layer
    # outputs: d99 BL thickness based on 99%
    #          Uiw locally re-constructed inviscid vel at the wall
    #          Ue velocity at the edge
    #          Pe pressure at the edge
    def BL_edge(self):
        P0 = self.P + 0.5*(self.U**2 + self.V**2) # total pressure
        Ui = np.sqrt(2*(np.amax(P0, axis = 1)[:, np.newaxis]
                    - self.P) - self.V**2) # reconstucted inviscid velocity
        self.Uiw = Ui[:, 0] # inviscid velocity at the wall
        # Pre-allocation 
        self.d99 = np.empty(self.nx)
        self.Ue = np.empty(self.nx)
        self.Pe = np.empty(self.nx)
        for i in range(0, self.nx):
            ind2 = np.argmax(self.U[i, :]/Ui[i, :] > 0.99) # 0.99 corresponds to 99% BL thickness
            ind1 = ind2 - 1
            shift = self.y[ind2] - self.y[ind1]
            f1 = self.U[i, ind1]/Ui[i, ind1]
            f2 = self.U[i, ind2]/Ui[i, ind2]
            # delta_99 BL thickness
            self.d99[i] =  self.y[ind1] + shift*((0.99-f1)/(f2-f1))
           # Edge quantities at delta_99
            self.Ue[i] = np.interp(self.d99[i], self.y, self.U[i, :])
            self.Pe[i] = np.interp(self.d99[i], self.y, self.P[i, :]) 
#___________________________________________________________________________________________________
# > This method computes the BL thickness definitions
    # outputs: d1  displacement thickness
    #          d2  momentum thickness
    #          d2v wall-normal momentum thickness
    def BL_Thickness(self):
        Dd1  = 1.0 - self.U/self.Uiw[:, np.newaxis]
        Dd2  = self.U*(1.0 - self.U)/self.Uiw[:, np.newaxis]**2
        Dd2v = self.V*(1.0 - self.U)/self.Uiw[:, np.newaxis]**2
        # numerical integration in the wall-normal direction
        self.d1  = integ.MP_Integral(self.y, self.d99, Dd1, 1.5)
        self.d2  = integ.MP_Integral(self.y, self.d99, Dd2, 1.5)
        self.d2v = integ.MP_Integral(self.y, self.d99, Dd2v, 1.5)
#___________________________________________________________________________________________________
# > This method computes Reynolds number
    # inputs:  Uscl velocity scale
    # outputs: Rex  
    #          Red1  
    #          Red2
    def BL_Reynolds(self, Uscl):
        self.Rex = self.x*Uscl/self.nu
        self.Red1 = self.d1*Uscl/self.nu
        self.Red2 = self.d2*Uscl/self.nu        
#___________________________________________________________________________________________________
# > This method computes Cf and friction scales
    # outputs: Cf
    #          utau friction velocity 
    #          dnu viscous length scale
    #          yp wall-normal in plus units
    #          up velocity in length scale
    def BL_Cf(self):
        self.Cf = self.nu*(self.U[:, 1] - self.U[:, 0])/(self.y[1]-self.y[0])
        self.utau = np.sqrt(self.Cf)
        self.dnu = self.nu/self.utau
#___________________________________________________________________________________________________
# > This method computes the plus units
    # input: ind index of the streamwise position
    # outputs: yp wall-normal direction in plus units
    #          up vel in plus units
    def WallUnits(self, ind):
        yp = self.y/self.dnu[ind]
        up = self.U[ind, :]/self.utau[ind]
        u2p = self.u2[ind, :]/self.utau[ind]**2
        v2p = self.v2[ind, :]/self.utau[ind]**2
        w2p = self.w2[ind, :]/self.utau[ind]**2
        uvp = self.uv[ind, :]/self.utau[ind]**2
        return yp, up, u2p, v2p, w2p, uvp
#___________________________________________________________________________________________________
# > This method computes the mean velocity in the viscous sublayer
    # outputs: yp_vsl
    #          up_vsl
    def ViscousSL(self, ind):
        yp, _, _, _, _, _ = self.WallUnits(ind)
        lim1 = np.argmax(yp>12.)  
        yvsl = yp[:lim1]
        uvsl = yvsl
        return yvsl, uvsl         
#___________________________________________________________________________________________________
# > This method computes the mean velocity in the log layer
    # outputs: yp_ll
    #          up_ll
    def LogLaw(self, ind):
        kap = 0.41
        B = 5.2
        yp, _, _, _, _, _ = self.WallUnits(ind)
        lim1 = np.argmax(yp>30.) 
        lim2 = np.argmax(yp>800.)
        yll = yp[lim1:lim2]
        ull = 1/kap*np.log(yll) + B
        return yll, ull  
#___________________________________________________________________________________________________
# > This method computes turbulnet kinetic energy
    # outputs: TKE      
    def BL_TKE(self):
        self.TKE = 0.5*(self.u2 + self.v2 + self.w2)
        self.ITKE = integ.MP_Integral(self.y, self.d99, self.TKE, 1.5)
#___________________________________________________________________________________________________
# > This method detects the limits of transition and turbulence
    # inputs:  Reynolds number based on x
    # outputs: ind_LT: index transition begins
    #          ind_TT: index turbulence begins  
    def BL_IndexDetect(self):
        self.BL_Reynolds(self.Ue)
        ind_LT = np.argmax(self.Rex >= 4.5*1e5)
        ind_TT = np.argmax(self.Rex >= 6.7*1e5)
        return ind_LT, ind_TT
#___________________________________________________________________________________________________
#                                       Plotting
# > This method plots the mean streamwise velocity in wall units    
    def Plot_U_plus(self, ind, save):
        yp, up, _, _, _, _ = self.WallUnits(ind)
        yvsl, uvsl = self.ViscousSL(ind)
        yll, ull = self.LogLaw(ind)
        plt.figure(101, (4.5, 3))
        plt.semilogx(yp, up, 'k')
        plt.semilogx(yvsl, uvsl, '--b', linewidth = 0.5)
        plt.semilogx(yll, ull, '--b', linewidth = 0.5)
        plt.ylim(0, 23)
        plt.xlim(1e-1, 1e3)
        plt.xlabel(r'$y^+$', fontsize=10)
        plt.ylabel(r'$U^+$', fontsize=10)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        if save == 1:
            plt.savefig(self.dat+'/Uplus.png', dpi=300, bbox_inches='tight', format='png')
# > This method plots the Reynolds stress components 
    def Plot_ReynoldsStress_plus(self, ind, save):
        yp, _, u2p, v2p, w2p, uvp = self.WallUnits(ind)
        plt.figure(102, (4.5, 3))
        plt.plot(yp, u2p, 'k', label=r'$\overline{uv}^+$')  
        plt.plot(yp, v2p, '--k', label=r'$\overline{vv}^+$')  
        plt.plot(yp, w2p, '-.k', label=r'$\overline{ww}^+$')  
        plt.plot(yp, uvp, ':k', label=r'$\overline{uv}^+$')  
        plt.plot([0, 50], [0, 0], 'k', linewidth=0.25) 
        plt.legend(fontsize=10)
        plt.xlim(0, 50)
        plt.xlabel(r'$y^+$', fontsize=10)
        plt.ylabel(r'$\overline{u_i u_j}^+$', fontsize=10)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        if save == 1:
            plt.savefig(self.dat+'/uvplus.png', dpi=300, bbox_inches='tight', format='png')

# > This method plots Cf vs Red2
    def Plot_CfRed2(self, save):
        ind_LT, ind_TT= self.BL_IndexDetect()
        self.BL_Reynolds(self.Ue)
        self.BL_Cf()
        Cf_lam = 0.332/self.Rex**(0.5)
        # Cf_Turb = 0.029/self.Rex**(0.2)
        Cf_Turb = 0.013/self.Red2**(0.25)
        plt.figure(201, (4.5, 3))
        plt.plot(self.Red2, 1e3*self.Cf, 'k')
        plt.plot(self.Red2, 1e3*Cf_lam, '--b', linewidth = 0.5)
        plt.plot(self.Red2, 1e3*Cf_Turb, '--b', linewidth = 0.5)
        plt.plot([self.Red2[ind_LT], self.Red2[ind_LT]], [0, 2.8], ':k', linewidth = 0.5)
        plt.plot([self.Red2[ind_TT], self.Red2[ind_TT]], [0, 2.8], ':k', linewidth = 0.5)
        plt.xlim(self.Red2[0], 1120)
        plt.ylim(0, 2.8)
        plt.xlabel(r'$Re_{\delta_2}$', fontsize=10)
        plt.ylabel(r'$C_f/2 \: \left(\times 10^3\right)$', fontsize=10)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        if save == 1:
         plt.savefig(self.dat+'/CfRed2.png', dpi=300, bbox_inches='tight', format='png')
# > This method plots Cf vs Rex
    def Plot_CfRex(self, save):
        ind_LT, ind_TT= self.BL_IndexDetect()
        self.BL_Reynolds(self.Ue)
        self.BL_Cf()
        Cf_lam = 0.332/self.Rex**(0.5)
        # Cf_Turb = 0.029/self.Rex**(0.2)
        Cf_Turb = 0.013/self.Red2**(0.25)
        plt.figure(202, (4.5, 3))
        plt.plot(self.Rex*1e-5, 1e3*self.Cf, 'k')
        plt.plot(self.Rex*1e-5, 1e3*Cf_lam, '--b', linewidth = 0.5)
        plt.plot(self.Rex*1e-5, 1e3*Cf_Turb, '--b', linewidth = 0.5)
        plt.plot([self.Rex[ind_LT]*1e-5, self.Rex[ind_LT]*1e-5], [0, 2.8], ':k', linewidth = 0.5)
        plt.plot([self.Rex[ind_TT]*1e-5, self.Rex[ind_TT]*1e-5], [0, 2.8], ':k', linewidth = 0.5)
        plt.xlim(self.Rex[0]*1e-5, 7.7)
        plt.ylim(0, 2.8)
        plt.xlabel(r'$Re_{x} \: \left(\times 10^{-5}\right)$', fontsize=10)
        plt.ylabel(r'$C_f/2 \: \left(\times 10^3\right)$', fontsize=10)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        if save == 1:
            plt.savefig(self.dat+'/CfRex.png', dpi=300, bbox_inches='tight', format='png')
# > This method plots the wall-normal integral of TKE
    def Plot_TKE(self, save):
        self.BL_Reynolds(self.Ue)
        self.BL_TKE()
        self.BL_Cf()
        plt.figure(301, (4.5, 3))
        plt.plot(1e-5*self.Rex, 1e3*self.ITKE, 'k')
        plt.plot([1e-5*self.Rex[49], 1e-5*self.Rex[49]],[0, 0.25], 'r', linewidth=0.25)
        plt.plot([1e-5*self.Rex[59], 1e-5*self.Rex[59]],[0, 0.25], 'r', linewidth=0.25)
        plt.xlim(3.8, 8)
        plt.xlabel(r'$Re_{x} \: \left(\times 10^{-5}\right)$', fontsize=10)
        plt.ylabel(r'$\int_0^\delta k dy \: \left(\times 10^{3}\right)$', fontsize=10)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        if save == 1:
            plt.savefig(self.dat+'/ITKE.png', dpi=300, bbox_inches='tight', format='png')
        
        plt.figure(302, (1.7, 1.5))
        plt.plot(1e-5*self.Rex, 1e3*self.ITKE, 'k')
        plt.plot([1e-5*self.Rex[49], 1e-5*self.Rex[49]],[0.0005, 0.001], 'r', linewidth=0.25)
        plt.plot([1e-5*self.Rex[59], 1e-5*self.Rex[59]],[0.0005, 0.001], 'r', linewidth=0.25)
        plt.xlim(3.9, 4.1)
        plt.ylim(0.0005, 0.001)
        plt.xticks(fontsize = 8) 
        plt.yticks(fontsize = 8)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        if save == 1:
            plt.savefig(self.dat+'/ITKE_inset.png', dpi=300, bbox_inches='tight', format='png')
        
# > This method plots the boundary layer thicknesses
    def Plot_BL(self, save):
        plt.figure(301, (9, 3))
        plt.plot(1e-5*self.Rex, self.d99, 'k', label=r'$\delta_{99}$') 
        plt.plot(1e-5*self.Rex, self.d1, '-.k', label=r'$\delta_{1}$') 
        plt.plot(1e-5*self.Rex, self.d2, '--k', label=r'$\delta_{2}$') 
        plt.xlim(3.8, 8)
        plt.xlabel(r'$Re_{x} \: \left(\times 10^{-5}\right)$', fontsize=10)
        plt.ylabel(r'$y$', fontsize=10)
        plt.grid(color='gray', linestyle=':', linewidth=0.1)
        plt.legend(fontsize=10)
        if save == 1:    
            plt.savefig(self.dat+'/BL.png', dpi=300, bbox_inches='tight', format='png')
        