#####################################################################
## Filename:	iXAtom_Class_Systematics.py
## Author:		B. Barrett
## Description: Systematics class definition for iXAtom analysis package
##				Contains all AI systematics attributes and methods
## Version:		3.2.4
## Last Mod:	09/07/2020
##===================================================================
## Change Log:
## 19/01/2020 - Systematics class defined				
##            - Adding methods for computing systematic effects on
##				atom interferometers due to the one-photon light shift,
##				two-photon light shift, and the magnetic gradient shift.
## 20/04/2020 - Moved systematics class to a dedicated file.
## 03/09/2020 - Created new version (v2.3.4) for general updates to
##				systematics class.
##			  - Added calculations of phase shifts due to Parasitic lines
##				(phase modulator case), wavefront distortion (simple case)
##				and Mach-Zehnder Asymmetry. The latter should be checked against
##				the sensitivity function approx of ONERA.
##			  - Updated calculation of TPLS. The phase uncertainty of the exact
##				formula still needs to be checked.
##			  - Updated plotting function based on ICE systematics class (v1.1)
##				This still needs to be debugged when using variables other
##				than TOF and T.
#####################################################################

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy.interpolate import interp1d
from scipy             import integrate

import iXAtom_Class_Physics as iXC_Phys
import iXAtom_Utilities     as iXUtils

class Systematics(iXC_Phys.Physics):
	#################################################################
	## Class for computing atom interferometer systematic effects
	## Inherits all attributes and methods from class: Physics
	#################################################################

	def __init__(self, iax, RunPars, SysPars):
		"""Get parameters needed to compute systematics. Some parameters are contained within RunPars,
		while others are input using the dictionary SysPars. Each instance of the Systematics class
		is specific to a given Raman axis, hence iax is required as input.
		ARGUMENTS:
		\t iax     (int)  - Index corresponding to specific Raman axis (0,1,2: X,Y,Z)
		\t RunPars (obj)  - Instance of RunParameters class
		\t SysPars (dict) - Systematic model parameters for specified Raman axis
		"""

		super().__init__(RunPars)

		## Set attributes according to those present in SysPars
		for key, val in SysPars.items():
			setattr(self, key, val)

		## Set derived AI parameters
		self.SetAIParameters()

	################# End of Systematics.__init__() #################
	#################################################################

	def SetAIParameters(self):
		"""Set derived three-pulse atom interferometer parameters."""

		## AI geometry
		self.tau1   = self.taupio2									## 1st pulse lengths (s)
		self.tau2   = self.taupi									## 2nd pulse lengths (s)
		self.tau3   = self.taupio2									## 3rd pulse lengths (s)
		self.t1     = self.TOF 										## 1st pulse times (s)
		self.t2     = self.t1 + self.T + self.tau1 + 0.5*self.tau2	## 2nd pulse times (s)
		self.t3     = self.t2 + self.T + 0.5*self.tau2 + self.tau3	## 3rd pulse times (s)
		self.Ttotal = self.t3 - self.t1								## Total interrogation times (s)
		self.Teff   = np.sqrt((self.T + self.tau2)*(self.T + \
			(2./self.Omegaeff)*np.tan(self.Omegaeff*self.tau1/2.))) ## Effective interrogation times (s)
		self.Seff   = self.keff*self.Teff**2						## Effective scale factor (rad/m/s^2)
		# self.vSel   = np.array([0.5*(self.deltakU - self.deltakD)/self.keff, \
		# 	0.1*self.Omegaeff/self.keff])	 						## Selected velocity (ignoring TPLS) (m/s)
		# self.vSel   = np.array([self.TOF*self.gLocal, 
		# 	0.1*self.Omegaeff/self.keff])							## Selected velocity (assuming zero initial) (m/s)
		self.sigvD  = np.sqrt(2.*self.kBoltz*self.Temp/self.MRb) 	## Doppler velocity standard deviation (m/s)

		## AI geometry
		self.TiltZ = 45.*(np.pi/180.)								## Angle between the Z-axis and the vertical [rad]

		# self.v0x[:,1] = self.sigvD[:,0]
		# self.v0y[:,1] = self.sigvD[:,0]

	############## End of Systematics.SetAIParameters() #############
	#################################################################

	def COMTrajectory(self, iax, ks, t):
		"""Compute atomic center-of-mass position and velocity at time t during a three-pulse interferometer.
		The expression for the COM position is:
		  z(t <  t1) = z0 + v0*t + 0.5*g*t^2 
		  z(t >= t1) = z0 + v0*t + 0.5*ks*vR*(t-t1) + 0.5*g*t^2 
		Note the time is the true time-of-flight (relative to molasses release).
		Set ks = 0 to obtain the undiffracted COM trajectory.
		"""

		if t > self.t1[iax]:
			## COM position (m)
			self.zCOM = self.z0[iax,0] + self.v0[iax,0]*t + 0.5*ks*self.vR*(t - self.t1[iax]) + 0.5*self.a[iax,0]*t**2
			## COM velocity (m/s)
			self.vCOM = self.v0[iax,0] + 0.5*ks*self.vR + self.a[iax,0]*t
		else:
			## COM position (m)
			self.zCOM = self.z0[iax,0] + self.v0[iax,0]*t + 0.5*self.a[iax,0]*t**2
			self.vCOM = self.v0[iax,0] + self.a[iax,0]*t

		## COM position uncertainty (m)
		self.dzCOM = np.sqrt(self.z0[iax,1]**2 + (self.v0[iax,1]*t)**2)
		## COM velocity uncertainty (m/s)
		self.dvCOM = self.v0[iax,1]

	############### End of Systematics.COMTrajectory() ##############
	#################################################################

	def OnePhotonLightShift(self, iax, ks):
		"""Compute the one-photon light shift on a three-pulse atom interferometer.
		The one-photon light shift varies linearly with power and quasi-linearly with time-of-flight
		due to cloud expansion in the beam. The one-photon frequency shift can be modeled as:
			fOPLS(P,t) = d(fOPLS)/dt*t + d(fOPLS)/dP*P
		where P is the power in the beam, and t is the time-of-flight.
		"""

		## Frequency shifts
		# self.omegaOPLS1  = 2*np.pi*(self.f0_OPLS[iax,0] + self.dfdt_OPLS[iax,0]*self.t1[iax] + self.dfdP_OPLS[iax,0]*self.P[iax,0])
		# self.omegaOPLS3  = 2*np.pi*(self.f0_OPLS[iax,0] + self.dfdt_OPLS[iax,0]*self.t3[iax] + self.dfdP_OPLS[iax,0]*self.P[iax,0])
		self.omegaOPLS1  = 0.5*(self.deltakU + self.deltakD) - self.omegaHF - self.omegaR
		self.omegaOPLS3  = 0.5*(self.deltakU + self.deltakD) - self.omegaHF - self.omegaR

		## Uncertainty in frequency shifts
		# self.domegaOPLS1 = 2*np.pi*np.sqrt(self.f0_OPLS[iax,1]**2 + (self.dfdt_OPLS[iax,1]*self.t1[iax])**2 + (self.dfdP_OPLS[iax,1]*self.P[iax,0])**2)
		# self.domegaOPLS3 = 2*np.pi*np.sqrt(self.f0_OPLS[iax,1]**2 + (self.dfdt_OPLS[iax,1]*self.t3[iax])**2 + (self.dfdP_OPLS[iax,1]*self.P[iax,0])**2)

		self.domegaOPLS1 = 2.*np.pi*1.E3
		self.domegaOPLS3 = 2.*np.pi*1.E3

		## Phase shift and uncertainty
		self.pOPLS   = \
			self.omegaOPLS1/self.Rabi1[iax,0]*np.tan(self.Rabi1[iax,0]*self.tau1[iax]/2) - \
			self.omegaOPLS3/self.Rabi3[iax,0]*np.tan(self.Rabi3[iax,0]*self.tau3[iax]/2)
		self.dpOPLS  = np.sqrt(
			(self.domegaOPLS1/self.Rabi1[iax,0]*np.tan(self.Rabi1[iax,0]*self.tau1[iax]/2))**2 + \
			(self.domegaOPLS3/self.Rabi3[iax,0]*np.tan(self.Rabi3[iax,0]*self.tau3[iax]/2))**2)

	########### End of Systematics.OnePhotonLightShift() ############
	#################################################################

	def TwoPhotonLightShift(self, iax, ks, Form='Approx'):
		"""Compute the two-photon light shift on a three-pulse atom interferometer.
		The two-photon light shift varies linearly with power and inversely with time-of-flight
		due to the Doppler shift. In the limit where the Doppler shift is much larger than the Rabi
		frequency and the recoil frequency, the two-photon frequency shift is given by: 
			fTPLS(P,t) = -Omegaeff(P)**2/(4*omegaD(t)),
		where omegaD(t) = keff*(v0 + g*t) is the Doppler shift.
		"""

		self.COMTrajectory(iax, ks, self.t1[iax])
		omegaD1  		= ks*self.keff*self.vCOM			## Doppler shift at t = t1
		domegaD1 		= self.keff*self.dvCOM 				## Uncertainty in omegaD1
		omegaB1  		= 4.*np.pi*self.alphaB*(self.beta0[iax,0] + self.beta1[iax,0]*self.zCOM) ## Zeeman shift (Delta mF = 2) at t = t1

		delta1_kU_mF0  	= 2.*omegaD1						## +k counter-propagating Delta mF =  0 transition at t1
		delta1_kU_mFp2 	= delta1_kU_mF0 + 2.*omegaB1		## +k counter-propagating Delta mF = +2 transition at t1
		delta1_kU_mFm2 	= delta1_kU_mF0 - 2.*omegaB1		## +k counter-propagating Delta mF = -2 transition at t1

		delta1_kD_mF0  	= 2.*(omegaD1 + 2.*self.omegaR)		## -k counter-propagating Delta mF =  0 transition at t1
		delta1_kD_mFp2 	= delta1_kD_mF0 + 2.*omegaB1 		## -k counter-propagating Delta mF = +2 transition at t1
		delta1_kD_mFm2 	= delta1_kD_mF0 - 2.*omegaB1		## -k counter-propagating Delta mF = -2 transition at t1

		delta1_kCo_mF0 	= omegaD1 + self.omegaR 			## Co-proprogating Delta mF =  0 transition at t1
		delta1_kCo_mFm2	= delta1_kCo_mF0 - omegaB1 			## Co-proprogating Delta mF = +2 transition at t1
		delta1_kCo_mFp2	= delta1_kCo_mF0 + omegaB1 			## Co-proprogating Delta mF = -2 transition at t1

		Omega1_kU_mF0	= self.Rabi1[iax,0]
		dOmega1_kU_mF0 	= self.Rabi1[iax,1]
		Omega1_kU_mF2  	= self.Rabi1_mF2[iax,0]
		dOmega1_kU_mF2  = self.Rabi1_mF2[iax,1]
		Omega1_kCo_mF0 	= self.Rabi1_mF0[iax,0]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))
		dOmega1_kCo_mF0	= self.Rabi1_mF0[iax,1]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))
		Omega1_kCo_mF2 	= self.Rabi1_mF2[iax,0]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))
		dOmega1_kCo_mF2	= self.Rabi1_mF2[iax,1]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))

		self.COMTrajectory(iax, ks, self.t3[iax])
		omegaD3  		= ks*self.keff*self.vCOM 			## Doppler shift at t = t3
		domegaD3 		= self.keff*self.dvCOM 				## Uncertainty in omegaD3
		omegaB3  		= 4.*np.pi*self.alphaB*(self.beta0[iax,0] + self.beta1[iax,0]*self.zCOM) ## Zeeman shift (Delta mF = 2) at t = t3

		delta3_kU_mF0  	= 2.*omegaD3						## +k counter-propagating Delta mF =  0 transition at t3
		delta3_kU_mFp2 	= delta3_kU_mF0 + 2.*omegaB3		## +k counter-propagating Delta mF = +2 transition at t3
		delta3_kU_mFm2 	= delta3_kU_mF0 - 2.*omegaB3		## +k counter-propagating Delta mF = -2 transition at t3

		delta3_kD_mF0  	= 2.*(omegaD3 + 2.*self.omegaR)		## -k counter-propagating Delta mF =  0 transition at t3
		delta3_kD_mFp2 	= delta3_kD_mF0 + 2.*omegaB3 		## -k counter-propagating Delta mF = +2 transition at t3
		delta3_kD_mFm2 	= delta3_kD_mF0 - 2.*omegaB3		## -k counter-propagating Delta mF = -2 transition at t3

		delta3_kCo_mF0 	= omegaD3 + self.omegaR 			## Co-proprogating Delta mF =  0 transition at t3
		delta3_kCo_mFm2	= delta3_kCo_mF0 - omegaB3			## Co-proprogating Delta mF = +2 transition at t3
		delta3_kCo_mFp2	= delta3_kCo_mF0 + omegaB3 			## Co-proprogating Delta mF = -2 transition at t3

		# delta3_kU_mF0   = 2.*omegaD3					## +k counter-propagating transition at t3
		# delta3_kD_mF0   = 2.*omegaD3 + 4.*self.omegaR	## -k counter-propagating transition at t3
		# delta3_kCo_mF0  = omegaD3 + self.omegaR 		## Co-proprogating Delta mF=0 transition at t3
		# delta3_kCo_mFm2 = delta3_kCo_mF0 - 4.*np.pi*self.alphaB*self.beta0[iax,0] ## Co-proprogating Delta mF=2 transition at t3
		# delta3_kCo_mFp2 = delta3_kCo_mF0 + 4.*np.pi*self.alphaB*self.beta0[iax,0] ## Co-proprogating Delta mF=2 transition at t3

		Omega3_kU_mF0  	= self.Rabi3[iax,0]
		dOmega3_kU_mF0 	= self.Rabi3[iax,1]
		Omega3_kU_mF2  	= self.Rabi3_mF2[iax,0]
		dOmega3_kU_mF2  = self.Rabi3_mF2[iax,1]
		Omega3_kCo_mF0 	= self.Rabi3_mF0[iax,0]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))
		dOmega3_kCo_mF0	= self.Rabi3_mF0[iax,1]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))
		Omega3_kCo_mF2 	= self.Rabi3_mF2[iax,0]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))
		dOmega3_kCo_mF2	= self.Rabi3_mF2[iax,1]*np.cos(self.Deltak*(self.zM[iax,0] - self.zCOM))

		if Form == 'Exact':
			## Exact formula for frequency shift
			self.omegaTPLS1 = 0.5*( \
				(np.sign(delta1_kU_mF0 )*np.sqrt(Omega1_kU_mF0**2 + delta1_kU_mF0**2) - delta1_kU_mF0) + \
				(np.sign(delta1_kD_mF0 )*np.sqrt(Omega1_kU_mF0**2 + delta1_kD_mF0**2) - delta1_kD_mF0) + \
				(np.sign(delta1_kCo_mF0)*np.sqrt(Omega1_kCo_mF0**2 + delta1_kCo_mF0**2) - delta1_kCo_mF0) + \
				(np.sign(delta1_kCo_mFm2)*np.sqrt(Omega1_kCo_mF2**2 + delta1_kCo_mFm2**2) - delta1_kCo_mFm2) + \
				(np.sign(delta1_kCo_mFp2)*np.sqrt(Omega1_kCo_mF2**2 + delta1_kCo_mFp2**2) - delta1_kCo_mFp2))
			self.omegaTPLS3 = 0.5*( \
				(np.sign(delta3_kU_mF0 )*np.sqrt(Omega3_kU_mF0**2 + delta3_kU_mF0**2) - delta3_kU_mF0) + \
				(np.sign(delta3_kD_mF0 )*np.sqrt(Omega3_kU_mF0**2 + delta3_kD_mF0**2) - delta3_kD_mF0) + \
				(np.sign(delta3_kCo_mF0)*np.sqrt(Omega3_kCo_mF0**2 + delta3_kCo_mF0**2) - delta3_kCo_mF0) + \
				(np.sign(delta3_kCo_mFm2)*np.sqrt(Omega3_kCo_mF2**2 + delta3_kCo_mFm2**2) - delta3_kCo_mFm2) + \
				(np.sign(delta3_kCo_mFp2)*np.sqrt(Omega3_kCo_mF2**2 + delta3_kCo_mFp2**2) - delta3_kCo_mFp2))

			## Exact uncertainty in frequency shift
			self.domegaTPLS1 = 0.5*np.sqrt( \
				(Omega1_kU_mF0*(1./np.sqrt(Omega1_kU_mF0**2 + delta1_kU_mF0**2) + \
					1./np.sqrt(Omega1_kU_mF0**2 + delta1_kD_mF0**2))*dOmega1_kU_mF0)**2 + \
				(Omega1_kCo_mF0/np.sqrt(Omega1_kCo_mF0**2 + delta1_kCo_mF0**2)*dOmega1_kCo_mF0)**2 + \
				(Omega1_kCo_mF2*(1./np.sqrt(Omega1_kCo_mF2**2 + delta1_kCo_mFm2**2) + \
					1./np.sqrt(Omega1_kCo_mF2**2 + delta1_kCo_mFp2**2))*dOmega1_kCo_mF2)**2 + \
				((2.*(delta1_kU_mF0  /np.sqrt(Omega1_kU_mF0**2     + delta1_kU_mF0**2  ) - 1.) + \
				  2.*(delta1_kD_mF0  /np.sqrt(Omega1_kU_mF0**2     + delta1_kD_mF0**2  ) - 1.) + \
				     (delta1_kCo_mF0 /np.sqrt(Omega1_kCo_mF0**2 + delta1_kCo_mF0**2 ) - 1.) + \
				     (delta1_kCo_mFm2/np.sqrt(Omega1_kCo_mF2**2 + delta1_kCo_mFm2**2) - 1.) + \
				     (delta1_kCo_mFp2/np.sqrt(Omega1_kCo_mF2**2 + delta1_kCo_mFp2**2) - 1.))*domegaD1)**2)
			self.domegaTPLS3 = 0.5*np.sqrt( \
				(Omega3_kU_mF0*(1./np.sqrt(Omega3_kU_mF0**2 + delta3_kU_mF0**2) + \
					1./np.sqrt(Omega3_kU_mF0**2 + delta3_kD_mF0**2))*dOmega3_kU_mF0)**2 + \
				(Omega3_kCo_mF0/np.sqrt(Omega3_kCo_mF0**2 + delta3_kCo_mF0**2)*dOmega3_kCo_mF0)**2 + \
				(Omega3_kCo_mF2*(1./np.sqrt(Omega3_kCo_mF2**2 + delta3_kCo_mFm2**2) + \
					1./np.sqrt(Omega3_kCo_mF2**2 + delta3_kCo_mFp2**2))*dOmega3_kCo_mF2)**2 + \
				((2.*(delta3_kU_mF0  /np.sqrt(Omega3_kU_mF0**2     + delta3_kU_mF0**2  ) - 1.) + \
				  2.*(delta3_kD_mF0  /np.sqrt(Omega3_kU_mF0**2     + delta3_kD_mF0**2  ) - 1.) + \
				     (delta3_kCo_mF0 /np.sqrt(Omega3_kCo_mF0**2 + delta3_kCo_mF0**2 ) - 1.) + \
				     (delta3_kCo_mFm2/np.sqrt(Omega3_kCo_mF2**2 + delta3_kCo_mFm2**2) - 1.) + \
				     (delta3_kCo_mFp2/np.sqrt(Omega3_kCo_mF2**2 + delta3_kCo_mFp2**2) - 1.))*domegaD3)**2)
		else:
			## Approximate formula for frequency shift when delta11, delta12 >> Rabi1, delta13 >> RabiCo1,
			self.omegaTPLS1 = 0.25*( \
				Omega1_kU_mF0**2*(1./delta1_kU_mF0 + 1./delta1_kD_mF0) + \
				Omega1_kU_mF2**2*(1./delta1_kU_mFp2 + 1./delta1_kU_mFm2 + 1./delta1_kD_mFp2 + 1./delta1_kD_mFm2) + \
				Omega1_kCo_mF2**2*(1./delta1_kCo_mFm2 + 1./delta1_kCo_mFp2) + \
				Omega1_kCo_mF0**2/delta1_kCo_mF0)
			self.omegaTPLS3 = 0.25*( \
				Omega3_kU_mF0**2*(1./delta3_kU_mF0 + 1./delta3_kD_mF0) + \
				Omega3_kU_mF2**2*(1./delta3_kU_mFp2 + 1./delta3_kU_mFm2 + 1./delta3_kD_mFp2 + 1./delta3_kD_mFm2) + \
				Omega3_kCo_mF2**2*(1./delta3_kCo_mFm2 + 1./delta3_kCo_mFp2) + \
				Omega3_kCo_mF0**2/delta3_kCo_mF0)

			## Approximate uncertainty in frequency shift
			self.domegaTPLS1 = 0.25*np.sqrt( \
				(2.*Omega1_kU_mF0*dOmega1_kU_mF0*(1./delta1_kU_mF0 + 1./delta1_kD_mF0))**2 + (2.*Omega1_kCo_mF0*dOmega1_kCo_mF0/delta1_kCo_mF0)**2 + \
				(2.*Omega1_kCo_mF2*dOmega1_kCo_mF2*(1./delta1_kCo_mFm2 + 1./delta1_kCo_mFp2))**2 + (Omega1_kU_mF0**2/delta1_kU_mF0**2*(2.*domegaD1))**2 + \
				(Omega1_kU_mF0**2/delta1_kD_mF0**2*(2.*domegaD1))**2 + (Omega1_kCo_mF0**2/delta1_kCo_mF0**2*domegaD1)**2 + \
				(Omega1_kCo_mF2**2/delta1_kCo_mFm2**2*domegaD1)**2 + (Omega1_kCo_mF2**2/delta1_kCo_mFp2**2*domegaD1)**2)
			self.domegaTPLS3 = 0.25*np.sqrt( \
				(2.*Omega3_kU_mF0*dOmega3_kU_mF0*(1./delta3_kU_mF0 + 1./delta3_kD_mF0))**2 + (2.*Omega3_kCo_mF0*dOmega3_kCo_mF0/delta3_kCo_mF0)**2 + \
				(2.*Omega3_kCo_mF2*dOmega3_kCo_mF2*(1./delta3_kCo_mFm2 + 1./delta3_kCo_mFp2))**2 + (Omega3_kU_mF0**2/delta3_kU_mF0**2*(2.*domegaD3))**2 + \
				(Omega3_kU_mF0**2/delta3_kD_mF0**2*(2.*domegaD3))**2 + (Omega3_kCo_mF0**2/delta3_kCo_mF0**2*domegaD3)**2 + \
				(Omega3_kCo_mF2**2/delta3_kCo_mFm2**2*domegaD3)**2 + (Omega3_kCo_mF2**2/delta3_kCo_mFp2**2*domegaD3)**2)

		## Phase shift and uncertainty
		self.pTPLS  = self.omegaTPLS1/Omega1_kU_mF0*np.tan(Omega1_kU_mF0*self.tau1[iax]/2.) - self.omegaTPLS3/Omega3_kU_mF0*np.tan(Omega3_kU_mF0*self.tau3[iax]/2.)
		# self.dpTPLS = np.sqrt( \
		# 	(self.domegaTPLS1/Omega1_kU_mF0*np.tan(Omega1_kU_mF0*self.tau1[iax]/2.))**2 + (self.domegaTPLS3/Omega3_kU_mF0*np.tan(Omega3_kU_mF0*self.tau3[iax]/2.))**2)
		self.dpTPLS = 0.25*np.sqrt( \
			((self.omegaTPLS1/Omega1_kU_mF0/np.cos(Omega1_kU_mF0*self.tau1[iax]/2.)**2*self.tau1[iax]/2. - \
			(Omega1_kCo_mF0**2/delta1_kCo_mF0 + Omega1_kCo_mF2**2/delta1_kCo_mFm2 + Omega1_kCo_mF2**2/delta1_kCo_mFp2)/Omega1_kU_mF0**2)*dOmega1_kU_mF0)**2 + \
			((2.*Omega1_kU_mF0**2*(1./delta1_kU_mF0**2 + 1./delta1_kD_mF0**2) + Omega1_kCo_mF0**2/delta1_kCo_mF0**2 + \
			Omega1_kCo_mF2**2*(1./delta1_kCo_mFm2**2 + 1./delta1_kCo_mFp2**2))*domegaD1/Omega1_kU_mF0*np.tan(Omega1_kU_mF0*self.tau1[iax]/2.))**2 + \
			((self.omegaTPLS3/Omega3_kU_mF0/np.cos(Omega3_kU_mF0*self.tau3[iax]/2.)**2*self.tau3[iax]/2. - \
			(Omega3_kCo_mF0**2/delta3_kCo_mF0 + Omega3_kCo_mF2**2/delta3_kCo_mFm2 + Omega3_kCo_mF2**2/delta3_kCo_mFp2)/Omega3_kU_mF0**2)*dOmega3_kU_mF0)**2 + \
			((2.*Omega3_kU_mF0**2*(1./delta3_kU_mF0**2 + 1./delta3_kD_mF0**2) + Omega3_kCo_mF0**2/delta3_kCo_mF0**2 + \
			Omega3_kCo_mF2**2*(1./delta3_kCo_mFm2**2 + 1./delta3_kCo_mFp2**2))*domegaD3/Omega3_kU_mF0*np.tan(Omega3_kU_mF0*self.tau3[iax]/2.))**2)

	########### End of Systematics.TwoPhotonLightShift() ############
	#################################################################

	def BModel(self, iax, ks, t):
		"""Model for the magnetic field containing a spatial gradient + curvature: 
			B(t) = beta0 + beta1*zCOM + beta2*zCOM**2,
		where zCOM is the center-of-mass position of the atoms at time t (the true time-of-flight relative
		to molasses release). Set ks = 0 to obtain the undiffracted trajectory.
		"""

		self.COMTrajectory(iax, ks, t)

		B  = self.beta0[iax,0] + self.beta1[iax,0]*self.zCOM + self.beta2[iax,0]*self.zCOM**2
		dB = np.sqrt(self.beta0[iax,1]**2 + (self.beta1[iax,1]*self.zCOM)**2 + (self.beta1[iax,0]*self.dvCOM*t)**2 + \
			(self.beta2[iax,1]*self.zCOM**2)**2 + (self.beta2[iax,0]*(2.*self.vCOM*self.dvCOM*t**2))**2)

		return np.array([B, dB])

	################## End of Systematics.BModel() ##################
	#################################################################

	def QuadraticZeemanShift(self, iax, ks):
		"""Analytical computation of the phase shift due to the 2nd-order Zeeman shift in the presence of
		a temporally and spatially-varying B-field according the following model:
			B(t)   = beta0 + beta1*(zk + vk*t + 0.5*a*t^2) + 0.5*beta2*(zk + vk*t + 0.5*a*t^2)^2 + BEddy*exp(-gEddy*t)
		In the absence of a curvature (beta2=0):
			B^2(t) = beta0^2
				   + 2*beta0*beta1*(zk + vk*t + 0.5*a*t^2)
				   + 2*beta0*BEddy*exp(-gEddy*t)
				   + 2*BEddy*beta1*(zk + vk*t + 0.5*a*t^2)*exp(-gEddy*t)
				   + beta1^2*(zk^2 + 2*zk*vk*t + zk*a*t^2 + vk^2*t^2 + vk*a*t^3 + 0.25*a^2*t^4)
				   + BEddy^2*exp(-2*gEddy*t)
		where t is the true time-of-flight relative to molasses release.
		"""

		t1      = self.t1[iax]
		T       = self.T[iax]
		tau1    = self.tau1[iax]
		tau2    = self.tau2[iax]
		Omega1  = self.Rabi1[iax,0]

		K       = 2.*np.pi*self.KClock  		## Clock shift parameter (rad/s/G^2)
		B, dB   = self.BModel(iax, ks, t1)		## B-field at t = t1 (G)
		## Trajectory computed in BModel above
		zk      = self.zCOM						## COM position just after first pulse (m)
		vk      = self.vCOM + 0.5*ks*self.vR 	## COM velocity just after first pulse (m/s)
		dvk     = self.dvCOM					## COM velocity uncertainty (m/s)
		a       = self.a[iax,0]					## Local acceleration
		beta0   = abs(self.beta0[iax,0])		## B-field offset (G)
		dbeta0  = self.beta0[iax,1]
		beta1   = self.beta1[iax,0]				## B-field gradient (G/m)
		dbeta1  = self.beta1[iax,1]
		beta2   = self.beta2[iax,0]				## B-field curvature (G/m^2)
		dbeta2  = self.beta2[iax,1]

		G0, G1, G2, G3, G4 = self.gPolyIntegrals(t1, T, tau1, Omega1)

		ZT1     = vk*G1 + 0.5*a*G2
		dZT1    = dvk*G1
		ZT2     = 2.*zk*vk*G1 + vk**2*G2 + vk*a*G3 + 0.25*a**2*G4
		dZT2    = dvk*(2.*zk*G1 + 2.*vk*G2 + a*G3)

		## Shift due to constant parts of B^2
		pbeta0  = K*(beta0**2 + 2.*beta0*beta1*zk + beta1**2*zk**2)*G0
		dpbeta0 = K*np.sqrt((2.*dbeta0*beta0)**2 + (2.*dbeta0*beta1*zk)**2 + (2.*beta0*dbeta1*zk)**2 + (2.*dbeta1*beta1*zk**2)**2)*G0

		## Shift due to B-gradient
		pbeta1  = 2.*K*beta0*beta1*ZT1 + K*beta1**2*ZT2
		dpbeta1 = K*np.sqrt((2.*dbeta0*beta1*ZT1)**2 + (2.*dbeta1*(beta0*ZT1 + beta1*ZT2))**2) # + (2.*beta0*beta1*dZT1 + beta1**2*dZT2)**2)

		## Shift due to B-curvature
		pbeta2  = K*beta0*beta2*ZT2
		dpbeta2 = K*np.sqrt((dbeta0*beta2*ZT2)**2 + (beta0*dbeta2*ZT2)**2 + (beta0*beta2*dZT2)**2)

		## Total phase shift due to quadratic Zeeman effect
		self.pQZ  = pbeta0 + pbeta1 + pbeta2
		self.dpQZ = np.sqrt(dpbeta0**2 + dpbeta1**2 + dpbeta2**2)
		## Total frequency shift due to quadratic Zeeman effect
		self.omegaQZ  = K*B**2
		self.domegaQZ = 2.*K*abs(B)*dB

	########### End of Systematics.QuadraticZeemanShift() ###########
	#################################################################

	def MagneticForceShift(self, iax, ks):
		"""Compute the phase shift due to the magnetic force from a spatially-varying B-field
		with a gradient (beta1).
		"""

		self.COMTrajectory(iax, ks, self.t1[iax])

		## Shift due to the force of the magnetic gradient on the clock state
		self.pBF  = -(2./3.)*ks*self.keff*self.Lambda*self.beta1[iax,0]**2*(self.vCOM*self.T[iax] + self.a[iax,0]*self.T[iax]**2)*self.T[iax]**2
		self.dpBF =  (2./3.)*self.keff*self.Lambda*np.sqrt((2.*self.beta1[iax,1]*self.beta1[iax,0]*(self.vCOM*self.T[iax] + self.a[iax,0]*self.T[iax]**2))**2 \
			+ (self.beta1[iax,0]**2*self.dvCOM*self.T[iax])**2)*self.T[iax]**2

	############ End of Systematics.MagneticForceShift() ############
	#################################################################

	def GravityGradientShift(self, iax, ks):
		"""Compute the gravity gradient shift on a three-pulse atom interferometer."""

		self.COMTrajectory(iax, ks, self.t1[iax])

		## Shift due to the gravity gradient
		self.pGG  = ks*self.keff*self.Tii[iax,0]*(self.vCOM*self.T[iax] + (7/12)*self.a[iax,0]*self.T[iax]**2)*self.T[iax]**2
		self.dpGG = np.sqrt((self.Tii[iax,1]/self.Tii[iax,0]*self.pGG)**2 + (self.keff*self.Tii[iax,0]*self.dvCOM*self.T[iax]**3)**2)

	########### End of Systematics.GravityGradientShift() ###########
	#################################################################

	def CoriolisShift(self, iax, ks):
		"""Compute the Coriolis shift on a three-pulse atom interferometer."""

		self.COMTrajectory(iax, ks, self.t1[iax])

		k       = np.zeros(3)
		v       = np.array(self.v0[:,0] + self.a[:,0]*self.T[iax])
		dv      = np.array(self.v0[:,1] + self.a[:,1]*self.T[iax])

		k[iax]  = self.keff
		v[iax]  = self.vCOM + self.a[iax,0]*self.T[iax]
		dv[iax] = self.dvCOM

		kXv     = np.cross(k, v)
		dkXv    = np.cross(k, dv)

		## Shift due to the Coriolis effect
		self.pCor  = -2.*ks*np.dot(kXv, self.Omega[:,0])*self.T[iax]**2
		self.dpCor =  2.*np.sqrt(np.dot(dkXv**2, self.Omega[:,0]**2) + np.dot(kXv**2, self.Omega[:,1]**2))*self.T[iax]**2

	############## End of Systematics.CoriolisShift() ###############
	#################################################################

	def ParasiticLinesShift(self, iax, ks):
		"""Compute the phase shift due to parasitic laser lines."""

		Delta2 = self.Delta + 2.*np.pi*266.65*1.E+6			## Detuning from F = 2 -> 2' (rad/s)
		Delta1 = self.Delta + 2.*np.pi*424.60*1.E+6			## Detuning from F = 2 -> 1' (rad/s)
		Deltaw = self.deltakU if ks == 1. else self.deltakD
		Deltak = 2.*ks*Deltaw/self.cLight					## Change in wavevector (rad/m)

		mList  = [m for m in range(-2,3+1)]					## Parasitic line indices
		nm     = len(mList) - 1								## Number of pairs to consider
		im0    = int(nm/2)									## Center line index

		Em     = np.sqrt(self.IRel[:,0])					## Relative E-field magnitudes
		dEm    = 0.5*self.IRel[:,1]/self.IRel[:,0]			## Uncertainty in relative E-field magnitudes
		sList  = [(-1.)**m if m < 0 else 1 for m in mList]	## List of E-field signs
		Em     = sList*Em

		Omega  = np.zeros(nm)
		dOmega = np.zeros(nm)
		for im in range(nm):
			Omega[im]  = (Em[im+1]*Em[im])/(Em[im0+1]*Em[im0]) * \
				(1./(Delta2 + mList[im]*Deltaw) + 1./(3.*(Delta1 + mList[im]*Deltaw)))/(1./Delta2 + 1./(3.*Delta1)) * self.Rabi1[iax,0]
			dOmega[im] = np.sqrt((dEm[im+1]/Em[im+1])**2 + (dEm[im]/Em[im])**2)*abs(Omega[im])

		self.COMTrajectory(iax, ks, self.t1[iax])

		zA  = self.zCOM - self.zM[iax,0]
		zB  = zA + (self.vCOM + ks*self.vR)*(self.T[iax] + self.tau2[iax]) + 0.5*self.a[iax,0]*(self.T[iax] + self.tau2[iax])**2
		zC  = zA + self.vCOM*(self.T[iax] + self.tau2[iax]) + 0.5*self.a[iax,0]*(self.T[iax] + self.tau2[iax])**2
		zD  = zA + (self.vCOM + 0.5*ks*self.vR)*(self.Ttotal[iax]) + 0.5*self.a[iax,0]*(self.Ttotal[iax])**2

		eps  = 1.E-6
		phi  = lambda z: np.angle(np.sum([Omega[im]*np.exp(1j*mList[im]*Deltak*z) for im in range(nm)]))
		dphi = lambda z: np.angle(np.sum([(Omega[im]+dOmega[im])*np.exp(1j*mList[im]*Deltak*z) for im in range(nm)]))

		self.pPL  = phi(zA) - phi(zB) - phi(zC) + phi(zD)

		dzM       = self.zM[iax,1]
		dzvCOM    = self.dvCOM*(self.T[iax] + self.tau2[iax])
		dpzM      = abs(phi(zA + dzM) - phi(zB - dzM) - phi(zC - dzM) + phi(zD + dzM) - self.pPL)
		dpvCOM    = abs(phi(zA) - phi(zB - dzvCOM) - phi(zC - dzvCOM) + phi(zD + 2.*dzvCOM) - self.pPL)
		dpOmega   = abs(dphi(zA) - dphi(zB) - dphi(zC) + dphi(zD) - self.pPL)

		self.dpPL = 0.*np.sqrt(dpzM**2 + dpvCOM**2 + dpOmega**2)

	########### End of Systematics.ParasiticLinesShift() ############
	#################################################################

	def WavefrontDistortionShift(self, iax, ks):
		"""Compute the phase shift due to wavefront distortion."""

		## Mirror radius of curvature due to non-zero surface flatness (m)
		RM   = self.DM[0]**2/(2*self.FlatM[0])
		## Uncertainty in radius of curavature (m)
		dRM  = np.sqrt((2.*self.DM[1]/self.DM[0])**2 + (self.FlatM[1]/self.FlatM[0])**2)*RM

		sigv = np.sqrt(2.*self.kBoltz*self.Temp[iax,0]/self.MRb)
		self.pWD  = 0.5*ks*self.keff*sigv**2*self.Teff[iax]**2/RM
		self.dpWD = np.sqrt((self.Temp[iax,1]/self.Temp[iax,0])**2 + (dRM/RM)**2)*abs(self.pWD)

	######### End of Systematics.WavefrontDistortionShift() #########
	#################################################################

	def ContrastModel(self, delta, Omega1, Omega2, Omega3, tau1, tau2, tau3):
		"""Model for contrast of a Mach-Zehnder interferometer due to an asymmetry in Rabi frequencies."""

		omega1 = np.sqrt(delta**2 + Omega1**2)
		omega2 = np.sqrt(delta**2 + Omega2**2)
		omega3 = np.sqrt(delta**2 + Omega3**2)
		sin1   = np.sin(omega1*tau1/2.)
		sin2   = np.sin(omega2*tau2/2.)
		sin3   = np.sin(omega3*tau3/2.)
		cos1   = np.cos(omega1*tau1/2.)
		cos3   = np.cos(omega3*tau3/2.)

		return 4.*Omega1*Omega2**2*Omega3*sin1*sin2**2*sin3/(omega1*omega2*omega3)**2 * \
			(delta**2*sin1*sin3 + omega1*omega3*cos1*cos3)

	############### End of Systematics.ContrastModel() ##############
	#################################################################

	def PhaseModel(self, delta, Omega1, Omega3, tau1, tau3):
		"""Model for phase shift of a Mach-Zehnder interferometer due to an asymmetry in Rabi frequencies."""

		omega1 = np.sqrt(delta**2 + Omega1**2)
		omega3 = np.sqrt(delta**2 + Omega3**2)
		sin1   = np.sin(omega1*tau1/2.)
		sin3   = np.sin(omega3*tau3/2.)
		cos1   = np.cos(omega1*tau1/2.)
		cos3   = np.cos(omega3*tau3/2.)

		return np.angle(cos1 - (delta*sin1/omega1)*1j) + np.angle(cos3 + (delta*sin3/omega3)*1j)

	################ End of Systematics.PhaseModel() ################
	#################################################################

	def MachZehnderAsymmetryShift(self, iax, ks):
		"""Compute the phase shift due to the asymmetry of the Mach-Zehnder interferometer."""

		self.OnePhotonLightShift(iax, ks)
		self.TwoPhotonLightShift(iax, ks)
		self.COMTrajectory(iax, ks, self.t1[iax])

		if ks == 1:
			delta = self.deltakU
		elif ks == -1:
			delta = self.deltakD
		else:
			delta = self.deltaSel

		delta -= self.omegaHF + self.omegaR + self.omegaTPLS1 + self.omegaOPLS1
		Deltav = delta/(ks*self.keff) - self.vCOM

		# print(self.omegaOPLS1/(2*np.pi))
		# print(self.omegaTPLS1/(2*np.pi))
		# print(delta/(ks*self.keff))
		# print(self.vCOM)
		# print(Deltav)
		# print()

		Omega1 = self.Rabi1[iax,0]
		Omega3 = self.Rabi3[iax,0]
		Omega2 = 0.5*(Omega1 + Omega3)
		tau1   = self.tau1[iax]
		tau2   = self.tau2[iax]
		tau3   = self.tau3[iax]
		sigv   = np.sqrt(2.*self.kBoltz*self.Temp[iax,0]/self.MRb)
		vL     = -8.*sigv
		vR     = +8.*sigv

		epsabs = 1.E-06
		epsrel = 1.E-06
		limit  = 100

		## Phase and contrast for selected velocity
		f   = lambda v:  np.exp(-(v/sigv)**2)/(np.sqrt(np.pi)*sigv)
		fC  = lambda v: f(v)*self.ContrastModel(ks*self.keff*(v-Deltav), Omega1, Omega2, Omega3, tau1, tau2, tau3)
		fCP = lambda v: fC(v)*self.PhaseModel(ks*self.keff*(v-Deltav), Omega1, Omega3, tau1, tau3)
		C   = integrate.quad(fC, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		P   = integrate.quad(fCP, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/C

		eps   = 1.E-3

		## Phase and contrast uncertainty due to selected velocity
		dfC   = lambda v: f(v)*self.ContrastModel(ks*self.keff*(v-(1.+eps)*Deltav), Omega1, Omega2, Omega3, tau1, tau2, tau3)
		dfCP  = lambda v: dfC(v)*self.PhaseModel(ks*self.keff*(v-(1.+eps)*Deltav), Omega1, Omega3, tau1, tau3)
		dCdv  = abs((integrate.quad(dfC, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0] - C)/(eps*Deltav))
		dPdv  = abs((integrate.quad(dfCP, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/C - P)/(eps*Deltav))
		dC    = dCdv*self.dvCOM

		## Phase uncertainty due to Rabi frequencies
		dfC   = lambda v: f(v)*self.ContrastModel(ks*self.keff*(v-Deltav), (1.+eps)*Omega1, Omega2, Omega3, tau1, tau2, tau3)
		dfCP  = lambda v: dfC(v)*self.PhaseModel(ks*self.keff*(v-Deltav), (1.+eps)*Omega1, Omega3, tau1, tau3)
		dPdO1 = abs((integrate.quad(dfCP, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/C - P)/(eps*Omega1))

		dfC   = lambda v: f(v)*self.ContrastModel(ks*self.keff*(v-Deltav), Omega1, (1.+eps)*Omega2, Omega3, tau1, tau2, tau3)
		dfCP  = lambda v: dfC(v)*self.PhaseModel(ks*self.keff*(v-Deltav), Omega1, Omega3, tau1, tau3)
		dPdO2 = abs((integrate.quad(dfCP, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/C - P)/(eps*Omega2))

		dfC   = lambda v: f(v)*self.ContrastModel(ks*self.keff*(v-Deltav), Omega1, Omega2, (1.+eps)*Omega3, tau1, tau2, tau3)
		dfCP  = lambda v: dfC(v)*self.PhaseModel(ks*self.keff*(v-Deltav), Omega1, (1.+eps)*Omega3, tau1, tau3)
		dPdO3 = abs((integrate.quad(dfCP, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/C - P)/(eps*Omega3))

		## Total phase uncertainty
		dP    = np.sqrt((P*dC/C**2)**2 + (dPdv*self.dvCOM)**2 + (dPdO1**2 + 0.25*dPdO2**2)*self.Rabi1[iax,1]**2 + (dPdO3**2 + 0.25*dPdO2**2)*self.Rabi1[iax,1]**2)

		self.pMZA  = P
		self.dpMZA = dP

		# v = np.linspace(vL, vR, num=201)
		# p = self.PhaseModel(ks*self.keff*(v-Deltav), Omega1, Omega3, tau1, tau3)
		# c = self.ContrastModel(ks*self.keff*(v-Deltav), Omega1, Omega2, Omega3, tau1, tau2, tau3)

		# plt.plot(v*1.E3, p, color='red', marker='None', linestyle='-')
		# plt.plot(v*1.E3, c*p, color='black', marker='None', linestyle='-')
		# # plt.plot(v*1.E3, f(v)/500., color='blue', marker='None', linestyle='-')
		# plt.ylabel(r'$\phi$  (rad)')

		# # plt.plot(v*1.E3, c, color='red', marker='None', linestyle='-')
		# # plt.plot(v*1.E3, f(v)*np.sqrt(np.pi)*sigv, color='blue', marker='None', linestyle='-')
		# # plt.ylabel(r'$C$')
		# # plt.axhline(C,-1,1,color='black')

		# plt.xlabel(r'$v$  (mm/s)')
		# plt.axvline(0,-1,1,color='gray')
		# plt.axhline(0,-1,1,color='gray')

		# plt.show()

	######### End of Systematics.MachZehnderAsymmetryShift() ########
	#################################################################

	def TotalShift(self, iax, ks):
		"""Compute the total shift on a three-pulse atom interferometer."""

		self.OnePhotonLightShift(iax, ks)
		self.TwoPhotonLightShift(iax, ks)
		self.QuadraticZeemanShift(iax, ks)
		self.MagneticForceShift(iax, ks)
		self.GravityGradientShift(iax, ks)
		self.CoriolisShift(iax, ks)
		self.WavefrontDistortionShift(iax, ks)
		self.ParasiticLinesShift(iax, ks)
		self.MachZehnderAsymmetryShift(iax, ks)

		self.pTot  = self.pOPLS + self.pTPLS + self.pQZ + self.pBF + self.pGG + \
			 self.pCor + self.pPL +  self.pWD + 0.*self.pMZA
		self.dpTot = np.sqrt(self.dpOPLS**2 + self.dpTPLS**2 + self.dpQZ**2 + self.dpBF**2 + self.dpGG**2 + \
			self.dpCor**2 + self.dpPL**2 + self.dpWD**2 + 0.*self.dpMZA**2)

		# self.pTot  = self.pTPLS
		# self.dpTot = self.dpTPLS

	############## End of Systematics.CoriolisShift() ###############
	#################################################################

	def ComputeSystematics(self, iax, PrintSys=True, Units='rad'):
		"""Compute systematics and fill values in a list of dataframes."""

		self.SetAIParameters()

		self.SysPhases = ['pOPLS', 'pTPLS', 'pQZ', 'pBF', 'pGG', 'pCor', 'pPL', 'pWD', 'pMZA', 'pTot']
		self.SysVars   = ['omegaOPLS1', 'omegaOPLS3', 'omegaTPLS1', 'omegaTPLS3', 'omegaQZ'] + self.SysPhases
		self.SysDF     = [pd.DataFrame([], columns=['Values', 'Errors'], index=self.SysVars) for ik in range(4)]

		iaxLabels = ['X', 'Y', 'Z']
		ikLabels  = ['kU', 'kD', 'kInd', 'kDep']
		for ik in range(0,4):
			if ik < 2:
				self.TotalShift(iax, float(1-2*ik))

				self.SysDF[ik]['Values'] = [getattr(self, var) for var in self.SysVars]
				self.SysDF[ik]['Errors'] = [getattr(self, 'd'+var) for var in self.SysVars]
			else:
				if ik == 2:
					self.SysDF[ik]['Values'] = 0.5*(self.SysDF[0]['Values'] + self.SysDF[1]['Values'])
					self.SysDF[ik]['Errors'] = 0.5*np.sqrt(self.SysDF[0]['Errors']**2 + self.SysDF[1]['Errors']**2 + \
						2.*self.kIndCorr[iax]*self.SysDF[0]['Errors']*self.SysDF[1]['Errors'])
				else:
					self.SysDF[ik]['Values'] = 0.5*(self.SysDF[0]['Values'] - self.SysDF[1]['Values'])
					self.SysDF[ik]['Errors'] = 0.5*np.sqrt(self.SysDF[0]['Errors']**2 + self.SysDF[1]['Errors']**2 - \
						2.*self.kIndCorr[iax]*self.SysDF[0]['Errors']*self.SysDF[1]['Errors'])


		pmask = [s[0] == 'p' for s in list(self.SysVars)]

		for ik in range(0,4):
			if Units == 'mrad':
				self.SysDF[ik].loc[pmask] /= 1.E-3
			elif Units == 'm/s^2':
				self.SysDF[ik].loc[pmask] /= self.Seff[iax]
			elif Units == 'ug':
				self.SysDF[ik].loc[pmask] /= self.Seff[iax]*self.gLocal*1.E-6
			elif Units == 'Hz/s':
				self.SysDF[ik].loc[pmask] /= 2.*np.pi*self.Teff[iax]**2

		if PrintSys:
			df = pd.DataFrame([], columns=['kU_Val', 'kU_Err', 'kD_Val', 'kD_Err', 'kInd_Val', 'kInd_Err', 'kDep_Val', 'kDep_Err'],
				index=self.SysPhases)

			for ik in range(4):
				df[ikLabels[ik]+'_Val'] = self.SysDF[ik]['Values'].loc[pmask]
				df[ikLabels[ik]+'_Err'] = self.SysDF[ik]['Errors'].loc[pmask]

			pd.set_option('display.max_rows', 10)
			pd.set_option('display.max_columns', 8)
			pd.set_option('display.expand_frame_repr', False)
			print('-------------------------------- {} Systematics ({}) --------------------------------'.format(iaxLabels[iax], Units))
			print(df)

	############ End of Systematics.ComputeSystematics() ############
	#################################################################

	def PlotSystematics(self, iax, xName, xList, Units='rad', SysType='All'):
		"""Plot systematics for a specific axis as a function of a specific AI variable."""

		iXUtils.SetDefaultPlotOptions()

		if SysType == 'All':
			(nRows, nCols) = (3,3)
			fig  = plt.figure(figsize=(nCols*4,nRows*2), constrained_layout=True)
			gs   = fig.add_gridspec(nrows=nRows, ncols=nCols)
			ax00 = fig.add_subplot(gs[0,0])
			ax10 = fig.add_subplot(gs[1,0])
			ax20 = fig.add_subplot(gs[2,0])
			ax01 = fig.add_subplot(gs[0,1])
			ax11 = fig.add_subplot(gs[1,1])
			ax21 = fig.add_subplot(gs[2,1])
			ax_2 = fig.add_subplot(gs[:,2])
			axs  = np.array([[ax00, ax01, ax_2], [ax10, ax11, ax_2], [ax20, ax21, ax_2]])

			axs[0,0].get_shared_x_axes().join(*axs[:,0])
			axs[0,1].get_shared_x_axes().join(*axs[:,1])
			axs[0,0].set_xticklabels([])
			axs[0,1].set_xticklabels([])
			axs[1,0].set_xticklabels([])
			axs[1,1].set_xticklabels([])
		else:
			(nRows, nCols) = (1,1)
			fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(6,4), constrained_layout=True)

		nx    = len(xList)
		pOPLS = np.zeros((4,2,nx)) ## [ik,iErr,iv]
		pTPLS = np.zeros((4,2,nx))
		pQZ   = np.zeros((4,2,nx))
		pBF   = np.zeros((4,2,nx))
		pGG   = np.zeros((4,2,nx))
		pCor  = np.zeros((4,2,nx))
		pPL   = np.zeros((4,2,nx))
		pWD   = np.zeros((4,2,nx))
		pMZA  = np.zeros((4,2,nx))
		pTot  = np.zeros((4,2,nx))

		for ix in range(nx):
			## Compute systematics for each value of AI variable

			if type(getattr(self, xName)) is list or type(getattr(self, xName)) == np.ndarray:
				if len(getattr(self, xName).shape) > 1:
					getattr(self, xName)[iax] = xList[ix]
				else:
					getattr(self, xName)[iax] = xList[ix]
			else:
				setattr(self, xName, xList[ix])

			self.ComputeSystematics(iax, PrintSys=False, Units=Units)

			for ik in range(0,4):
				pOPLS[ik,0,ix] = self.SysDF[ik].loc['pOPLS','Values']
				pOPLS[ik,1,ix] = self.SysDF[ik].loc['pOPLS','Errors']
				pTPLS[ik,0,ix] = self.SysDF[ik].loc['pTPLS','Values']
				pTPLS[ik,1,ix] = self.SysDF[ik].loc['pTPLS','Errors']
				pQZ[ik,0,ix]   = self.SysDF[ik].loc['pQZ','Values']
				pQZ[ik,1,ix]   = self.SysDF[ik].loc['pQZ','Errors']
				pBF[ik,0,ix]   = self.SysDF[ik].loc['pBF','Values']
				pBF[ik,1,ix]   = self.SysDF[ik].loc['pBF','Errors']
				# pGG[ik,0,ix]   = self.SysDF[ik].loc['pGG','Values']
				# pGG[ik,1,ix]   = self.SysDF[ik].loc['pGG','Errors']
				pCor[ik,0,ix]  = self.SysDF[ik].loc['pCor','Values']
				pCor[ik,1,ix]  = self.SysDF[ik].loc['pCor','Errors']
				pPL[ik,0,ix]   = self.SysDF[ik].loc['pPL','Values']
				pPL[ik,1,ix]   = self.SysDF[ik].loc['pPL','Errors']
				pWD[ik,0,ix]   = self.SysDF[ik].loc['pWD','Values']
				pWD[ik,1,ix]   = self.SysDF[ik].loc['pWD','Errors']
				# pMZA[ik,0,ix]  = self.SysDF[ik].loc['pMZA','Values']
				# pMZA[ik,1,ix]  = self.SysDF[ik].loc['pMZA','Errors']
				pTot[ik,0,ix]  = self.SysDF[ik].loc['pTot','Values']
				pTot[ik,1,ix]  = self.SysDF[ik].loc['pTot','Errors']

		if Units == 'rad':
			symbLabel = r'$\phi$'
			unitLabel = '(rad)'
		elif Units == 'mrad':
			symbLabel = r'$\phi$'
			unitLabel = '(mrad)'
		elif Units == 'ug':
			symbLabel = r'$a$'
			unitLabel = r'($\mu g$)'
		elif Units == 'm/s^2':
			symbLabel = r'$a$'
			unitLabel = r'(m/s$^2$)'
		elif Units == 'Hz/s':
			symbLabel = r'$\alpha$'
			unitLabel = '(Hz/s)'

		colors   = ['red', 'blue', 'darkorange', 'purple']
		lLabels  = [r'$+k$', r'$-k$', r'$k_{\rm ind}$', r'$k_{\rm dep}$']
		plotOpts = {'Color': 'red', 'Linestyle': '-', 'Marker': 'None', 'Title': 'None',
			'xLabel': 'None', 'yLabel': 'None', 'LegLabel': 'None', 'Legend': False,
			'LegLocation': 'best'}

		if xName == 'T':
			xLabel = r'$T$  (ms)'
			xScale = 1.E3
		elif xName == 'TOF':
			xLabel = r'$TOF$  (ms)'
			xScale = 1.E3
		elif xName == 'v0':
			xLabel = r'$v_z$  (ms)'
			xScale = 1.E3
		elif xName == 'zM':
			xLabel = r'$z_M$  (mm)'
			xScale = 1.E3
		else:
			xLabel = xName
			xScale = 1.

		if SysType == 'All':
			for ik in range(0,4):
				plotOpts['Color']    = colors[ik]
				plotOpts['xLabel']   = 'None'
				plotOpts['LegLabel'] = 'None'
				plotOpts['Legend']   = False
				# plotOpts['yLabel']   = symbLabel+r'$_{\rm OPLS}$  '+unitLabel
				# iXUtils.CustomPlot(axs[0,0], plotOpts, xList*xScale, pOPLS[ik,0])
				# axs[0,0].fill_between(xList*xScale, pOPLS[ik,0] - pOPLS[ik,1], pOPLS[ik,0] + pOPLS[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['yLabel']   = symbLabel+r'$_{\rm TPLS}$  '+unitLabel
				iXUtils.CustomPlot(axs[0,0], plotOpts, xList*xScale, pTPLS[ik,0])
				axs[0,0].fill_between(xList*xScale, pTPLS[ik,0] - pTPLS[ik,1], pTPLS[ik,0] + pTPLS[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['yLabel']   = symbLabel+r'$_{\rm QZ}$  '+unitLabel
				iXUtils.CustomPlot(axs[1,0], plotOpts, xList*xScale, pQZ[ik,0])
				axs[1,0].fill_between(xList*xScale, pQZ[ik,0] - pQZ[ik,1], pQZ[ik,0] + pQZ[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['xLabel']   = xLabel
				plotOpts['yLabel']   = symbLabel+r'$_{\rm BF}$  '+unitLabel
				iXUtils.CustomPlot(axs[2,0], plotOpts, xList*xScale, pBF[ik,0])
				axs[2,0].fill_between(xList*xScale, pBF[ik,0] - pBF[ik,1], pBF[ik,0] + pBF[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['xLabel']   = 'None'
				plotOpts['yLabel']   = symbLabel+r'$_{\rm Cor}$  '+unitLabel
				iXUtils.CustomPlot(axs[0,1], plotOpts, xList*xScale, pCor[ik,0])
				axs[0,1].fill_between(xList*xScale, pCor[ik,0] - pCor[ik,1], pCor[ik,0] + pCor[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['yLabel']   = symbLabel+r'$_{\rm PL}$  '+unitLabel
				iXUtils.CustomPlot(axs[1,1], plotOpts, xList*xScale, pPL[ik,0])
				axs[1,1].fill_between(xList*xScale, pPL[ik,0] - pPL[ik,1], pPL[ik,0] + pPL[ik,1], color=colors[ik], alpha=0.5)

				# plotOpts['yLabel']   = symbLabel+r'$_{\rm GG}$  '+unitLabel
				# iXUtils.CustomPlot(axs[1,1], plotOpts, xList*xScale, pGG[ik,0])
				# axs[1,1].fill_between(xList*xScale, pGG[ik,0] - pGG[ik,1], pGG[ik,0] + pGG[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['xLabel']   = xLabel
				plotOpts['yLabel']   = symbLabel+r'$_{\rm WD}$  '+unitLabel
				iXUtils.CustomPlot(axs[2,1], plotOpts, xList*xScale, pWD[ik,0])
				axs[2,1].fill_between(xList*xScale, pWD[ik,0] - pWD[ik,1], pWD[ik,0] + pWD[ik,1], color=colors[ik], alpha=0.5)

				# plotOpts['xLabel']   = xLabel
				# plotOpts['yLabel']   = symbLabel+r'$_{\rm MZA}$  '+unitLabel
				# iXUtils.CustomPlot(axs[2,1], plotOpts, xList*xScale, pMZA[ik,0])
				# axs[2,1].fill_between(xList*xScale, pMZA[ik,0] - pMZA[ik,1], pMZA[ik,0] + pMZA[ik,1], color=colors[ik], alpha=0.5)

				plotOpts['yLabel']   = symbLabel+r'$_{\rm Total}$  '+unitLabel
				plotOpts['LegLabel'] = lLabels[ik]
				plotOpts['Legend']   = True
				iXUtils.CustomPlot(axs[0,2], plotOpts, xList*xScale, pTot[ik,0])
				axs[0,2].fill_between(xList*xScale, pTot[ik,0] - pTot[ik,1], pTot[ik,0] + pTot[ik,1], color=colors[ik], alpha=0.5)
		else:

			plotOpts['xLabel']   = xLabel
			if SysType == 'OPLS':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm OPLS}$  '+unitLabel
				yList = pOPLS
			elif SysType == 'TPLS':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm TPLS}$  '+unitLabel
				yList = pTPLS
			elif SysType == 'QZ':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm QZ}$  '+unitLabel
				yList = pQZ
			elif SysType == 'BF':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm BF}$  '+unitLabel
				yList = pBF
			elif SysType == 'Cor':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm Cor}$  '+unitLabel
				yList = pCor
			elif SysType == 'PL':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm PL}$  '+unitLabel
				yList = pPL
			elif SysType == 'WD':
				plotOpts['yLabel'] = symbLabel+r'$_{\rm WD}$  '+unitLabel
				yList = pWD
			else:
				plotOpts['yLabel'] = symbLabel+r'$_{\rm Total}$  '+unitLabel
				yList = pTot

			for ik in range(0,4):
				plotOpts['Color']    = colors[ik]
				plotOpts['LegLabel'] = lLabels[ik]
				plotOpts['Legend']   = True
				iXUtils.CustomPlot(axs, plotOpts, xList*xScale, yList[ik,0])
				axs.fill_between(xList*xScale, yList[ik,0] - yList[ik,1], yList[ik,0] + yList[ik,1], color=colors[ik], alpha=0.5)

		plt.show()

	############## End of Systematics.PlotSystematics() #############
	#################################################################

#####################################################################
##################### End of class Systematics ######################
#####################################################################

#####################################################################
############### Test routines for class Systematics #################

def TestSystematics():
	class RunPars:
		def __init__(self):
			self.RamanDetuning	= -1.21E9
			self.RamanTOF		= 15.0E-3
			self.RamanT 		= 20.0E-3
			self.RamanpiX		= 6.08E-6
			self.RamanpiY		= 6.08E-6
			self.RamanpiZ		= 6.08E-6
			self.kUpFrequency	= 6.835098427E+9
			self.kDownFrequency	= 6.834287203E+9
			self.SelectionFreqs	= [6.834650111E+9, 6.834650111E+9]

	runPars = RunPars()
	phys    = iXC_Phys.Physics(runPars)

	# vSel = np.pi*(runPars.kUpFrequency - runPars.kDownFrequency)/phys.keff

	theta   = np.pi*phys.Latitude/180.
	alpha   = np.pi*phys.Heading/180.
	Omega   = np.array([np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), np.cos(theta)])*phys.OmegaE

	sysPars = {
		'kIndCorr':		np.array([0.90, 0.90, 0.90]),		## Correlation coefficient for kInd phase shifts
		'z0':			np.array([
							[0., 0.],
							[0., 0.],
							[0., 0.]]),						## Initial position relative to mirrors (m)			
		'v0':			np.array([
							[0.0, 2*np.pi*500./phys.keff],
							[0.0, 2*np.pi*500./phys.keff],
							[0.0, 2*np.pi*500./phys.keff]]),## Initial (selected) velocity (m/s)
		'a':			np.array([
							[phys.gLocal, 0.],
							[phys.gLocal, 0.],
							[phys.gLocal, 0.]]),			## Acceleration (m/s^2)
		'Tii':			np.array([
							[phys.Txx, 0.01*phys.Txx],
							[phys.Tyy, 0.01*phys.Tyy],
							[phys.Tzz, 0.01*phys.Tzz]]),	## Gravity gradients (s^-2)
		'Omega':		np.array([
							[Omega[0], 0.01*Omega[0]],
							[Omega[1], 0.01*Omega[1]],
							[Omega[2], 0.01*Omega[2]]]),	## Rotation rates (Hz)
		'Temp':			np.array([
							[2.5E-6, 0.5E-6],
							[2.5E-6, 0.5E-6],
							[2.5E-6, 0.5E-6]]),				## Sample temperature (K)
		'Rabi1': 		np.array([
							[0.468, 0.002],
							[0.468, 0.002],
							[0.450, 0.050]])*1.E6,			## Counter-propagating Rabi frequency at t = t1 (rad/s)
		'Rabi3': 		np.array([
							[0.468, 0.002],
							[0.468, 0.002],
							[0.450, 0.050]])*1.E6,			## Counter-propagating Rabi frequency at t = t3 (rad/s)
		'Rabi1_mF0': 	np.array([
							[0.250, 0.010],
							[0.250, 0.010],
							[0.250, 0.010]])*1.E6,			## Co-propagating Rabi frequency for Delta mF = 0 at t = t1 (rad/s)
		'Rabi3_mF0': 	np.array([
							[0.250, 0.010],
							[0.250, 0.010],
							[0.250, 0.010]])*1.E6,			## Co-propagating Rabi frequency for Delta mF = 0 at t = t3 (rad/s)
		'Rabi1_mF2': 	np.array([
							[0.250/3, 0.003],
							[0.250/3, 0.003],
							[0.250/3, 0.003]])*1.E6,		## Co-propagating Rabi frequency for Delta mF = 2 at t = t3 (rad/s)
		'Rabi3_mF2': 	np.array([
							[0.250/3, 0.003],
							[0.250/3, 0.003],
							[0.250/3, 0.003]])*1.E6,		## Co-propagating Rabi frequency for Delta mF = 2 at t = t3 (rad/s)
		'P':			np.array([
							[0.000, 0.001],
							[0.000, 0.001],
							[0.000, 0.001]]),				## Power in Raman beam [V]
		'f0_OPLS':		np.array([
							[0.000, 0.001],
							[0.000, 0.001],
							[0.000, 0.001]]),				## One-photon light shift offset [Hz]
		'dfdt_OPLS':	np.array([
							[0.000, 0.001],
							[0.000, 0.001],
							[0.000, 0.001]]), 				## One-photon light shift time derivative [Hz/s]
		'dfdP_OPLS':	np.array([
							[0.000, 0.001],
							[0.000, 0.001],
							[0.000, 0.001]]),				## One-photon light shift power derivative [Hz/V]
		'beta0':		np.array([
							[0.142, 0.001],
							[0.142, 0.001],
							[0.142, 0.001]]),				## Magnetic bias field (G)
		'beta1':		np.array([
							[0.026, 0.005],
							[0.026, 0.005],
							[0.026, 0.005]]),				## Magnetic field gradient (G/m)
		'beta2':		np.array([
							[0.000, 0.001],
							[0.000, 0.001],
							[0.000, 0.001]]),				## Magnetic field curvature (G/m^2)
		'DM':			np.array([0.050, 0.001]),			## Mirror diameter (m)
		'FlatM':		np.array([780.E-9/2.,780.E-9/4.]),	## Mirror flatness (m)
		'zM':			np.array([
							[0.15, 0.001],
							[0.15, 0.001],
							[0.074, 0.001]]),				## Relative mirror position (m)
		'IRel': 		np.array([
							[0.01, 0.01], [0.01, 0.01], [1.00, 0.01],
							[0.55, 0.01], [0.01, 0.01], [0.01, 0.01]]),	## Relative intensities of parasitic lines (order=-2,-1,...,+3)
		# 'IRel': 		np.array([
		# 					[0.066, 0.004], [0.614, 0.02], [1.00, 0.02],
		# 					[0.636, 0.02], [0.062, 0.004], [0.001, 0.0015]]),	## Relative intensities of parasitic lines (order=-2,-1,...,+3)
	}

	iax   = 2
	units = 'mrad' ## ('rad', 'mrad', m/s^2', 'ug', 'Hz/s')
	sys   = Systematics(iax, runPars, sysPars)
	# sys.ComputeSystematics(iax, PrintSys=True, Units=units)
	# sys.PlotSystematics(iax, 'T', np.linspace(5.E-3, 48.E-3, num=100, endpoint=True), Units=units, SysType='PL')
	sys.PlotSystematics(iax, 'TOF', np.linspace(10.E-3, 50.E-3, num=40, endpoint=True), Units=units, SysType='TPLS')
	# sys.PlotSystematics(iax, 'v0', np.linspace(0.E-3, 100.E-3, num=40, endpoint=True), Units=units, SysType='TPLS')
	# sys.PlotSystematics(iax, 'zM', np.linspace(0.E-3, 44.E-3, num=40, endpoint=True), Units=units, SysType='PL')
	# sys.PlotSystematics(iax, 'Rabi1', np.linspace(0.4E+6, 0.45E+6, num=50, endpoint=True), units, SysType='All')

	# sys.tau1[iax] = 0.5*np.pi/sys.Rabi1[iax,0]
	# sys.tau3[iax] = 0.5*np.pi/sys.Rabi3[iax,0]

	# form = 'Exact'
	# form = 'Approx'
	# sys.TwoPhotonLightShift(iax,  1, Form=form)
	# pTPLS_kU  = sys.pTPLS
	# dpTPLS_kU = sys.dpTPLS

	# print(pTPLS_kU)
	# print(dpTPLS_kU)

	# sys.TwoPhotonLightShift(iax, -1, Form=form)
	# pTPLS_kD  = sys.pTPLS
	# dpTPLS_kD = sys.dpTPLS

	# pTPLS_kDep_F  = 0.5*(pTPLS_kU - pTPLS_kD)
	# pTPLS_kInd_F  = 0.5*(pTPLS_kU + pTPLS_kD)
	# dpTPLS_kDep_F = 0.5*np.sqrt(dpTPLS_kU**2 + dpTPLS_kD**2)
	# dpTPLS_kInd_F = 0.5*np.sqrt(dpTPLS_kU**2 + dpTPLS_kD**2)

	# r = 2.
	# sys.Rabi1[iax,0] /= r
	# sys.Rabi3[iax,0] /= r
	# sys.Rabi1_mF0[iax,0] /= r
	# sys.Rabi3_mF0[iax,0] /= r
	# sys.Rabi1_mF2[iax,0] /= r
	# sys.Rabi3_mF2[iax,0] /= r

	# sys.tau1[iax] = 0.5*np.pi/sys.Rabi1[iax,0]
	# sys.tau3[iax] = 0.5*np.pi/sys.Rabi3[iax,0]

	# sys.TwoPhotonLightShift(iax,  1, Form=form)
	# pTPLS_kU  = sys.pTPLS
	# dpTPLS_kD = sys.dpTPLS

	# sys.TwoPhotonLightShift(iax, -1, Form=form)
	# pTPLS_kD  = sys.pTPLS
	# dpTPLS_kD = sys.dpTPLS

	# pTPLS_kDep_H  = 0.5*(pTPLS_kU - pTPLS_kD)
	# pTPLS_kInd_H  = 0.5*(pTPLS_kU + pTPLS_kD)
	# dpTPLS_kDep_H = 0.5*np.sqrt(dpTPLS_kU**2 + dpTPLS_kD**2)
	# dpTPLS_kInd_H = 0.5*np.sqrt(dpTPLS_kU**2 + dpTPLS_kD**2)

	# pTPLS_kDep_D  = (r*pTPLS_kDep_H - pTPLS_kDep_F)/(r-1.)
	# pTPLS_kInd_D  = (r*pTPLS_kInd_H - pTPLS_kInd_F)/(r-1.)
	# dpTPLS_kDep_D = np.sqrt(r*dpTPLS_kDep_H**2 + dpTPLS_kDep_F**2)/(r-1.)
	# dpTPLS_kInd_D = np.sqrt(r*dpTPLS_kInd_H**2 + dpTPLS_kInd_F**2)/(r-1.)

	# print('pTPLS_kDep_F = {:.4f}({:.1E}) rad'.format(pTPLS_kDep_F, dpTPLS_kDep_F))
	# print('pTPLS_kInd_F = {:.4f}({:.1E}) rad'.format(pTPLS_kInd_F, dpTPLS_kInd_F))
	# print('pTPLS_kDep_H = {:.4f}({:.1E}) rad'.format(pTPLS_kDep_H, dpTPLS_kDep_H))
	# print('pTPLS_kInd_H = {:.4f}({:.1E}) rad'.format(pTPLS_kInd_H, dpTPLS_kInd_H))
	# print('pTPLS_kDep_D = {:.4f}({:.1E}) rad'.format(pTPLS_kDep_D, dpTPLS_kDep_D))
	# print('pTPLS_kInd_D = {:.4f}({:.1E}) rad'.format(pTPLS_kInd_D, dpTPLS_kInd_D))

	# sys.MachZehnderAsymmetryShift(iax, 1)

#################### End of TestSystematics() #######################
#####################################################################

if __name__ == '__main__':
	TestSystematics()