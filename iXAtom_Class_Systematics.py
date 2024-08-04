#####################################################################
## Filename:	iXAtom_Class_Systematics.py
## Author:		B. Barrett
## Description: Systematics class definition for iXAtom analysis package
##				Contains all AI systematics attributes and methods
## Version:		3.2.5
## Last Mod:	29/11/2020
#####################################################################

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy            	import integrate
from scipy.special    	import jv, binom
from scipy.interpolate	import griddata

import iXAtom_Class_Physics as iXC_Phys
import iXAtom_Utilities     as iXUtils

class Systematics(iXC_Phys.Physics):
	#################################################################
	## Class for computing atom interferometer systematic effects
	## Inherits all attributes and methods from class: Physics
	#################################################################

	def __init__(self, RunPars, SysInput, SysOptions):
		"""Get input needed to compute systematics and model parameters.
		Some input is contained within RunPars, while others are specified in SysInput.
		Each instance of the Systematics class is specific to a given Raman axis,
		hence iax is required as input.
		ARGUMENTS:
		\t RunPars  (obj)  - Instance of RunParameters class
		\t SysInput (dict) - Input parameters for model calculations (specified for each Raman axis)
		\t SysOpts  (dict) - Systematic evaluation options
		"""

		super().__init__(RunPars)

		## Initialize quantities contained in SysInput
		self.kIndCorr	= np.ones(3)					## Correlation coefficient for kInd phase shifts
		self.r0 		= np.zeros((3,2))				## Initial position relative to mirrors (m)
		self.v0			= np.zeros((3,2))				## Initial (selected) velocity (m/s)
		self.Temp 		= np.zeros((3,2))				## Sample temperature (K)
		self.Rabi1_kU	= np.zeros((3,2))				## Counter-propagating Rabi frequency at t = t1 (rad/s)
		self.Rabi3_kU	= np.zeros((3,2))				## Counter-propagating Rabi frequency at t = t3 (rad/s)
		self.Rabi1_kCo  = np.zeros((3,2))				## Co-propagating Rabi frequency for Delta mF = 0 at t = t1 (rad/s)
		self.Rabi3_kCo  = np.zeros((3,2))				## Co-propagating Rabi frequency for Delta mF = 0 at t = t3 (rad/s)
		self.beta0 		= np.zeros((3,2))				## Magnetic bias field (G)
		self.beta1 		= np.zeros((3,2))				## Magnetic field gradient (G/m)
		self.beta2		= np.zeros((3,2))				## Magnetic field curvature (G/m^2)
		self.wBeam		= np.zeros(2)					## 1/e^2 beam waist (m)
		self.DM			= np.zeros(2)					## Mirror diameter (m)
		self.FlatM		= np.zeros(2)					## Mirror flatness (m)
		self.zM			= np.zeros((3,2))				## Relative mirror position (m)
		self.IRatio		= np.zeros(2)					## Intensities ratio between main Raman lines

		self.iaxLabels  = ['X', 'Y', 'Z']
		self.ikLabels   = ['kU', 'kD', 'kInd', 'kDep']

		## Set attributes according to those present in SysInput
		for key, val in SysInput.items():
			setattr(self, key, val)

		## Add systematic evaluation options
		self.SysOpts 	= SysOptions

		## Calibrate B-field
		_, self.BInt = self.BFieldCalibration()

		## Set derived AI parameters
		self.SetAIParameters()

	################# End of Systematics.__init__() #################
	#################################################################

	def SetAIParameters(self):
		"""Set derived three-pulse atom interferometer parameters."""

		## AI timing and scale factors
		self.tau1   = 0.5*self.taupi								## 1st pulse lengths (s)
		self.tau2   = self.taupi									## 2nd pulse lengths (s)
		self.tau3   = 0.5*self.taupi								## 3rd pulse lengths (s)
		self.t1     = self.TOF 										## 1st pulse times (s)
		self.t2     = self.t1 + self.T + self.tau1 + 0.5*self.tau2	## 2nd pulse times (s)
		self.t3     = self.t2 + self.T + 0.5*self.tau2 + self.tau3	## 3rd pulse times (s)
		self.Ttotal = self.t3 - self.t1								## Total interrogation times (s)

		# self.Omega1 = self.Rabi1_kU[:,0]							## Rabi frequencies during Raman pulse 1 (rad/s)
		# self.Omega2 = 0.5*(self.Rabi1_kU[:,0] + self.Rabi3_kU[:,0])	## Rabi frequencies during Raman pulse 2 (rad/s)
		# self.Omega3 = self.Rabi3_kU[:,0]							## Rabi frequencies during Raman pulse 3 (rad/s)

		Omega1 = self.Rabi1_kU[:,0]
		Omega3 = self.Rabi3_kU[:,0]
		self.Teff   = np.sqrt((self.T + self.tau2)*(self.T + np.tan(Omega1*self.tau1/2.)/Omega1 \
			+ np.tan(Omega3*self.tau3/2.)/Omega3)) 					## Effective interrogation times (s)
		self.Seff   = self.keff*self.Teff**2 						## Effective interferometer scale factors (rad/m/s^2)

		self.sigvD  = np.sqrt(2.*self.kBoltz*self.Temp/self.MRb) 	## Doppler velocity standard deviation (m/s)

	############## End of Systematics.SetAIParameters() #############
	#################################################################

	def TrajectoryModel(self, iax, ks, t):
		"""Compute atomic center-of-mass position and velocity at time t during a three-pulse interferometer.
		The expression for the COM position is:
		  z(t <  t1) = z0 + v0*t + 0.5*a*t^2 
		  z(t >= t1) = z0 + v0*t + 0.5*ks*vR*(t-t1) + 0.5*a*t^2 
		Note the time is the true time-of-flight (relative to molasses release).
		Set ks = 0 to obtain the undiffracted COM trajectory.
		"""

		## Position (m) and velocity (m/s) of initial state |F=1,mF=0>
		r = self.r0[iax,0] + self.v0[iax,0]*t + 0.5*self.aBody[iax]*t**2
		v = self.v0[iax,0] + self.aBody[iax]*t

		## Position (m) and velocity uncertainty (m/s)
		dr = np.sqrt(self.r0[iax,1]**2 + (self.v0[iax,1]*t)**2)
		dv = self.v0[iax,1]

		## COM position (m) and velocity (m/s)
		if t > self.t1[iax]:
			rCOM = r + 0.5*ks*self.vR*(t - self.t1[iax])
			vCOM = v + 0.5*ks*self.vR
		else:
			rCOM = r
			vCOM = v

		## COM position (m) and velocity uncertainty (m/s)
		drCOM = dr
		dvCOM = dv

		return np.array([rCOM, drCOM, vCOM, dvCOM, r, dr, v, dv])

	############## End of Systematics.TrajectoryModel() #############
	#################################################################

	def BModel(self, iax, ks, t):
		"""Model for the magnetic field containing a spatial gradient + curvature: 
			B(t) = beta0 + beta1*rCOM + beta2*rCOM**2,
		where rCOM is the center-of-mass position of the atoms at time t (the true time-of-flight relative
		to molasses release). Set ks = 0 to obtain the undiffracted trajectory.
		"""

		rCOM, drCOM = self.TrajectoryModel(iax, ks, t)[:2]

		if self.SysOpts['UseBCal']:
			if iax == 0:
				Q1 = 'Bx_Offset'
				Q2 = 'Bx_Grad'
			elif iax == 1:
				Q1 = 'By_Offset'
				Q2 = 'By_Grad'
			else: ## iax == 2
				Q1 = 'Bz_Offset'
				Q2 = 'Bz_Grad'

			beta0  = griddata((self.BInt['thetaX'], self.BInt['thetaZ']), self.BInt[Q1], ([self.thetaX*180./np.pi], [self.thetaZ*180./np.pi]), method='nearest')[0]
			beta1  = griddata((self.BInt['thetaX'], self.BInt['thetaZ']), self.BInt[Q2], ([self.thetaX*180./np.pi], [self.thetaZ*180./np.pi]), method='nearest')[0]
			dbeta0 = self.BInt['d'+Q1]
			dbeta1 = self.BInt['d'+Q2]
		else:
			beta0  = self.beta0[iax,0]
			dbeta0 = self.beta0[iax,1]
			beta1  = self.beta1[iax,0]
			dbeta1 = self.beta1[iax,1]

		# print('thetaX, thetaZ, beta0, beta1 = {:.1f}, {:.1f} deg, {:.5f} G, {:.3f} G/m'.format(self.thetaX*180./np.pi, self.thetaZ*180./np.pi, beta0, beta1))

		beta2  = self.beta2[iax,0]
		dbeta2 = self.beta2[iax,1]

		B      = beta0 + beta1*rCOM + beta2*rCOM**2
		dB     = np.sqrt(dbeta0**2 + (dbeta1*rCOM)**2 + (beta1*drCOM)**2 + (dbeta2*rCOM**2)**2 + (beta2*(2.*rCOM*drCOM))**2)
		BGrad  = beta1 + 2.*beta2*rCOM
		dBGrad = np.sqrt((dbeta1)**2 + (2.*beta2*drCOM)**2 + (2.*dbeta2*rCOM)**2)

		return np.array([B, dB, BGrad, dBGrad])

	################## End of Systematics.BModel() ##################
	#################################################################

	def RabiModel(self, iax, ks, t):
		"""Model for the effective Rabi frequency as a function of t. Set ks = 0 for co-propagating Rabi frequency,
		otherwise method returns counter-propagating (kU) case.
		Set SysOpts['RabiModel'] = 'Gaussian' for the Rabi frequency in a tilted, Gaussian beam.
		Set SysOpts['RabiModel'] = 'PL' for the Rabi frequency produced by parasitic lines.
		Set SysOpts['RabiModel'] = 'None' to recover input Rabi frequencies: Rabi1_kU, Rabi3_kU, Rabi1_kCo, Rabi3_kCo
		"""

		self.SetAIParameters()

		if self.SysOpts['RabiModel'] == 'Gaussian':
			## Get cloud position in the body frame at time t1 and time t
			p1 = np.zeros(3) ## iaxx
			p  = np.zeros(3) ## iaxx
			for iaxx in range(3):
				p1[iaxx] = self.TrajectoryModel(iaxx, 0., self.t1[iaxx])[0]
				p[iaxx]  = self.TrajectoryModel(iaxx, 0., t)[0]

			## Project cloud position into the plane of beam iax at times t1 and t:
			if iax == 0:
				r1 = np.sqrt(p1[1]**2 + p1[2]**2)
				r  = np.sqrt(p[1]**2 + p[2]**2)
			elif iax == 1:
				r1 = np.sqrt(p1[2]**2 + p1[0]**2)
				r  = np.sqrt(p[2]**2 + p[0]**2)
			else: ## iax == 2:
				r1 = np.sqrt(p1[0]**2 + p1[1]**2)
				r  = np.sqrt(p[0]**2 + p[1]**2)

			w, dw = self.wBeam							## 1/e^2 beam waist and uncertainty (m)
			I1 = np.exp(-2.*(r1/w)**2)					## Relative intensity at time t1
			I  = np.exp(-2.*(r/w)**2)					## Relative intensity at time t
			dI = 4.*(r/w)**2*I*dw/w						## Uncertainty in relative intensity

		elif self.SysOpts['RabiModel'] == 'PL':

			[NLoop, MLoop, nN, nM, N0, M0, NOffset] = self.InitializeParasiticLines(iax, ks)[2:]

			Deltak	= ks*(2./self.cLight)*self.OmegaRF			## List of effective wavevectors (rad/m)
			Delta1  = self.Delta + self.delta31 + self.OmegaRF	## Raman detuning from F = 2 -> 1' (rad/s)
			Delta2  = self.Delta + self.delta32 + self.OmegaRF	## Raman detuning from F = 2 -> 2' (rad/s)

			I00 = self.EPPLN[N0+NOffset[0],M0]*self.EPPLN[N0+NOffset[1],M0+1]*(1./abs(Delta2[N0,M0]) + 1./(3.*abs(Delta1[N0,M0])))
			INM = np.zeros((nN, nM))
			for N in NLoop:
				for M in MLoop:
					INM[N0+N,M0+M] = self.EPPLN[N0+NOffset[0]+N,M0+M]*self.EPPLN[N0+NOffset[1]+N,M0+M+1]*(1./abs(Delta2[N0+N,M0+M]) + 1./(3.*abs(Delta1[N0+N,M0+M])))/I00

			rCOM, drCOM = self.TrajectoryModel(iax, ks, t)[:2]

			zA  = rCOM - self.zM[iax,0]
			dzA = np.sqrt(drCOM**2 + self.zM[iax,1]**2)
			drI = self.IRatio[1]/self.IRatio[0] ## Relative uncertainty in Rabi frequencies (unitless)

			I   = np.abs(np.sum([[INM[N0+N,M0+M]*np.cos(Deltak[N0+N,M0+M]*zA) for N in NLoop] for M in MLoop]))
			dI  = np.abs(np.sqrt(np.sum([[(drI*INM[N0+N,M0+M]*np.cos(Deltak[N0+N,M0+M]*zA))**2 + \
				(INM[N0+N,M0+M]*Deltak[N0+N,M0+M]*np.sin(Deltak[N0+N,M0+M]*zA)*dzA)**2 for N in NLoop] for M in MLoop])))

		else: ## self.SysOpts['RabiModel'] == 'None':

			if ks == 0:
				if t <= self.t1[iax]:
					OmegaR, dOmegaR = self.Rabi1_kCo[iax]	## Rabi frequency and uncertainty at time t (rad/s)
				elif t <= self.t2[iax]:
					OmegaR, dOmegaR = 0.5*(self.Rabi1_kCo[iax] + self.Rabi3_kCo[iax])
				else:
					OmegaR, dOmegaR = self.Rabi3_kCo[iax]
			else:
				if t <= self.t1[iax]:
					OmegaR, dOmegaR = self.Rabi1_kU[iax]	## Rabi frequency and uncertainty at time t (rad/s)
				elif t <= self.t2[iax]:
					OmegaR, dOmegaR = 0.5*(self.Rabi1_kU[iax] + self.Rabi3_kU[iax])
				else:
					OmegaR, dOmegaR = self.Rabi3_kU[iax]

		if self.SysOpts['RabiModel'] == 'Gaussian' or self.SysOpts['RabiModel'] == 'PL':
			if ks == 0.:
				OmegaR  = self.Rabi1_kCo[iax,0]*I/I1		## Rabi frequency at time t (rad/s)
				dOmegaR = np.sqrt((self.Rabi1_kCo[iax,1]*I/I1)**2 + (self.Rabi1_kCo[iax,0]*dI/I1)**2) ## Rabi frequency uncertainty (rad/s)
			else:
				OmegaR  = self.Rabi1_kU[iax,0]*I/I1			## Rabi frequency at time t (rad/s)
				dOmegaR = np.sqrt((self.Rabi1_kU[iax,1]*I/I1)**2 + (self.Rabi1_kU[iax,0]*dI/I1)**2) ## Rabi frequency uncertainty (rad/s)

		return np.array([OmegaR, dOmegaR])

	################ End of Systematics.RabiModel() #################
	#################################################################

	def OnePhotonLightShiftModel(self, iax, ks, t):
		"""Model for the one-photon light shift as a function of t."""

		Omega, dOmega = self.RabiModel(iax, ks, t)		## Rabi frequency (rad/s)
		I2 = self.I2Raman(Omega, 1./self.IRatio[0])		## Intensity of Raman beam 2
		I1 = self.IRatio[0]*I2							## Intensity of Raman beam 1

		## Scalar, vector and tensor light shifts for mF = 0 and linearly polarized light
		deltaS, deltaV, deltaT = self.deltaAC(I1, I2, 0., self.Delta, 0.)
		omegaOPLS  = deltaS + deltaV + deltaT			## Total one-photon light shift (rad/s)
		domegaOPLS = (dOmega/Omega)*abs(omegaOPLS)		## Uncertainty in OPLS (rad/s)

		return np.array([omegaOPLS, domegaOPLS, Omega, dOmega])

	######## End of Systematics.OnePhotonLightShiftModel() ##########
	#################################################################

	def TwoPhotonLightShiftModel(self, iax, ks, t):
		"""Model for the two-photon light shift as a function of t.
		SysOpts['TPLS_Form'] == 'Exact' uses the exact formula for the TPLS.
		SysOpts['TPLS_Form'] == 'Approx' uses an approximate formula for the TPLS.
		"""
		v, dv   		= self.TrajectoryModel(iax, ks, t)[6:]	## Velocity of initial state at time t
		omegaD  		= ks*self.keff*v						## Doppler shift at time t (rad/s)
		domegaD 		= self.keff*dv							## Uncertainty in Doppler shift (rad/s)

		B, dB			= self.BModel(iax, ks, t)[:2]			## Magnetic field at time t (G)
		omegaB 			= 2.*np.pi*self.alphaB*abs(B)			## First-order Zeeman shift (Delta mF = 2) at time t (rad/s)
		# domegaB			= 2.*np.pi*self.alphaB*dB				## Uncertainty in Zeeman shift (rad/s)

		delta_kU_mF0  	= 2.*omegaD								## +k counter-propagating Delta mF =  0 transition
		delta_kU_mFp2 	= delta_kU_mF0 + 2.*omegaB				## +k counter-propagating Delta mF = +2 transition
		delta_kU_mFm2 	= delta_kU_mF0 - 2.*omegaB				## +k counter-propagating Delta mF = -2 transition
		# ddelta_kU		= 2.*domegaD							## Uncertainty in counter-propagating transition

		delta_kD_mF0  	= 2.*(omegaD + 2.*self.omegaR)			## -k counter-propagating Delta mF =  0 transition
		delta_kD_mFp2 	= delta_kD_mF0 + 2.*omegaB 				## -k counter-propagating Delta mF = +2 transition
		delta_kD_mFm2 	= delta_kD_mF0 - 2.*omegaB				## -k counter-propagating Delta mF = -2 transition

		delta_kCo_mF0 	= omegaD + self.omegaR 					## Co-proprogating Delta mF =  0 transition
		delta_kCo_mFm2	= delta_kCo_mF0 - omegaB				## Co-proprogating Delta mF = +2 transition
		delta_kCo_mFp2	= delta_kCo_mF0 + omegaB				## Co-proprogating Delta mF = -2 transition
		# ddelta_kCo		= domegaD								## Uncertainty in co-propagating transition

		rOmega_kU_mF20  = 0.084									## Theoretical Rabi frequency ratio (Omega_kU_mF2 / Omega_kU_mF0)
		rOmega_kCo_mF20 = 0.322									## Theoretical Rabi frequency ratio (Omega_kCo_mF2 / Omega_kU_mF0)

		Omega_kU_mF0,  dOmega_kU_mF0  = self.RabiModel(iax, ks, t)
		Omega_kCo_mF0, dOmega_kCo_mF0 = self.RabiModel(iax, 0., t)

		Omega_kU_mF2  	= rOmega_kU_mF20*Omega_kU_mF0
		dOmega_kU_mF2	= rOmega_kU_mF20*dOmega_kU_mF0
		Omega_kCo_mF2 	= rOmega_kCo_mF20*Omega_kU_mF0
		dOmega_kCo_mF2	= rOmega_kCo_mF20*dOmega_kU_mF0

		if self.SysOpts['TPLS_Form'] == 'Exact':
			## Exact formula for frequency shift
			omegaTPLS = 0.5*( \
				(np.sign(delta_kU_mF0  )*np.sqrt(Omega_kU_mF0**2  + delta_kU_mF0**2  ) - delta_kU_mF0  ) + \
				(np.sign(delta_kU_mFp2 )*np.sqrt(Omega_kU_mF2**2  + delta_kU_mFp2**2 ) - delta_kU_mFp2 ) + \
				(np.sign(delta_kU_mFm2 )*np.sqrt(Omega_kU_mF2**2  + delta_kU_mFm2**2 ) - delta_kU_mFm2 ) + \
				(np.sign(delta_kD_mF0  )*np.sqrt(Omega_kU_mF0**2  + delta_kD_mF0**2  ) - delta_kD_mF0  ) + \
				(np.sign(delta_kD_mFp2 )*np.sqrt(Omega_kU_mF2**2  + delta_kD_mFp2**2 ) - delta_kD_mFp2 ) + \
				(np.sign(delta_kD_mFm2 )*np.sqrt(Omega_kU_mF2**2  + delta_kD_mFm2**2 ) - delta_kD_mFm2 ) + \
				(np.sign(delta_kCo_mF0 )*np.sqrt(Omega_kCo_mF0**2 + delta_kCo_mF0**2 ) - delta_kCo_mF0 ) + \
				(np.sign(delta_kCo_mFm2)*np.sqrt(Omega_kCo_mF2**2 + delta_kCo_mFm2**2) - delta_kCo_mFm2) + \
				(np.sign(delta_kCo_mFp2)*np.sqrt(Omega_kCo_mF2**2 + delta_kCo_mFp2**2) - delta_kCo_mFp2))

			## Exact uncertainty in frequency shift
			domegaTPLS = 0.5*np.sqrt( \
				(Omega_kU_mF0*( \
					np.sign(delta_kU_mF0  )/np.sqrt(Omega_kU_mF0**2  + delta_kU_mF0**2  ) + \
					np.sign(delta_kD_mF0  )/np.sqrt(Omega_kU_mF0**2  + delta_kD_mF0**2  ))*dOmega_kU_mF0 )**2 + \
				(Omega_kU_mF2*( \
					np.sign(delta_kU_mFp2 )/np.sqrt(Omega_kU_mF2**2  + delta_kU_mFp2**2 ) + \
					np.sign(delta_kU_mFm2 )/np.sqrt(Omega_kU_mF2**2  + delta_kU_mFm2**2 ) + \
					np.sign(delta_kD_mFp2 )/np.sqrt(Omega_kU_mF2**2  + delta_kD_mFp2**2 ) + \
					np.sign(delta_kD_mFm2 )/np.sqrt(Omega_kU_mF2**2  + delta_kD_mFm2**2 ))*dOmega_kU_mF2 )**2 + \
				(Omega_kCo_mF0*( \
					np.sign(delta_kCo_mF0 )/np.sqrt(Omega_kCo_mF0**2 + delta_kCo_mF0**2 ))*dOmega_kCo_mF0)**2 + \
				(Omega_kCo_mF2*( \
					np.sign(delta_kCo_mFm2)/np.sqrt(Omega_kCo_mF2**2 + delta_kCo_mFm2**2) + \
					np.sign(delta_kCo_mFp2)/np.sqrt(Omega_kCo_mF2**2 + delta_kCo_mFp2**2))*dOmega_kCo_mF2)**2)# + \
				# ((delta_kU_mF0  /np.sqrt(Omega_kU_mF0**2  + delta_kU_mF0**2  ) - 1.)*ddelta_kU)**2  + \
				# ((delta_kD_mF0  /np.sqrt(Omega_kU_mF0**2  + delta_kD_mF0**2  ) - 1.)*ddelta_kU)**2  + \
				# ((delta_kCo_mF0 /np.sqrt(Omega_kCo_mF0**2 + delta_kCo_mF0**2 ) - 1.)*ddelta_kCo)**2 + \
				# ((delta_kCo_mFm2/np.sqrt(Omega_kCo_mF2**2 + delta_kCo_mFm2**2) - 1.)*ddelta_kCo)**2 + \
				# ((delta_kCo_mFp2/np.sqrt(Omega_kCo_mF2**2 + delta_kCo_mFp2**2) - 1.)*ddelta_kCo)**2)

		else: ## self.SysOpts['TPLS_Form'] == 'Approx'

			## Approximate formula for frequency shift when delta11, delta12 >> Rabi1_kU, delta13 >> RabiCo1,
			omegaTPLS = 0.25*( \
				Omega_kU_mF0**2 *(1./delta_kU_mF0   + 1./delta_kD_mF0) + \
				Omega_kU_mF2**2 *(1./delta_kU_mFp2  + 1./delta_kU_mFm2 + 1./delta_kD_mFp2 + 1./delta_kD_mFm2) + \
				Omega_kCo_mF2**2*(1./delta_kCo_mFm2 + 1./delta_kCo_mFp2) + \
				Omega_kCo_mF0**2*(1./delta_kCo_mF0))

			## Approximate uncertainty in frequency shift
			domegaTPLS = 0.25*np.sqrt( \
				(2.*Omega_kU_mF0*dOmega_kU_mF0*(1./delta_kU_mF0 + 1./delta_kD_mF0))**2 + \
				(2.*Omega_kU_mF2*dOmega_kU_mF2*(1./delta_kU_mFp2 + 1./delta_kU_mFm2 + 1./delta_kD_mFp2 + 1./delta_kD_mFm2))**2 + \
				(2.*Omega_kCo_mF0*dOmega_kCo_mF0*(1./delta_kCo_mF0))**2 + \
				(2.*Omega_kCo_mF2*dOmega_kCo_mF2*(1./delta_kCo_mFm2 + 1./delta_kCo_mFp2))**2 + \
				(Omega_kU_mF0**2*(2.*domegaD)*(1./delta_kU_mF0**2 + 1./delta_kD_mF0**2))**2 + \
				(Omega_kU_mF2**2*(2.*domegaD)*(1./delta_kU_mFp2**2 + 1./delta_kU_mFm2**2 + 1./delta_kD_mFp2**2 + 1./delta_kD_mFm2**2))**2 + \
				(Omega_kCo_mF0**2*(domegaD)*(1./delta_kCo_mF0**2))**2 + \
				(Omega_kCo_mF2**2*(domegaD)*(1./delta_kCo_mFm2**2 + 1./delta_kCo_mFp2**2))**2)

		return np.array([omegaTPLS, domegaTPLS, Omega_kU_mF0, dOmega_kU_mF0, Omega_kCo_mF0, dOmega_kCo_mF0])

	######## End of Systematics.TwoPhotonLightShiftModel() ##########
	#################################################################

	def DetuningModel(self, iax, ks, t):
		"""Model for the counter-propagating Raman detuning as a function of t. Set ks = 0 for co-propagating Rabi frequency.
		SysOpts['RTFreq'] = 'On' sets Raman frequency equal to recoil shift + true Doppler shift in body frame.
		SysOpts['RTFreq'] = 'Off' sets Raman frequency equal to recoil shift + Doppler shift in the lab frame.
		SysOpts['DetuningType'] = 'Full' includes OPLS, TPLS, and Zeeman shifts.
		SysOpts['DetuningType'] = 'OPLS' includes only OPLS.
		SysOpts['DetuningType'] = 'Basic' includes no light shifts or Zeeman shifts.
		"""

		v, dv   = self.TrajectoryModel(iax, ks, t)[6:]	## Velocity of initial state at time t
		omegaD  = ks*self.keff*v						## Doppler shift at time t (rad/s)
		domegaD = self.keff*dv							## Uncertainty in Doppler shift (rad/s)

		omegaOPLS, domegaOPLS = self.OnePhotonLightShiftModel(iax, ks, t)[:2]
		omegaTPLS, domegaTPLS = self.TwoPhotonLightShiftModel(iax, ks, t)[:2]

		B, dB	 = self.BModel(iax, ks, t)[:2]			## Magnetic field at time t (G)
		omegaQZ	 = 2.*np.pi*self.KClock*B**2 			## Quadratic Zeeman shift (Delta mF = 0) (rad/s)
		domegaQZ = 4.*np.pi*self.KClock*abs(B)*dB 		## Uncertainty in omegaQZ (rad/s)

		if self.SysOpts['RTFreq'] == 'On':
			omegaL  = self.omegaHF + omegaD + self.omegaR
			domegaL = 2.*np.pi*1.E3
		else: ## self.SysOpts['RTFreq'] == 'Off':
			omegaL  = self.omegaHF + ks*self.keff*self.aLocal[iax]*t + self.omegaR
			domegaL = 2.*np.pi*1.E3

		if self.SysOpts['DetuningType'] == 'Full':
			delta  = omegaL - (self.omegaHF + omegaD + self.omegaR + omegaOPLS + omegaTPLS + omegaQZ)
			ddelta = np.sqrt(domegaD**2 + domegaOPLS**2 + domegaTPLS**2 + domegaQZ**2)
		elif self.SysOpts['DetuningType'] == 'OPLS':
			delta  = omegaL - (self.omegaHF + omegaD + self.omegaR + omegaOPLS)
			ddelta = np.sqrt(domegaD**2 + domegaOPLS**2)
		else: ## self.SysOpts['DetuningType'] == 'Basic':
			delta  = omegaL - (self.omegaHF + omegaD + self.omegaR)
			ddelta = domegaD

		return np.array([delta, ddelta, omegaD, domegaD, omegaOPLS, domegaOPLS, omegaTPLS, domegaTPLS, omegaQZ, domegaQZ, omegaL, domegaL])

	############### End of Systematics.DetuningModel() ##############
	#################################################################

	def RamanFreqShifts(self, iax, ks):
		"""Compute Raman resonance frequency shifts at t = t1, t2, and t3."""

		self.delta1, self.ddelta1 = self.DetuningModel(iax, ks, self.t1[iax])[:2]
		self.delta2, self.ddelta2 = self.DetuningModel(iax, ks, self.t2[iax])[:2]
		self.delta3, self.ddelta3 = self.DetuningModel(iax, ks, self.t3[iax])[:2]

	############## End of Systematics.RamanFreqShifts() #############
	#################################################################

	def ContrastModel(self, iax, ks):
		"""Model of the contrast of a Mach-Zehnder interferometer as a function of Rabi frequency and detuning."""

		omega1, domega1 = self.RabiModel(iax, ks, self.t1[iax])
		omega2, domega2 = self.RabiModel(iax, ks, self.t2[iax])
		omega3, domega3 = self.RabiModel(iax, ks, self.t3[iax])

		if self.SysOpts['kCoTrans'] == 'On':
			omega_kCo1 	= self.RabiModel(iax, 0., self.t1[iax])[0]
			omega_kCo2 	= self.RabiModel(iax, 0., self.t2[iax])[0]
			omega_kCo3 	= self.RabiModel(iax, 0., self.t3[iax])[0]
		else:
			omega_kCo1 	= 0.
			omega_kCo2 	= 0.
			omega_kCo3 	= 0.

		delta1, omegaD1 = self.DetuningModel(iax, ks, self.t1[iax])[[0,2]]
		delta2, omegaD2 = self.DetuningModel(iax, ks, self.t2[iax])[[0,2]]
		delta3, omegaD3 = self.DetuningModel(iax, ks, self.t3[iax])[[0,2]]

		Omega1	= np.sqrt(delta1**2 + omega1**2)
		Omega2	= np.sqrt(delta2**2 + omega2**2)
		Omega3	= np.sqrt(delta3**2 + omega3**2)

		tau1 	= self.tau1[iax]
		tau2 	= self.tau2[iax]
		tau3 	= self.tau3[iax]

		s1		= np.sin(Omega1*tau1/2.)
		s2 		= np.sin(Omega2*tau2/2.)
		s3 		= np.sin(Omega3*tau3/2.)
		c1 		= np.cos(Omega1*tau1/2.)
		c2		= np.cos(Omega2*tau2/2.)
		c3		= np.cos(Omega3*tau3/2.)
		fC 		= delta1*delta3*s1*s3 + Omega1*Omega3*c1*c3

		DC1		= 2.*omega2**2*omega3/((Omega1*Omega2*Omega3)**2)*s2**2*s3 \
				* (omega1/Omega1*s1*(omega1*tau1*delta1*delta3*c1*s3 - omega1*tau1*Omega1*Omega3*s1*c3 + 2.*omega1*Omega3*c1*c3) \
				+ (2.*s1 + omega1**2*tau1/Omega1*c1 - 4.*omega1**2/Omega1**2*s1) * fC)
		DC2		= 4.*omega1*omega3/((Omega1*Omega2*Omega3)**2)*s1*s2*s3 \
				* (2.*omega2*s2 + omega2**3*tau2/Omega2*c2 - 2.*omega2**3/Omega2**2*s2) * fC
		DC3		= 2.*omega1*omega2**2/((Omega1*Omega2*Omega3)**2)*s1*s2**2 \
				* (omega3/Omega3*s3*(omega3*tau3*delta1*delta3*s1*c3 - omega3*tau3*Omega1*Omega3*c1*s3 + 2.*omega3*Omega1*c1*c3) \
				+ (2.*s3 + omega3**2*tau1/Omega3*c1 - 4.*omega3**2/Omega3**2*s3) * fC)
		dC		= np.sqrt((DC1*domega1)**2 + (DC2*domega2)**2 + (DC3*domega3)**2)

		if self.SysOpts['VelocityAvg'] == 'On':
			## Compute velocity-averaged contrast
			if self.SysOpts['RamanRegime'] == 'SD':
				def Cv(v):
					kv = self.keff*v
					d1 = delta1 + kv
					d2 = delta2 + kv
					d3 = delta3 + kv
					O1 = np.sqrt(d1**2 + omega1**2)
					O2 = np.sqrt(d2**2 + omega2**2)
					O3 = np.sqrt(d3**2 + omega3**2)
					s1 = np.sin(O1*tau1/2.)
					s2 = np.sin(O2*tau2/2.)
					s3 = np.sin(O3*tau3/2.)
					c1 = np.cos(O1*tau1/2.)
					c3 = np.cos(O3*tau3/2.)

					return 4.*omega1*omega2**2*omega3/((O1*O2*O3)**2)*s1*s2**2*s3 * (d1*d3*s1*s3 + O1*O3*c1*c3)

			else: ## self.SysOpts['RamanRegime'] == 'DD':
				def Cv(v):
					kv = self.keff*v
					c10_1, c21_1 = self.RamanDiffraction(tau1, omega1, delta1 + kv, omegaD1 + kv, self.omegaR, OmegakCo=omega_kCo1)
					_,     c21_2 = self.RamanDiffraction(tau2, omega2, delta2 + kv, omegaD2 + kv, self.omegaR, OmegakCo=omega_kCo2)
					c10_3, c21_3 = self.RamanDiffraction(tau3, omega3, delta3 + kv, omegaD3 + kv, self.omegaR, OmegakCo=omega_kCo3)

					return 4.*abs(c10_1*c21_1*c21_2**2*c10_3*c21_3)

			sigv = self.sigvD[iax,0]
			vL   = -4.*sigv
			vR   = +4.*sigv
			tol  = 1.E-3
			lim  = 100

			NCv  = lambda v: np.exp(-(v/sigv)**2)/(np.sqrt(np.pi)*sigv) * Cv(v)
			C    = integrate.quad(NCv, vL, vR, epsabs=tol, epsrel=tol, limit=lim)[0]

		else: ## self.SysOpts['VelocityAvg'] == 'Off':

			if self.SysOpts['RamanRegime'] == 'SD':
				C	= 4.*omega1*omega2**2*omega3/((Omega1*Omega2*Omega3)**2)*s1*s2**2*s3 * fC
			else: ## self.SysOpts['RamanRegime'] == 'DD':
				c10_1, c21_1 = self.RamanDiffraction(tau1, omega1, delta1, omegaD1, self.omegaR, OmegakCo=omega_kCo1)
				_,     c21_2 = self.RamanDiffraction(tau2, omega2, delta2, omegaD2, self.omegaR, OmegakCo=omega_kCo2)
				c10_3, c21_3 = self.RamanDiffraction(tau3, omega3, delta3, omegaD3, self.omegaR, OmegakCo=omega_kCo3)

				C 	= 4.*abs(c10_1*c21_1*c21_2**2*c10_3*c21_3)

		return np.array([C, dC])

	############### End of Systematics.ContrastModel() ##############
	#################################################################

	def OnePhotonLightShift(self, iax, ks):
		"""Compute the one-photon light shift on a three-pulse atom interferometer."""

		self.omegaOPLS1, self.domegaOPLS1, Omega1, dOmega1 = self.OnePhotonLightShiftModel(iax, ks, self.t1[iax])
		self.omegaOPLS3, self.domegaOPLS3, Omega3, dOmega3 = self.OnePhotonLightShiftModel(iax, ks, self.t3[iax])

		Theta1 = 0.5*Omega1*self.tau1[iax]
		Theta3 = 0.5*Omega3*self.tau3[iax]

		## Phase shift and uncertainty
		self.pOPLS  = self.omegaOPLS1/Omega1*np.tan(Theta1) - self.omegaOPLS3/Omega3*np.tan(Theta3)
		# self.dpOPLS = np.sqrt((self.domegaOPLS1/Omega1*np.tan(Theta1))**2 + (self.domegaOPLS3/Omega3*np.tan(Theta3))**2)
		self.dpOPLS = np.sqrt( \
			(self.domegaOPLS1/Omega1*np.tan(Theta1))**2 + (self.domegaOPLS3/Omega3*np.tan(Theta3))**2 + \
			(self.omegaOPLS1*dOmega1*(Theta1/np.cos(Theta1)**2 - np.tan(Theta1))/Omega1**2)**2 + \
			(self.omegaOPLS3*dOmega3*(Theta3/np.cos(Theta3)**2 - np.tan(Theta3))/Omega3**2)**2)

	########### End of Systematics.OnePhotonLightShift() ############
	#################################################################

	def TwoPhotonLightShift(self, iax, ks):
		"""Compute the two-photon light shift on a three-pulse atom interferometer.
		The two-photon light shift varies linearly with power and inversely with time-of-flight
		due to the Doppler shift. In the limit where the Doppler shift is much larger than the Rabi
		frequency and the recoil frequency, the two-photon frequency shift is given by: 
			fTPLS(P,t) = -Omegaeff(P)**2/(4*omegaD(t)),
		where omegaD(t) = keff*(v0 + g*t) is the Doppler shift.
		"""

		self.omegaTPLS1, self.domegaTPLS1, Omega1, dOmega1 = self.TwoPhotonLightShiftModel(iax, ks, self.t1[iax])[:4]
		self.omegaTPLS3, self.domegaTPLS3, Omega3, dOmega3 = self.TwoPhotonLightShiftModel(iax, ks, self.t3[iax])[:4]

		Theta1 = 0.5*Omega1*self.tau1[iax]
		Theta3 = 0.5*Omega3*self.tau3[iax]

		## Phase shift and uncertainty
		self.pTPLS  = self.omegaTPLS1/Omega1*np.tan(Theta1) - self.omegaTPLS3/Omega3*np.tan(Theta3)
		self.dpTPLS = np.sqrt( \
			(self.domegaTPLS1/Omega1*np.tan(Theta1))**2 + (self.domegaTPLS3/Omega3*np.tan(Theta3))**2 + \
			(self.omegaTPLS1*dOmega1*(Theta1/np.cos(Theta1)**2 - np.tan(Theta1))/Omega1**2)**2 + \
			(self.omegaTPLS3*dOmega3*(Theta3/np.cos(Theta3)**2 - np.tan(Theta3))/Omega3**2)**2)

	########### End of Systematics.TwoPhotonLightShift() ############
	#################################################################

	def QuadraticZeemanShiftNum(self, iax, ks):
		"""Numerical computation of the phase shift due to the 2nd-order Zeeman shift by
		numerically integrating the sensitivity function with BModel(t)^2.
		"""

		t1L = self.t1[iax]
		t1R = t1L + self.tau1[iax]
		t2L = t1R + self.T[iax]
		t2R = t2L + self.tau2[iax]
		t3L = t2R + self.T[iax]
		t3R = t3L + self.tau3[iax]

		gkws   = {'t0': self.t1[iax], 'T': self.T[iax], 'tau1': self.tau1[iax]}
		epsabs = 1.E-06
		epsrel = 1.E-06
		limit  = 100

		f   = lambda t: self.gSensitivity(t, **gkws)*(self.BModel(iax, ks, t)[0])**2
		I1  = integrate.quad(f, t1L, t1R, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		I2  = integrate.quad(f, t1R, t2L, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		I3  = integrate.quad(f, t2L, t2R, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		I4  = integrate.quad(f, t2R, t3L, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		I5  = integrate.quad(f, t3L, t3R, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

		df  = lambda t: 2*self.gSensitivity(t, **gkws)*abs(self.BModel(iax, ks, t)[0])*self.BModel(iax, ks, t)[1]
		dI1 = integrate.quad(df, t1L, t1R, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		dI2 = integrate.quad(df, t1R, t2L, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		dI3 = integrate.quad(df, t2L, t2R, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		dI4 = integrate.quad(df, t2R, t3L, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		dI5 = integrate.quad(df, t3L, t3R, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]

		self.pQZ  = 2*np.pi*self.KClock*(I1 + I2 + I3 + I4 + I5)
		self.dpQZ = 2*np.pi*self.KClock*abs(dI1 + dI2 + dI3 + dI4 + dI5)

		# self.pQZ  = 2.*np.pi*self.KClock*(I2 + I4)
		# self.dpQZ = 2.*np.pi*self.KClock*abs(dI2 + dI4)

	######### End of Systematics.QuadraticZeemanShiftNum() ##########
	#################################################################

	def QuadraticZeemanShift(self, iax, ks):
		"""Analytical computation of the phase shift due to the 2nd-order Zeeman shift in the presence of
		a spatially-varying B-field according the following model:
			B(t)   = beta0 + beta1*(zk + vk*t + 0.5*a*t**2) + 0.5*beta2*(zk + vk*t + 0.5*a*t**2)**2
		In the absence of a curvature (beta2=0):
			B^2(t) = beta0**2
				   + 2*beta0*beta1*(zk + vk*t + 0.5*a*t**2)
				   + beta1**2*(zk**2 + 2*zk*vk*t + (zk*a + vk**2)*t**2 + vk*a*t**3 + 0.25*a^2*t**4)
		Here, t is the true time-of-flight relative to molasses release, t1 is the time of the first Raman pulse, 
		zk = z0 + v0*t1 is the position of the atoms at the first pulse minus the gravitational part,
		vk = v0 + 0.5*ks*vR is the velocity at the first pulse minus the gravitational part.
		The gravitational term is common to the entire trajectory, so its contribution is accounted for automatically
		in the integrals of the sensitivity function.
		"""

		t1      = self.t1[iax]
		T       = self.T[iax]
		tau1    = self.tau1[iax]
		Omega1  = self.RabiModel(iax, ks, t1)[0]

		zk 		= self.r0[iax,0] + self.v0[iax,0]*t1 	## Position after 1st pulse minus gravity (m)
		vk      = self.v0[iax,0] + 0.5*ks*self.vR 		## Velocity after 1st pulse minus gravity (m/s)
		dvk     = self.v0[iax,1]						## Velocity uncertainty (m/s)
		a       = self.aBody[iax]						## Body acceleration (m/s^2)

		K       = 2.*np.pi*self.KClock  				## Clock shift parameter (rad/s/G^2)
		beta0   = abs(self.beta0[iax,0])				## B-field offset (G)
		dbeta0  = self.beta0[iax,1]
		beta1   = self.beta1[iax,0]						## B-field gradient (G/m)
		dbeta1  = self.beta1[iax,1]
		beta2   = self.beta2[iax,0]						## B-field curvature (G/m^2)
		dbeta2  = self.beta2[iax,1]

		G0, G1, G2, G3, G4 = self.gPolyIntegrals(t1, T, tau1, Omega1)

		ZT1     = vk*G1 + 0.5*a*G2
		dZT1    = dvk*G1
		ZT2     = 2.*zk*vk*G1 + vk**2*G2 + vk*a*G3 + 0.25*a**2*G4
		dZT2    = dvk*(2.*zk*G1 + 2.*vk*G2 + a*G3)

		## Shift due to constant parts of B**2
		pbeta0  = K*G0*(beta0**2 + 2.*beta0*beta1*zk + beta1**2*zk**2)
		dpbeta0 = K*G0*np.sqrt((2.*dbeta0*beta0)**2 + (2.*dbeta0*beta1*zk)**2 + (2.*beta0*dbeta1*zk)**2 + (2.*dbeta1*beta1*zk**2)**2)

		## Shift due to B-gradient
		pbeta1  = K*(2.*beta0*beta1*ZT1 + beta1**2*ZT2)
		dpbeta1 = K*np.sqrt((2.*dbeta0*beta1*ZT1)**2 + (2.*dbeta1*(beta0*ZT1 + beta1*ZT2))**2 + (2.*beta0*beta1*dZT1 + beta1**2*dZT2)**2)

		## Shift due to B-curvature
		pbeta2  = K*beta0*beta2*ZT2
		dpbeta2 = K*np.sqrt((dbeta0*beta2*ZT2)**2 + (beta0*dbeta2*ZT2)**2 + (beta0*beta2*dZT2)**2)

		## Total phase shift due to quadratic Zeeman effect
		self.pQZ  = pbeta0 + pbeta1 + pbeta2
		self.dpQZ = np.sqrt(dpbeta0**2 + dpbeta1**2 + dpbeta2**2)

	########### End of Systematics.QuadraticZeemanShift() ###########
	#################################################################

	def MagneticForceShift(self, iax, ks):
		"""Compute the phase shift due to the magnetic force from a spatially-varying B-field
		with a gradient (beta1).
		"""

		vCOM, dvCOM = self.TrajectoryModel(iax, ks, self.t1[iax])[[2,3]]

		## Shift due to the force of the magnetic gradient on the clock state
		self.pBF  = -(2./3.)*ks*self.keff*self.Lambda*self.beta1[iax,0]**2*(vCOM*self.T[iax] + self.aBody[iax]*self.T[iax]**2)*self.T[iax]**2
		self.dpBF =  (2./3.)*self.keff*self.Lambda*np.sqrt((2.*self.beta1[iax,1]*self.beta1[iax,0]*(vCOM*self.T[iax] + self.aBody[iax]*self.T[iax]**2))**2 \
			+ (self.beta1[iax,0]**2*dvCOM*self.T[iax])**2)*self.T[iax]**2

	############ End of Systematics.MagneticForceShift() ############
	#################################################################

	def GravityGradientShift(self, iax, ks):
		"""Compute the gravity gradient shift on a three-pulse atom interferometer."""

		vCOM, dvCOM = self.TrajectoryModel(iax, ks, self.t1[iax])[[2,3]]

		## Shift due to the gravity gradient
		self.pGG  = ks*self.keff*self.TBody[iax,iax]*(vCOM*self.T[iax] + (7./12.)*self.aBody[iax]*self.T[iax]**2)*self.T[iax]**2
		self.dpGG = np.sqrt((self.dTBody[iax,iax]/self.TBody[iax,iax]*self.pGG)**2 + (self.keff*self.TBody[iax,iax]*dvCOM*self.T[iax]**3)**2)

	########### End of Systematics.GravityGradientShift() ###########
	#################################################################

	def CoriolisShift(self, iax, ks):
		"""Compute the Coriolis shift on a three-pulse atom interferometer."""

		vCOM, dvCOM = self.TrajectoryModel(iax, ks, self.t1[iax])[[2,3]]

		k       = np.zeros(3)
		v       = np.array(self.v0[:,0] + self.aBody*self.T[iax])
		dv      = np.array(self.v0[:,1])

		k[iax]  = self.keff
		v[iax]  = vCOM + self.aBody[iax]*self.T[iax]
		dv[iax] = dvCOM

		kXv     = np.cross(k, v)
		dkXv    = np.cross(k, dv)

		## Shift due to the Coriolis effect
		self.pCor  = -2.*ks*np.dot(kXv, self.OmegaBody)*self.T[iax]**2
		self.dpCor =  2.*np.sqrt(np.dot(dkXv**2, self.OmegaBody**2) + np.dot(kXv**2, self.dOmegaBody**2))*self.T[iax]**2

	############## End of Systematics.CoriolisShift() ###############
	#################################################################

	def InitializeParasiticLines(self, iax, ks):
		"""Initialize parameters for computing Rabi frequency and phase shift due to parasitic laser lines.
		Parameters values are associated with a specific EOM type."""

		OmegaL = self.DetuningModel(iax, ks, self.t1[iax])[10]

		if self.SysOpts['PL_EOMType'] == 'PM':
			## RF frequency injected in phase modulator (rad/s)
			# if ks == 1.:
			# 	OmegaPM = self.deltakU[iax]
			# else:
			# 	OmegaPM = self.deltakD[iax]
			
			OmegaPM = OmegaL

			## List of parasitic line indices
			NList   = np.arange(0,0+1)
			MList   = np.arange(-4,4+1)
			nN 		= len(NList)
			nM 		= len(MList)
			N0 		= int(nN/2)
			M0 		= int(nM/2)
			NLoop   = list(range(N0,N0+1))
			MLoop   = list(range(-M0-1,M0-1+1))
			NOffset = [0,0]

			## List of RF frequencies
			self.OmegaRF = np.array([[M*OmegaPM for M in MList] for N in NList])

			## Relative electric field amplitudes
			E          = np.zeros((nN, nM))
			E[N0,M0-3] = 10**(-30.0/20)	## -20.4 GHz
			E[N0,M0-2] = 10**(-12.0/20)	## -13.6 GHz
			E[N0,M0-1] = 10**( -2.0/20)	##  -6.8 GHz
			E[N0,M0  ] = 10**(  0.0/20)	##   0.0 GHz
			E[N0,M0+1] = 10**( -2.0/20)	##  +6.8 GHz
			E[N0,M0+2] = 10**(-12.0/20)	## +13.6 GHz
			E[N0,M0+3] = 10**(-30.0/20)	## +20.4 GHz

			beta  = 0.625
			## Model of electric field amplitude
			APPLN = lambda N,M,b: jv(M,2*b)

			## Model of electric field phase (minus control phase and overall sign)
			## This phase cancels for all lines resonant with the Raman transition (M' = M)
			# pPPLN = lambda N,M: M*np.pi/2

			## Sign of electric field amplitude
			sPPLN = lambda N,M: np.sign(APPLN(N,M,beta))

		else: ## self.SysOpts['PL_EOMType'] == 'IQ'

			## RF frequencies injected in IQ modulator (rad/s)
			OmegaIQ1 = 2.*np.pi*0.955E9
			# if ks == 1.:
			# 	OmegaIQ2 = OmegaIQ1 + self.deltakU[iax]
			# else:
			# 	OmegaIQ2 = OmegaIQ1 + self.deltakD[iax]

			OmegaIQ2 = OmegaIQ1 + OmegaL

			## List of parasitic line indices
			NList   = np.arange(-4,4+1)
			MList   = np.arange(-2,2+1)
			nN 		= len(NList)
			nM 		= len(MList)
			N0 		= int(nN/2)
			M0 		= int(nM/2)
			NLoop   = list(range(-N0-2,N0-2+1))
			MLoop   = list(range(-M0-1,M0-1+1))
			NOffset = [2,1]

			## List of RF frequencies 
			self.OmegaRF = np.array([[N*OmegaIQ1 + M*OmegaIQ2 for M in MList] for N in NList])

			## Relative electric field amplitudes
			E   	     = np.zeros((nN, nM))
			E[N0-4,M0-1] = 10**(-34.5/20)	## -11.8 GHz
			E[N0-4,M0  ] = 10**(-40.0/20)	##  -4.0 GHz
			E[N0-3,M0-1] = 10**(-44.0/20)	## -10.8 GHz
			E[N0-3,M0  ] = 10**(-37.0/20)	##  -3.0 GHz
			E[N0-3,M0+1] = 10**(-29.7/20)	##  +4.8 GHz
			E[N0-2,M0-1] = 10**(-34.5/20)	##  -9.8 GHz
			E[N0-2,M0  ] = 10**(-26.4/20)	##  -2.0 GHz
			E[N0-2,M0+1] = 10**(-37.0/20)	##  +5.8 GHz
			E[N0-1,M0-1] = 10**(-25.0/20)	##  -8.8 GHz
			E[N0-1,M0  ] = 10**(-32.0/20)	##  -1.0 GHz
			E[N0-1,M0+1] = 10**(-21.6/20)	##  +6.8 GHz
			E[N0  ,M0-2] = 10**(-43.0/20)	## -15.6 GHz
			E[N0  ,M0-1] = 10**(-35.5/20)	##  -7.8 GHz
			E[N0  ,M0  ] = 10**(-20.9/20)	##   0.0 GHz
			E[N0  ,M0+1] = 10**(-24.3/20)	##  +7.8 GHz
			E[N0  ,M0+2] = 10**(-19.8/20)	## +15.6 GHz
			E[N0+1,M0-1] = 10**(-38.0/20)	##  -6.8 GHz
			E[N0+1,M0  ] = 10**(-17.4/20)	##  +1.0 GHz
			E[N0+1,M0+1] = 10**(-1.97/20)	##  +8.8 GHz, primary line 2
			E[N0+2,M0-1] = 10**(-33.0/20)	##  -5.8 GHz
			E[N0+2,M0  ] = 10**(  0.0/20)	##  +2.0 GHz, primary line 1
			E[N0+2,M0+1] = 10**(-22.5/20)	##  +9.8 GHz 
			E[N0+3,M0-1] = 10**(-19.7/20)	##  -4.8 GHz
			E[N0+3,M0  ] = 10**(-29.7/20)	##  +3.0 GHz
			E[N0+3,M0+1] = 10**(-26.0/20)	## +10.8 GHz
			E[N0+4,M0  ] = 10**(-28.6/20)	##  +4.0 GHz
			E[N0+4,M0+1] = 10**(-21.5/20)	## +11.8 GHz

			beta1 = np.pi*0.55
			beta2 = np.pi*0.23

			## Model of electric field amplitude
			APPLN = lambda N,M,b1,b2: np.sum([[np.cos((n+m+1)*np.pi/2.)*np.cos((N+M-n-m+1)*np.pi/2.)*np.cos((n+m-1)*np.pi/4.)*np.cos((N+M-n-m-1)*np.pi/4.) \
				* jv(n,b1)*jv(N-n,b1)*jv(m,b2)*jv(M-m,b2) for m in range(-5,5)] for n in range(-5,5)])

			## Model of electric field phase (minus control phases and overall sign)
			## This phase cancels for all lines resonant with the Raman transition (N' = N-1, M' = M+1) 
			# pPPLN = lambda N,M: -(N+M)*np.pi/4

			## Sign of electric field amplitude
			sPPLN = lambda N,M: np.sign(APPLN(N,M,beta1,beta2))

		self.EPPLN = np.array([[sPPLN(N,M)*E[N0+N,M0+M] for M in MList] for N in NList])
		# self.EPPLN = np.array([[E[N0+N,M0+M] for M in MList] for N in NList])

		return [NList, MList, NLoop, MLoop, nN, nM, N0, M0, NOffset]

	######## End of Systematics.InitializeParasiticLines() ##########
	#################################################################

	def ParasiticLinesShift(self, iax, ks):
		"""Compute the phase shift due to parasitic laser lines for a specific EOM type."""

		NLoop, MLoop, nN, nM, N0, M0, NOffset = self.InitializeParasiticLines(iax, ks)[2:]

		Deltak	= ks*(2./self.cLight)*self.OmegaRF			## List of effective wavevectors (rad/m)
		delta31 = 2.*np.pi*424.60*1.E+6						## Hyperfine splitting (F' = 1 <-> 3) (rad/s)
		delta32 = 2.*np.pi*266.65*1.E+6						## Hyperfine splitting (F' = 2 <-> 3) (rad/s)
		Delta1  = self.Delta + delta31 + self.OmegaRF		## Raman detuning from F = 2 -> 1' (rad/s)
		Delta2  = self.Delta + delta32 + self.OmegaRF		## Raman detuning from F = 2 -> 2' (rad/s)

		# if self.SysOpts['Print']:
		# 	print('---------------- iax = {}, ks = {} ----------------'.format(iax, int(ks)))
		# 	print('N\tM\tINM (dB)\tsign\tDel1\tDel2\tFreq (GHz)\tPeriod (mm)')

		I00 = self.EPPLN[N0+NOffset[0],M0]*self.EPPLN[N0+NOffset[1],M0+1]*(1./abs(Delta2[N0,M0]) + 1./(3.*abs(Delta1[N0,M0])))
		INM = np.zeros((nN, nM))

		for N in NLoop:
			for M in MLoop:
				INM[N0+N,M0+M] = self.EPPLN[N0+NOffset[0]+N,M0+M]*self.EPPLN[N0+NOffset[1]+N,M0+M+1] * \
					(1./abs(Delta2[N0+N,M0+M]) + 1./(3.*abs(Delta1[N0+N,M0+M])))/I00

				# if self.SysOpts['Print']:
				# 	if abs(INM[N0+N,M0+M]) > 1.E-3:
				# 		print('{}\t{}\t{:.1f}\t\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\t\t{:.2f}'.format(N, M, 10.*np.log10(abs(INM[N0+N,M0+M])), \
				# 			np.sign(INM[N0+N,M0+M]), Delta1[N0+N,M0+M]/(2*np.pi*1.E9), Delta2[N0+N,M0+M]/(2*np.pi*1.E9), \
				# 			self.OmegaRF[N0+N,M0+M]/(2*np.pi*1.E9), \
				# 			self.cLight/abs(self.OmegaRF[N0+N,M0+M]/np.pi)*1.E3 if abs(self.OmegaRF[N0+N,M0+M]) > 0 else np.inf))

		rCOM, drCOM, vCOM, dvCOM = self.TrajectoryModel(iax, ks, self.t1[iax])[:4]

		zA  = rCOM - self.zM[iax,0]
		zB  = zA + vCOM*(self.T[iax] + self.tau2[iax]) + 0.5*self.aBody[iax]*(self.T[iax] + self.tau2[iax])**2
		zC  = zA + (vCOM + ks*self.vR)*(self.T[iax] + self.tau2[iax]) + 0.5*self.aBody[iax]*(self.T[iax] + self.tau2[iax])**2
		zD  = zA + (vCOM + 0.5*ks*self.vR)*(self.Ttotal[iax]) + 0.5*self.aBody[iax]*(self.Ttotal[iax])**2

		dzA = np.sqrt(drCOM**2 + self.zM[iax,1]**2)
		dzB = np.sqrt(dzA**2 + (dvCOM*(self.T[iax] + self.tau2[iax]))**2)
		dzC = np.sqrt(dzA**2 + (dvCOM*(self.T[iax] + self.tau2[iax]))**2)
		dzD = np.sqrt(dzA**2 + (dvCOM*self.Ttotal[iax])**2)

		drI = self.IRatio[1]/self.IRatio[0] ## Relative uncertainty in Rabi frequencies (unitless)

		def phiPL(z, dz):
			c   = (1./I00)*np.sum([[INM[N0+N,M0+M]*np.exp(1j*Deltak[N0+N,M0+M]*z) for N in NLoop] for M in MLoop])
			a   = np.real(c)
			b   = np.imag(c)
			da  = (1./I00)*np.sqrt(np.sum([[(drI*INM[N0+N,M0+M]*np.cos(Deltak[N0+N,M0+M]*z))**2 + \
				(INM[N0+N,M0+M]*Deltak[N0+N,M0+M]*np.sin(Deltak[N0+N,M0+M]*z)*dz)**2 for N in NLoop] for M in MLoop]))
			db  = (1./I00)*np.sqrt(np.sum([[(drI*INM[N0+N,M0+M]*np.sin(Deltak[N0+N,M0+M]*z))**2 + \
				(INM[N0+N,M0+M]*Deltak[N0+N,M0+M]*np.cos(Deltak[N0+N,M0+M]*z)*dz)**2 for N in NLoop] for M in MLoop]))

			phi  = np.angle(c)
			dphi = np.abs(np.sqrt((b*da)**2 + (a*db)**2)/c**2)
			
			return phi, dphi

		phiA, dphiA = phiPL(zA, dzA)
		phiB, dphiB = phiPL(zB, dzB)
		phiC, dphiC = phiPL(zC, dzC)
		phiD, dphiD = phiPL(zD, dzD)

		self.pPL  = phiA - phiB - phiC + phiD
		self.dpPL = np.sqrt(dphiA**2 + dphiB**2 + dphiC**2 + dphiD**2)

	########### End of Systematics.ParasiticLinesShift() ############
	#################################################################

	def WavefrontCurvatureShift(self, iax, ks):
		"""Compute the phase shift due to wavefront curvature."""

		# ## Mirror radius of curvature due to non-zero surface flatness (m)
		# RM   = self.DM[0]**2/(2.*self.FlatM[0])
		# ## Uncertainty in radius of curavature (m)
		# dRM  = np.sqrt((2.*self.DM[1]/self.DM[0])**2 + (self.FlatM[1]/self.FlatM[0])**2)*RM

		sigv = np.sqrt(2.*self.kBoltz*self.Temp[iax,0]/self.MRb)
		# self.pWC  = 0.5*ks*self.keff*(sigv*self.Teff[iax])**2/RM
		# self.dpWC = np.sqrt((self.Temp[iax,1]/self.Temp[iax,0])**2 + (dRM/RM)**2)*abs(self.pWC)

		zR, dzR = self.zR[iax]				## Rayleigh range of Raman beams (m)
		zC, dzC = self.zC[iax]				## Distance of atoms from collimator (m)
		zM, dzM = self.zM[iax]				## Distance of atoms from mirror (m)

		z   = abs(zC) + abs(zM)				## Distance to evaluate the wavefront curvature (m)
		dz  = np.sqrt(dzC**2 + dzM**2)		## Uncertainty in z (m)

		R   = z*(1 + (zR/z)**2)				## Wavefront curvature assuming focus at lense (m)
		dR  = np.sqrt(((1. - (zR/z)**2)*dz)**2 + (2.*zR/z*dzR)**2)

		t   = np.array([self.t1[iax], self.t2[iax], self.t3[iax]])
		phi = np.zeros(3)

		for i in range(3):
			## Get cloud position in the body frame at each pulse time
			p = np.zeros(3) ## iaxx
			for iaxx in range(3):
				p[iaxx] = self.TrajectoryModel(iaxx, 0., t[i])[0]

			## Project cloud position into the plane of beam iax
			if iax == 0:
				r = np.sqrt(p[1]**2 + p[2]**2)
			elif iax == 1:
				r = np.sqrt(p[2]**2 + p[0]**2)
			else: ## iax == 2:
				r = np.sqrt(p[0]**2 + p[1]**2)

			phi[i] = 0.5*ks*self.keff/R*((sigv*t[i])**2 + r**2)

		self.pWC  = phi[0] - 2*phi[1] + phi[2]
		# self.pWC  = ks*self.keff*(sigv*self.Teff[iax])**2/R
		self.dpWC = np.sqrt((self.Temp[iax,1]/self.Temp[iax,0])**2 + (dR/R)**2)*abs(self.pWC)

	######### End of Systematics.WavefrontCurvatureShift() ##########
	#################################################################

	def MachZehnderAsymmetryShift(self, iax, ks):
		"""Compute the phase shift due to the asymmetry of the Mach-Zehnder interferometer."""

		omegaL1, domegaL1 = self.DetuningModel(iax, ks, self.t1[iax])[[10,11]]
		omegaL3, domegaL3 = self.DetuningModel(iax, ks, self.t3[iax])[[10,11]]

		if ks == 1.:
			RamanL1 = self.deltakD[iax]
			RamanL3 = RamanL1 + self.alphakD[iax]*self.Ttotal[iax]
		elif ks == -1.:
			RamanL1 = self.deltakU[iax]
			RamanL3 = RamanL1 + self.alphakU[iax]*self.Ttotal[iax]
		else:
			RamanL1 = self.omegaHF
			RamanL3 = RamanL1

		delta1  = RamanL1 - omegaL1
		delta3  = RamanL3 - omegaL3
		ddelta1 = domegaL1
		ddelta3 = domegaL3

		# print('ks, omegaL1, RamanL1, delta1 = {}, {:.6E}, {:.6E}, {:.3E}'.format(ks, RamanL1/(2*np.pi), omegaL1/(2*np.pi), delta1/(2*np.pi)))
		# print('ks, omegaL3, RamanL3, delta3 = {}, {:.6E}, {:.6E}, {:.3E}'.format(ks, RamanL3/(2*np.pi), omegaL3/(2*np.pi), delta3/(2*np.pi)))

		omega1, domega1 = self.RabiModel(iax, ks, self.t1[iax])
		omega3, domega3 = self.RabiModel(iax, ks, self.t3[iax])

		Omega1 = np.sqrt(omega1**2 + delta1**2)
		Omega3 = np.sqrt(omega3**2 + delta3**2)
		dOmega1 = np.sqrt((omega1*domega1)**2 + (delta1*ddelta1)**2)/Omega1
		dOmega3 = np.sqrt((omega3*domega3)**2 + (delta3*ddelta3)**2)/Omega3

		Theta1 = Omega1*self.tau1[iax]/2.
		Theta3 = Omega3*self.tau3[iax]/2.

		self.pMZA  = delta3/Omega3*np.tan(Theta3) - delta1/Omega1*np.tan(Theta1)
		self.dpMZA = np.sqrt( \
			(ddelta1/Omega1*np.tan(Theta1))**2 + (ddelta3/Omega3*np.tan(Theta3))**2 + \
			(delta1*dOmega1*(Theta1/np.cos(Theta1)**2 - np.tan(Theta1))/Omega1**2)**2 + \
			(delta3*dOmega3*(Theta3/np.cos(Theta3)**2 - np.tan(Theta3))/Omega3**2)**2)

		# self.OnePhotonLightShift(iax, ks)
		# self.TwoPhotonLightShift(iax, ks)

		# if ks == 1:
		# 	omegaL1 = self.deltakU[iax]
		# elif ks == -1:
		# 	omegaL1 = self.deltakD[iax]
		# else:
		# 	omegaL1 = self.deltaSel

		# omegaD1  = omegaL1 - (self.omegaHF + self.omegaR + self.omegaOPLS1)
		# DeltaD1 = omegaD1 - self.omegaD1 	## True Doppler shift - input estimate

		# Deltav1 = delta1/(ks*self.keff) ## Velocity error due to light shifts (m/s)
		# sigv   = self.sigvD[iax,0]
		# vL     = -4.*sigv
		# vR     = +4.*sigv

		# omega  = lambda delta, Omega: np.sqrt(delta**2 + Omega**2)
		# sin    = lambda omega, tau: np.sin(omega*tau/2.)
		# cos    = lambda omega, tau: np.cos(omega*tau/2.)
		# tan    = lambda omega, tau: np.tan(omega*tau/2.)

		# def C(v):
		# 	"""Contrast of a Mach-Zehnder interferometer due to an asymmetry in Rabi frequencies."""

		# 	delta  = ks*self.keff*v
		# 	omega1 = omega(delta, Omega1)
		# 	omega2 = omega(delta, Omega2)
		# 	omega3 = omega(delta, Omega3)
		# 	s1     = sin(omega1, tau1)
		# 	s2     = sin(omega2, tau2)
		# 	s3     = sin(omega3, tau3)
		# 	c1     = cos(omega1, tau1)
		# 	c3     = cos(omega3, tau3)

		# 	return 4.*Omega1*Omega2**2*Omega3*s1*s2**2*s3/((omega1*omega2*omega3)**2)*(delta**2*s1*s3 + omega1*omega3*c1*c3)

		# def dCdv(v):
		# 	"""Derivative of contrast with respect to velocity."""

		# 	delta  = ks*self.keff*v
		# 	omega1 = omega(delta, Omega1)
		# 	omega2 = omega(delta, Omega2)
		# 	omega3 = omega(delta, Omega3)
		# 	s1     = sin(omega1, tau1)
		# 	s2     = sin(omega2, tau2)
		# 	s3     = sin(omega3, tau3)
		# 	c1     = cos(omega1, tau1)
		# 	c3     = cos(omega3, tau3)
		# 	t1     = tan(omega1, tau1)
		# 	t2     = tan(omega2, tau2)
		# 	t3     = tan(omega3, tau3)

		# 	dC1    = ks*self.keff*delta*C(delta)*(0.5*tau1/(omega1*t1) + tau2/(omega2*t2) + 0.5*tau3/(omega3*t3) \
		# 		- 2./omega1**2 - 2./omega2**2 - 2./omega3**2)
		# 	dC2    = ks*self.keff*delta*Omega1*Omega2**2*Omega3*s1*s2**2*s3/((omega1*omega2*omega3)**2) \
		# 		* (8.*s1*s3 - 2.*c1*s3/omega1*(Omega1**2*tau1) - 2.*s1*c3/omega3*(Omega3**2*tau3) + 4.*c1*c3*(omega1/omega3 + omega1/omega3))

		# 	return dC1 + dC2

		# def P(v):
		# 	"""Phase shift of a Mach-Zehnder interferometer due to an asymmetry in Rabi frequencies."""

		# 	delta  = ks*self.keff*v
		# 	omega1 = omega(delta, Omega1)
		# 	omega3 = omega(delta, Omega3)
		# 	t1     = tan(omega1, tau1)
		# 	t3     = tan(omega3, tau3)

		# 	return np.arctan(delta/omega3*t3) - np.arctan(delta/omega1*t1) + 0.5*delta*(tau1 - tau3)

		# def dPdv(v):
		# 	"""Derivative of phase with respect to delta."""

		# 	delta  = ks*self.keff*v
		# 	omega1 = omega(delta, Omega1)
		# 	omega3 = omega(delta, Omega3)
		# 	c1     = cos(omega1, tau1)
		# 	c3     = cos(omega3, tau3)
		# 	t1     = tan(omega1, tau1)
		# 	t3     = tan(omega3, tau3)

		# 	dP1dv  = ks*self.keff*(delta**2*tau1/(2.*c1**2) + Omega1**2/omega1*t1)/(omega1**2 + delta**2*t1**2)
		# 	dP3dv  = ks*self.keff*(delta**2*tau3/(2.*c3**2) + Omega3**2/omega3*t3)/(omega3**2 + delta**2*t3**2)

		# 	return dP3dv - dP1dv - 0.5*(tau1 - tau3)

		# epsabs = 1.E-6
		# epsrel = 1.E-6
		# limit  = 100
		# ## Phase and contrast averaging over velocity
		# f      = lambda v: np.exp(-(v/sigv)**2)/(np.sqrt(np.pi)*sigv)
		# fC     = lambda v: f(v)*C(v-Deltav1)
		# fCP    = lambda v: f(v)*C(v-Deltav1)*P(v-Deltav1)
		# c      = integrate.quad(fC, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]
		# phi    = integrate.quad(fCP, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/c

		# ## Phase uncertainty due to velocity
		# dfdv   = lambda v: (-2.*v/sigv**2)*f(v)
		# dphidv = lambda v: (dfdv(v)*C(v-Deltav1) + f(v)*dCdv(v-Deltav1))*P(v-Deltav1) + f(v)*C(v-Deltav1)*dPdv(v-Deltav1)
		# dphi   = abs(integrate.quad(dphidv, vL, vR, epsabs=epsabs, epsrel=epsrel, limit=limit)[0]/c)*self.v0[iax,1]

		# self.pMZA  = phi
		# self.dpMZA = dphi

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

	def ScaleFactorShift(self, iax, ks):
		"""Compute the phase shift due to imperfections in the interferometer scale factor.
		These expressions assume the laser frequency is jumped by 0.5*alpha*(T+2*tau) between laser pulses
		in a phase continuous manner, and the jumps are pretriggered symmetrically about the center of the interferometer.
		This frequency profile simulates a phase continuous frequency chirp of alpha, but creates a discrepancy between
		the scale factors corresponding to the kinetic and laser components of the phase shift."""

		tau1 = self.tau1[iax]
		tau2 = self.tau2[iax]
		tau3 = self.tau3[iax]
		T    = self.T[iax]

		Omega1, dOmega1 = self.RabiModel(iax, ks, self.t1[iax])
		# Omega2, dOmega2 = self.RabiModel(iax, ks, self.t2[iax])
		Omega3, dOmega3 = self.RabiModel(iax, ks, self.t3[iax])

		Theta1    = 0.5*Omega1*tau1
		# Theta2    = 0.5*Omega2*tau2
		Theta3    = 0.5*Omega3*tau3
		DeltaT2   = (T + tau2)*(np.tan(Theta3)/Omega3 - np.tan(Theta1)/Omega1)
		dDeltaT2  = (T + tau2)*(np.sqrt( \
			(Theta1/(Omega3*np.cos(Theta3))**2 - np.tan(Theta3)/Omega3**2)**2*dOmega3**2 + \
			(Theta3/(Omega1*np.cos(Theta1))**2 - np.tan(Theta1)/Omega1**2)**2*dOmega1**2))

		if self.SysOpts['RTFreq'] == 'On':
			alpha = -ks*self.keff*self.aBody[iax] ## Chirp rate (rad/s)
		else: ## self.SysOpts['RTFreq'] == 'Off':
			alpha = -ks*self.keff*self.aLocal[iax] ## Chirp rate (rad/s)

		## Scale factor phase shift
		self.pSF  = alpha*DeltaT2
		self.dpSF = abs(alpha)*dDeltaT2

	############## End of Systematics.ScaleFactorShift() ############
	#################################################################

	def NonLinearityShift(self, iax, ks):
		"""Compute the phase shift due to the non-linearity of the RF chain. The non-linearity was calibrated
		and fit to a 7th-order polynomial. This routine converts the calibration into a phase using analytical
		integrals of the sensitivity function."""

		omega0 = 2.*np.pi*6.834E9 ## Offset frequency (rad/s)
		omegaL = self.DetuningModel(iax, ks, self.t1[iax])[10] ## Raman laser frequency (rad/s)
		f0     = (omegaL - omega0)/(2.E6*np.pi) ## Start frequency (MHz)

		if self.SysOpts['RTFreq'] == 'On':
			alpha = ks*self.keff*self.aBody[iax]/(2.E6*np.pi) ## Chirp rate (MHz/s)
		else: ## self.SysOpts['RTFreq'] == 'Off':
			alpha = ks*self.keff*self.aLocal[iax]/(2.E6*np.pi) ## Chirp rate (MHz/s)

		beta = alpha/f0

		## RF non-linearity fit function (f between -6 to +6 MHz):
		## y = np.sum([c[n] f^n for n in range(7)]) (returns y in rad)
		## Fit parameters
		# c0: -1.99453736 +/- 1.8745e-04 (0.01%) (init = -1.994537)
		# c1:  0.35320970 +/- 1.0526e-04 (0.03%) (init = 0.3532097)
		# c2:  0.01782360 +/- 6.0050e-05 (0.34%) (init = 0.0178236)
		# c3:  0.00311870 +/- 1.0894e-05 (0.35%) (init = 0.0031187)
		# c4:  2.8362e-04 +/- 4.3369e-06 (1.53%) (init = 0.0002836242)
		# c5: -4.5948e-05 +/- 2.5132e-07 (0.55%) (init = -4.594789e-05)
		# c6: -8.2253e-06 +/- 8.2416e-08 (1.00%) (init = -8.225258e-06)

		c    = np.empty((7,2))
		c[0] = np.array([-1.99453736, 1.8745E-04])
		c[1] = np.array([+0.35320970, 1.0526E-04])
		c[2] = np.array([+0.01782360, 6.0050E-05])
		c[3] = np.array([+0.00311870, 1.0894E-05])
		c[4] = np.array([+2.8362E-04, 4.3369E-06])
		c[5] = np.array([-4.5948E-05, 2.5132E-07])
		c[6] = np.array([-8.2253E-06, 8.2416E-08])

		Omega1, dOmega1 = self.RabiModel(iax, ks, self.t1[iax])
		eps  = 1.E-3
		dG   = self.dgPolyIntegrals(self.T[iax], self.tau1[iax], Omega1)
		dGp  = self.dgPolyIntegrals(self.T[iax], self.tau1[iax], (1. + eps)*Omega1)
		dGdO = (dGp - dG)/(eps*Omega1)

		N     = len(dG)
		phi   = np.array([c[n,0]*f0**n*np.sum([dG[k]*binom(n,k)*beta**k for k in range(n+1)]) for n in range(N)])
		dphi1 = np.array([phi[n]*c[n,1]/c[n,0] for n in range(N)])
		dphi2 = np.array([c[n,0]*f0**n*np.sqrt(np.sum([(dGdO[k]*dOmega1*binom(n,k)*beta**k)**2 for k in range(n+1)])) for n in range(N)])

		self.pNL  = np.sum(phi)
		self.dpNL = np.sqrt(np.sum(dphi1**2 + dphi2**2))

	############# End of Systematics.NonLinearityShift() ############
	#################################################################

	def TotalShift(self, iax, ks):
		"""For selected systematic effects, compute the total systematic shift."""

		self.RamanFreqShifts(iax, ks)

		if 'All' in self.SysOpts['SysList']:
			self.OnePhotonLightShift(iax, ks)
			self.TwoPhotonLightShift(iax, ks)
			self.QuadraticZeemanShift(iax, ks)
			# self.QuadraticZeemanShiftNum(iax, ks)
			self.MagneticForceShift(iax, ks)
			self.GravityGradientShift(iax, ks)
			self.CoriolisShift(iax, ks)
			self.WavefrontCurvatureShift(iax, ks)
			self.ParasiticLinesShift(iax, ks)
			self.ScaleFactorShift(iax, ks)
			self.NonLinearityShift(iax, ks)
			# self.MachZehnderAsymmetryShift(iax, ks)

			self.pTot  = self.pOPLS + self.pTPLS + self.pQZ + self.pBF + self.pGG + \
				self.pCor + self.pPL + self.pWC + self.pNL + self.pSF# + self.pMZA 
			self.dpTot = np.sqrt(self.dpOPLS**2 + self.dpTPLS**2 + self.dpQZ**2 + self.dpBF**2 + self.dpGG**2 + \
				self.dpCor**2 + self.dpPL**2 + self.dpWC**2 + self.dpNL**2 + self.dpSF**2)# + self.dpMZA**2)
		else:
			self.pTot  = 0.
			self.dpTot = 0.
			for sysType in self.SysOpts['SysList']:
				if sysType == 'OPLS':
					self.OnePhotonLightShift(iax, ks)
				elif sysType == 'TPLS':
					self.TwoPhotonLightShift(iax, ks)
				elif sysType == 'QZ':
					self.QuadraticZeemanShift(iax, ks)
					# self.QuadraticZeemanShiftNum(iax, ks)
				elif sysType == 'BF':
					self.MagneticForceShift(iax, ks)
				elif sysType == 'GG':
					self.GravityGradientShift(iax, ks)
				elif sysType == 'Cor':
					self.CoriolisShift(iax, ks)
				elif sysType == 'WC':
					self.WavefrontCurvatureShift(iax, ks)
				elif sysType == 'PL':
					self.ParasiticLinesShift(iax, ks)
				elif sysType == 'MZA':
					self.MachZehnderAsymmetryShift(iax, ks)
				elif sysType == 'SF':
					self.ScaleFactorShift(iax, ks)
				elif sysType == 'NL':
					self.NonLinearityShift(iax, ks)

				self.pTot  += getattr(self, 'p'+sysType)
				self.dpTot += getattr(self, 'dp'+sysType)**2

			self.dpTot = np.sqrt(self.dpTot)

	################ End of Systematics.TotalShift() ################
	#################################################################

	def SetSystematicList(self):
		"""Set list of selected systematics based on option 'SysList'."""

		if 'All' in self.SysOpts['SysList']:
			# self.SysPhases = ['pOPLS', 'pTPLS', 'pQZ', 'pBF', 'pGG', 'pCor', 'pPL', 'pWC', 'pNL', 'pSF', 'pMZA', 'pTot']
			self.SysPhases = ['pSF', 'pWC', 'pTPLS', 'pPL', 'pOPLS', 'pCor', 'pQZ', 'pNL', 'pTot']
			self.SysVars   = ['omegaOPLS1', 'omegaOPLS3', 'omegaTPLS1', 'omegaTPLS3'] + self.SysPhases
		else:
			self.SysPhases = []
			self.SysVars   = []
			for sysType in self.SysOpts['SysList'] + ['Tot']:
				if sysType == 'OPLS':
					self.SysVars += ['omegaOPLS1', 'omegaOPLS3']
				elif sysType == 'TPLS':
					self.SysVars += ['omegaTPLS1', 'omegaTPLS3']
				self.SysPhases += ['p'+sysType]
			self.SysVars += self.SysPhases

		self.nSys  = len(self.SysPhases)
		self.SysDF = [pd.DataFrame([], columns=['Values', 'Errors', 'Values_Scaled', 'Errors_Scaled'], index=self.SysVars) for ik in range(4)]

	############### End of Systematics.SetSystematicList() ###############
	#################################################################

	def ComputeSystematics(self, iax):
		"""Compute systematics for k-cases (kU,kD,kInd,kDep) and fill values in a list of dataframes."""

		self.RotateToBodyFrame()
		self.SetAIParameters()

		for ik in range(0,4):
			if ik < 2:
				self.TotalShift(iax, float(1-2*ik))

				self.SysDF[ik]['Values'] = [getattr(self, var) for var in self.SysVars]
				self.SysDF[ik]['Errors'] = [getattr(self, 'd'+var) for var in self.SysVars]
			else:
				if ik == 2:
					self.SysDF[ik]['Values'] = 0.5*(self.SysDF[0]['Values'] + self.SysDF[1]['Values'])
					self.SysDF[ik]['Errors'] = 0.5*np.sqrt(self.SysDF[0]['Errors']**2 + self.SysDF[1]['Errors']**2 \
						+ 2.*self.kCorrCoeff[iax]*np.sign(self.SysDF[0]['Values'])*self.SysDF[0]['Errors']*np.sign(self.SysDF[1]['Values'])*self.SysDF[1]['Errors'])
				else:
					self.SysDF[ik]['Values'] = 0.5*(self.SysDF[0]['Values'] - self.SysDF[1]['Values'])
					self.SysDF[ik]['Errors'] = 0.5*np.sqrt(self.SysDF[0]['Errors']**2 + self.SysDF[1]['Errors']**2 \
						- 2.*self.kCorrCoeff[iax]*np.sign(self.SysDF[0]['Values'])*self.SysDF[0]['Errors']*np.sign(self.SysDF[1]['Values'])*self.SysDF[1]['Errors'])

			self.SysDF[ik]['Values_Scaled'] = self.SysDF[ik].copy()['Values']
			self.SysDF[ik]['Errors_Scaled'] = self.SysDF[ik].copy()['Errors']

		pmask = [s[0] == 'p' for s in self.SysVars]

		for ik in range(0,4):
			if self.SysOpts['Units'] == 'mrad':
				self.SysDF[ik]['Values_Scaled'].loc[pmask] /= 1.E-3
				self.SysDF[ik]['Errors_Scaled'].loc[pmask] /= 1.E-3
			elif self.SysOpts['Units'] == 'm/s^2':
				self.SysDF[ik]['Values_Scaled'].loc[pmask] /= self.Seff[iax]
				self.SysDF[ik]['Errors_Scaled'].loc[pmask] /= self.Seff[iax]
			elif self.SysOpts['Units'] == 'ug':
				self.SysDF[ik]['Values_Scaled'].loc[pmask] /= self.Seff[iax]*self.gLocal*1.E-6
				self.SysDF[ik]['Errors_Scaled'].loc[pmask] /= self.Seff[iax]*self.gLocal*1.E-6
			elif self.SysOpts['Units'] == 'Hz/s':
				self.SysDF[ik]['Values_Scaled'].loc[pmask] /= 2.*np.pi*self.Teff[iax]**2
				self.SysDF[ik]['Errors_Scaled'].loc[pmask] /= 2.*np.pi*self.Teff[iax]**2

	############ End of Systematics.ComputeSystematics() ############
	#################################################################

	def PrintSystematics(self, iax):
		"""Print table of selected systematic effects for a given axis."""

		logging.info('iXC_Sys::Printing systematics for iax = {}...'.format(iax))

		self.SetSystematicList()
		self.ComputeSystematics(iax)

		df = pd.DataFrame([], columns=['kU_Val', 'kU_Err', 'kD_Val', 'kD_Err', 'kInd_Val', 'kInd_Err', 'kDep_Val', 'kDep_Err'],
			index=self.SysPhases)
		pmask = [s[0] == 'p' for s in self.SysVars]

		for ik in range(4):
			df[self.ikLabels[ik]+'_Val'] = self.SysDF[ik]['Values_Scaled'].loc[pmask]
			df[self.ikLabels[ik]+'_Err'] = self.SysDF[ik]['Errors_Scaled'].loc[pmask]

		pd.set_option('display.max_rows', 12)
		pd.set_option('display.max_columns', 8)
		pd.set_option('display.expand_frame_repr', False)
		print('--------------------------------- {} Systematics ({}) ---------------------------------'.format(self.iaxLabels[iax], self.SysOpts['Units']))
		print(df)
		print('--------------------------------------------------------------------------------------')

		if self.SysOpts['Export']:
			self.ExportSystematics(iax, df)

		return df

	############# End of Systematics.PrintSystematics() #############
	#################################################################

	def ExportSystematics(self, iax, df):
		"""Export systematics table to file. Units of systematics is determined by SysOptions['Units']."""

		logging.info('iXC_Sys::Exporting systematics table for iax = {}...'.format(iax))

		fileName = self.SysOpts['FilePrefix']+'-'+self.iaxLabels[iax]+'.txt'
		filePath = os.path.join(self.SysOpts['Folder'], fileName)

		iXUtils.WriteDataFrameToFile(df, self.SysOpts['Folder'], filePath, True, True, '%5.3E')

	############# End of Systematics.ExportSystematics() ############
	#################################################################

	def GetSystematicLabels(self):
		"""Define labels for x,y,z-axes, legends and units according to the requested systematics."""

		if self.SysOpts['Plot'] == '1D':
			nD = 1
		else: ## self.SysOpts['Plot'] == '2D':
			nD = 2

		labels = [None for iD in range(2)]
		scales = [1.E0 for iD in range(2)]

		for iD in range(nD):
			if self.SysOpts['PlotVariables'][iD] == 'T':
				labels[iD] = r'$T$  (ms)'
				scales[iD] = 1.E3
			elif self.SysOpts['PlotVariables'][iD] == 'TOF':
				labels[iD] = r'$TOF$  (ms)'
				scales[iD] = 1.E3
			elif self.SysOpts['PlotVariables'][iD] == 'v0':
				labels[iD] = r'$v_0$  (mm/s)'
				scales[iD] = 1.E3
			elif self.SysOpts['PlotVariables'][iD] == 'zM':
				labels[iD] = r'$z_M$  (mm)'
				scales[iD] = 1.E3
			elif self.SysOpts['PlotVariables'][iD] == 'TiltX':
				labels[iD] = r'$\theta_x$  (deg)'
				scales[iD] = 1.E0
			elif self.SysOpts['PlotVariables'][iD] == 'TiltZ':
				labels[iD] = r'$\theta_z$  (deg)'
				scales[iD] = 1.E0
			else:
				labels[iD] = self.SysOpts['PlotVariables'][iD]
				scales[iD] = 1.

		[xLabel, yLabel] = labels
		[xScale, yScale] = scales

		if self.SysOpts['Units'] == 'rad':
			symbol = r'$\phi$'
			unit   = '  (rad)'
		elif self.SysOpts['Units'] == 'mrad':
			symbol = r'$\phi$'
			unit   = '  (mrad)'
		elif self.SysOpts['Units'] == 'ug':
			symbol = r'$a$'
			unit   = r'  ($\mu g$)'
		elif self.SysOpts['Units'] == 'm/s^2':
			symbol = r'$a$'
			unit   = r'  (m/s$^2$)'
		elif self.SysOpts['Units'] == 'Hz/s':
			symbol = r'$\alpha$'
			unit   = '  (Hz/s)'

		subscript = ['' for iSys in range(self.nSys)]
		
		for iSys in range(self.nSys):
			if self.SysPhases[iSys] == 'pOPLS':
				subscript[iSys] = r'$_{OPLS}$'
			elif self.SysPhases[iSys] == 'pTPLS':
				subscript[iSys] = r'$_{TPLS}$'
			elif self.SysPhases[iSys] == 'pQZ':
				subscript[iSys] = r'$_{QZ}$'
			elif self.SysPhases[iSys] == 'pBF':
				subscript[iSys] = r'$_{BF}$'
			elif self.SysPhases[iSys] == 'pGG':
				subscript[iSys] = r'$_{GG}$'
			elif self.SysPhases[iSys] == 'pCor':
				subscript[iSys] = r'$_{Cor}$'
			elif self.SysPhases[iSys] == 'pPL':
				subscript[iSys] = r'$_{PL}$'
			elif self.SysPhases[iSys] == 'pWC':
				subscript[iSys] = r'$_{WC}$'
			elif self.SysPhases[iSys] == 'pMZA':
				subscript[iSys] = r'$_{MZA}$'
			elif self.SysPhases[iSys] == 'pSF':
				subscript[iSys] = r'$_{SF}$'
			elif self.SysPhases[iSys] == 'pNL':
				subscript[iSys] = r'$_{NL}$'
			elif self.SysPhases[iSys] == 'pTot':
				subscript[iSys] = r'$_{Total}$'

		zLabels = [symbol + subscript[iSys] + unit for iSys in range(self.nSys)]
		lLabels = [r'$+k$', r'$-k$', r'$k_{\rm ind}$', r'$k_{\rm dep}$']

		return xLabel, yLabel, zLabels, lLabels, xScale, yScale

	############ End of Systematics.GetSystematicLabels() ###########
	#################################################################

	def PlotSystematics1D(self, iax):
		"""Plot systematics for a specific axis as a function of plot variable 1."""

		logging.info('iXC_Sys::Plotting systematics in 1D for iax = {}...'.format(iax))

		xName = self.SysOpts['PlotVariables'][0]

		if xName not in vars(self).keys():
			logging.error('iXC_Sys::Plot variable 1 ({}) not an attribute of Systematics class'.format(xName))
			logging.error('iXC_Sys::Aborting...')

		if type(getattr(self, xName)) is list:
			logging.error('iXC_Sys::Plot variable 1 ({}) is an attribute of type "list". Convert to nd.array in the code.'.format(xName))
			logging.error('iXC_Sys::Aborting...')

		## Turn off printing function
		self.SysOpts['Print'] = False

		self.SetSystematicList()

		if 'All' in self.SysOpts['SysList']:
			(nRows, nCols) = (3,3)
			fig  = plt.figure(figsize=(nCols*5,nRows*3), constrained_layout=True)
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
			if self.nSys <= 3:
				(nRows, nCols) = (1, self.nSys)
			elif self.nSys <= 6:
				(nRows, nCols) = (2, int(self.nSys/2 + 0.5))
			else:
				(nRows, nCols) = (3, 3)

			axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*5,nRows*3), sharex='col', constrained_layout=True)[1]
			axs = np.reshape(axs, nRows*nCols) ## Reshape as 1D array

		## List of independent variables to plot
		xList = np.linspace(self.SysOpts['PlotRanges'][0][0], self.SysOpts['PlotRanges'][0][1], num=self.SysOpts['PlotPoints'][0], endpoint=True)
		nx    = len(xList)

		## List to store systematics
		pSys  = np.zeros((self.nSys,8,2,nx)) ## [iSys,ik,iErr,iVal]

		## Compute systematics for each value in xList
		for ix in range(nx):
			## Set independent variable value
			if type(getattr(self, xName)) == np.ndarray:
				if getattr(self, xName).ndim == 1:
					getattr(self, xName)[iax] = xList[ix]
				else: ## ndim == 2
					getattr(self, xName)[iax,0] = xList[ix]
			else:
				setattr(self, xName, xList[ix])

			self.ComputeSystematics(iax)

			for iSys in range(self.nSys):
				for ik in range(0,4):
					pSys[iSys,ik  ,0,ix] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Values_Scaled']
					pSys[iSys,ik  ,1,ix] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Errors_Scaled']
					pSys[iSys,ik+4,0,ix] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Values']
					pSys[iSys,ik+4,1,ix] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Errors']

		xLabel, _, zLabels, lLabels, xScale, _ = self.GetSystematicLabels()

		colors   = ['red', 'blue', 'darkorange', 'purple']
		plotOpts = {'Color': 'red', 'Linestyle': '-', 'Marker': 'None', 'Title': 'None',
			'xLabel': 'None', 'yLabel': 'None', 'LegLabel': 'None', 'Legend': False,
			'LegLocation': 'best'}

		if 'All' in self.SysOpts['SysList']:
			iSysList = [0,1,2,3,5,6] ## ['pSF', 'pWC', 'pTPLS', 'pPL', 'pOPLS', 'pCor', 'pQZ', 'pNL', 'pTot']
			for ik in range(0,4):
				plotOpts['Color']    = colors[ik]
				plotOpts['xLabel']   = 'None'
				plotOpts['LegLabel'] = 'None'
				plotOpts['Legend']   = False

				(r,c) = (0,0)
				for iSys in iSysList[:2]:
					plotOpts['yLabel'] = zLabels[iSys]
					iXUtils.CustomPlot(axs[r,c], plotOpts, xList*xScale, pSys[iSys,ik,0])
					axs[r,c].fill_between(xList*xScale, pSys[iSys,ik,0] - pSys[iSys,ik,1], pSys[iSys,ik,0] + pSys[iSys,ik,1], color=colors[ik], alpha=0.5)
					c += 1

				(r,c) = (1,0)
				for iSys in iSysList[2:4]:
					plotOpts['yLabel'] = zLabels[iSys]
					iXUtils.CustomPlot(axs[r,c], plotOpts, xList*xScale, pSys[iSys,ik,0])
					axs[r,c].fill_between(xList*xScale, pSys[iSys,ik,0] - pSys[iSys,ik,1], pSys[iSys,ik,0] + pSys[iSys,ik,1], color=colors[ik], alpha=0.5)
					c += 1

				(r,c) = (2,0)
				plotOpts['xLabel'] = xLabel
				for iSys in iSysList[4:6]:
					plotOpts['yLabel'] = zLabels[iSys]
					iXUtils.CustomPlot(axs[r,c], plotOpts, xList*xScale, pSys[iSys,ik,0])
					axs[r,c].fill_between(xList*xScale, pSys[iSys,ik,0] - pSys[iSys,ik,1], pSys[iSys,ik,0] + pSys[iSys,ik,1], color=colors[ik], alpha=0.5)
					c += 1

				plotOpts['yLabel']   = zLabels[-1]
				plotOpts['LegLabel'] = lLabels[ik]
				plotOpts['Legend']   = True
				iXUtils.CustomPlot(axs[0,2], plotOpts, xList*xScale, pSys[-1,ik,0])
				axs[0,2].fill_between(xList*xScale, pSys[-1,ik,0] - pSys[-1,ik,1], pSys[-1,ik,0] + pSys[-1,ik,1], color=colors[ik], alpha=0.5)
		else:
			plotOpts['xLabel'] = xLabel

			for ik in range(0,4):
				plotOpts['Color'] = colors[ik]
				for iSys in range(self.nSys):
					if int(iSys/nCols) == nRows-1:
						plotOpts['xLabel']   = xLabel
					else:
						plotOpts['xLabel']   = 'None'

					if iSys == self.nSys-1:
						plotOpts['LegLabel'] = lLabels[ik]
						plotOpts['Legend']   = True
					else:
						plotOpts['LegLabel'] = 'None'
						plotOpts['Legend']   = False

					plotOpts['yLabel'] = zLabels[iSys]
					iXUtils.CustomPlot(axs[iSys], plotOpts, xList*xScale, pSys[iSys,ik,0])
					axs[iSys].fill_between(xList*xScale, pSys[iSys,ik,0] - pSys[iSys,ik,1], pSys[iSys,ik,0] + pSys[iSys,ik,1], color=colors[ik], alpha=0.5)

		if self.SysOpts['ShowPlot']:
			plt.show()

		if self.SysOpts['Export']:
			self.ExportSystematics1DPlot(iax, xList, pSys[:,[4,5,6,7]]) ## Export parameters in SI units and systematics in rad

	############# End of Systematics.PlotSystematics1D() ############
	#################################################################

	def PlotSystematics2D(self, iax):
		"""Plot systematics for a specific axis as a function of plot variables 1 & 2."""

		logging.info('iXC_Sys::Plotting systematics in 2D for iax = {}...'.format(iax))

		[xName, yName] = self.SysOpts['PlotVariables']

		if xName not in vars(self).keys():
			logging.error('iXC_Sys::Plot variable 1 ({}) not an attribute of Systematics class'.format(xName))
			logging.error('iXC_Sys::Aborting...')

		if type(getattr(self, xName)) is list:
			logging.error('iXC_Sys::Plot variable 1 ({}) is an attribute of type "list". Convert to nd.array in the code.'.format(xName))
			logging.error('iXC_Sys::Aborting...')

		if yName not in vars(self).keys():
			logging.error('iXC_Sys::Plot variable 2 ({}) not an attribute of Systematics class'.format(yName))
			logging.error('iXC_Sys::Aborting...')

		if type(getattr(self, yName)) is list:
			logging.error('iXC_Sys::Plot variable 2 ({}) is an attribute of type "list". Convert to nd.array in the code.'.format(yName))
			logging.error('iXC_Sys::Aborting...')

		## Turn off printing function
		self.SysOpts['Print'] = False

		self.SetSystematicList()

		if 'All' in self.SysOpts['SysList']:
			(nRows, nCols) = (3,3)
			fig  = plt.figure(figsize=(nCols*5,nRows*3), constrained_layout=True)
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
			if self.nSys <= 3:
				(nRows, nCols) = (1, self.nSys)
			elif self.nSys <= 6:
				(nRows, nCols) = (2, int(self.nSys/2 + 0.5))
			else:
				(nRows, nCols) = (3, 3)

			axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*5,nRows*3), sharex='col', constrained_layout=True)[1]
			axs = np.reshape(axs, nRows*nCols) ## Reshape as 1D array

		## List of independent variables to plot
		xList = np.linspace(self.SysOpts['PlotRanges'][0][0], self.SysOpts['PlotRanges'][0][1], num=self.SysOpts['PlotPoints'][0], endpoint=True)
		yList = np.linspace(self.SysOpts['PlotRanges'][1][0], self.SysOpts['PlotRanges'][1][1], num=self.SysOpts['PlotPoints'][1], endpoint=True)
		nx    = len(xList)
		ny    = len(yList)

		## List to store systematics
		pSys  = np.zeros((self.nSys,8,2,nx,ny)) ## [iSys,ik,iErr,iVal]

		## Compute systematics for each value in xList and yList
		for ix in range(nx):
			## Set x variable values
			if type(getattr(self, xName)) == np.ndarray:
				if getattr(self, xName).ndim == 1:
					getattr(self, xName)[iax] = xList[ix]
				else: ## ndim == 2
					getattr(self, xName)[iax,0] = xList[ix]
			else:
				setattr(self, xName, xList[ix])

			for iy in range(ny):
				## Set y variable values
				if type(getattr(self, yName)) == np.ndarray:
					if getattr(self, yName).ndim == 1:
						getattr(self, yName)[iax] = yList[iy]
					else: ## ndim == 2
						getattr(self, yName)[iax,0] = yList[iy]
				else:
					setattr(self, yName, yList[iy])

				self.ComputeSystematics(iax)

				for iSys in range(self.nSys):
					for ik in range(0,4):
						pSys[iSys,ik  ,0,ix,iy] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Values_Scaled']
						pSys[iSys,ik  ,1,ix,iy] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Errors_Scaled']
						pSys[iSys,ik+4,0,ix,iy] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Values']
						pSys[iSys,ik+4,1,ix,iy] = self.SysDF[ik].loc[self.SysPhases[iSys], 'Errors']

				# print(xList[ix], yList[iy], self.aBody, pSys[-1,-1,0,ix,iy])
				# print(round(self.TiltX), round(self.TiltZ), round(self.thetaX*180./np.pi), round(self.thetaZ*180./np.pi), self.aBody, pSys[-1,3,0,ix,iy])# self.SysDF[3].loc['pTot', 'Values_Scaled'])

		xLabel, yLabel, zLabels, _, xScale, yScale = self.GetSystematicLabels()

		colors   = ['red', 'blue', 'darkorange', 'purple']
		plotOpts = {'Color': 'red', 'Linestyle': '-', 'Marker': 'None', 'Title': 'None',
			'xLabel': 'None', 'yLabel': 'None', 'LegLabel': 'None', 'Legend': False,
			'LegLocation': 'best'}

		if self.SysOpts['Plot2D_Var'] == 'kU':
			ik = 0
		if self.SysOpts['Plot2D_Var'] == 'kD':
			ik = 1
		if self.SysOpts['Plot2D_Var'] == 'kInd':
			ik = 2
		else: ## self.SysOpts['Plot2D_Var'] == 'kDep':
			ik = 3

		if 'All' in self.SysOpts['SysList']:
			iSysList = [0,1,2,3,5,6] ## ['pSF', 'pWC', 'pTPLS', 'pPL', 'pOPLS', 'pCor', 'pQZ', 'pNL', 'pTot']

			(r,c) = (0,0)
			for iSys in iSysList[:2]:
				cp = axs[r,c].contourf(xList*xScale, yList*yScale, np.transpose(pSys[iSys,ik,0]), levels=100, cmap=plt.get_cmap('viridis'))
				plt.colorbar(cp, ax=axs[r,c])
				axs[r,c].set_ylabel(yLabel)
				axs[r,c].set_title(zLabels[iSys])
				axs[r,c].grid(b=True, which='both', axis='both', color='0.95', linestyle='-')
				c += 1

			(r,c) = (1,0)
			for iSys in iSysList[2:4]:
				cp = axs[r,c].contourf(xList*xScale, yList*yScale, np.transpose(pSys[iSys,ik,0]), levels=100, cmap=plt.get_cmap('viridis'))
				plt.colorbar(cp, ax=axs[r,c])
				axs[r,c].set_ylabel(yLabel)
				axs[r,c].set_title(zLabels[iSys])
				axs[r,c].grid(b=True, which='both', axis='both', color='0.95', linestyle='-')
				c += 1

			(r,c) = (2,0)
			for iSys in iSysList[4:6]:
				cp = axs[r,c].contourf(xList*xScale, yList*yScale, np.transpose(pSys[iSys,ik,0]), levels=100, cmap=plt.get_cmap('viridis'))
				plt.colorbar(cp, ax=axs[r,c])
				axs[r,c].set_xlabel(xLabel)
				axs[r,c].set_ylabel(yLabel)
				axs[r,c].set_title(zLabels[iSys])
				axs[r,c].grid(b=True, which='both', axis='both', color='0.95', linestyle='-')
				c += 1

			(r,c) = (0,2)
			iSys = -1
			cp = axs[r,c].contourf(xList*xScale, yList*yScale, np.transpose(pSys[iSys,ik,0]), levels=100, cmap=plt.get_cmap('viridis'))
			plt.colorbar(cp, ax=axs[r,c])
			axs[r,c].set_xlabel(xLabel)
			axs[r,c].set_ylabel(yLabel)
			axs[r,c].set_title(zLabels[iSys])
			axs[r,c].grid(b=True, which='both', axis='both', color='0.95', linestyle='-')
		else:
			for iSys in range(self.nSys):
				if int(iSys/nCols) == nRows-1:
					axs[iSys].set_xlabel(xLabel)
				axs[iSys].set_ylabel(yLabel)
				axs[iSys].set_title(zLabels[iSys])

				cp = axs[iSys].contourf(xList*xScale, yList*yScale, np.transpose(pSys[iSys,ik,0]), levels=100, cmap=plt.get_cmap('viridis'))
				plt.colorbar(cp, ax=axs[iSys]) # Add a colorbar to the density plot
				axs[iSys].grid(b=True, which='both', axis='both', color='0.95', linestyle='-')

		if self.SysOpts['ShowPlot']:
			plt.show()

		if self.SysOpts['Export']:
			self.ExportSystematics2DPlot(iax, xList, yList, pSys[:,[4,5,6,7]]) ## Export parameters in SI units and systematics in rad

	############# End of Systematics.PlotSystematics2D() ############
	#################################################################

	def ExportSystematics1DPlot(self, iax, xList, pSys):
		"""Export to file the systematic phases (in rad) as a function of PlotVariable 1 (in SI units)."""

		logging.info('iXC_Sys::Exporting systematics in 1D Plot for iax = {}...'.format(iax))

		ikList = [3]
		iik = -1
		for ik in ikList:
			iik += 1
			d1 = {self.SysOpts['PlotVariables'][0]: xList}
			d2 = {self.SysPhases[iSys]: pSys[iSys,ik,0] for iSys in range(self.nSys)}
			d3 = {'d'+self.SysPhases[iSys]: pSys[iSys,ik,1] for iSys in range(self.nSys)}
			d1.update(d2)
			d1.update(d3)
			df = pd.DataFrame(data=d1)

			fileName = self.SysOpts['FilePrefix']+'-'+self.iaxLabels[iax]+'-'+self.ikLabels[ik]+'-vs-'+self.SysOpts['PlotVariables'][0]+'.txt'
			filePath = os.path.join(self.SysOpts['Folder'], fileName)
			if iik == 0:
				## Set the order of columns to write to file
				z = list(zip(list(d2.keys()), list(d3.keys())))
				columns = [self.SysOpts['PlotVariables'][0]] + [n for l in z for n in l]

			iXUtils.WriteDataFrameToFile(df, self.SysOpts['Folder'], filePath, True, False, '%5.3E', Columns=columns)

	########## End of Systematics.ExportSystematics1DPlot() #########
	#################################################################

	def ExportSystematics2DPlot(self, iax, xList, yList, pSys):
		"""Export to file the systematic phases (in rad) as a function of PlotVariables 1 & 2 (in SI units)."""

		logging.info('iXC_Sys::Exporting systematics in 2D Plot for iax = {}...'.format(iax))

		ikList = [3]
		iik = -1
		Y, X = np.meshgrid(yList, xList)

		for ik in ikList:
			iik += 1
			d1 = {self.SysOpts['PlotVariables'][0]: X.flatten()}
			d2 = {self.SysOpts['PlotVariables'][1]: Y.flatten()}
			d3 = {self.SysPhases[iSys]: pSys[iSys,ik,0].flatten() for iSys in range(self.nSys)}
			d4 = {'d'+self.SysPhases[iSys]: pSys[iSys,ik,1].flatten() for iSys in range(self.nSys)}
			d1.update(d2)
			d1.update(d3)
			d1.update(d4)
			df = pd.DataFrame(data=d1)

			fileName = self.SysOpts['FilePrefix'] + '-' + self.iaxLabels[iax] + '-' + self.ikLabels[ik]+'-vs-' \
				+ self.SysOpts['PlotVariables'][0] + '-vs-' + self.SysOpts['PlotVariables'][1] + '.txt'
			filePath = os.path.join(self.SysOpts['Folder'], fileName)
			if iik == 0:
				## Set the order of columns to write to file
				z = list(zip(list(d3.keys()), list(d4.keys())))
				columns = self.SysOpts['PlotVariables'] + [n for l in z for n in l]

			iXUtils.WriteDataFrameToFile(df, self.SysOpts['Folder'], filePath, True, False, '%5.3E', Columns=columns)

	########## End of Systematics.ExportSystematics2DPlot() #########
	#################################################################

	def SetParameterList(self):
		"""Set list of selected parameters based on option 'ParList'."""

		if 'All' in self.SysOpts['ParList']:
			self.Pars = ['rCOM', 'vCOM', 'B', 'BGrad', 'OmegaR', 'Contrast']
		else:
			self.Pars = []
			for par in self.SysOpts['ParList']:
				self.Pars += [par]

		self.nPars = len(self.Pars)
		self.ParDF = [pd.DataFrame([], columns=['Values', 'Errors', 'Units'], index=self.Pars) for _ in range(3)]

	############# End of Systematics.SetParameterList() #############
	#################################################################

	def EvaluateParameterModels(self, iax, ks, t):
		"""Evaluate models for selected parameters ('rCOM', 'vCOM', 'B', 'BGrad', 'OmegaR', 'Contrast', 'All')."""

		if 'All' in self.SysOpts['ParList']:
			self.rCOM, self.drCOM, self.vCOM, self.dvCOM = self.TrajectoryModel(iax, ks, t)[:4]
			self.B, self.dB, self.BGrad, self.dBGrad = self.BModel(iax, ks, t)
			self.OmegaR, self.dOmegaR = self.RabiModel(iax, ks, t)
			self.Contrast, self.dContrast = self.ContrastModel(iax, ks)
			self.Symbols = [r'$r_{\rm COM}$', r'$v_{\rm COM}$', r'$B_{0}$', r'$\nabla B$', r'$\Omega_{\rm eff}$', r'$C$']
			self.Units   = ['mm', 'mm/s', 'G', 'G/m', 'kHz', 'arb.']
			self.yScales = [1.E3, 1.E3, 1.E0, 1.E0, 1./(2.*np.pi*1.E3), 1.E0]
		else:
			self.Symbols = []
			self.Units   = []
			self.yScales = []
			for par in self.SysOpts['ParList']:
				if par == 'rCOM':
					self.rCOM, self.drCOM = self.TrajectoryModel(iax, ks, t)[[0,1]]
					self.Symbols += [r'$r_{\rm COM}$']
					self.Units   += ['mm']
					self.yScales += [1.E3]
				elif par == 'vCOM':
					self.vCOM, self.dvCOM = self.TrajectoryModel(iax, ks, t)[[2,3]]
					self.Symbols += [r'$v_{\rm COM}$']
					self.Units   += ['mm/s']
					self.yScales += [1.E3]
				elif par == 'B':
					self.B, self.dB = self.BModel(iax, ks, t)[:2]
					self.Symbols += [r'$B_{0}$']
					self.Units   += ['G']
					self.yScales += [1.E0]
				elif par == 'BGrad':
					self.BGrad, self.dBGrad = self.BModel(iax, ks, t)[2:]
					self.Symbols += [r'$\nabla B$']
					self.Units   += ['G/m']
					self.yScales += [1.E0]
				elif par == 'OmegaR':
					self.OmegaR, self.dOmegaR = self.RabiModel(iax, ks, t)
					self.Symbols += [r'$\Omega_{\rm eff}$']
					self.Units   += ['kHz']
					self.yScales += [1./(2.*np.pi*1.E3)]
				elif par == 'delta1' or par == 'delta2' or par == 'delta3':
					self.RamanFreqShifts(iax, ks)
					if par == 'delta1':
						self.Symbols += [r'$\delta_1$']
					elif par == 'delta2':
						self.Symbols += [r'$\delta_2$']
					else: ## par == 'delta3':
						self.Symbols += [r'$\delta_3$']
					self.Units   += ['kHz']
					self.yScales += [1./(2.*np.pi*1.E3)]
				elif par == 'Contrast':
					self.Contrast, self.dContrast = self.ContrastModel(iax, ks)
					self.Symbols += [r'$C$']
					self.Units   += ['arb.']
					self.yScales += [1.E0]

	######### End of Systematics.EvaluateParameterModels() ##########
	#################################################################

	def ComputeParameters(self, iax, ks, t):
		"""Compute parameters for specific Raman axis iax and k-case (kU,kD,kCo), and fill values in a list of dataframes."""

		self.RotateToBodyFrame()
		self.SetAIParameters()

		self.EvaluateParameterModels(iax, ks, t)

		self.ParDF[iax]['Values'] = [getattr(self, par) for par in self.Pars]
		self.ParDF[iax]['Errors'] = [getattr(self, 'd'+par) for par in self.Pars]		
		self.ParDF[iax]['Units']  = self.Units

	############ End of Systematics.ComputeParameters() #############
	#################################################################

	def PrintParameters(self, iaxList, ks):
		"""Print table of selected parameters for a given k-sign (-1,1,0)."""

		logging.info('iXC_Sys::Printing parameters for iax = {}, ks = {}...'.format(iaxList, ks))

		self.SetParameterList()

		columns = [[self.iaxLabels[iax]+'_Val',self.iaxLabels[iax]+'_Err'] for iax in iaxList]
		columns = [n for l in columns for n in l] + ['Units']
		df = pd.DataFrame([], columns=columns, index=self.Pars)

		for iax in iaxList:
			self.ComputeParameters(iax, ks, self.t1[iax])

			df[self.iaxLabels[iax]+'_Val'] = self.ParDF[iax]['Values']
			df[self.iaxLabels[iax]+'_Err'] = self.ParDF[iax]['Errors']
			df['Units'] = self.ParDF[iax]['Units']

		pd.set_option('display.max_rows', 12)
		pd.set_option('display.max_columns', 8)
		pd.set_option('display.expand_frame_repr', False)
		print('--------------------------------- ks = {} Parameters ----------------------------------'.format(ks))
		print(df)
		print('--------------------------------------------------------------------------------------')

	############# End of Systematics.PrintParameters() ##############
	#################################################################

	def GetParameterLabels(self):
		"""Define labels for plot axes, legends and units according to the requested parameters."""

		if self.SysOpts['PlotVariable'] == 't':
			xLabel = r'$t$  (ms)'
			xScale = 1.E3
		elif self.SysOpts['PlotVariable'] == 'TOF':
			xLabel = r'TOF  (ms)'
			xScale = 1.E3
		elif self.SysOpts['PlotVariable'] == 'zM':
			xLabel = r'$z_M$  (mm)'
			xScale = 1.E3
		elif self.SysOpts['PlotVariable'] == 'TiltX':
			xLabel = r'$\theta_X$  (deg)'
			xScale = 1.
		elif self.SysOpts['PlotVariable'] == 'TiltZ':
			xLabel = r'$\theta_Z$  (deg)'
			xScale = 1.
		else:
			xLabel = self.SysOpts['PlotVariable']
			xScale = 1.

		yLabels = [self.Symbols[iPar] + '  (' + self.Units[iPar] + ')' for iPar in range(self.nPars)]
		lLabels = [[r'$X, +k$', r'$Y, +k$', r'$Z, +k$'], [r'$X, -k$', r'$Y, -k$', r'$Z, -k$']]

		return xLabel, yLabels, lLabels, xScale

	############# End of Systematics.GetParameterLabels() ###########
	#################################################################

	def PlotParameters(self, iaxList, ks):
		"""Plot selected parameters for a specific Raman axes and k-signs (-1,+1,0) as a function of the specified plot variable."""

		logging.info('iXC_Sys::Plotting parameters for iax = {}, ks = {}...'.format(iaxList, ks))

		xName = self.SysOpts['PlotVariable']

		if xName != 't':
			if xName not in vars(self).keys():
				logging.error('iXC_Sys::Plot variable ({}) not an attribute of Systematics class'.format(xName))
				logging.error('iXC_Sys::Aborting...')
				quit()

		## Turn off printing function
		self.SysOpts['Print'] = False

		self.SetParameterList()

		if 'All' in self.SysOpts['ParList']:
			(nRows, nCols) = (2,3)
		else:
			if self.nPars <= 3:
				(nRows, nCols) = (1, self.nPars)
			elif self.nPars <= 6:
				(nRows, nCols) = (2, int(self.nPars/2 + 0.5))
			else:
				(nRows, nCols) = (2,3)

		axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*5,nRows*3), sharex='col', constrained_layout=True)[1]
		axs = np.reshape(axs, nRows*nCols) ## Reshape as 1D array

		## List of independent variables to plot
		xList = np.linspace(self.SysOpts['PlotRange'][0], self.SysOpts['PlotRange'][1], num=self.SysOpts['PlotPoints'], endpoint=True)
		nx    = len(xList)

		## List to store parameter values
		pList = np.zeros((self.nPars,3,2,nx)) ## [iPar,iax,iErr,iVal]

		## Compute systematics for each value in xList
		for ix in range(nx):
			for iax in iaxList:
				for iPar in range(self.nPars):
					## Set independent variable value
					if xName == 't':
						self.ComputeParameters(iax, ks, xList[ix])
					elif type(getattr(self, xName)) == float or type(getattr(self, xName)) == np.float64:
						setattr(self, xName, xList[ix])
						self.ComputeParameters(iax, ks, self.t1[iax])
					elif type(getattr(self, xName)) == np.ndarray:
						if getattr(self, xName).ndim == 1:
							getattr(self, xName)[iax] = xList[ix]
						else: ## ndim == 2
							getattr(self, xName)[iax,0] = xList[ix]
						self.ComputeParameters(iax, ks, self.t1[iax])
					else:
						logging.error('iXC_Sys::Cannot set plot variable ({}) for some reason... Aborting!'.format(xName))
						quit()

					pList[iPar,iax,0,ix] = self.ParDF[iax].loc[self.Pars[iPar], 'Values']
					pList[iPar,iax,1,ix] = self.ParDF[iax].loc[self.Pars[iPar], 'Errors']

		xLabel, yLabels, lLabels, xScale = self.GetParameterLabels()

		colors   = ['darkgreen', 'blue', 'red']
		plotOpts = {'Color': 'red', 'Linestyle': '-', 'Marker': 'None', 'Title': 'None',
			'xLabel': 'None', 'yLabel': 'None', 'LegLabel': 'None', 'Legend': False,
			'LegLocation': 'best'}

		ik = int((1-ks)/2)
		for iax in iaxList:
			plotOpts['Color'] = colors[iax]
			for iPar in range(self.nPars):
				if int(iPar/nCols) == nRows-1:
					plotOpts['xLabel']   = xLabel
				else:
					plotOpts['xLabel']   = 'None'

				if iPar == self.nPars-1:
					plotOpts['LegLabel'] = lLabels[ik][iax]
					plotOpts['Legend']   = True
				else:
					plotOpts['LegLabel'] = 'None'
					plotOpts['Legend']   = False

				plotOpts['yLabel'] = yLabels[iPar]
				iXUtils.CustomPlot(axs[iPar], plotOpts, xScale*xList, self.yScales[iPar]*pList[iPar,iax,0])
				axs[iPar].fill_between(xScale*xList, self.yScales[iPar]*(pList[iPar,iax,0] - pList[iPar,iax,1]),  self.yScales[iPar]*(pList[iPar,iax,0] + pList[iPar,iax,1]), color=colors[iax], alpha=0.5)

		plt.show()

		if self.SysOpts['Export']:
			for iax in iaxList:
				self.ExportParameters(iax, ik, xList, pList)

	############### End of Systematics.PlotParameters() #############
	#################################################################

	def ExportParameters(self, iax, ik, xList, pList):
		"""Export to file selected systematic parameters as a function of PlotVariable (both in SI units)."""

		logging.info('iXC_Sys::Exporting parameters for iax = {}...'.format(iax))

		d1 = {self.SysOpts['PlotVariable']: xList}
		d2 = {self.Pars[iPar]: self.yScales[iPar]*pList[iPar,iax,0] for iPar in range(self.nPars)}
		d3 = {'d'+self.Pars[iPar]: self.yScales[iPar]*pList[iPar,iax,1] for iPar in range(self.nPars)}
		d1.update(d2)
		d1.update(d3)
		df = pd.DataFrame(data=d1)

		fileName = self.SysOpts['FilePrefix']+'-'+self.iaxLabels[iax]+'-'+self.ikLabels[ik]+'-vs-'+self.SysOpts['PlotVariable']+'.txt'
		filePath = os.path.join(self.SysOpts['Folder'], fileName)

		## Set the order of columns to write to file
		z = list(zip(list(d2.keys()), list(d3.keys())))
		columns = [self.SysOpts['PlotVariable']] + [n for l in z for n in l]

		iXUtils.WriteDataFrameToFile(df, self.SysOpts['Folder'], filePath, True, False, '%7.5E', Columns=columns)

	############# End of Systematics.ExportParameters() #############
	#################################################################

#####################################################################
##################### End of class Systematics ######################
#####################################################################

def PrintTriadSystematics(SysTables, SysOpts, RunPars):
	"""Combine systematics from each axis into new dataframe, compute shifts for the vector norm, and print table."""

	logging.info('iXC_Sys::Printing systematic summary for accelerometer triad...')

	columns = ['X_Val', 'X_Err', 'Y_Val', 'Y_Err', 'Z_Val', 'Z_Err', 'Norm_Val', 'Norm_Err']
	TriadDF = pd.DataFrame([], columns=columns)

	iaxList = [0,1,2]
	labels = ['X','Y','Z']
	for iax in iaxList:
		TriadDF[labels[iax]+'_Val'] = SysTables[iax]['kDep_Val']
		TriadDF[labels[iax]+'_Err'] = SysTables[iax]['kDep_Err']

	phys = iXC_Phys.Physics(RunPars)
	phys.RotateToBodyFrame()

	TriadDF['Norm_Val'] = np.sum([phys.aBody[iax]*SysTables[iax]['kDep_Val'].to_numpy() for iax in iaxList], axis=0)/phys.gLocal
	TriadDF['Norm_Err'] = np.sqrt(np.sum([(phys.aBody[iax]*SysTables[iax]['kDep_Err'].to_numpy())**2 for iax in iaxList], axis=0))/phys.gLocal

	pd.set_option('display.max_rows', 12)
	pd.set_option('display.max_columns', 8)
	pd.set_option('display.expand_frame_repr', False)
	print('------------------------------- Triad Systematics ({}) -------------------------------'.format(SysOpts['Units']))
	print(TriadDF)
	print('--------------------------------------------------------------------------------------')

################### End of PrintTriadSystematics ####################
#####################################################################