#####################################################################
## Filename:	iXAtom_Class_Physics.py
## Author:		B. Barrett
## Description: Physics class definition for iXAtom analysis package
## Version:		3.2.5
## Last Mod:	09/07/2020
#####################################################################

import datetime as dt
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz

from scipy.interpolate	import interp1d, griddata
from scipy.integrate 	import solve_ivp

import iXAtom_Utilities as iXUtils

class Physics:
	#################################################################
	## Class for useful physical constants and formulas
	#################################################################

	def __init__(self, RunPars):
		"""Define and set physical constants."""

		twopi			= 2.*np.pi
		deg2rad 		= np.pi/180.

		## Physical constants
		self.cLight		= 2.99792458E+08					## Speed of light (m/s)
		self.mu0		= 2.E-7*twopi 						## Permeability of vacuum (N/A^2)
		self.epsilon0	= 1./(self.mu0*self.cLight**2)		## Permittivity of vacuum (F/m)
		self.hPlanck	= 6.62606896E-34					## Planck's constant (J*s)
		self.hbar		= self.hPlanck/twopi 				## Reduced Planck's constant (J*s)
		self.kBoltz		= 1.3806503E-23 	 				## Boltzmann's constant (J/K)
		self.muB		= 9.274009994E-28					## Bohr magneton (J/G)
		self.muN		= 5.050783699E-31					## Nuclear magneton (J/G)
		self.OmegaE 	= 7.2921150E-5						## Rotation rate of the Earth (rad/s)
		
		## Location-dependent parameters
		self.Latitude 	= 44.804							## Latitude  (deg North)
		self.Longitude	= -0.605							## Longitude (deg East)
		self.Heading	= -19.8								## Compass direction (deg) (0 = true North, 90 = East, -90 = West, etc.)
		self.Height		= 21.								## Height above sea level (m)
		self.gLocal		= 9.805642							## Local gravity (m/s^2)
		self.aLocal		= np.array([0., 0., -self.gLocal])	## Local acceleration vector (m/s^2)
		self.Tzz		= 3086.0E-9							## Local vertical gravity gradient (s^-2)

		## Gravity gradient tensor in Earth-centered frame: +x: east, +y:north, +z: up (s^-2)
		self.TLocal		= np.diagflat([0.5*self.Tzz, 0.5*self.Tzz, self.Tzz])
		## Rotation vector in Earth-centered frame: +x: east, +y: north, +z: up (rad/s)
		self.OmegaLocal	= np.array([0., np.cos(self.Latitude*deg2rad), np.sin(self.Latitude*deg2rad)])*self.OmegaE

		## Rotate kinematic vectors from Earth-centered to lab frame
		RMatrix			= self.RotationMatrix((90.-self.Heading)*deg2rad, 0., 0.)
		self.OmegaLocal	= np.dot(RMatrix, self.OmegaLocal)	## Rotation vector in lab frame (rad/s)
		self.TLocal		= np.dot(RMatrix, self.TLocal)		## Gravity gradient tensor in lab frame (s^-2)

		## Orientation-dependent parameters
		self.TiltX		= RunPars.TiltX						## Tilt of sensor head about x-axis (deg)
		if RunPars.SoftwareVersion >= 3.3:
			self.TiltZ  = RunPars.TiltZ						## Tilt of sensor head about z-axis (deg) (i.e. relative to Heading)
		else:
			self.TiltZ  = 180.								## Tilt of sensor head about z-axis (deg) (i.e. relative to Heading)

		## Compute kinematic quantities in body frame
		self.RotateToBodyFrame()

		## 87Rb properties
		self.MRb      	= 1.443160648E-25					## Atomic mass of 87Rb (kg)
		self.tauLife   	= 26.2348E-9						## Lifetime of 87Rb 5P_{3/2} (s)
		self.Gamma     	= 1/(twopi*self.tauLife)			## Spontaneous decay rate of 87Rb 5P_{3/2} (Hz)
		self.omegaD2   	= twopi*384.2304844685E12			## Frequency for 87Rb D2 transition 5S_{1/2} -> 5P_{3/2} (rad/s) 
		self.omegaHF   	= twopi*6.834682610904290E9 		## Hyperfine splitting of 87Rb 5S_{1/2} (rad/s)
		self.dD2       	= 3.584244E-29 						## Reduced electric dipole moment for 87Rb 5S_{1/2} -> 5P_{3/2} (C.m)
		# self.dD2 		= np.sqrt((3*twopi*self.epsilon0*self.hbar*self.cLight**3)/(self.omegaD2**3*self.tauLife))

		## Frequency omegaNM corresponds to ground-excited state transitions between F = N, F'= M
		self.omega10   	= self.omegaD2 + twopi*(+4271.677E6 - 302.074E6)
		self.omega11   	= self.omegaD2 + twopi*(+4271.677E6 - 229.852E6)
		self.omega12   	= self.omegaD2 + twopi*(+4271.677E6 -  72.911E6)
		self.omega21   	= self.omegaD2 + twopi*(-2563.006E6 - 229.852E6)
		self.omega22   	= self.omegaD2 + twopi*(-2563.006E6 -  72.911E6)
		self.omega23   	= self.omegaD2 + twopi*(-2563.006E6 + 193.741E6)

		self.delta31 	= twopi*424.60*1.E+6				## Hyperfine splitting (F' = 1 <-> 3) (rad/s)
		self.delta32 	= twopi*266.65*1.E+6				## Hyperfine splitting (F' = 2 <-> 3) (rad/s)

		## Magnetic properties
		self.gS			= 2.0023193043622					## Electron spin g-factor
		self.gL 		= 0.99999369						## Electron orbital g-factor
		self.gI    		= -0.0009951414						## Nuclear g-factor
		self.gJ			= 2.00233113						## Fine structure Lande g-factor (87Rb 5S_{1/2})
		# S, L 			= [0.5, 0.]							## Electron spin and oribital angular momentum (87Rb 5S_{1/2})
		J, I			= [0.5, 1.5]						## Electron total angular momentum and nuclear spins (87Rb 5S_{1/2})
		F1 				= abs(J - I)						## Total angular momentum for hyperfine ground state
		F2 				= abs(J + I)						## Hyperfine structure Lande g-factors (87Rb 5S_{1/2})
		self.gF1		= self.gJ*(F1*(F1 + 1.) - I*(I + 1.) + J*(J + 1.))/(2.*F1*(F1 + 1.)) + self.gI*(F1*(F1 + 1.) + I*(I + 1.) - J*(J + 1.))/(2.*F1*(F1 + 1.))
		self.gF2		= self.gJ*(F2*(F2 + 1.) - I*(I + 1.) + J*(J + 1.))/(2.*F2*(F2 + 1.)) + self.gI*(F2*(F2 + 1.) + I*(I + 1.) - J*(J + 1.))/(2.*F2*(F2 + 1.))
		self.alphaB		= (self.gF2 - self.gF1)*self.muB/self.hPlanck ## First-order Zeeman shift (Hz/G)
		self.KClock 	= 575.146				 		 	## Clock shift of 87Rb 5S_{1/2} (Hz/G^2)
		self.Lambda 	= self.hPlanck*self.KClock/self.MRb	## Magnetic force coefficient (rad/m^2/s^2/G^2)

		## Interferometer parameters loaded from RunPars
		self.Delta 		= twopi*RunPars.RamanDetuning		## Raman laser detuning (rad/s)
		if RunPars.SoftwareVersion >= 3.4:
			self.deltakU = twopi*np.array(RunPars.RamankUFreq)		## Raman kU frequency (rad/s)
			self.deltakD = twopi*np.array(RunPars.RamankDFreq)		## Raman kD frequency (rad/s)
			self.alphakU = twopi*np.array(RunPars.RamankUChirp) 	## Raman kU chirp rate (rad/s^2)
			self.alphakD = twopi*np.array(RunPars.RamankDChirp) 	## Raman kD chirp rate (rad/s^2)
		else:
			self.deltakU = twopi*np.full(3, RunPars.kUpFrequency)	## Raman kU frequency (rad/s)
			self.deltakD = twopi*np.full(3, RunPars.kDownFrequency)	## Raman kD frequency (rad/s)
			self.alphakU = twopi*np.full(3, RunPars.kUpChirpRate) 	## Raman kU chirp rate (rad/s^2)			
			self.alphakD = twopi*np.full(3, RunPars.kDownChirpRate) ## Raman kD chirp rate (rad/s^2)			

		self.deltaSel	= twopi*RunPars.SelectionFreqs[1]	## Raman selection frequency (rad/s)
		self.deltaSum	= 0.5*(self.deltakU + self.deltakD)	## Raman half-sum frequencies (rad/s)
		self.deltaDiff	= 0.5*(self.deltakU - self.deltakD)	## Raman half-difference frequencies (rad/s)
		self.TOF		= np.full(3, RunPars.RamanTOF)		## Raman times-of-flight (relative to molasses) (s)
		self.T			= np.full(3, RunPars.RamanT) 		## Raman interrogation times (s)
		self.taupi		= np.array([RunPars.RamanpiX, RunPars.RamanpiY, RunPars.RamanpiZ]) ## Raman pi-pulse durations (s)

		## Atom interferometer parameters
		self.omegaL2	= self.omega23 + self.Delta			## Raman laser frequency 2 (rad/s)
		self.omegaL1	= self.omegaL2 + twopi*6.83468E9 	## Raman laser frequency 1 (rad/s)
		self.k1			= self.omegaL1/self.cLight			## Raman laser wavenumber 1 (rad/m)
		self.k2			= self.omegaL2/self.cLight			## Raman laser wavenumber 2 (rad/m)
		self.keff		= self.k1 + self.k2  				## Counter-propagating Raman wavenumber (rad/m)
		self.Deltak		= abs(self.k1 - self.k2)			## Co-propagating Raman wavenumber (rad/m)
		self.omegaD		= self.keff*self.gLocal*self.TOF	## Doppler shift (rad/s) (assuming vertical orientation)
		self.omegaR		= self.hbar*self.keff**2/(2*self.MRb)## Recoil frequency (rad/s)
		self.vR  		= self.hbar*self.keff/self.MRb		## Recoil velocity (m/s)
		self.tau1		= 0.5*self.taupi					## Raman pulse 1 durations (s)
		self.tau2		= self.taupi						## Raman pulse 2 durations (s)
		self.tau3		= 0.5*self.taupi					## Raman pulse 3 durations (s)
		self.Omegaeff	= np.pi/self.taupi					## Effective counter-propagating Rabi frequencies (rad/s)
		self.Omega1		= self.Omegaeff						## Rabi frequencies during Raman pulse 1 (rad/s)
		self.Omega2		= self.Omegaeff						## Rabi frequencies during Raman pulse 2 (rad/s)
		self.Omega3		= self.Omegaeff						## Rabi frequencies during Raman pulse 3 (rad/s)
		self.Ttotal		= 2.0*self.T + self.tau1 + self.tau2 + self.tau3 ## Total interrogation times (s)
		self.Teff 		= np.sqrt((self.T + self.tau2)*(self.T + np.tan(self.Omega1*self.tau1/2.)/self.Omega1 \
			+ np.tan(self.Omega3*self.tau3/2.)/self.Omega3))## Effective interrogation times (s)
		self.Seff 		= self.keff*self.Teff**2 			## Effective interferometer scale factors (rad/m/s^2)
		self.I21Ratio 	= 1.76 								## Raman beam intensity ratio (I2/I1)

	################### End of Physics.__init__() ###################
	#################################################################

	def RotateToBodyFrame(self):
		"""Rotate frame of reference from 'lab' frame to 'body' frame.
		Set kinematic quantities in the body frame."""

		## Convert tilt angle of sensor head about X-axis to radians in the correct quadrant
		if (self.TiltX >= -90. and self.TiltX <= 90.) or (self.TiltX >= 270. and self.TiltX <= 360.):
			self.thetaX	= np.arcsin(np.sin(self.TiltX*np.pi/180.))
		else:
			self.thetaX	= np.arccos(np.cos(self.TiltX*np.pi/180.))

		## Convert tilt angle of sensor head about Z-axis to radians in the correct quadrant
		if (self.TiltZ >= -90. and self.TiltZ <= 90.) or (self.TiltZ >= 270. and self.TiltZ <= 360.):
			self.thetaZ = np.arcsin(np.sin(-self.TiltZ*np.pi/180.))
		else:
			self.thetaZ	= np.arccos(np.cos(-self.TiltZ*np.pi/180.))

		# ## Convert tilt angle of sensor head about X-axis to radians in the correct quadrant
		# if self.TiltX >= 0. and self.TiltX < 180.:
		# 	self.thetaX	= np.arccos(np.cos(self.TiltX*np.pi/180.))
		# else:
		# 	self.thetaX	= np.arcsin(np.sin(self.TiltX*np.pi/180.))

		# ## Convert tilt angle of sensor head about Z-axis to radians in the correct quadrant
		# if self.TiltZ >= 0. and self.TiltZ < 180.:
		# 	self.thetaZ	= np.arccos(np.cos(-self.TiltZ*np.pi/180.))
		# else:
		# 	self.thetaZ = np.arcsin(np.sin(-self.TiltZ*np.pi/180.))

		RMatrix			= self.RotationMatrix(self.thetaZ, self.thetaX, 0.) ## Rotation matrix
		self.aBody		= np.dot(RMatrix, self.aLocal)		## Acceleration vector in rotated frame (m/s^2)
		self.OmegaBody	= np.dot(RMatrix, self.OmegaLocal)	## Rotation vector in rotated frame (rad/s)
		self.dOmegaBody	= 0.05*self.OmegaBody				## Rotation vector uncertainty (rad/s)
		self.TBody		= np.dot(RMatrix, self.TLocal)		## Gravity gradient tensor in rotated frame (s^-2)
		self.dTBody		= 0.05*self.TBody					## Gravity gradient tensor unceratinty (s^-2)

	############### End of Physics.RotateToBodyFrame() ##############
	#################################################################

	@staticmethod
	def RotationMatrix(alpha, beta, gamma):
		"""Compute rotation matrix for extrinsic rotation sequence: Z1-X2-Z3, with proper Euler angles: alpha (Z1), beta (X2), gamma (Z3)."""

		c1 = np.cos(alpha)
		c2 = np.cos(beta)
		c3 = np.cos(gamma)
		s1 = np.sin(alpha)
		s2 = np.sin(beta)
		s3 = np.sin(gamma)
		R  = np.zeros((3,3))

		R[0,0] = +c1*c3 - c2*s1*s3
		R[0,1] = -c1*s3 - c2*c3*s1
		R[0,2] = +s1*s2
		R[1,0] = +c3*s1 + c1*c2*s3
		R[1,1] = +c1*c2*c3 - s1*s3
		R[1,2] = -c1*s2
		R[2,0] = +s2*s3
		R[2,1] = +c3*s2
		R[2,2] = +c2

		return R

	#################### End of RotationMatrix() ####################
	#################################################################

	@staticmethod
	def SymmetryAxisRotationMatrix(theta):
		"""Compute rotation matrix for a rotation about the XY-symmetry axis through an angle theta (relative to vertical).
		This is equivalent to performing an extrinsic rotation sequence: Z1-X2-Z3 with proper Euler angles: alpha = 45 deg, beta=theta, gamma = 0."""

		c1 = 1./np.sqrt(2.)
		c2 = np.cos(theta)
		s1 = 1./np.sqrt(2.)
		s2 = np.sin(theta)
		R  = np.zeros((3,3))

		R[0,0] = +c1
		R[0,1] = -c2*s1
		R[0,2] = +s1*s2
		R[1,0] = +s1
		R[1,1] = +c1*c2
		R[1,2] = -c1*s2
		R[2,0] = 0.
		R[2,1] = +s2
		R[2,2] = +c2

		return R

	############## End of SymmetryAxisRotationMatrix() ##############
	#################################################################

	def ClassicalTrajectory(self, ks, r0, v0, t):
		# """Compute classical center-of-mass trajectory of a body in the local (Earth-centered) navigation frame
		# assuming a constant acceleration and rotation rate."""
		"""Compute atomic center-of-mass position and velocity at time t during a three-pulse interferometer.
		The expression for the COM position is:
		  z(t <  t1) = z0 + v0*t + 0.5*a*t^2 
		  z(t >= t1) = z0 + v0*t + 0.5*ks*vR*(t-t1) + 0.5*a*t^2 
		Note the time is the true time-of-flight (relative to molasses release).
		Set ks = 0 to obtain the undiffracted COM trajectory.
		"""

		## COM position (m) and velocity (m/s)
		rCOM = np.zeros(3)
		vCOM = np.zeros(3)

		# for iax in range(3):
		# 	if t > self.TOF[iax]:
		# 		rCOM[iax] = r0[iax] + v0[iax]*t + 0.5*ks*self.vR*(t - self.TOF[iax]) + 0.5*self.aBody[iax]*t**2
		# 		vCOM[iax] = v0[iax] + 0.5*ks*self.vR + self.aBody[iax]*t
		# 	else:
		# 		rCOM[iax] = r0[iax] + v0[iax]*t + 0.5*self.aBody[iax]*t**2
		# 		vCOM[iax] = v0[iax] + self.aBody[iax]*t

		TOF = self.TOF[0]
		if t > TOF:
			rCOM = r0 + v0*t + 0.5*ks*self.vR*(t - TOF) + 0.5*self.aBody*t**2
			vCOM = v0 + 0.5*ks*self.vR + self.aBody*t
		else:
			rCOM = r0 + v0*t + 0.5*self.aBody*t**2
			vCOM = v0 + self.aBody*t

		return np.array([rCOM, vCOM])

	################# End of ClassicalTrajectory() ##################
	#################################################################

	@staticmethod
	def gSensitivity(t, t0, T, tau1, Omega1=0., tau2=0., Omega2=0., tau3=0., Omega3=0.):
		"""Compute three-pulse atom interferometer sensitivity function."""

		if tau2 <= 0.:
			tau2 = 2*tau1
		if tau3 <= 0.:
			tau3 = tau1
		if Omega1 <= 0.:
			Omega1 = np.pi/(2.*tau1)
		if Omega2 <= 0.:
			Omega2 = np.pi/tau2
		if Omega3 <= 0.:
			Omega3 = Omega1

		t1L = t0
		t1R = t1L + tau1
		t2L = t1R + T
		t2R = t2L + tau2
		tM  = t2L + tau2/2
		t3L = t2R + T
		t3R = t3L + tau3

		if t <= t1L or t >= t3R:
			return 0.
		if t1L < t and t <= t1R:
			return -np.sin(Omega1*(t-t1L))/np.sin(Omega1*tau1)
		if t1R < t and t <= t2L:
			return -1.
		if t2L < t and t <= t2R:
			return np.sin(Omega2*(t-tM))/np.sin(Omega2*tau2/2)
		if t2R < t and t <= t3L:
			return +1.
		if t3L < t and t <= t3R:
			return -np.sin(Omega3*(t-t3R))/np.sin(Omega3*tau3)

	################ End of Physics.gSensitivity() ##################
	#################################################################

	# @staticmethod
	# def fResponse(t, T, tau):
	# 	"""Compute atom interferometer response function."""

	# 	Omega = np.pi/(2*tau)
	# 	OmegaInv = 1./Omega

	# 	if t <= 0 or t >= 2*T + 4*tau:
	# 		return 0
	# 	if 0 < t and t <= tau:
	# 		return OmegaInv*(1 - np.cos(Omega*t))
	# 	if tau < t and t <= T + tau:
	# 		return t - tau + OmegaInv
	# 	if T + tau < t and t <= T + 3*tau:
	# 		return T + OmegaInv*(1 - np.cos(Omega*(t-T)))
	# 	if T + 3*tau < t and t <= 2*T + 3*tau:
	# 		return 2*T + 3*tau + OmegaInv - t
	# 	if 2*T + 3*tau < t and t <= 2*T + 4*tau:
	# 		return OmegaInv*(1 - np.cos(Omega*(t-2*T)))

	# ################## End of Physics.fResponse() ###################
	# #################################################################

	@staticmethod
	def fResponse(t, t0, T, tau1, Omega1=0., tau2=0., Omega2=0., tau3=0., Omega3=0.):
		"""Compute three-pulse atom interferometer response function."""

		if tau2 <= 0.:
			tau2 = 2*tau1
		if tau3 <= 0.:
			tau3 = tau1
		if Omega1 <= 0.:
			Omega1 = np.pi/(2*tau1)
		if Omega2 <= 0.:
			Omega2 = np.pi/tau2
		if Omega3 <= 0.:
			Omega3 = Omega1

		t1L = t0
		t1R = t1L + tau1
		t2L = t1R + T
		t2R = t2L + tau2
		tM  = t2L + 0.5*tau2
		t3L = t2R + T
		t3R = t3L + tau3

		if t <= t1L:
			return 0.
		if t1L < t and t <= t1R:
			return (1.-np.cos(Omega1*(t-t1L)))/(Omega1*np.sin(Omega1*tau1))
		if t1R < t and t <= t2L:
			return t - t1L + np.tan(Omega1*tau1/2.)/Omega1
		if t2L < t and t <= t2R:
			return T + np.tan(Omega1*tau1/2.)/Omega1 - (np.cos(Omega2*tau2/2.) - np.cos(Omega2*(t - tM)))/(Omega2*np.sin(Omega2*tau2/2.))
		if t2R < t and t <= t3L:
			return -(t - t3L) + np.tan(Omega1*tau1/2.)/Omega1
		if t3L < t and t <= t3R:
			return np.tan(Omega1*tau1/2.)/Omega1 + (np.cos(Omega3*tau3) - np.cos(Omega3*(t-t3R)))/(Omega3*np.sin(Omega3*tau3))
		else: #t > t3R
			return np.tan(Omega1*tau1/2.)/Omega1 - np.tan(Omega3*tau3/2.)/Omega3

	################## End of Physics.fResponse() ###################
	#################################################################

	@staticmethod
	def gPolyIntegrals(t0, T, tau, Omega=0.):
		"""Compute sensitivity function integrals of the form g(t-t0)*t^n*dt up to order n = 4.
		These results are computed analytically from the sensitivity function assuming equal Rabi frequencies.
		"""

		if Omega <= 0.:
			Omega = np.pi/(2.*tau)

		## G0 = Integral[g(t-t0) dt] (s)
		G0 = 0.
		## G1 = Integral[g(t-t0)*t^1 dt] (s^2)
		G1 = (T + 2.*tau)*(T + (2./Omega)*np.tan(Omega*tau/2))
		## G2 = Integral[g(t-t0)*t^2 dt] (s^3)
		G2 = 2.*(t0 + T + 2.*tau)*G1
		## G3 = Integral[g(t-t0)*t^3 dt] (s^4)
		G3 = (T + 2.*tau)*( \
			((7./2.)*T**2 + 6.*T*t0 + 3.*t0**2 + 13.*T*tau + 12.*t0*tau + 13.*tau**2 - 6./Omega**2)*T + \
			(8.*T**2 + 12.*T*t0 + 6.*t0**2 + 29.*T*tau + 24.*t0*tau + 29.*tau**2 - 12./Omega**2)*np.tan(Omega*tau/2.)/Omega + \
			3.*tau*(T + tau)/(Omega*np.tan(Omega*tau/2.)))
 		## G4 = Integral[g(t-t0)*t^4 dt] (s^5)
		G4 = 2.*(T + 2.*tau)*(t0 + T + 2.*tau)*( \
			(3.*T**2 + 4.*T*t0 + 2.*t0**2 + 10.*T*tau + 8.*t0*tau + 10.*tau**2 - 12./Omega**2)*T + \
			4.*(2.*T**2 + 2.*T*t0 + t0**2 + 5.*T*tau + 4.*t0*tau + 5.*tau**2 - 6./Omega**2)*np.tan(Omega*tau/2.)/Omega + \
			12.*tau*(T + tau)/(np.sin(Omega*tau)*Omega))

		return np.array([G0, G1, G2, G3, G4])

	################ End of Physics.gPolyIntegrals() ################
	#################################################################

	@staticmethod
	def dgPolyIntegrals(T, tau, Omega=0.):
		"""Compute sensitivity function integrals of the form dg/dt(t)*t^n*dt up to order n = 6.
		These results are computed analytically from the derivative of the sensitivity function
		assuming equal Rabi frequencies.
		"""

		if Omega <= 0.:
			Omega = np.pi/(2.*tau)

		Omtau = Omega*tau
		OmT   = Omega*T

		## dG0 = Integral[dg/dt(t)*t^0 dt] (s^0)
		dG0 = 0.
		## dG1 = Integral[dg/dt(t)*t^1 dt] (s^1)
		dG1 = 0.
		## dG1 = Integral[dg/dt(t)*t^2 dt] (s^2)
		dG2 = 2.*(T + 2.*tau)*(T + (2./Omega)*np.tan(Omtau/2.))
		## dG2 = Integral[dg/dt(t)*t^3 dt] (s^3)
		dG3 = 3.*(T + 2.*tau)*dG2
		## dG4 = Integral[dg/dt(t)*t^4 dt] (s^4)
		dG4 = 2.*(T + 2.*tau)*((7.*T**2 + 26.*T*tau + 26.*tau**2 - 12./Omega**2)*T + \
			8.*(2.*T**2 + 8.*T*tau + 8.*tau**2 - 3./Omega**2)*np.tan(Omtau/2.)/Omega + 12.*tau*(T + tau)/(Omega*np.tan(Omtau)))
		## dG5 = Integral[dg/dt(t)*t^5 dt] (s^5)
		dG5 = 10.*(T + 2.*tau)*((3.*T**2 + 10.*T*tau + 10.*tau**2 - 12./Omega**2)*T + \
			8.*(T**2 + 4.*T*tau + 4.*tau**2 - 3./Omega**2)*np.tan(Omtau/2.)/Omega + 12.*tau*(T + tau)/(Omega*np.tan(Omtau)))
		## dG6 = Integral[dg/dt(t)*t^6 dt] (s^6)
		dG6 = 2.*(T + 2.*tau)/Omega**5*( \
			OmT*(360. - 780.*Omtau**2 + 31.*OmT**4 + 214.*OmT**3*Omtau + 363.*Omtau**4 + OmT**2*(577.*Omtau**2 - 210.) + OmT*Omtau*(726.*Omtau**2 - 780.)) - \
			6.*(120. - 260.*Omtau**2 + 16.*OmT**4 + 93.*OmT**3*Omtau + 121.*Omtau**4 + OmT**2*(214.*Omtau**2 - 80.) + OmT*Omtau*(242.*Omtau**2 - 260.))/np.tan(Omtau) + \
			48.*(15. - 40.*Omtau**2 + 2.*OmT**4 + 16.*OmT**3*Omtau + 32.*Omtau**4 + 2.*OmT**2*(24.*Omtau**2 - 5.) + 8.*OmT*Omtau*(8.*Omtau**2 - 5.)/np.sin(Omtau)))

		return np.array([dG0, dG1, dG2, dG3, dG4, dG5, dG6])

	################ End of Physics.dgPolyIntegrals() ###############
	#################################################################

	@staticmethod
	def gExpIntegrals(t0, T, tau, gamma, Omega=0.):
		"""Compute sensitivity function integrals of the form g(t-t0)*t^n*exp(-g*t)*dt.
		These results are computed analytically from integrals of the sensitivity function.		
		For n = 0, the full sensitivity function (assuming equal Rabi frequencies) is used.
		For n > 0, the integral is non-trivial (even for Mathematica), so a simple 'square'
		sensitivity function is used (which ignores pulse width effects).
		"""

		if Omega <= 0.:
			Omega = np.pi/(2.*tau)

		if gamma > 0.:
			## Texp = Integral[g(t-t1)*exp(-gamma*t) dt] (s)
			G0Exp = Omega/(gamma**2 + Omega**2)*np.exp(-gamma*(t0 + 2.*T + 4.*tau))*(1. - np.exp(gamma*(T + 2.*tau)))* \
				( np.exp(gamma*tau)*(-1. + np.exp(gamma*T))*(Omega/gamma) \
				+ (1. + np.exp(gamma*(T + 2.*tau)) - np.exp(gamma*tau)*(1. + np.exp(gamma*T))*np.cos(Omega*tau))/np.sin(Omega*tau))
			G1Exp = (1./gamma**2)*np.exp(-gamma*(t0 + 2.*T + 4.*tau))*(1. - np.exp(gamma*(T + 2.*tau))) * \
				(-1. + np.exp(gamma*(T + 2.*tau))*(1. + gamma*t0) - gamma*(t0 + 2.*T + 4.*tau))
			G2Exp = (1./gamma**3)*np.exp(-gamma*(2.*t0 + 3.*T + 6.*tau)) * \
				(-np.exp(gamma*(t0 + 3.*T + 6.*tau))*(2. + gamma*t0*(2. + gamma*t0)) \
				+ 2.*np.exp(gamma*(t0 + 2.*T + 4.*tau))*(2. + gamma*(t0 + T + 2.*tau)*(2. + gamma*(t0 + T + 2.*tau))) \
				- np.exp(gamma*(t0 + T + 2.*tau))*(2. + gamma*(t0 + 2.*T + 4.*tau)*(2. + gamma*(t0 + 2.*T + 4.*tau))))
		else:
			G0Exp = 0.
			G1Exp = 0.
			G2Exp = 0.

		return np.array([G0Exp, G1Exp, G2Exp])

	################ End of Physics.gExpIntegrals() #################
	#################################################################

	@staticmethod
	def fPolyIntegrals(t, t0, T, tau1, Omega1=0., tau2=0., Omega2=0., tau3=0., Omega3=0.):
		"""Compute response function integrals of the form f(t-t0)*t^n*dt up to order n = 0.
		These results are computed analytically from the general response function assuming different
		Rabi frequencies.
		"""

		if tau2 <= 0.:
			tau2 = 2*tau1
		if tau3 <= 0.:
			tau3 = tau1
		if Omega1 <= 0.:
			Omega1 = np.pi/(2*tau1)
		if Omega2 <= 0.:
			Omega2 = np.pi/tau2
		if Omega3 <= 0.:
			Omega3 = Omega1

		## F0 = Integral[f(t-t0) dt] (s^2) (otherwise known as Teff)
		F0 = T*(T + tau2) + np.tan(Omega1*tau1/2.)/Omega1*(2.*T + tau1 + tau2 + tau2) \
			+ tau1/(Omega1*np.tan(Omega1*tau1)) - tau2/(Omega2*np.tan(Omega2*tau2/2.)) + tau3/(Omega3*np.tan(Omega3*tau3)) \
			- 1./Omega1**2 + 2./Omega2**2 - 1./Omega3**2 \
			+ t0*(np.tan(Omega1*tau1/2.)/Omega1 - np.tan(Omega3*tau3/2.)/Omega3)

		return F0

	################ End of Physics.fPolyIntegrals() ################
	#################################################################

	def gFactors(self, F_Init, F_Final):
		"""Return Lande g-factors."""

		gF_Init  = self.gF1 if F_Init  == 1. else self.gF2
		gF_Final = self.gF1 if F_Final == 1. else self.gF2

		return [gF_Init, gF_Final]

	################### End of Physics.gFactors() ###################
	#################################################################

	def fZeeman(self, F_Init, mF_Init, F_Final, mF_Final, B):
		"""Given a magnetic field B, compute the frequency shift f due to the 1st order Zeeman effect."""

		[gF_Init, gF_Final] = self.gFactors(F_Init, F_Final)

		return (mF_Final*gF_Final - mF_Init*gF_Init)*self.muB*B/self.hPlanck

	#################### End of Physics.fZeeman() ###################
	#################################################################

	def BZeeman(self, F_Init, mF_Init, F_Final, mF_Final, f):
		"""Given a frequency shift f, compute the magnetic field B expected from the 1st order Zeeman effect."""

		[gF_Init, gF_Final] = self.gFactors(F_Init, F_Final)
	    
		DmFgF = (mF_Final*gF_Final - mF_Init*gF_Init)
		if abs(DmFgF) > 0.:
			B = self.hPlanck*f/(DmFgF*self.muB)
		else:
			B = np.inf

		return B

	#################### End of Physics.BZeeman() ###################
	#################################################################

	def fBreitRabi(self, F_Init, mF_Init, F_Final, mF_Final, B):
		"""Given a magnetic field B, compute the frequency shift f from the Breit-Rabi formula."""

		sInit  = -1. if F_Init  == 1. else 1.
		sFinal = -1. if F_Final == 1. else 1.

		EHF    = self.hbar*self.omegaHF
		x      = (self.gJ - self.gI)*self.muB*B/EHF
		EInit  = mF_Init*self.gI*self.muB*B  + sInit*EHF*(mF_Init*x + x**2)/4.
		EFinal = mF_Final*self.gI*self.muB*B + sFinal*EHF*(mF_Final*x + x**2)/4.

		return (EFinal - EInit)/self.hPlanck

	################### End of self.fBreitRabi() ####################
	#################################################################

	def BBreitRabi(self, F_Init, mF_Init, F_Final, mF_Final, f):
		"""Given a frequency shift f, compute the expected magnetic field B from the Breit-Rabi formula."""

		sInit  = -1. if F_Init  == 1. else 1.
		sFinal = -1. if F_Final == 1. else 1.

		EHF = self.hbar*self.omegaHF

		## Solve for roots of a*B**2 + b*B + c == 0
		a = (sFinal - sInit)*((self.gJ - self.gI)*self.muB)**2/(4.*EHF)
		b = (mF_Final*((1. - sFinal/4.)*self.gI + sFinal*self.gJ/4.) - mF_Init*((1. - sInit/4.)*self.gI + sInit*self.gJ/4.))*self.muB
		c = -self.hPlanck*f

		if mF_Final > 0 and mF_Init > 0:
			B = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
		else:
			B = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

		return B

	################### End of self.BBreitRabi() ####################
	#################################################################

	def alphaDynamic(self, omega):
		"""Compute the scalar, vector, and tensor components of the dynamic polarizability
		for a given species and optical frequency.
		\t omega (float) - Optical frequency of Raman beam (rad/s)
		"""

		omega10 = self.omega10
		omega11 = self.omega11
		omega12 = self.omega12
		omega21 = self.omega21
		omega22 = self.omega22
		omega23 = self.omega23
		d2oh    = self.dD2**2/self.hbar

		## Scalar polarizability [(C.m)^2/J = (C.m)^2/(kg.m/s^2) = C.m^3/(kg.m^2/(s^2.C)) = C.m/(V/m^2)]     (V=kg·m^2/(s^2·C))
		alphaS_F1 = d2oh*(+(omega10)/( 9*(omega10**2 - omega**2)) + (5*omega11)/(18*(omega11**2 - omega**2)) + (5*omega12)/(18*(omega12**2 - omega**2)))
		alphaS_F2 = d2oh*(+(omega21)/(30*(omega21**2 - omega**2)) + (  omega22)/( 6*(omega22**2 - omega**2)) + (7*omega23)/(15*(omega23**2 - omega**2)))

		## Vector polarizability
		alphaV_F1 = d2oh*(-(omega10)/( 6*(omega10**2 - omega**2)) - (5*omega11)/(24*(omega11**2 - omega**2)) + (5*omega12)/(24*(omega12**2 - omega**2)))
		alphaV_F2 = d2oh*(-(omega21)/(20*(omega21**2 - omega**2)) - (  omega22)/(12*(omega22**2 - omega**2)) + (7*omega23)/(15*(omega23**2 - omega**2)))

		## Tensor polarizability
		alphaT_F1 = d2oh*(-(omega10)/( 9*(omega10**2 - omega**2)) + (5*omega11)/(36*(omega11**2 - omega**2)) - (  omega12)/(36*(omega12**2 - omega**2)))
		alphaT_F2 = d2oh*(-(omega21)/(30*(omega21**2 - omega**2)) + (  omega22)/( 6*(omega22**2 - omega**2)) - (2*omega23)/(15*(omega23**2 - omega**2)))

		return [alphaS_F1, alphaS_F2, alphaV_F1, alphaV_F2, alphaT_F1, alphaT_F2]

	################ End of Physics.alphaDynamic() ##################
	#################################################################

	def deltaAC(self, I1, I2, A, Delta, mF, kdotB=1., zetadotB=0., returnAlphas=False):
		"""Compute the scalar, vector and tensor light shifts for a given Raman field config.
		ARGUMENTS:
		\t I1          (float) - Intensity of Raman beam 1 (W/m^2)
		\t I2          (float) - Intensity of Raman beam 2 (W/m^2)
		\t A           (float) - Degree of circular polarization (+1: RHC, -1: LHC, 0: linear)
		\t Delta       (float) - Detuning of Raman beams from F=2 -> F'=3 transition (rad/s)
		\t mF          (float) - Magnetic quantum number (-1,0,1)
		\t kdotB       (float) - Dot-product of k (wavevector) unit vector and B-field unit vector
		\t zetadotB    (float) - Dot-product of zeta (polarization) unit vector and B-field unit vector
		\t returnAlphas (bool) - Flag for returning list of polarizabilities
		"""

		omegaL2 = self.omega23 + Delta
		omegaL1 = omegaL2 + 2*np.pi*6834.68E6

		[aS_F1L1, aS_F2L1, aV_F1L1, aV_F2L1, aT_F1L1, aT_F2L1] = self.alphaDynamic(omegaL1)
		[aS_F1L2, aS_F2L2, aV_F1L2, aV_F2L2, aT_F1L2, aT_F2L2] = self.alphaDynamic(omegaL2)

		# I = 2*self.cLight*self.epsilon0*(E0/2)**2
		# (E2/2)**2 = I2/(2*self.cLight*self.epsilon0)

		cScal  = 1./(2*self.cLight*self.epsilon0*self.hbar)
		cVect  = cScal*kdotB*A*mF
		cTens  = cScal*(3*zetadotB**2 - 1)

		cF1    = 1.
		cF2    = 1.
		deltaS = -cScal*(I1*(cF2*aS_F2L1 - cF1*aS_F1L1) + I2*(cF2*aS_F2L2 - cF1*aS_F1L2))
		cF1    = 0.50
		cF2    = 0.25
		deltaV = -cVect*(I1*(cF2*aV_F2L1 - cF1*aV_F1L1) + I2*(cF2*aV_F2L2 - cF1*aV_F1L2))
		cF1    = (3*mF**2 - 2)/2.
		cF2    = (3*mF**2 - 6)/12.
		deltaT = -cTens*(I1*(cF2*aT_F2L1 - cF1*aT_F1L1) + I2*(cF2*aT_F2L2 - cF1*aT_F1L2))

		if returnAlphas:
			return np.array([deltaS, deltaV, deltaT, 
				np.array([aS_F1L1, aS_F2L1, aV_F1L1, aV_F2L1, aT_F1L1, aT_F2L1]),
				np.array([aS_F1L2, aS_F2L2, aV_F1L2, aV_F2L2, aT_F1L2, aT_F2L2])], dtype=object)
		else:
			return np.array([deltaS, deltaV, deltaT])

	################### End of Physics.deltaAC() ####################
	#################################################################

	def I2Raman(self, Omega, I21Ratio):
		"""Compute the intensity of the main Raman beam (I2) based on Rabi frequency for a given species."""

		deff2 = (self.dD2**2)/3.	## Square of the effective far off-resonant dipole moment
		# Omega = np.pi/taupi			## Rabi frequency (rad/s)

		# return (self.hPlanck*self.epsilon0*self.cLight/deff2)*(self.hbar*abs(self.Delta)*np.sqrt(I21Ratio))/taupi
		return (2.*self.hbar**2*self.epsilon0*self.cLight/deff2)*abs(self.Delta)*np.sqrt(I21Ratio)*Omega

	################### End of Physics.I2Raman() ####################
	#################################################################

	@staticmethod
	def RamanDiffraction(t, Omega, delta, omegaD, omegaR, OmegakCo=0., Regime='DD'):
		"""Compute the complex transition amplitudes for a Raman pulse of duration t. 
		Solves the TDSE in the single-diffraction (Regime == 'SD') or double-diffraction (Regime == 'DD') regime.
		'Omega' = Rabi frequency, 'delta' = Raman detuning, 'omegaD' = Doppler shift, 'omegaR' = recoil shift in rad/s,
		'OmegakCo' = Rabi frequency for co-propagating transitions. 
		Returns only the transition amplitudes of states |1,p0> and |2,p0+hk>.
		"""

		if Regime == 'DD':
			nMax = 4 ## Truncation index for double diffraction regim (must be positive and even!)
		else: ## Regime == 'SD'
			nMax = 0 ## Truncation index for single diffraction

		## Initial conditions: state initially in |1,0>
		if OmegakCo <= 0.:
			y0 = np.zeros(2*(nMax+1), dtype=complex)
			y0[nMax] = 1. + 0.*1j
		else:
			y0 = np.zeros(4*(nMax+1), dtype=complex)
			y0[2*nMax] = 1. + 0.*1j

		## Convert delta from traditional definition: delta = omegaL1 + omegaL2 - (omegaHF + omegaAC + omegaD + omegaR)
		## to that defined in the state equations: delta = omegaL1 + omegaL2 - (omegaHF + omegaAC)
		delta += omegaD + omegaR

		if OmegakCo <= 0.:
			## Define RHS including only counter-propagating transitions
			def dydt(t, y):
				"""RHS of double Raman diffraction equations (including co-propagating transitions)
				y = [|F=1,n=-nMax>, |2,-nMax+1>, ..., |1,0>, |2,1>, ..., |1,nMax>, |2,nMax+1>, |2,0>]"""

				dy = np.zeros(2*(nMax+1), dtype=complex)
				nList = np.arange(-nMax, nMax+1, 2) ## e.g. [-1,1], [-3,-1,1,3]
				m = -2
				for n in nList:
					m += 2
					if nMax == 0:
						## D(|1,n>, t)
						dy[m]   = np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+1]
						## D(|2,n+1>, t)
						dy[m+1] = np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m]
					elif n == -nMax:
						## D(|1,n>, t)
						dy[m]   = np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+1]
						## D(|2,n+1>, t)
						dy[m+1] = np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m] \
								+ np.exp(-1j*( delta + omegaD + (2*n+3)*omegaR)*t)*y[m+2]
					elif n == nMax:
						## D(|1,n>, t)
						dy[m]   = np.exp(-1j*(-delta - omegaD - (2*n-1)*omegaR)*t)*y[m-1] \
								+ np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+1]
						## D(|2,n+1>, t)
						dy[m+1] = np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m]
					else:
						## D(|1,n>, t)
						dy[m]   = np.exp(-1j*(-delta - omegaD - (2*n-1)*omegaR)*t)*y[m-1] \
								+ np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+1]
						## D(|2,n+1>, t)
						dy[m+1] = np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m] \
								+ np.exp(-1j*( delta + omegaD + (2*n+3)*omegaR)*t)*y[m+2]
				dy *= 1j*0.5*Omega

				return dy
		else:
			## Define RHS including both co- and counter-propagating transitions
			def dydt(t, y):
				"""RHS of double Raman diffraction equations. 
				y = [|F=1,n=-nMax>, |2,-nMax>, |1,-nMax+1>, |2,-nMax+1>, ..., |1,0>, |2,0>, ..., |1,nMax>, |2,nMax>, |1,nMax+1>, |2,nMax+1>]"""

				dy = np.zeros(4*(nMax+1), dtype=complex)
				nList = np.arange(-nMax, nMax+1, 2) ## e.g. nMax = 0: [0], nMax=2: [-2,0,2], nMax=4: [-4,-2,0,2,4]
				m = -4
				for n in nList:
					m += 4
					if nMax == 0:
						## D(|1,0>, t)
						dy[m]   = 1j*0.5*Omega*np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+3] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+1]
						## D(|2,0>, t)
						dy[m+1] = 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m]
						## D(|1,1>, t)
						dy[m+2] = 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+3]
						## D(|2,1>, t)
						dy[m+3] = 1j*0.5*Omega*np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m+2]
					elif n == -nMax:
						## D(|1,n>, t)
						dy[m]   = 1j*0.5*Omega*np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+3] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+1]
						## D(|2,n>, t)
						dy[m+1] = 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m]
						## D(|1,n+1>, t)
						dy[m+2] = 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+3]
						## D(|2,n+1>, t)
						dy[m+3] = 1j*0.5*Omega*np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m] \
								+ 1j*0.5*Omega*np.exp(-1j*( delta + omegaD + (2*n+3)*omegaR)*t)*y[m+4] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m+2]
					elif n == nMax:
						## D(|1,n>, t)
						dy[m]   = 1j*0.5*Omega*np.exp(-1j*(-delta - omegaD - (2*n-1)*omegaR)*t)*y[m-1] \
								+ 1j*0.5*Omega*np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+3] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+1]
						## D(|2,n>, t)
						dy[m+1] = 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m]
						## D(|1,n+1>, t)
						dy[m+2] = 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+3]
						## D(|2,n+1>, t)
						dy[m+3] = 1j*0.5*Omega*np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m+2]
					else:
						## D(|1,n>, t)
						dy[m]   = 1j*0.5*Omega*np.exp(-1j*(-delta - omegaD - (2*n-1)*omegaR)*t)*y[m-1] \
								+ 1j*0.5*Omega*np.exp(-1j*(-delta + omegaD + (2*n+1)*omegaR)*t)*y[m+3] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+1]
						## D(|2,n>, t)
						dy[m+1] = 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m]
						## D(|1,n+1>, t)
						dy[m+2] = 1j*0.5*OmegakCo*np.exp(-1j*(-delta)*t)*y[m+3]
						## D(|2,n+1>, t)
						dy[m+3] = 1j*0.5*Omega*np.exp(-1j*( delta - omegaD - (2*n+1)*omegaR)*t)*y[m] \
								+ 1j*0.5*Omega*np.exp(-1j*( delta + omegaD + (2*n+3)*omegaR)*t)*y[m+4] \
								+ 1j*0.5*OmegakCo*np.exp(-1j*( delta)*t)*y[m+2]

				return dy

		soln = solve_ivp(dydt, (0., t), y0, method='RK45', t_eval=None)

		if OmegakCo <= 0.:
			c10  = soln.y[nMax,-1]          ## Coefficient of state |1,p0>
			c21  = soln.y[nMax+1,-1]        ## Coefficient of state |2,p0+hk>
		else:
			c10 = soln.y[2*nMax,-1]    		## Coefficient of state |1,p0>
			c21 = soln.y[2*nMax+3,-1]       ## Coefficient of state |2,p0+hk>

		return np.array([c10, c21], dtype=complex)

	############## End of Physics.RamanDiffraction() ################
	#################################################################

	def SetETGTABParameters(self, TidePars):
		"""Set parameters for tidal anomaly software ETGTAB."""

		self.Latitude  = TidePars['Latitude'] 	## [deg]
		## Ellipsoidal latitude referring to Geodetic Reference System 1980.

		self.Longitude = TidePars['Longitude'] 	## [deg] positive east of Greenwich
		## Ellipsoidal longitude referring to Geodetic Reference System 1980.

		self.Height    = TidePars['Height'] 	## [m]
		## Ellipsoidal height referring to Geodetic Reference System 1980.

		self.Gravity   = TidePars['Gravity']	## [m/s**2]
		## Local gravity (only used for tidal tilt computations)
		## If unknown, set to zero and program will use GRS80 reference value

		self.Azimuth   = TidePars['Azimuth']	## [deg]
		## Azimuthal angle clockwise from North.
		## Only used for tidal tilt and horizontal strain computations.

		self.TidalComp = TidePars['TidalComp']
		## Tidal component to output. Possible values:
		## 'TidalPotential', 'VerticalAcceleration', 'HorizontalAcceleration', 'VerticalDisplacement', 'HorizontalDisplacement',
		## 'VerticalStrain', 'HorizontalStrain', 'ArealStrain', 'ShearStrain', 'VolumeStrain', 'OceanTides'
		if self.TidalComp == 'TidalPotential':
		# IC =-1: tidal potential, geodetic coefficients in m**2/s**2.
			self.IC = -1
		elif self.TidalComp == 'VerticalAcceleration':
		# IC = 0: vertical tidal acceleration (gravity tide), geodetic coefficients in nm/s**2 (positive down).
			self.IC = 0
		elif self.TidalComp == 'HorizontalAcceleration':
		# IC = 1: horizontal tidal acceleration (tidal tilt) in azimuth DAZ, geodetic coefficients in milli-arcsec
			self.IC = 1
		elif self.TidalComp == 'VerticalDisplacement':
		# IC = 2: vertical tidal displacement, geodetic coefficients in mm.
			self.IC = 2
		elif self.TidalComp == 'HorizontalDisplacement':
		# IC = 3: horizontal tidal displacement in azimuth DAZ, geodetic coefficients in mm.
			self.IC = 3
		elif self.TidalComp == 'VerticalStrain':
		# IC = 4: vertical tidal strain, geodetic coefficients in 10**-9 = nstr.
			self.IC = 4
		elif self.TidalComp == 'HorizontalStrain':
		# IC = 5: horizontal tidal strain in azimuth DAZ, geodetic coefficients in 10**-9 = nstr.
			self.IC = 5
		elif self.TidalComp == 'ArealStrain':
		# IC = 6: areal tidal strain, geodetic coefficients  in 10**-9 = nstr.
			self.IC = 6
		elif self.TidalComp == 'ShearStrain':
		# IC = 7: shear tidal strain, geodetic coefficients  in 10**-9 = nstr.
			self.IC = 7
		elif self.TidalComp == 'VolumeStrain':
		# IC = 8: volume tidal strain, geodetic coefficients in 10**-9 = nstr.
			self.IC = 8
		elif self.TidalComp == 'OceanTides':
		# IC = 9: ocean tides, geodetic coefficients in mm.
			self.IC = 9
		else:
			logging.warning('Physics::SetETGTABParameters::Tidal component {} not recognized...'.format(self.TidalComp))
			logging.warning('Physics::SetETGTABParameters::Setting to default: "VerticalAcceleration"')
			self.TidalComp = 'VerticalAcceleration'
			self.IC = 0

		self.PrintLevel = TidePars['PrintLevel']
		## Tidal data print level. Possible values: 'None', 'Some', 'All'
		if self.PrintLevel == 'None':
		# IPRINT = 0: tidal potential development will not be printed.
			self.IPRINT = 0
		elif self.PrintLevel == 'None':
		# IPRINT = 1: geodetic coefficients and astronomical elements will be printed only.
			self.IPRINT = 1
		elif self.PrintLevel == 'None':
		# IPRINT = 2: geodetic coefficients, astronomical elements, and tidal potential development will be printed
			self.IPRINT = 2
		else:
			logging.warning('Physics::SetETGTABParameters::Print level {} not recognized...'.format(self.PrintLevel))
			logging.warning('Physics::SetETGTABParameters::Setting to default: "None".')
			self.PrintLevel = 'None'
			self.IPRINT = 0

		self.TimeStart  = TidePars['TimeStart']
		## Start date and time for tidal computation in UTC: [YYYY, MM, DD, HH]

		self.TimeSpan   = TidePars['TimeSpan'] ## [hrs]
		## Time span for tidal computation

		self.TimeStep   = TidePars['TimeStep'] ## [s]
		## Time step for tidal computation (only 300 or 3600 s allowed?)

		self.TidalModel = TidePars['TidalModel']
		## Tidal potential model to be used. Possible values:
		## 'Doodson1921', 'CTE1973', 'Tamura1987', 'Buellesfeld1985'
		if self.TidalModel == 'Doodson1921':
		# IMODEL = 0: DOODSON 1921 model with 378 waves
			self.IMODEL = 0
		elif self.TidalModel == 'CTE1973':
		# IMODEL = 1: CARTWRIGHT-TAYLER-EDDEN 1973 model with 505 waves
			self.IMODEL = 1
		elif self.TidalModel == 'Tamura1987':
		# IMODEL = 2: TAMURA 1987 model with 1200 waves
			self.IMODEL = 2
		elif self.TidalModel == 'Buellesfeld1985':
		# IMODEL = 3: BUELLESFELD 1985 model with 665 waves
			self.IMODEL = 3
		else:
			logging.warning('Physics::SetETGTABParameters::Tidal model {} not recognized...'.format(self.TidalModel))
			logging.warning('Physics::SetETGTABParameters::Setting to default: "Tamura1987".')
			self.TidalModel = 'Tamura1987'
			self.IMODEL = 2

		self.EarthModel = TidePars['EarthModel']
		## Earth model to be used. Possible values: 'Elastic', 'Rigid'
		## Attention: 'Rigid' should only be used for comparison with model tides computed from ephemeris programs.
		## Attention: For read world predictions, always use 'Elastic'
		if self.EarthModel == 'Elastic':
		# IRIGID = 0 for elastic Earth model tides
			self.IRIGID = 0
		elif self.EarthModel == 'Rigid':
		# IRIGID = 1 for rigid Earth model tides
			self.IRIGID = 1
		else:
			logging.warning('Physics::SetETGTABParameters::Earth model {} not recognized...'.format(self.EarthModel))
			logging.warning('Physics::SetETGTABParameters::Setting to default: "Elastic".')
			self.TidalModel = 'Elastic'
			self.IRIGID = 0

	############### End of self.SetETGTABParameters() ###############
	#################################################################

	def WriteETGTABParameters(self, TidePars):
		"""Write input parameters for tidal anomaly software ETGTAB to file."""

		##===========================================================
		## Sample input for ETGTAB:
		##===========================================================
		## Ellipsoidal latitude  in degree:            44.804
		## Ellipsoidal longitude in degree:            -0.605
		## Ellipsoidal height in meter:                21.000
		## Gravity in m/s**2:                           9.806
		## Azimuth in degree clockwise from north:      0.000
		## Earth tide component (-1...9):                   0
		## Printout of tidal potential (0...2):             0 
		## Initial epoch(YYMMDDHH):                      2019   12   05   18
		## Number of hours:                                38
		## Time interval in secs:                         300
		## Tidal potential (0...3):                         2
		## Tides for a rigid Earth (IRIGID=1):              0
		##===========================================================

		self.SetETGTABParameters(TidePars)

		with open(os.path.join(self.ETGTAB_Folder, self.ETGTAB_Input), 'w') as f:
			f.write('Ellipsoidal latitude  in degree:            {:5.3f}\n'.format(self.Latitude))
			f.write('Ellipsoidal longitude in degree:            {:5.3f}\n'.format(self.Longitude))
			f.write('Ellipsoidal height in meter:                {:5.3f}\n'.format(self.Height))
			f.write('Gravity in m/s**2:                           {:4.3f}\n'.format(self.Gravity))
			f.write('Azimuth in degree clockwise from north:      {:4.3f}\n'.format(self.Azimuth))
			f.write('Earth tide component (-1...9):                   {:d}\n'.format(self.IC))
			f.write('Printout of tidal potential (0...2):             {:d}\n'.format(self.IPRINT)) 
			f.write('Initial epoch(YYMMDDHH):                      {:04d}   {:02d}   {:02d}   {:02d}\n'.format(*self.TimeStart))
			f.write('Number of hours:                              {:4d}\n'.format(self.TimeSpan))
			f.write('Time interval in secs:                        {:4d}\n'.format(self.TimeStep))
			f.write('Tidal potential (0...3):                         {:d}\n'.format(self.IMODEL))
			f.write('Tides for a rigid Earth (IRIGID=1):              {:d}\n'.format(self.IRIGID))
			f.write(' Gravimetric tides for Black Forest Observatory\n')
			f.write(' Parameters from LaCoste Romberg G249F\n')
			f.write('  120 days (8.12.1988 - 11.4.1989)\n')
			f.write('\n\n\n\n\n\n\n')
			f.write('14\n')
			f.write('    1  285 LONG       1.1500    0.0000\n')
			f.write('  286  428 Q1         1.1429   -0.3299\n')
			f.write('  429  488 O1         1.1464    0.0791\n')
			f.write('  489  537 M1         1.1537    0.0583\n')
			f.write('  538  592 P1K1       1.1349    0.3174\n')
			f.write('  593  634 J1         1.1537   -0.0078\n')
			f.write('  635  739 OO1        1.1538   -0.0531\n')
			f.write('  740  839 2N2        1.1560    2.5999\n')
			f.write('  840  890 N2         1.1761    2.5446\n')
			f.write('  891  947 M2         1.1840    2.0816\n')
			f.write('  948  987 L2         1.1906    0.1735\n')
			f.write('  988 1121 S2K2       1.1865    0.6230\n')
			f.write(' 1122 1204 M3         1.0584   -0.0814\n')
			f.write(' 1205 1214 M4         0.5777   92.2440\n')
			f.write(' \n')

	############## End of self.WriteETGTABParameters() ##############
	#################################################################

	def ReadETGTABResults(self):
		"""Read ETGTAB output into a dataframe."""

		df = pd.read_csv(os.path.join(self.ETGTAB_Folder, self.ETGTAB_Output), \
			sep='\s+', header=None, engine='python', skiprows=46, error_bad_lines=False, warn_bad_lines=False)

		df.columns = ['Y', 'M', 'D', 'H', 'm', 's', 'gAnomaly']
		nRows = df.shape[0]

		self.TidalDF = pd.DataFrame([])
		self.TidalDF['UTCTime'] = [dt.datetime(df['Y'].iloc[r], df['M'].iloc[r], df['D'].iloc[r], df['H'].iloc[r], df['m'].iloc[r], df['s'].iloc[r], tzinfo=pytz.UTC) for r in range(nRows)]
		self.TidalDF['RelTime']  = [(self.TidalDF['UTCTime'].iloc[r] - self.TidalDF['UTCTime'].iloc[0]).total_seconds() for r in range(nRows)]
		self.TidalDF['gAnomaly'] = df['gAnomaly']*1.E-9 ## Convert to m/s^2

	################ End of self.ReadETGTABResults() ################
	#################################################################

	def RunETGTAB(self, TidePars, ReExecute=True):
		"""Execute ETGTAB tidal gravity anomaly software."""

		# self.ETGTAB_Folder = os.path.abspath('ETGTAB\\')
		self.ETGTAB_Folder = '.\\ETGTAB\\'
		self.ETGTAB_Input  = 'ETGTAB.INP'
		self.ETGTAB_Output = 'ETGTAB.OUT'
		self.ETGTAB_exe    = 'ETGTAB-v3.0.2018.exe'

		if ReExecute:
			self.WriteETGTABParameters(TidePars)

			os.chdir('ETGTAB')
			if os.path.exists(self.ETGTAB_Output):
				os.system('del /Q '+self.ETGTAB_Output)
			os.system(self.ETGTAB_exe)
			os.chdir('..')

		self.ReadETGTABResults()

	#################### End of self.RunETGTAB() ####################
	#################################################################

	def GetTideModel(self, tData, Location, Recompute=False):
		"""Compute tidal gravity anomaly in m/s^2 at specific times.
		ARGUMENTS:
		\t tData     (list) - Timestamps (in seconds since epoch 01/01/1970) at which to compute tides.
		\t Location  (dict) - Key:value pairs defining location at which to compute tides.
		\t Recompute (bool) - Flag for recomputing tides for new times or location.
		"""

		## Reference time is start time rounded down to the nearest hour
		t0       = np.floor(tData[0]/3600.)*3600.
		dt0_UTC  = dt.datetime.fromtimestamp(t0, tz=pytz.UTC)
		## Time span is the requested span rounded up to the nearest hour + 1
		timeSpan = int(np.ceil((tData[-1] - tData[0])/3600.)) + 1

		TidePars = {
			'Latitude':		Location['Latitude'],	## [deg]
			'Longitude': 	Location['Longitude'],	## [deg]
			'Height':		Location['Height'],		## [m]
			'Gravity':		self.gLocal,			## [m/s**2]
			'Azimuth':		0.,						## [deg]
			'TidalComp':	'VerticalAcceleration', ## 'VerticalAcceleration'
			'PrintLevel':	'None',					## 'None', 'Some', 'All'
			'TimeStart':	[dt0_UTC.year, dt0_UTC.month, dt0_UTC.day, dt0_UTC.hour],
			'TimeSpan':		timeSpan, 				## [hrs]
			'TimeStep':		300, 					## [s] 300 or 3600 only
			'TidalModel':	'Tamura1987',			## 'Doodson1921', 'CTE1973', 'Tamura1987', 'Buellesfeld1985'
			'EarthModel':	'Elastic'				## 'Rigid', 'Elastic'
		}

		self.RunETGTAB(TidePars, Recompute)

		nTide = len(self.TidalDF['gAnomaly'])
		## Times relative to reference time at which to compute tides
		tTide = np.array([(self.TidalDF['UTCTime'].iloc[i] - dt0_UTC).total_seconds() for i in range(nTide)])
		gTide = self.TidalDF['gAnomaly'].to_numpy()

		fTide = interp1d(tTide, gTide, kind='quadratic')
		gTide = fTide(tData - tData[0])

		return gTide

	#################### End of GetTidalModel() #####################
	#################################################################

	@staticmethod
	def BFieldCalibration():
		"""Calibration for magnetic field offset and gradient as a function of tilt angles.
		Note that the tilt angles here are already in the correct quadrant:
			thetaX = np.arcsin(np.sin( TiltX*ni.pi/180.))*180./np.pi
			thetaZ = np.arcsin(np.sin(-TiltZ*ni.pi/180.))*180./np.pi
		"""

		B = np.empty((0,14), dtype=float) ## [thetaX, thetaZ, BxOff, dBxOff, BxGrad, dBxGrad, ...]

		## Units: [deg, deg, G, G, G/m, G/m, ...]
		B = np.append(B, np.array([[  0., 45.,  0.14437,0.00010,+0.000,0.050,  0.14302,0.00019,+0.000,0.050,  0.14952,0.00004,-0.047,0.008]]), axis=0)
		B = np.append(B, np.array([[-30., 45.,  0.14614,0.00010,+0.765,0.052,  0.14445,0.00004,-0.627,0.020,  0.14777,0.00002,-0.138,0.005]]), axis=0)
		B = np.append(B, np.array([[+30., 45.,  0.14157,0.00011,-0.176,0.058,  0.14001,0.00009,+0.641,0.046,  0.14922,0.00008,+0.060,0.015]]), axis=0)
		B = np.append(B, np.array([[+45., 45.,  0.14054,0.00005,-0.036,0.021,  0.13883,0.00007,+0.449,0.025,  0.14857,0.00004,+0.218,0.009]]), axis=0)
		B = np.append(B, np.array([[-44., 45.,  0.14631,0.00006,+0.717,0.021,  0.14503,0.00007,-0.427,0.026,  0.14626,0.00004,-0.270,0.011]]), axis=0)
		B = np.append(B, np.array([[-60., 45.,  0.14661,0.00007,+0.551,0.022,  0.14551,0.00014,-0.269,0.047,  0.14430,0.00008,-0.544,0.033]]), axis=0)
		B = np.append(B, np.array([[+60., 45.,  0.13973,0.00009,+0.114,0.033,  0.13802,0.00009,+0.318,0.026,  0.14734,0.00008,+0.405,0.027]]), axis=0)

		B = np.append(B, np.array([[  0., 30.,  0.14443,0.00027,+0.000,0.050,  0.14353,0.00018,+0.000,0.050,  0.14645,0.00019,+0.042,0.035]]), axis=0)
		B = np.append(B, np.array([[+30., 30.,  0.14193,0.00006,-0.681,0.050,  0.13969,0.00016,+0.662,0.070,  0.14627,0.00025,+0.117,0.045]]), axis=0)
		B = np.append(B, np.array([[+60., 30.,  0.14100,0.00017,+0.083,0.101,  0.13745,0.00016,+0.301,0.070,  0.14395,0.00015,+0.317,0.064]]), axis=0)
		B = np.append(B, np.array([[-30., 30.,  0.14536,0.00005,+1.408,0.035,  0.14540,0.00020,-0.448,0.092,  0.14466,0.00015,-0.140,0.034]]), axis=0)
		B = np.append(B, np.array([[-60., 30.,  0.14572,0.00018,+0.845,0.088,  0.14646,0.00012,-0.150,0.035,  0.14074,0.00020,-0.855,0.087]]), axis=0)

		B = np.append(B, np.array([[  0.,  0.,  0.14372,0.00031,+0.000,0.050,  0.14466,0.00031,+0.000,0.050,  0.14638,0.00024,+0.027,0.040]]), axis=0)
		B = np.append(B, np.array([[+30.,  0.,  0.14363,0.00002,+0.000,0.050,  0.14067,0.00012,+0.690,0.052,  0.14596,0.00023,+0.062,0.044]]), axis=0)
		B = np.append(B, np.array([[+60.,  0.,  0.14334,0.00012,+0.000,0.050,  0.13766,0.00038,+0.381,0.114,  0.14415,0.00021,+0.348,0.094]]), axis=0)
		B = np.append(B, np.array([[-30.,  0.,  0.14426,0.00046,+0.000,0.050,  0.14700,0.00015,-0.279,0.055,  0.14442,0.00028,-0.171,0.058]]), axis=0)
		B = np.append(B, np.array([[-60.,  0.,  0.14391,0.00012,+0.000,0.050,  0.14870,0.00012,+0.049,0.023,  0.14157,0.00020,-0.384,0.076]]), axis=0)

		B = np.append(B, np.array([[  0.,-30.,  0.14348,0.00034,+0.000,0.050,  0.14536,0.00012,+0.000,0.050,  0.14655,0.00037,-0.006,0.063]]), axis=0)
		B = np.append(B, np.array([[+30.,-30.,  0.14440,0.00010,+0.882,0.071,  0.14194,0.00018,+0.760,0.086,  0.14610,0.00031,+0.064,0.061]]), axis=0)
		B = np.append(B, np.array([[+60.,-30.,  0.14560,0.00021,+0.159,0.114,  0.14006,0.00018,+0.237,0.069,  0.14410,0.00019,+0.214,0.094]]), axis=0)
		B = np.append(B, np.array([[-30.,-30.,  0.14186,0.00007,-1.043,0.053,  0.14758,0.00009,-0.162,0.036,  0.14473,0.00043,-0.068,0.082]]), axis=0)
		B = np.append(B, np.array([[-60.,-30.,  0.14113,0.00011,-0.524,0.050,  0.14885,0.00012,+0.159,0.028,  0.14167,0.00016,-0.232,0.057]]), axis=0)

		B = np.append(B, np.array([[  0.,-45.,  0.14238,0.00014,+0.000,0.050,  0.14567,0.00015,+0.000,0.050,  0.14642,0.00024,-0.017,0.044]]), axis=0)
		B = np.append(B, np.array([[+30.,-45.,  0.14444,0.00011,+0.521,0.057,  0.14300,0.00011,+0.880,0.067,  0.14620,0.00010,+0.035,0.019]]), axis=0)
		B = np.append(B, np.array([[+60.,-45.,  0.14600,0.00017,+0.131,0.061,  0.14096,0.00020,+0.487,0.091,  0.14406,0.00023,+0.121,0.111]]), axis=0)
		B = np.append(B, np.array([[-30.,-45.,  0.14055,0.00021,-0.560,0.117,  0.14750,0.00009,-0.158,0.042,  0.14449,0.00015,-0.101,0.028]]), axis=0)
		B = np.append(B, np.array([[-60.,-45.,  0.13952,0.00014,-0.295,0.052,  0.14849,0.00004,+0.247,0.011,  0.14157,0.00011,-0.188,0.041]]), axis=0)

		## Fill dataframe for data
		df_B = pd.DataFrame(data=B, columns=['thetaX', 'thetaZ', 'Bx_Offset', 'dBx_Offset', 'Bx_Grad', 'dBx_Grad', \
			'By_Offset', 'dBy_Offset', 'By_Grad', 'dBy_Grad', 'Bz_Offset', 'dBz_Offset', 'Bz_Grad', 'dBz_Grad'])

		## Generate 2D interpolants to B-Offsets and gradients
		nInt = 51
		x = df_B['thetaX'].to_numpy()
		y = df_B['thetaZ'].to_numpy()
		X, Y = np.meshgrid(np.linspace(x.min(), x.max(), nInt), np.linspace(y.min(), y.max(), nInt))

		## Fill dictionary with interpolants and mean uncertainties
		BInt = {'thetaX': X.flatten(), 'thetaZ': Y.flatten()}
		QList = ['Bx_Offset', 'By_Offset', 'Bz_Offset', 'Bx_Grad', 'By_Grad', 'Bz_Grad']
		for Q in QList:
			BInt[Q] = griddata((x, y), df_B[Q].to_numpy(), (X, Y), method='cubic', fill_value=0).flatten()
			BInt['d'+Q] = df_B['d'+Q].mean()

		return df_B, BInt

	################ End of self.BFieldCalibration() ################
	#################################################################

	@staticmethod
	def LightShiftCalibration():
		"""Calibration for one-photon light shift offset and gradient as a function of tilt angles.
		Note that the tilt angles here are already in the correct quadrant:
			thetaX = np.arcsin(np.sin( TiltX*ni.pi/180.))*180./np.pi
			thetaZ = np.arcsin(np.sin(-TiltZ*ni.pi/180.))*180./np.pi
		"""

		fLS = np.empty((0,14), dtype=float) ## [thetaX, thetaZ, fLSx_Offset, dfLSx_Offset, fLSx_Grad, dfLSx_Grad, ...]

		## Units: [deg, deg, kHz, kHz, kHz/mm, kHz/mm, ...]
		fLS = np.append(fLS, np.array([[  0., 45.,  +1.139,0.096,+0.000,0.100,  +1.132,0.155,+0.000,0.100,  +0.076,0.104,-0.007,0.019]]), axis=0)
		fLS = np.append(fLS, np.array([[-30., 45.,  +1.147,0.091,-0.287,0.048,  +0.757,0.059,+0.246,0.033,  +1.310,0.082,+0.444,0.020]]), axis=0)
		fLS = np.append(fLS, np.array([[+30., 45.,  +1.053,0.072,+0.192,0.038,  +1.174,0.134,-0.188,0.065,  +0.562,0.115,+0.006,0.020]]), axis=0)
		fLS = np.append(fLS, np.array([[+45., 45.,  +1.155,0.064,+0.171,0.025,  +1.550,0.173,-0.230,0.058,  +0.659,0.106,+0.035,0.023]]), axis=0)
		fLS = np.append(fLS, np.array([[-44., 45.,  +1.030,0.054,-0.151,0.020,  +1.164,0.081,+0.401,0.032,  +1.186,0.182,+0.331,0.048]]), axis=0)
		fLS = np.append(fLS, np.array([[-60., 45.,  +1.196,0.170,-0.118,0.050,  +0.948,0.156,+0.243,0.052,  +0.975,0.250,+0.246,0.104]]), axis=0)
		fLS = np.append(fLS, np.array([[+60., 45.,  +1.109,0.053,+0.157,0.018,  +0.792,0.231,+0.068,0.063,  +0.436,0.237,+0.000,0.080]]), axis=0)

		fLS = np.append(fLS, np.array([[  0., 30.,  +1.748,0.211,+0.000,0.100,  +1.451,0.075,+0.000,0.100,  +0.643,0.558,+0.017,0.100]]), axis=0)
		fLS = np.append(fLS, np.array([[+30., 30.,  +1.264,0.158,+0.366,0.121,  +1.683,0.157,+0.318,0.068,  +1.837,0.497,+0.140,0.088]]), axis=0)
		fLS = np.append(fLS, np.array([[+60., 30.,  +1.514,0.279,+0.301,0.161,  +1.092,0.331,-0.048,0.090,  +1.745,0.358,+0.317,0.142]]), axis=0)
		fLS = np.append(fLS, np.array([[-30., 30.,  +1.544,0.189,-0.532,0.140,  +1.494,0.115,-0.236,0.053,  +2.129,0.487,+0.867,0.104]]), axis=0)
		fLS = np.append(fLS, np.array([[-60., 30.,  +1.571,0.048,-0.286,0.023,  +2.138,0.736,-0.437,0.213,  -0.262,0.701,+0.293,0.296]]), axis=0)

		fLS = np.append(fLS, np.array([[  0.,  0.,  +1.511,0.198,+0.000,0.100,  +1.474,0.128,+0.000,0.100,  +1.172,0.416,-0.021,0.069]]), axis=0)
		fLS = np.append(fLS, np.array([[+30.,  0.,  +2.144,0.031,+0.000,0.100,  +1.135,0.247,+0.145,0.100,  +0.663,0.430,-0.083,0.080]]), axis=0)
		fLS = np.append(fLS, np.array([[+60.,  0.,  +1.808,0.185,+0.000,0.100,  +1.308,0.405,+0.125,0.119,  +1.412,0.392,+0.255,0.166]]), axis=0)
		fLS = np.append(fLS, np.array([[-30.,  0.,  +1.705,0.373,+0.000,0.100,  +1.540,0.246,-0.294,0.087,  +1.521,0.380,+0.485,0.076]]), axis=0)
		fLS = np.append(fLS, np.array([[-60.,  0.,  +1.762,0.181,+0.000,0.100,  +1.420,0.127,-0.104,0.023,  +1.316,0.425,+0.474,0.155]]), axis=0)

		fLS = np.append(fLS, np.array([[  0.,-30.,  +1.791,0.491,+0.000,0.100,  +1.434,0.060,+0.000,0.100,  +1.791,0.812,+0.209,0.137]]), axis=0)
		fLS = np.append(fLS, np.array([[+30.,-30.,  +2.079,0.245,-0.851,0.166,  +1.226,0.257,-0.213,0.119,  +0.811,0.276,+0.061,0.052]]), axis=0)
		fLS = np.append(fLS, np.array([[+60.,-30.,  +2.289,0.419,-0.511,0.212,  +1.276,0.240,-0.098,0.087,  +2.046,0.228,+0.567,0.106]]), axis=0)
		fLS = np.append(fLS, np.array([[-30.,-30.,  +1.622,0.156,+0.647,0.110,  +1.474,0.214,+0.230,0.082,  +0.972,0.457,+0.030,0.084]]), axis=0)
		fLS = np.append(fLS, np.array([[-60.,-30.,  +1.397,0.141,+0.309,0.060,  +1.565,0.159,+0.133,0.035,  +0.261,0.326,-0.103,0.115]]), axis=0)

		fLS = np.append(fLS, np.array([[  0.,-45.,  +2.312,0.171,+0.000,0.100,  +1.808,0.246,+0.000,0.100,  +1.281,0.709,+0.026,0.125]]), axis=0)
		fLS = np.append(fLS, np.array([[+30.,-45.,  +1.140,0.377,-0.316,0.190,  +1.326,0.216,-0.502,0.127,  +0.356,0.109,+0.006,0.022]]), axis=0)
		fLS = np.append(fLS, np.array([[+60.,-45.,  +1.151,0.330,-0.065,0.114,  +1.041,0.352,-0.259,0.159,  +0.563,0.275,-0.040,0.127]]), axis=0)
		fLS = np.append(fLS, np.array([[-30.,-45.,  +1.653,0.208,-0.470,0.112,  +1.641,0.351,+0.325,0.165,  +0.804,0.159,+0.092,0.030]]), axis=0)
		fLS = np.append(fLS, np.array([[-60.,-45.,  +1.363,0.097,+0.218,0.033,  +1.305,0.132,+0.048,0.034,  +1.966,0.175,+0.208,0.062]]), axis=0)

		df_fLS = pd.DataFrame(fLS, columns=['thetaX', 'thetaZ', 'fLSx_Offset', 'dfLSx_Offset', 'fLSx_Grad', 'dfLSx_Grad', \
			'fLSy_Offset', 'dfLSy_Offset', 'fLSy_Grad', 'dfLSy_Grad', 'fLSz_Offset', 'dfLSz_Offset', 'fLSz_Grad', 'dfLSz_Grad'])

		## Generate 2D interpolants to B-Offsets and gradients
		nInt = 51
		x = df_fLS['thetaX'].to_numpy()
		y = df_fLS['thetaZ'].to_numpy()
		X, Y = np.meshgrid(np.linspace(x.min(), x.max(), nInt), np.linspace(y.min(), y.max(), nInt))

		## Fill dictionary with interpolants and mean uncertainties
		fLSInt = {'thetaX': X.flatten(), 'thetaZ': Y.flatten()}
		QList = ['fLSx_Offset', 'fLSy_Offset', 'fLSz_Offset', 'fLSx_Grad', 'fLSy_Grad', 'fLSz_Grad']
		for Q in QList:
			fLSInt[Q] = griddata((x, y), df_fLS[Q].to_numpy(), (X, Y), method='cubic', fill_value=0).flatten()
			fLSInt['d'+Q] = df_fLS['d'+Q].mean()

		return df_fLS, fLSInt

	############## End of self.LightShiftCalibration() ##############
	#################################################################

	@staticmethod
	def RabiFreqCalibration(TOF):
		"""Calibration for Rabi frequency as a function of tilt angles at a given TOF.
		Note that the tilt angles here are already in the correct quadrant:
			thetaX = np.arcsin(np.sin( TiltX*ni.pi/180.))*180./np.pi
			thetaZ = np.arcsin(np.sin(-TiltZ*ni.pi/180.))*180./np.pi
		"""

		Om = np.empty((0,9), dtype=float) ## [thetaX, thetaZ, TOF, OmegaX, dOmegaX, OmegaY, dOmegaY, OmegaZ, dOmegaZ]

		## Units: [deg, deg, s, rad/us, rad/us, ...]
		Om = np.append(Om, np.array([[+54.7, +45.0, 0.020000, 0.320622,	0.002507,	0.291259, 	0.001998, 	0.339011, 	0.002337]]), axis=0)
		Om = np.append(Om, np.array([[+54.7, +45.0, 0.025000, 0.290664,	0.001243,	0.305805,	0.001879,	0.325630,	0.002461]]), axis=0)
		Om = np.append(Om, np.array([[+54.7, +45.0, 0.030000, 0.250738,	0.001496,	0.274142,	0.001965,	0.313684,	0.001752]]), axis=0)
		Om = np.append(Om, np.array([[+54.7, +45.0, 0.035000, 0.187087,	0.002649,	0.224150,	0.001368,	0.264876,	0.001364]]), axis=0)
		Om = np.append(Om, np.array([[+54.7, +45.0, 0.040000, 0.128791,	0.009619,	0.179052,	0.003772,	0.217344,	0.001171]]), axis=0)

		Om = np.append(Om, np.array([[+45.0, +30.0, 0.020000, 0.313495,	0.002556,	0.296660,	0.004074,	0.315533,	0.003208]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, +30.0, 0.025000, 0.283772,	0.002959,	0.279956,	0.004316,	0.313448,	0.001759]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, +30.0, 0.030000, 0.219086,	0.008353,	0.258429,	0.004361,	0.298236,	0.002000]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, +30.0, 0.035000, 0.156236,	0.026483,	0.239824,	0.048645,	0.264820,	0.003447]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, +30.0, 0.040000, 0.028485,	0.324748,	0.187376,	0.112447,	0.205933,	0.008868]]), axis=0)

		Om = np.append(Om, np.array([[-60.0, -30.0, 0.020000, 0.325739,	0.001363,	0.301818,	0.001566,	0.341487,	0.001910]]), axis=0)
		Om = np.append(Om, np.array([[-60.0, -30.0, 0.025000, 0.304783,	0.001823,	0.310976,	0.001613,	0.316221,	0.001362]]), axis=0)
		Om = np.append(Om, np.array([[-60.0, -30.0, 0.030000, 0.260469,	0.001136,	0.303136,	0.001516,	0.294262,	0.001367]]), axis=0)
		Om = np.append(Om, np.array([[-60.0, -30.0, 0.035000, 0.192103,	0.004521,	0.264609,	0.001356,	0.231853,	0.001767]]), axis=0)
		Om = np.append(Om, np.array([[-60.0, -30.0, 0.040000, 0.099114,	0.028206,	0.222289,	0.002202,	0.153076,	0.012309]]), axis=0)

		Om = np.append(Om, np.array([[-45.0, -30.0, 0.020000, 0.325422,	0.007961,	0.319539,	0.028579,	0.388969,	0.019160]]), axis=0)
		Om = np.append(Om, np.array([[-45.0, -30.0, 0.025000, 0.282726,	0.001584,	0.277728,	0.001148,	0.330130,	0.001557]]), axis=0)
		Om = np.append(Om, np.array([[-45.0, -30.0, 0.030000, 0.220609,	0.002144,	0.246529,	0.000928,	0.311669,	0.001408]]), axis=0)
		Om = np.append(Om, np.array([[-45.0, -30.0, 0.035000, 0.167039,	0.009080,	0.213011,	0.002678,	0.272486,	0.001287]]), axis=0)
		Om = np.append(Om, np.array([[-45.0, -30.0, 0.040000, 0.063120,	0.090988,	0.160182,	0.012288,	0.212860,	0.002907]]), axis=0)

		Om = np.append(Om, np.array([[-30.0, -30.0, 0.020000, 0.303773,	0.001933,	0.274827,	0.001231,	0.354435,	0.001546]]), axis=0)
		Om = np.append(Om, np.array([[-30.0, -30.0, 0.025000, 0.289956,	0.001876,	0.247963,	0.001568,	0.342391,	0.001687]]), axis=0)
		Om = np.append(Om, np.array([[-30.0, -30.0, 0.030000, 0.218523,	0.002188,	0.213150,	0.001586,	0.333450,	0.001561]]), axis=0)
		Om = np.append(Om, np.array([[-30.0, -30.0, 0.035000, 0.110883,	0.015656,	0.156419,	0.008274,	0.309668,	0.001356]]), axis=0)
		Om = np.append(Om, np.array([[-30.0, -30.0, 0.040000, 0.010620,	0.704005,	0.098215,	0.045200,	0.277560,	0.000804]]), axis=0)

		Om = np.append(Om, np.array([[-15.0, -30.0, 0.020000, 0.261592,	0.005194,	0.252377,	0.001969,	0.348636,	0.001699]]), axis=0)
		Om = np.append(Om, np.array([[-15.0, -30.0, 0.025000, 0.247711,	0.003105,	0.227776,	0.002840,	0.357241,	0.001765]]), axis=0)
		Om = np.append(Om, np.array([[-15.0, -30.0, 0.030000, 0.193059,	0.004480,	0.182621,	0.006777,	0.349733,	0.001837]]), axis=0)
		Om = np.append(Om, np.array([[-15.0, -30.0, 0.035000, 0.139956,	0.019132,	0.131053,	0.013919,	0.327040,	0.001264]]), axis=0)
		Om = np.append(Om, np.array([[-15.0, -30.0, 0.040000, 0.031272,	0.391726,	0.018480,	0.459056,	0.326186,	0.001350]]), axis=0)

		Om = np.append(Om, np.array([[+15.0, -30.0, 0.020000, 0.308422,	0.087442,	0.204495,	0.024419,	0.310632,	0.001594]]), axis=0)
		Om = np.append(Om, np.array([[+15.0, -30.0, 0.025000, 0.265674,	0.030757,	0.202132,	0.010424,	0.320568,	0.001031]]), axis=0)
		Om = np.append(Om, np.array([[+15.0, -30.0, 0.030000, 0.219006,	0.009915,	0.143640,	0.017670,	0.304548,	0.001105]]), axis=0)
		Om = np.append(Om, np.array([[+15.0, -30.0, 0.035000, 0.134538,	0.123965,	0.092820,	0.077211,	0.292258,	0.000974]]), axis=0)
		Om = np.append(Om, np.array([[+15.0, -30.0, 0.040000, 0.050052,	0.310798,	0.015149,	0.339617,	0.263930,	0.000995]]), axis=0)

		Om = np.append(Om, np.array([[+30.0, -30.0, 0.020000, 0.275600,	0.002249,	0.237906,	0.002096,	0.314469,	0.001195]]), axis=0)
		Om = np.append(Om, np.array([[+30.0, -30.0, 0.025000, 0.263176,	0.001304,	0.197983,	0.003696,	0.301723,	0.001101]]), axis=0)
		Om = np.append(Om, np.array([[+30.0, -30.0, 0.030000, 0.186914,	0.003327,	0.151200,	0.007516,	0.272211,	0.000823]]), axis=0)
		Om = np.append(Om, np.array([[+30.0, -30.0, 0.035000, 0.118562,	0.019066,	0.097149,	0.030509,	0.224869,	0.001992]]), axis=0)
		Om = np.append(Om, np.array([[+30.0, -30.0, 0.040000, 0.012385,	0.392394,	0.017078,	0.743406,	0.180523,	0.005332]]), axis=0)

		Om = np.append(Om, np.array([[+45.0, -30.0, 0.020000, 0.319103,	0.001660,	0.234111,	0.001432,	0.304770,	0.001221]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, -30.0, 0.025000, 0.287475,	0.001615,	0.212732,	0.002378,	0.281245,	0.000995]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, -30.0, 0.030000, 0.208787,	0.002931,	0.167252,	0.005353,	0.242140,	0.001469]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, -30.0, 0.035000, 0.144472,	0.008812,	0.116381,	0.017787,	0.181172,	0.005252]]), axis=0)
		Om = np.append(Om, np.array([[+45.0, -30.0, 0.040000, 0.050813,	0.139960,	0.025531,	8.912652,	0.116847,	0.014406]]), axis=0)

		Om = np.append(Om, np.array([[+60.0, -30.0, 0.020000, 0.337338,	0.003379,	0.245807,	0.009172,	0.304773,	0.004240]]), axis=0)
		Om = np.append(Om, np.array([[+60.0, -30.0, 0.025000, 0.294892,	0.003944,	0.235705,	0.034936,	0.283349,	0.003731]]), axis=0)
		Om = np.append(Om, np.array([[+60.0, -30.0, 0.030000, 0.254743,	0.004592,	0.193920,	0.021518,	0.219386,	0.008563]]), axis=0)
		Om = np.append(Om, np.array([[+60.0, -30.0, 0.035000, 0.184032,	0.021722,	0.131029,	0.054981,	0.159485,	0.029760]]), axis=0)
		Om = np.append(Om, np.array([[+60.0, -30.0, 0.040000, 0.107295,	0.141383,	0.086473,	0.205919,	0.031771,	0.661769]]), axis=0)

		## Fill dataframe for data
		df_Om = pd.DataFrame(data=Om, columns=['thetaX', 'thetaZ', 'TOF', 'OmegaX', 'dOmegaX', 'OmegaY', 'dOmegaY', 'OmegaZ', 'dOmegaZ'])

		## Generate 2D interpolants to Rabi frequencies
		nInt = 51
		mask = df_Om['TOF'] == TOF
		x = df_Om.loc[mask, 'thetaX'].to_numpy()
		print(x)
		y = df_Om.loc[mask, 'thetaZ'].to_numpy()
		# x = df_Om['thetaX'].to_numpy()
		# y = df_Om['thetaZ'].to_numpy()
		X, Y = np.meshgrid(np.linspace(x.min(), x.max(), nInt), np.linspace(y.min(), y.max(), nInt))

		## Fill dictionary with interpolants and mean uncertainties
		OmInt = {'thetaX': X.flatten(), 'thetaZ': Y.flatten()}
		QList = ['OmegaX', 'OmegaY', 'OmegaZ']
		for Q in QList:
			OmInt[Q] = griddata((x, y), df_Om.loc[mask, Q].to_numpy(), (X, Y), method='cubic', fill_value=np.nan).flatten()
			OmInt['d'+Q] = df_Om.loc[mask, ['d'+Q]].mean()

		return df_Om, OmInt

	############### End of self.RabiFreqCalibration() ###############
	#################################################################

	@staticmethod
	def PlotCalibrations(df_B, fLSdf, thetaX=None, thetaZ=None):
		"""Plot calibrations for B-field and one-photon light shift as a function of tilt angles."""

		iXUtils.SetDefaultPlotOptions()

		nRows, nCols = (2,2)
		axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nRows*5,nCols*3), sharex='col', constrained_layout=True)[1]

		plotOpts = {'Color': 'black', 'Linestyle': 'None', 'Marker': '.', 'Title': 'None',
			'xLabel': 'None', 'yLabel': 'None', 'LegLabel': None, 'Legend': False, 'LegLocation': 'best'}

		if thetaZ != None and thetaX == None:
			mask   = (df_B['thetaZ'] == thetaZ)
			df_B    = df_B.loc[mask]
			fLSdf  = fLSdf.loc[mask]
			xList  = df_B['thetaX'].to_numpy()
			xLabel = r'$\theta_x$  (deg)'
		elif thetaZ == None and thetaX != None:
			mask   = (df_B['thetaX'] == thetaX)
			df_B    = df_B.loc[mask]
			fLSdf  = fLSdf.loc[mask]
			xList  = df_B['thetaZ'].to_numpy()
			xLabel = r'$\theta_z$  (deg)'

		plotOpts['Color']    = 'darkgreen'
		plotOpts['LegLabel'] = 'X'
		plotOpts['yLabel']   = r'$B_0$  (G)'
		iXUtils.CustomPlot(axs[0,0], plotOpts, xList, df_B['Bx_Offset'].to_numpy(), df_B['dBx_Offset'].to_numpy())
		plotOpts['yLabel']   = r'$\nabla B$  (G/m)'
		iXUtils.CustomPlot(axs[0,1], plotOpts, xList, df_B['Bx_Grad'].to_numpy(), df_B['dBx_Grad'].to_numpy())

		plotOpts['Color']    = 'blue'
		plotOpts['LegLabel'] = 'Y'
		plotOpts['yLabel']   = r'$B_0$  (G)'
		iXUtils.CustomPlot(axs[0,0], plotOpts, xList, df_B['By_Offset'].to_numpy(), df_B['dBy_Offset'].to_numpy())
		plotOpts['yLabel']   = r'$\nabla B$  (G/m)'
		iXUtils.CustomPlot(axs[0,1], plotOpts, xList, df_B['By_Grad'].to_numpy(), df_B['dBy_Grad'].to_numpy())

		plotOpts['Color']    = 'red'
		plotOpts['LegLabel'] = 'Z'
		plotOpts['yLabel']   = r'$B_0$  (G)'
		iXUtils.CustomPlot(axs[0,0], plotOpts, xList, df_B['Bz_Offset'].to_numpy(), df_B['dBz_Offset'].to_numpy())
		plotOpts['yLabel']   = r'$\nabla B$  (G/m)'
		iXUtils.CustomPlot(axs[0,1], plotOpts, xList, df_B['Bz_Grad'].to_numpy(), df_B['dBz_Grad'].to_numpy())

		plotOpts['Color']    = 'darkgreen'
		plotOpts['LegLabel'] = 'X'
		plotOpts['xLabel']   = xLabel
		plotOpts['yLabel']   = r'$\delta_{\rm LS}$  (kHz)'
		iXUtils.CustomPlot(axs[1,0], plotOpts, xList, fLSdf['fLSx_Offset'].to_numpy(), fLSdf['dfLSx_Offset'].to_numpy())
		plotOpts['yLabel']   = r'$\nabla \delta_{\rm LS}$  (kHz/mm)'
		iXUtils.CustomPlot(axs[1,1], plotOpts, xList, fLSdf['fLSx_Grad'].to_numpy(), fLSdf['dfLSx_Grad'].to_numpy())

		plotOpts['Color']    = 'blue'
		plotOpts['LegLabel'] = 'Y'
		plotOpts['yLabel']   = r'$\delta_{\rm LS}$  (kHz)'
		iXUtils.CustomPlot(axs[1,0], plotOpts, xList, fLSdf['fLSy_Offset'].to_numpy(), fLSdf['dfLSy_Offset'].to_numpy())
		plotOpts['yLabel']   = r'$\nabla \delta_{\rm LS}$  (kHz/mm)'
		iXUtils.CustomPlot(axs[1,1], plotOpts, xList, fLSdf['fLSy_Grad'].to_numpy(), fLSdf['dfLSy_Grad'].to_numpy())

		plotOpts['Color']    = 'red'
		plotOpts['LegLabel'] = 'Z'
		plotOpts['yLabel']   = r'$\delta_{\rm LS}$  (kHz)'
		iXUtils.CustomPlot(axs[1,0], plotOpts, xList, fLSdf['fLSz_Offset'].to_numpy(), fLSdf['dfLSz_Offset'].to_numpy())
		plotOpts['yLabel']   = r'$\nabla \delta_{\rm LS}$  (kHz/mm)'
		iXUtils.CustomPlot(axs[1,1], plotOpts, xList, fLSdf['fLSz_Grad'].to_numpy(), fLSdf['dfLSz_Grad'].to_numpy())

		axs[0,1].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		axs[1,1].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))

		plt.show()

	################# End of self.PlotCalibrations() ################
	#################################################################

	@staticmethod
	def PlotCalibration2D(df, Int, Quantity='B_Offset', TOF=0.):
		"""Plot calibrations for B-field, light shifts, or Rabi frequencies as a function of tilt angles."""

		iXUtils.SetDefaultPlotOptions()

		x = df['thetaX'].to_numpy()
		y = df['thetaZ'].to_numpy()
		if Quantity == 'B_Offset':
			z1 = df['Bx_Offset'].to_numpy()
			z2 = df['By_Offset'].to_numpy()
			z3 = df['Bz_Offset'].to_numpy()
		elif Quantity == 'B_Grad':
			z1 = df['Bx_Grad'].to_numpy()
			z2 = df['By_Grad'].to_numpy()
			z3 = df['Bz_Grad'].to_numpy()
		elif Quantity == 'LS_Offset':
			z1 = df['fLSx_Offset'].to_numpy()
			z2 = df['fLSy_Offset'].to_numpy()
			z3 = df['fLSz_Offset'].to_numpy()
		elif Quantity == 'LS_Grad':
			z1 = df['fLSx_Grad'].to_numpy()
			z2 = df['fLSy_Grad'].to_numpy()
			z3 = df['fLSz_Grad'].to_numpy()
		else: ## Quantity == 'Omega':
			mask = df['TOF'] == TOF
			x  = df.loc[mask, 'thetaX'].to_numpy()
			y  = df.loc[mask, 'thetaZ'].to_numpy()
			z1 = df.loc[mask, 'OmegaX'].to_numpy()
			z2 = df.loc[mask, 'OmegaY'].to_numpy()
			z3 = df.loc[mask, 'OmegaZ'].to_numpy()

		print(x)

		fig = plt.figure(figsize=(3*5.,5.), constrained_layout=True)
		ax1 = fig.add_subplot(1, 3, 1, projection='3d')
		ax2 = fig.add_subplot(1, 3, 2, projection='3d')
		ax3 = fig.add_subplot(1, 3, 3, projection='3d')

		ax1.scatter(x, y, z1, color='black')
		ax2.scatter(x, y, z2, color='black')
		ax3.scatter(x, y, z3, color='black')

		## 2D interpolations
		X = Int['thetaX']
		Y = Int['thetaZ']
		if Quantity == 'B_Offset':
			Z1 = Int['Bx_Offset']
			Z2 = Int['By_Offset']
			Z3 = Int['Bz_Offset']
		elif Quantity == 'B_Grad':
			Z1 = Int['Bx_Grad']
			Z2 = Int['By_Grad']
			Z3 = Int['Bz_Grad']
		elif Quantity == 'LS_Offset':
			Z1 = Int['fLSx_Offset']
			Z2 = Int['fLSy_Offset']
			Z3 = Int['fLSz_Offset']
		elif Quantity == 'LS_Grad':
			Z1 = Int['fLSx_Grad']
			Z2 = Int['fLSy_Grad']
			Z3 = Int['fLSz_Grad']
		else: ## Quantity == 'Omega':
			Z1 = Int['OmegaX']
			Z2 = Int['OmegaY']
			Z3 = Int['OmegaZ']

		# ax1.scatter(X, Y, Z1, color='red', alpha=0.5)
		# ax2.scatter(X, Y, Z2, color='green', alpha=0.5)
		# ax3.scatter(X, Y, Z3, color='blue', alpha=0.5)

		ax1.scatter(X, Y, Z1, c=Z1, cmap=plt.get_cmap('summer'), s=10, alpha=0.5)
		ax2.scatter(X, Y, Z2, c=Z1, cmap=plt.get_cmap('autumn'), s=10, alpha=0.5)
		ax3.scatter(X, Y, Z3, c=Z1, cmap=plt.get_cmap('winter'), s=10, alpha=0.5)

		ax1.set_xlabel(r'$\theta_x$  (deg)')
		ax1.set_ylabel(r'$\theta_z$  (deg)')
		ax2.set_xlabel(r'$\theta_x$  (deg)')
		ax2.set_ylabel(r'$\theta_z$  (deg)')
		ax3.set_xlabel(r'$\theta_x$  (deg)')
		ax3.set_ylabel(r'$\theta_z$  (deg)')
		if Quantity == 'B_Offset':
			ax1.set_zlabel(r'$B_x$  (G)')
			ax2.set_zlabel(r'$B_y$  (G)')
			ax3.set_zlabel(r'$B_z$  (G)')
		elif Quantity == 'B_Grad':
			ax1.set_zlabel(r'$\partial B_x$  (G/m)')
			ax2.set_zlabel(r'$\partial B_y$  (G/m)')
			ax3.set_zlabel(r'$\partial B_z$  (G/m)')
		elif Quantity == 'LS_Offset':
			ax1.set_zlabel(r'$\delta_{\rm LS, x}$  (kHz)')
			ax2.set_zlabel(r'$\delta_{\rm LS, y}$  (kHz)')
			ax3.set_zlabel(r'$\delta_{\rm LS, z}$  (kHz)')
		elif Quantity == 'LS_Grad':
			ax1.set_zlabel(r'$\partial \delta_{\rm LS, x}$  (kHz/mm)')
			ax2.set_zlabel(r'$\partial \delta_{\rm LS, y}$  (kHz/mm)')
			ax3.set_zlabel(r'$\partial \delta_{\rm LS, z}$  (kHz/mm)')
		else: ## Quantity == 'Omega':
			ax1.set_zlabel(r'$\Omega_X$  (rad/$\mu$s)')
			ax2.set_zlabel(r'$\Omega_Y$  (rad/$\mu$s)')
			ax3.set_zlabel(r'$\Omega_Z$  (rad/$\mu$s)')

		plt.show()

	################ End of self.PlotCalibration2D() ################
	#################################################################

#####################################################################
####################### End of class Physics ########################
#####################################################################

#####################################################################
################# Test routines for Physics class ###################

def TestETGTAB():
	TidePars = {
		'Latitude':		phys.Latitude,			## [deg]
		'Longitude': 	phys.Longitude,			## [deg]
		'Height':		phys.Height,			## [m]
		'Gravity':		phys.gLocal,			## [m/s^2]
		'Azimuth':		0.,						## [deg]
		'TidalComp':	'VerticalAcceleration', ## 'TidalPotential', 'VerticalAcceleration', 'HorizontalAcceleration', 'VerticalDisplacement', 'HorizontalDisplacement', 'VerticalStrain', 'HorizontalStrain', 'ArealStrain', 'ShearStrain', 'VolumeStrain', 'OceanTides'
		'PrintLevel':	'None',					## 'None', 'Some', 'All'
		'TimeStart':	[2019, 12, 5, 18],		## [YYYY,MM,DD,HH]
		'TimeSpan':		50,  					## [hrs]
		'TimeStep':		300, 					## [s] 300 or 3600 only
		'TidalModel':	'Tamura1987',			## 'Doodson1921', 'CTE1973', 'Tamura1987', 'Buellesfeld1985'
		'EarthModel':	'Elastic'				## 'Rigid', 'Elastic'
	}

	phys.RunETGTAB(TidePars, True)

	# dates = mpl.dates.date2num(list(phys.TidalDF['DateTime']))
	# plt.plot_date(dates, phys.TidalDF['gAnomaly'].to_numpy())

	plt.plot(phys.TidalDF['RelTime'].to_numpy(), phys.TidalDF['gAnomaly'].to_numpy()/phys.gLocal)
	plt.show()

###################### End of TestETGTAB() ##########################
#####################################################################

def TestLightShifts():
	"""Test module for dynamic polarizabilities and AC Stark shifts."""

	# omegaL1 = phys.omega21_Rb - 2*np.pi*700.E6
	# omegaL2 = omegaL1 + 2*np.pi*6.834E9
	# omegaL1 = phys.omegaL1_Rb
	# omegaL2 = phys.omegaL2_Rb
	# alphaS_F1L1, alphaS_F2L1, alphaV_F1L1, alphaV_F2L1, alphaT_F1L1, alphaT_F2L1 = phys.alphaDynamic(omegaL1)
	# alphaS_F1L2, alphaS_F2L2, alphaV_F1L2, alphaV_F2L2, alphaT_F1L2, alphaT_F2L2 = phys.alphaDynamic(omegaL2)

	r21 = 1.76  ## Raman beam intensity ration (I2/I1)
	tau = 10.E-6 ## Raman pi-pulse length (s)
	I2  = phys.I2Raman(np.pi/tau, r21)
	# I2 = 2.5*10. ## Raman beam 2 intensity (W/m^2)
	I1 = I2/r21  ## Raman beam 1 intensity (W/m^2)
	A  = 0.      ## Degree of circular polarization
	deltaS_mFp1, deltaV_mFp1, deltaT_mFp1, alphas_L1, alphas_L2 = phys.deltaAC(I1, I2, A, phys.Delta, +1., returnAlphas=True)
	deltaS_mF0 , deltaV_mF0 , deltaT_mF0  = phys.deltaAC(I1, I2, A, phys.Delta,  0., returnAlphas=False)
	deltaS_mFm1, deltaV_mFm1, deltaT_mFm1 = phys.deltaAC(I1, I2, A, phys.Delta, -1., returnAlphas=False)

	alphaS_F1L1, alphaS_F2L1, alphaV_F1L1, alphaV_F2L1, alphaT_F1L1, alphaT_F2L1 = alphas_L1
	alphaS_F1L2, alphaS_F2L2, alphaV_F1L2, alphaV_F2L2, alphaT_F1L2, alphaT_F2L2 = alphas_L2

	print('---------------------- 87Rb ----------------------')
	print('I1 = {:.2f} W/m^2'.format(I1))
	print('I2 = {:.2f} W/m^2'.format(I2))
	print('IT = {:.2f} W/m^2'.format(I1+I2))
	print('Scalar alpha:  F = 1 \t{:.4f}\t{:.4f}\th.kHz/(V/cm)^2'.format(alphaS_F1L1/phys.hPlanck*10, alphaS_F1L2/phys.hPlanck*10))
	print('               F = 2 \t{:.4f}\t{:.4f}\th.kHz/(V/cm)^2'.format(alphaS_F2L1/phys.hPlanck*10, alphaS_F2L2/phys.hPlanck*10))
	print('Vector alpha:  F = 1 \t{:.4f}\t{:.4f}\th.kHz/(V/cm)^2'.format(alphaV_F1L1/phys.hPlanck*10, alphaV_F1L2/phys.hPlanck*10))
	print('               F = 2 \t{:.4f}\t{:.4f}\th.kHz/(V/cm)^2'.format(alphaV_F2L1/phys.hPlanck*10, alphaV_F2L2/phys.hPlanck*10))
	print('Tensor alpha:  F = 1 \t{:.4f}\t{:.4f}\th.kHz/(V/cm)^2'.format(alphaT_F1L1/phys.hPlanck*10, alphaT_F1L2/phys.hPlanck*10))
	print('               F = 2 \t{:.4f}\t{:.4f}\th.kHz/(V/cm)^2'.format(alphaT_F2L1/phys.hPlanck*10, alphaT_F2L2/phys.hPlanck*10))

	print('Scalar shift: mF = +1\t{:.4f}\tkHz'.format(deltaS_mFp1*1.E-3/(2*np.pi)))
	print('              mF =  0\t{:.4f}\tkHz'.format(deltaS_mF0 *1.E-3/(2*np.pi)))
	print('              mF = -1\t{:.4f}\tkHz'.format(deltaS_mFm1*1.E-3/(2*np.pi)))
	print('Vector shift: mF = +1\t{:.4f}\tkHz\t{:.4f}\tmG'.format(deltaV_mFp1*1.E-3/(2*np.pi), phys.hbar*deltaV_mFp1/phys.muB*1.E3))
	print('              mF =  0\t{:.4f}\tkHz\t{:.4f}\tmG'.format(deltaV_mF0 *1.E-3/(2*np.pi), phys.hbar*deltaV_mF0 /phys.muB*1.E3))
	print('              mF = -1\t{:.4f}\tkHz\t{:.4f}\tmG'.format(deltaV_mFm1*1.E-3/(2*np.pi), phys.hbar*deltaV_mFm1/phys.muB*1.E3))
	print('Tensor shift: mF = +1\t{:.4f}\tkHz'.format(deltaT_mFp1*1.E-3/(2*np.pi)))
	print('              mF =  0\t{:.4f}\tkHz'.format(deltaT_mF0 *1.E-3/(2*np.pi)))
	print('              mF = -1\t{:.4f}\tkHz'.format(deltaT_mFm1*1.E-3/(2*np.pi)))

################### End of TestLightShifts() ########################
#####################################################################

if __name__ == '__main__':
	class RunParameters:
		def __init__(self):
			self.SoftwareVersion= 3.4
			self.RamanDetuning	= -1.21E9
			self.RamanTOF		= 15.0E-3
			self.RamanT 		= 10.0E-3
			self.RamanpiX		= 12.0E-6
			self.RamanpiY		= 11.0E-6
			self.RamanpiZ      	= 10.0E-6
			self.RamankUFreq  	= [6.835005081E+9, 6.834401307E+9, 6.834999592E+9]
			self.RamankDFreq   	= [6.834387585E+9, 6.834985870E+9, 6.834398563E+9]
			self.RamankUChirp  	= [+1.472810000E+7, -1.436040000E+7, +1.444660000E+7]
			self.RamankDChirp 	= [-1.472810000E+7, +1.436040000E+7, -1.444660000E+7]
			self.SelectionFreqs	= [6.834650111E+9, 6.834650111E+9]
			self.TiltX   		= +54.7
			self.TiltZ   		= -45.0
			# self.TiltX   		= 0.
			# self.TiltZ   		= 0.

	runPars = RunParameters()
	phys    = Physics(runPars)

	# TestETGTAB()
	# TestLightShifts()

	# print(phys.omegaR)
	# print(phys.vR)
	# print(phys.keff)

	# RMatrix			= self.RotationMatrix(self.thetaZ, self.thetaX, 0.) ## Rotation matrix
	# self.aBody		= np.dot(RMatrix, self.aLocal)		## Acceleration vector in rotated frame (m/s^2)

	g = np.array([0., 0., -1.])*phys.gLocal
	thx, thz = np.array([phys.TiltX, -phys.TiltZ])*np.pi/180.
	R = phys.RotationMatrix(thz, thx, 0.)
	a = np.dot(R, g)
	print(a)

	# df_B, BInt = phys.BFieldCalibration()
	# df_fLS, fLSInt = phys.LightShiftCalibration()
	# TOF = 30.E-3
	# df_Om, OmInt = phys.RabiFreqCalibration(TOF)

	# phys.PlotCalibrations(df_B, fLSdf, thetaZ=45.)

	# phys.PlotCalibration2D(df_B, BInt, Quantity='B_Offset') ## 'B_Offset' or 'B_Grad'
	# phys.PlotCalibration2D(df_fLS, fLSInt, Quantity='LS_Grad') ## 'LS_Offset' or 'LS_Grad'
	# phys.PlotCalibration2D(df_Om, OmInt, Quantity='Omega', TOF=TOF) ## 'Omega'

	# ## Generate 2D interpolations
	# X, Y = np.meshgrid(np.linspace(x.min(), x.max(), 30), np.linspace(y.min(), y.max(), 30))
	# Z1 = griddata((x, y), z1, (X, Y), method='cubic', fill_value=0)

	# B = np.append(B, np.array([[+30.,  0.,  0.14363,0.00002,+0.000,0.050,  0.14067,0.00012,+0.690,0.052,  0.14596,0.00023,+0.062,0.044]]), axis=0)
	# B = np.append(B, np.array([[  0.,+30.,  0.14348,0.00034,+0.000,0.050,  0.14536,0.00012,+0.000,0.050,  0.14655,0.00037,-0.006,0.063]]), axis=0)

	# Q   = 'Bz_Grad'
	# thx = [0.]
	# thz = [30.]
	
	# BTest = griddata((BInt['thetaX'], BInt['thetaZ']), BInt[Q], (thx, thz), method='nearest')
	# print(BTest)