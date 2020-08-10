#####################################################################
## Filename:	iXAtom_Class_Physics.py
## Author:		B. Barrett
## Description: Physics class definition for iXAtom analysis package
## Version:		3.2.4
## Last Mod:	09/07/2020
##===================================================================
## Change Log:
## 02/12/2019 - Physics class defined.
##			  - Physical constants set during __init__ module.
##			  - Modules for magnetic field models defined and tested.
## 10/12/2019 - Added computation of tidal gravity anomaly (wrapper
##				for ETGTAB F77 code).
## 20/04/2020 - Moved systematics class to a dedicated file
##            - Updated __init__ method to include more physical
##				constants and spectral properties of 87Rb
## 09/07/2020 - Created new version (v3.2.4) to accomodate upgrades to
##				systematics class
#####################################################################

import datetime as dt
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz

from scipy.interpolate import interp1d

import iXAtom_Utilities as iXUtils

class Physics:
	#################################################################
	## Class for useful physical constants and formulas
	#################################################################

	def __init__(self, RunPars):
		"""Define and set physical constants."""

		twopi = 2*np.pi

		## Physical constants
		self.cLight    = 2.99792458E+08						## Speed of light (m/s)
		self.mu0       = 2E-7*twopi 						## Permeability of vacuum (N/A^2)
		self.epsilon0  = 1./(self.mu0*self.cLight**2)		## Permittivity of vacuum (F/m)
		self.hPlanck   = 6.62606896E-34						## Planck's constant (J*s)
		self.hbar	   = self.hPlanck/twopi 				## Reduced Planck's constant (J*s)
		self.kBoltz	   = 1.3806503E-23 		 				## Boltzmann's constant (J/K)
		self.muB       = 9.274009994E-28					## Bohr magneton (J/G)
		self.muN 	   = 5.050783699E-31					## Nuclear magneton (J/G)
		self.OmegaE    = 7.2921150E-5						## Rotation rate of the Earth (rad/s)

		## Location-dependent parameters
		self.Latitude  = 44.804								## Latitude  (deg North)
		self.Longitude = -0.605								## Longitude (deg East)
		self.Heading   = 45.								## Compass direction (deg, 0 = true North, 90 = true East, etc.)
		self.Height    = 21.								## Height above sea level (m)
		self.gLocal    = 9.805642							## Local gravity (m/s^2)
		self.Tzz       = 3086.0E-9							## Local vertical gravity gradient (s^-2)
		self.Tyy       = 0.5*self.Tzz						## Local north-south gravity gradient (s^-2)
		self.Txx       = 0.5*self.Tzz						## Local east-west gravity gradient (s^-2)
		self.dTzz      = 0.05*self.Tzz						## Uncertainty in vertical gravity gradient (s^-2)
		self.dTyy      = 0.05*self.Tyy						## Uncertainty in north-south gravity gradient (s^-2)
		self.dTxx      = 0.05*self.Txx						## Uncertainty in east-west gravity gradient (s^-2)

		## 87Rb properties
		self.MRb       = 1.443160648E-25					## Atomic mass of 87Rb (kg)
		self.tauLife   = 26.2348E-9							## Lifetime of 87Rb 5P_{3/2} (s)
		self.Gamma     = 1/(twopi*self.tauLife)				## Spontaneous decay rate of 87Rb 5P_{3/2} (Hz)
		self.omegaD2   = twopi*384.2304844685E+12			## Frequency for 87Rb D2 transition 5S_{1/2} -> 5P_{3/2} (rad/s) 
		self.omegaHF   = twopi*6.834682610904290E+09 		## Hyperfine splitting of 87Rb 5S_{1/2} (rad/s)
		self.dD2       = 3.584244E-29 						## Reduced electric dipole moment for 87Rb 5S_{1/2} -> 5P_{3/2} (C.m)
		# self.dD2_Rb   = np.sqrt((3*twopi*self.epsilon0*self.hbar*self.cLight**3)/(self.omegaD2**3*self.tauLife))

		## Frequency omegaNM corresponds to ground-excited state transitions between F = N, F'= M
		self.omega10   = self.omegaD2 + twopi*(+4271.677E+6 - 302.074E6)
		self.omega11   = self.omegaD2 + twopi*(+4271.677E+6 - 229.852E6)
		self.omega12   = self.omegaD2 + twopi*(+4271.677E+6 -  72.911E6)
		self.omega21   = self.omegaD2 + twopi*(-2563.006E+6 - 229.852E6)
		self.omega22   = self.omegaD2 + twopi*(-2563.006E+6 -  72.911E6)
		self.omega23   = self.omegaD2 + twopi*(-2563.006E+6 + 193.741E6)

		## Magnetic properties
		self.gS		   = 2.0023193043622					## Electron spin g-factor
		self.gL 	   = 0.99999369							## Electron orbital g-factor
		self.gI        = -0.0009951414						## Nuclear g-factor
		self.gJ		   = 2.00233113							## Fine structure Lande g-factor (87Rb 5S_{1/2})
		[S,L,J,I]	   = [0.5, 0., 0.5, 1.5]				## Electron and nuclear spins (87Rb 5S_{1/2})
		F1 			   = abs(J - I)							## Total angular momentum for hyperfine ground state
		F2 			   = abs(J + I)							## Hyperfine structure Lande g-factors (87Rb 5S_{1/2})
		self.gF1       = self.gJ*(F1*(F1 + 1.) - I*(I + 1.) + J*(J + 1.))/(2.*F1*(F1 + 1.)) + self.gI*(F1*(F1 + 1.) + I*(I + 1.) - J*(J + 1.))/(2.*F1*(F1 + 1.))
		self.gF2       = self.gJ*(F2*(F2 + 1.) - I*(I + 1.) + J*(J + 1.))/(2.*F2*(F2 + 1.)) + self.gI*(F2*(F2 + 1.) + I*(I + 1.) - J*(J + 1.))/(2.*F2*(F2 + 1.))
		self.alphaB    = 0.5*self.muB*(self.gF2 - self.gF1)/self.hPlanck     ## First-order Zeeman shift (Hz/G)
		self.KClock    = 575.146				 		 	## Clock shift of 87Rb 5S_{1/2} (Hz/G^2)
		self.Lambda    = self.hPlanck*self.KClock/self.MRb	## Magnetic force coefficient (rad/m^2/s^2/G^2)

		## Interferometer parameters loaded from RunPars
		self.Delta     = twopi*RunPars.RamanDetuning		## Raman laser detuning (rad/s)
		self.deltakU   = twopi*RunPars.kUpFrequency			## Raman kU frequency (rad/s)
		self.deltakD   = twopi*RunPars.kDownFrequency		## Raman kD frequency (rad/s)
		self.deltaSel  = twopi*RunPars.SelectionFreqs[1]	## Raman selection frequency (rad/s)		
		self.TOF       = np.full(3, RunPars.RamanTOF + 0.)	## Raman times-of-flight (relative to molasses) (s)
		self.T         = np.full(3, RunPars.RamanT) 		## Raman interrogation times (s)
		self.taupi 	   = np.array([RunPars.RamanpiX, RunPars.RamanpiY, RunPars.RamanpiZ]) ## Raman pi-pulse durations (s)
		self.taupio2   = 0.5*self.taupi  					## Raman pi/2-pulse durations (s)

		## Atom interferometer parameters
		self.omegaL2   = self.omega23 + self.Delta          ## Raman frequency 2 (rad/s)
		self.omegaL1   = self.omegaL2 + twopi*6.83468E9 	## Raman frequency 1 (rad/s)
		self.k1        = self.omegaL1/self.cLight			## Raman wavenumber 1 (rad/m)
		self.k2        = self.omegaL2/self.cLight			## Raman wavenumber 2 (rad/m)
		self.keff      = self.k1 + self.k2  				## Counter-propagating Raman wavenumber (rad/m)
		self.Deltak    = abs(self.k1 - self.k2)				## Co-propagating Raman wavenumber (rad/m)
		self.omegaD    = self.keff*self.gLocal*self.TOF 	## Doppler shift (rad/s) (assuming vertical orientation)
		self.omegaR    = self.hbar*self.keff**2/(2*self.MRb)## Recoil frequency (rad/s)
		self.vR        = self.hbar*self.keff/self.MRb       ## Recoil velocity (m/s)
		self.Omegaeff  = np.pi/self.taupi					## Effective Rabi frequencies (rad/s)
		self.Ttotal    = 2*(self.T + self.taupio2) + self.taupi ## Total interrogation times (s)
		self.Teff      = np.sqrt((self.T + self.taupi)*(self.T + (2./self.Omegaeff)*np.tan(self.Omegaeff*self.taupio2/2.))) ## Effective interrogation times (s)
		self.Seff      = self.keff*self.Teff**2 			## Effective AI scale factors (rad/m/s^2)
		self.IRatio    = 1.77 								## Raman beam intensity ratio (I2/I1)

	################### End of Physics.__init__() ###################
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
		tM  = t2L + 0.5*tau2
		t3L = t2R + T
		t3R = t3L + tau3

		if t <= t1L or t >= t3R:
			return 0.
		if t1L < t and t <= t1R:
			return -np.sin(Omega1*(t-t1L))/np.sin(Omega1*tau1)
		if t1R < t and t <= t2L:
			return -1.
		if t2L < t and t <= t2R:
			return np.sin(Omega2*(t-tM))/np.sin(Omega2*tau2)
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

		gF_Init  = self.gF1_Rb if F_Init  == 1. else self.gF2_Rb
		gF_Final = self.gF1_Rb if F_Final == 1. else self.gF2_Rb

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
		if abd(DmFgF) > 0.:
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
			return [deltaS, deltaV, deltaT, 
				[aS_F1L1, aS_F2L1, aV_F1L1, aV_F2L1, aT_F1L1, aT_F2L1],
				[aS_F1L2, aS_F2L2, aV_F1L2, aV_F2L2, aT_F1L2, aT_F2L2]]
		else:
			return [deltaS, deltaV, deltaT]

	################### End of Physics.deltaAC() ####################
	#################################################################

	def I2Raman(self, taupi):
		"""Compute the intensity of the main Raman beam (I2) based on effective Rabi frequency for a given species."""

		deff2 = (self.dD2**2)/3.	## Square of the effective far off-resonant dipole moment

		return (self.hPlanck*self.epsilon0*self.cLight/deff2)*(self.hbar*abs(self.Delta)*np.sqrt(self.IRatio)/taupi)

	################### End of Physics.I2Raman() ####################
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
			info.warning('Physics::SetETGTABParameters::Tidal component {} not recognized...'.format(self.TidalComp))
			info.warning('Physics::SetETGTABParameters::Setting to default: "VerticalAcceleration"')
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
			info.warning('Physics::SetETGTABParameters::Print level {} not recognized...'.format(self.PrintLevel))
			info.warning('Physics::SetETGTABParameters::Setting to default: "None".')
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
			info.warning('Physics::SetETGTABParameters::Tidal model {} not recognized...'.format(self.TidalModel))
			info.warning('Physics::SetETGTABParameters::Setting to default: "Tamura1987".')
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
			info.warning('Physics::SetETGTABParameters::Earth model {} not recognized...'.format(self.EarthModel))
			info.warning('Physics::SetETGTABParameters::Setting to default: "Elastic".')
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

		df = pd.read_csv(os.path.join(self.ETGTAB_Folder, self.ETGTAB_Output),
			sep='\s+', header=None, engine='python', skiprows=46, error_bad_lines=False, warn_bad_lines=False)

		df.columns = ['Y', 'M', 'D', 'H', 'm', 's', 'gAnomaly']
		(nRows, nCols) = df.shape

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
	def RotationMatrix(EulerAngles):
		"""Compute rotation matrix based on Euler angles."""

		cosphi   = np.cos(EulerAngles[0])
		costheta = np.cos(EulerAngles[1])
		cospsi   = np.cos(EulerAngles[2])
		sinphi   = np.sin(EulerAngles[0])
		sintheta = np.sin(EulerAngles[1])
		sinpsi   = np.sin(EulerAngles[2])

		RMatrix  = np.zeros((3,3))

		RMatrix[0,0] = +costheta*cospsi
		RMatrix[0,1] = +costheta*sinpsi
		RMatrix[0,2] = -sintheta
		RMatrix[1,0] = -cosphi*sinpsi + sinphi*sintheta*cospsi
		RMatrix[1,1] = +cosphi*cospsi + sinphi*sintheta*sinpsi
		RMatrix[1,2] = +sinphi*costheta
		RMatrix[2,0] = +sinphi*sinpsi + cosphi*sintheta*cospsi
		RMatrix[2,1] = -sinphi*cospsi + cosphi*sintheta*sinpsi
		RMatrix[2,2] = +cosphi*costheta

		return RMatrix

	#################### End of RotationMatrix() ####################
	#################################################################

	@staticmethod
	def SymmetryAxisRotationMatrix(theta):
		"""Compute rotation matrix for a rotation about the XY-symmetry axis through an angle theta."""

		costheta = np.cos(theta)
		sintheta = np.sin(theta)
		invsqrt2 = 1./np.sqrt(2.)

		RMatrix  = np.zeros((3,3))

		RMatrix[0,0] = +0.5*(1 + costheta)
		RMatrix[0,1] = -0.5*(1 - costheta)
		RMatrix[0,2] = -invsqrt2*sintheta
		RMatrix[1,0] = -0.5*(1 - costheta)
		RMatrix[1,1] = +0.5*(1 + costheta)
		RMatrix[1,2] = -invsqrt2*sintheta
		RMatrix[2,0] = +invsqrt2*sintheta
		RMatrix[2,1] = +invsqrt2*sintheta
		RMatrix[2,2] = +costheta

		return RMatrix

	############## End of SymmetryAxisRotationMatrix() ##############
	#################################################################

	def ClassicalTrajectory(self):
		"""Compute classical center-of-mass trajectory of a body in the local (Earth-centered) navigation frame
		assuming a constant acceleration and rotation rate."""

		pass

	################# End of ClassicalTrajectory() ##################
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
	# [alphaS_F1L1, alphaS_F2L1, alphaV_F1L1, alphaV_F2L1, alphaT_F1L1, alphaT_F2L1] = phys.alphaDynamic(omegaL1)
	# [alphaS_F1L2, alphaS_F2L2, alphaV_F1L2, alphaV_F2L2, alphaT_F1L2, alphaT_F2L2] = phys.alphaDynamic(omegaL2)

	r   = 1.77   ## Raman beam intensity ration (I2/I1)
	tau = 35.E-6 ## Raman pi-pulse length (s)
	I2  = phys.I2Raman(tau)
	# I2 = 2.5*10. ## Raman beam 2 intensity (W/m^2)
	I1 = I2/r    ## Raman beam 1 intensity (W/m^2)
	A  = 1.      ## Degree of circular polarization
	[deltaS_mFp1, deltaV_mFp1, deltaT_mFp1, alphas_L1, alphas_L2] = phys.deltaAC(I1, I2, A, phys.Delta, +1., returnAlphas=True)
	[deltaS_mF0 , deltaV_mF0 , deltaT_mF0 ] = phys.deltaAC(I1, I2, A, phys.Delta,  0., returnAlphas=False)
	[deltaS_mFm1, deltaV_mFm1, deltaT_mFm1] = phys.deltaAC(I1, I2, A, phys.Delta, -1., returnAlphas=False)

	[alphaS_F1L1, alphaS_F2L1, alphaV_F1L1, alphaV_F2L1, alphaT_F1L1, alphaT_F2L1] = alphas_L1
	[alphaS_F1L2, alphaS_F2L2, alphaV_F1L2, alphaV_F2L2, alphaT_F1L2, alphaT_F2L2] = alphas_L2

	print('---------------------- 87Rb ----------------------')
	print('I1 = {:.2f} W/m^2'.format(I1))
	print('I2 = {:.2f} W/m^2'.format(I2))
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
			self.RamanDetuning	= -1.2E9
			self.RamanT 		= 1.0E-3
			self.RamanTOF		= 1.5E-2
			self.RamanpiX		= 6.0E-6
			self.RamanpiY		= 6.0E-6
			self.RamanpiZ      	= 6.0E-6
			self.kUpFrequency   = 6.835098427E+9
			self.kDownFrequency = 6.834287203E+9
			self.SelectionFreqs = [6.834650111E+9, 6.834650111E+9]

	runPars = RunParameters()
	phys    = Physics(runPars)

	# TestETGTAB()
	TestLightShifts()