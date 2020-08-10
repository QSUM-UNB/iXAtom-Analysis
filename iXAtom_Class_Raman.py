#####################################################################
## Filename:	iXAtom_Class_Raman.py
## Author:		B. Barrett
## Description: Raman class definition for iXAtom analysis package
## Version:		3.2.4
## Last Mod:	24/04/2020
##===================================================================
## Change Log:
## 14/10/2019 - Raman class defined and bug tested
##				(for LabVIEW v3.1 data only)
## 17/10/2019 - Minor modifications for v3.2 compatibility
## 19/10/2019 - Added control for combining detector results with
##				different coefficients
## 22/10/2019 - Added RamanAnalysisLevel1(), RamanAnalysisLevel2(),
##				and RamanAnalysisLevel3() utility methods for
##				AnalysisLevels = 1, 2, 3
## 25/11/2019 - Added functionality for ProcessLevel = 1 within
##				AnalysisLevel = 3
## 29/11/2019 - Separated global plot options from RamanOpts to its own
##				dictionary 'PlotOpts' to facilite easy sharing with
##				other classes (e.g. Ramsey)
## 30/11/2019 - Minor modifications and bug fixes
## 07/01/2020 - Completed overhaul of Raman class to use lmfit module.
##			    This allows simple and versatile control of fitting
##				options including setting bounds, fixing parameters,
##				robust estimation of confidence intervals, etc.
## 14/01/2020 - Implemented Raman phase and offset noise estimation based
##				on an optimization of the log-likelihood distribution
##				using the Minimizer method of the lmfit module.
## 31/01/2020 - Implemented method RamanAnalysisLevel5 for correlating
##				analysis level = 3 results with monitor data.
## 22/02/2020 - Improved robustness of Raman fitting function to avoid
##				zero-contrast results.
## 01/04/2020 - Implemented improved TimeSeriesAnalysis method from iXUtils. 
## 04/04/2020 - Upgraded Raman AnalysisLevel3 to be handle runs with different
##				sets of Raman axes more intelligently.
## 24/04/2020 - Modified NegLogLikelihood and EstimateRamanNoise methods
##				to include estimation of contrast noise in addtion to phase
##				and offset noise. The minimization results seem a bit more
##				reliable when all noise sources are included.
#####################################################################

import copy
import datetime as dt
import lmfit as lm
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz

from scipy.optimize    import minimize
# from scipy.interpolate import interp1d

import iXAtom_Utilities           as iXUtils
import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Class_Detector      as iXC_Detect
import iXAtom_Class_Monitor       as iXC_Monitor
import iXAtom_Class_Physics       as iXC_Phys

class Raman(iXC_RunPars.RunParameters):
	#################################################################
	## Class for storing and processing Raman data
	## Inherits all attributes and methods from class: RunParameters
	#################################################################

	def __init__(self, WorkDir, Folder, RunNum, RamanOpts, PlotOpts, LoadRunParsFlag=True, RunPars=[]):
		"""Initialize Raman variables.
		ARGUMENTS:
		\t WorkDir   (str)  - Path to the top-level directory where dataset is located
		\t Folder    (str)  - Name of folder within WorkDir where dataset is located
		\t RunNum    (int)  - Run number of requested dataset
		\t RamanOpts (dict) - Key:value pairs controlling Raman options
		\t PlotOpts  (dict) - Key:value pairs controlling plot options
		\t LoadRunParsFlag (bool) - (Optional) Flag for loading run parameters from file (True) or setting them from input (False).
		\t RunPars         (list) - (Optional) Key:value pairs containing run parameters
		"""

		super().__init__(WorkDir, Folder, RunNum)
		if LoadRunParsFlag:
			super().LoadRunParameters()
		else:
			# Take parameters from input list 'RunPars'
			for key, val in RunPars:
				setattr(self, key, val)

		self.RamanOptions  = copy.deepcopy(RamanOpts)
		self.PlotOptions   = copy.deepcopy(PlotOpts)
		self.idCoeffs      = np.array([0.,0.,0.])
		self.SetDetectCoeffs(self.RamanOptions['DetectCoeffs'])

		self.RawData       = False
		self.RawDataFound  = False
		self.PostDataFound = False
		self.RawDataFiles  = ['Raman-Run{:02d}-AvgRatios-kU.txt'.format(self.Run), 'Raman-Run{:02d}-AvgRatios-kD.txt'.format(self.Run)]
		self.RawDataDF     = [[pd.DataFrame([]) for ik in range(2)] for iax in range(3)]
		self.PostDataDF    = [[pd.DataFrame([]) for ik in range(2)] for iax in range(3)]
		self.Outliers      = [[[] for ik in range(2)] for iax in range(3)]
		self.RawFitDict    = [[{} for ik in range(2)] for iax in range(3)]
		self.PostFitDict   = [[{} for ik in range(2)] for iax in range(3)]
		self.RawFitResult  = [[{} for ik in range(2)] for iax in range(3)]
		self.PostFitResult = [[{} for ik in range(2)] for iax in range(3)]

		self.Fit_method    = 'lm' ## 'lm': Levenberg-Marquardt, 'trf': Trust Region Reflective, 'dogbox': dogleg algorithm with rectangular trust regions
		self.Fit_ftol      = 1.E-6
		self.Fit_xtol      = 1.E-6
		self.Fit_maxfev    = 2000

		self.Min_method     = 'nelder'
		self.Min_tol        = 1.E-5
		self.Min_xatol      = 1.E-5
		self.Min_fatol      = 1.E-5
		self.Min_maxfev     = 10000
		self.Min_maxiter	= 10000

		self.PlotPath  	   = os.path.join(self.PlotOptions['PlotFolderPath'], 'Raman-Run{:02d}-RawData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

	################## End of Raman.__init__() ######################
	#################################################################

	def LoadRawRamanData(self):
		"""Load raw Raman data from data file into a Pandas dataframe."""

		logging.info('iXC_Raman::Loading raw Raman data for {}...'.format(self.RunString))

		nFilesFound = 0
		for ik in self.ikList:
			dataPath = os.path.join(self.RawFolderPath, self.RawDataFiles[ik])
			if os.path.exists(dataPath):
				nFilesFound += 1
				df = pd.read_csv(dataPath, sep='\t')
				for iax in self.iaxList:
					# Select rows of the imported data frame corresponding to each Raman axis
					if ik == 0: # kU
						self.RawDataDF[iax][ik] = df[df['RamanAxis'] == 2*iax]
					else: # kD
						self.RawDataDF[iax][ik] = df[df['RamanAxis'] == 2*iax+1]

		if nFilesFound == self.nk:
			self.RawDataFound = True
			self.GetRunTiming(self.RamanOptions['PrintRunTiming'], RawDF=True)
		else:
			self.RawDataFound = False
			logging.warning('iXC_Raman::LoadRawRamanData::Raw Raman data not found!')

	############### End of Raman.LoadRawRamanData() #################
	#################################################################

	def ParseRamanData(self, iax, ik):
		"""Parse Raman data from pandas dataframe.
		ARGUMENTS:
		\t iax (int) - Index representing Raman axis (X,Y,Z = 0,1,2)
		\t ik  (int) - Index representing k-vector (kU,kD = 0,1)
		RETURN:
		\t xData (np array) - List of independent variables (Raman laser chirp or phase)
		\t yData (np array) - List of measurements (atomic Ratio)
		\t yErr  (np array) - List of measurement errors
		"""

		if self.RawData:
			df = self.RawDataDF[iax][ik]
		else:
			df = self.PostDataDF[iax][ik]

		if self.ChirpedData:
			## Take absolute value of chirps
			xData = abs(df['Chirp'].to_numpy())
		else:
			xData = df['Phase'].to_numpy()

		if self.RawData:
			yData = df['RatioMean'].to_numpy()
			yErr  = df['RatioStdDev'].to_numpy()
		else:
			nPoints = len(xData)
			yData = np.zeros(nPoints)
			yErr  = np.zeros(nPoints)

			for id in self.idList:
				if self.idCoeffs[id] > 0:
					label = self.DetectNames[id][0]
					yData +=  self.idCoeffs[id]*df['RatioMean_'+label].to_numpy()
					yErr  += (self.idCoeffs[id]*df['RatioSDev_'+label].to_numpy())**2

			yErr = np.sqrt(yErr)

		return [xData, yData, yErr]

	################ End of Raman.ParseRamanData() ##################
	#################################################################

	@staticmethod
	def fFringe(x, xOffset, yOffset, Contrast, xScale):
		"""Function for fitting Raman fringe."""
		return yOffset - 0.5*Contrast*np.cos(xScale*(x - xOffset))

	################### End of Raman.fFringe() ######################
	#################################################################

	def ConstructFitModel(self, iax, ik):
		"""Construct fit model from lmfit Model class."""

		pInit = self.RamanOptions['FitParameters'][iax].copy()

		if self.ChirpedData:
			## Set initial fit scale factor according to AI scale factor for phase-continuous chirps
			pInit['xScale'].value = 2*np.pi*self.Teff[iax]**2
			## Add central fringe chirps to initial offset values:
			pInit['xOffset_kU'].value += self.RamanOptions['CentralFringeChirp'][iax]
			pInit['xOffset_kD'].value += self.RamanOptions['CentralFringeChirp'][iax]

		model = lm.Model(self.fFringe)
		if ik == 0:
			model.set_param_hint('xOffset', value=pInit['xOffset_kU'].value)
		else:
			model.set_param_hint('xOffset', value=pInit['xOffset_kD'].value)
		model.set_param_hint('yOffset',  value=pInit['yOffset'].value)
		model.set_param_hint('Contrast', value=pInit['Contrast'].value, min=pInit['Contrast'].min)
		model.set_param_hint('xScale',   value=pInit['xScale'].value, min=pInit['xScale'].min, vary=pInit['xScale'].vary)

		return model

	############### End of Raman.ConstructFitModel() ################
	#################################################################

	def FitRamanData(self, iax, xData, yData, yErr):
		"""Fit Raman data using lmfit module.
		ARGUMENTS:
		iax        (int) - Raman axis index
		xData (np.array) - Independent variable values
		yData (np.array) - Dependent variable values
		yErr  (np.array) - Errors on dependent values
		"""

		if np.dot(yErr, yErr) > 0.:
			weights = 1./yErr
		else:
			weights = np.ones(len(yErr))

		fit_kws  = {'xtol': self.Fit_xtol, 'ftol': self.Fit_ftol, 'maxfev': self.Fit_maxfev} 

		iFit = 1
		while iFit <= 3:
			ResultLS = self.FitModel.fit(yData, self.FitPars, x=xData, weights=weights, method='leastsq', fit_kws=fit_kws)
			iFit += 1

			## Note that ResultLS.residual is divided by the errors
			# yRes    = ResultLS.residual
			yRes     = yData - ResultLS.best_fit
			sRes     = np.std(yRes)
			dsRes    = sRes/np.sqrt(2*ResultLS.nfree)
			contrast = ResultLS.params['Contrast'].value
			message  = 'iXC_Raman::{} (ier = {}, nfev = {}, contrast = {:4.2E})'.format(ResultLS.message, ResultLS.ier, ResultLS.nfev, contrast)

			if ResultLS.success and ResultLS.ier <= 3 and contrast >= 0.005:
				logging.info(message)
				return [ResultLS, yRes, sRes, dsRes]
			# elif iFit == 2 or contrast < 0.005:
			# 	xMin = xData[yData.argmin()]
			# 	self.FitPars['xOffset'].value = xMin
			# 	logging.warning(message)
			# 	logging.warning('iXC_Raman::Trying fit again with x0 set to local minimum ({:.5E})...'.format(xMin))
			# else:
			# 	logging.warning(message)
			# 	logging.warning(ResultLS.lmdif_message)
			elif contrast < 0.005:
				if iFit == 2:
					if self.ChirpedData:
						self.FitPars['xOffset'].value -= np.pi/self.Teff[iax]**2
					else:
						self.FitPars['xOffset'].value -= np.pi
					logging.warning(message)
					logging.warning('iXC_Raman::Trying fit again with x0 shifted by -pi...')
				elif iFit == 3:
					if self.ChirpedData:
						self.FitPars['xOffset'].value += 2*np.pi/self.Teff[iax]**2
					else:
						self.FitPars['xOffset'].value += 2*np.pi
					logging.warning(message)
					logging.warning('iXC_Raman::Trying fit again with x0 shifted by +pi...')
				else:
					logging.warning(message)
					logging.warning(ResultLS.lmdif_message)

		return [ResultLS, yRes, sRes, dsRes]

	################# End of Ramsey.FitRamanData() ##################
	#################################################################

	@staticmethod
	def NegLogLikelihood(Pars, x, y):
		"""Define the negative log-likelihood for the fit model, including phase and offset noise."""

		p = Pars.valuesdict()

		deltax = x - p['phi0']
		fModel = p['yOffset'] - 0.5*abs(p['Contrast'])*np.cos(deltax)
		sigma2 = (0.5*p['Contrast']*np.sin(deltax))**2*np.exp(2*p['log_sigX']) + np.exp(2*p['log_sigY']) + (0.5*np.cos(deltax))**2*np.exp(2*p['log_sigC'])

		return np.sum((y - fModel)**2/sigma2 + np.log(sigma2))

	############### End of Raman.NegLogLikelihood() #################
	#################################################################

	def EstimateRamanNoise(self, xData, yData, ResultFit):
		"""Estimate phase and amplitude noise using lmfit module.
		ARGUMENTS:
		xData     (np.array)       - Independent variable values
		yData     (np.array)       - Dependent variable values
		ResultFit (lm.ModelResult) - Result of the non-linear least-squares fit
		"""

		logging.info('ICE_Raman::Estimating Raman noise parameters...')

		x0  = ResultFit.params['xOffset'].value
		dx0 = ResultFit.params['xOffset'].stderr
		y0  = ResultFit.params['yOffset'].value
		dy0 = ResultFit.params['yOffset'].stderr
		c   = ResultFit.params['Contrast'].value
		dc  = ResultFit.params['Contrast'].stderr
		s   = ResultFit.params['xScale'].value

		if self.ChirpedData:
			## Convert xData to phase and shift to zero
			xData = s*(xData - x0)
			x0    = 0.

		pInit = lm.Parameters()
		pInit.add_many(
			lm.Parameter('phi0',     value=x0, vary=False),
			lm.Parameter('yOffset',  value=y0, vary=False),
			lm.Parameter('Contrast', value=c,  vary=False),
			lm.Parameter('log_sigX', value=np.log(1.E-1), max=np.log(1.), min=np.log(1.E-4)),
			lm.Parameter('log_sigY', value=np.log(1.E-2), max=np.log(1.), min=np.log(1.E-5)),
			lm.Parameter('log_sigC', value=np.log(1.E-3), max=np.log(1.), min=np.log(1.E-6)))

		min_kws     = {'tol': self.Min_tol, 'options': {'xatol': self.Min_xatol, 'fatol': self.Min_fatol, 'maxfev': self.Min_maxfev, 'maxiter': self.Min_maxiter}}
		ResultNoise = lm.minimize(self.NegLogLikelihood, pInit, method=self.Min_method, args=(xData, yData), nan_policy='omit', **min_kws)

		logging.info('iXC_Raman::{} (nfev = {}, redchi = {:5.3E})'.format(ResultNoise.message, ResultNoise.nfev, ResultNoise.redchi))

		return ResultNoise

	############## End of Raman.EstimatePhaseNoise() ################
	#################################################################

	# def EstimateRamanNoise(self, iax, xData, yData, ResultLS):
	# 	"""Estimate phase and amplitude noise using lmfit module.
	# 	ARGUMENTS:
	# 	xData    (np.array)       - Independent variable values
	# 	yData    (np.array)       - Dependent variable values
	# 	ResultLS (lm.ModelResult) - Result of the non-linear least-squares fit
	# 	"""

	# 	logging.info('iXC_Raman::Estimating Raman noise parameters...')

	# 	x0  = ResultLS.params['xOffset'].value
	# 	dx0 = ResultLS.params['xOffset'].stderr
	# 	y0  = ResultLS.params['yOffset'].value
	# 	dy0 = ResultLS.params['yOffset'].stderr
	# 	c   = ResultLS.params['Contrast'].value
	# 	dc  = ResultLS.params['Contrast'].stderr
	# 	s   = ResultLS.params['xScale'].value

	# 	if type(dx0) is not float:
	# 		dx0 = 0.
	# 	if type(dy0) is not float:
	# 		dy0 = 0.
	# 	if type(dc) is not float:
	# 		dy0 = 0.

	# 	if self.ChirpedData:
	# 		## Convert xData to phase and shift to zero
	# 		xData = s*(xData - self.RamanOptions['CentralFringeChirp'][iax])
	# 		x0    = 0.

	# 	pInit = lm.Parameters()
	# 	pInit.add_many(
	# 		lm.Parameter('phi0',     value=x0, vary=False),
	# 		lm.Parameter('yOffset',  value=y0, vary=False),
	# 		lm.Parameter('Contrast', value=c, vary=False),
	# 		lm.Parameter('log_sigX', value=np.log(s*dx0 + 1.E-3)),
	# 		lm.Parameter('log_sigY', value=np.log(dy0 + 1.E-3)))

	# 	ResultML = lm.minimize(self.NegLogLikelihood, pInit, method='nelder', args=(xData, yData))

	# 	logging.info('iXC_Raman::{} (nfev = {}, redchi = {:5.3E})'.format(ResultML.message, ResultML.nfev, ResultML.redchi))

	# 	return ResultML

	# ################ End of Ramsey.FitRamseyData() ##################
	# #################################################################

	def AddAuxPars(self, iax, ik, ResultLS, ResultML, AuxData):
		"""Add auxiliary parameters to a Parameters object, chi**2, reduced-chi**2,
		std dev. of residuals, fit SNR, and noise parameter estimates.
		ARGUMENTS:
		iax      (int)			      - Index representing Raman axis
		ik       (int)			      - Index representing k-direction
		ResultLS (lm.ModelResult)     - Least-squares fit result
		ResultML (lm.MinimizerResult) - Likelihood optmization result
		AuxData  (dict)			      - Key:value pairs of auxiliary data to add
		RETURN:
		Pars     (lm.Parameters)      - Modified parameters
		"""

		Pars = ResultLS.params

		## Add auxiliary data
		resdev = lm.Parameter('ResDev', value=AuxData['ResDev'])
		resdev.init_value = 0.
		resdev.stderr = AuxData['ResErr']

		chisqr = lm.Parameter('ChiSqr', value=AuxData['ChiSqr'])
		chisqr.init_value = 0.
		chisqr.stderr = 2.*(resdev.stderr/resdev.value)*chisqr.value

		redchi = lm.Parameter('RedChiSqr', value=AuxData['RedChiSqr'])
		redchi.init_value = 0.
		redchi.stderr = 2.*(resdev.stderr/resdev.value)*redchi.value

		C   = ResultLS.params['Contrast'].value
		dC  = ResultLS.params['Contrast'].stderr
		# if type(dC) is not float:
		# 	dC = 0.

		SNR = lm.Parameter('SNR', value=C/resdev)
		SNR.init_value = 0.
		SNR.stderr = np.sqrt((dC/C)**2 + (resdev.stderr/resdev.value)**2)*SNR.value

		if self.ChirpedData:
			alpha0_init     = ResultLS.params['xOffset'].init_value
			alpha0          = ResultLS.params['xOffset'].value
			dalpha0         = ResultLS.params['xOffset'].stderr

			phi0_best       = 2*np.pi*self.kSign[iax][ik]*(alpha0 - self.RamanOptions['CentralFringeChirp'][iax])*self.Teff[iax]**2
			phi0_init       = 2*np.pi*self.kSign[iax][ik]*(alpha0_init - self.RamanOptions['CentralFringeChirp'][iax])*self.Teff[iax]**2
			phi0            = lm.Parameter('phi0', value=phi0_best)
			phi0.init_value = phi0_init
			phi0.stderr     = 2*np.pi*dalpha0*self.Teff[iax]**2

			gExp            = lm.Parameter('gExp', value=2*np.pi*abs(alpha0)/self.keff)
			gExp.init_value	= 2*np.pi*abs(alpha0_init)/self.keff
			gExp.stderr	    = 2*np.pi*dalpha0/self.keff
		else:
			phig            = self.Seff[iax]*self.gLocal
			n2pi            = np.floor(phig/(2.*np.pi))
			phi0_best		= ResultLS.params['xOffset'].value % (self.kSign[iax][ik]*2.*np.pi) + self.kSign[iax][ik]*n2pi*(2.*np.pi)
			phi0            = lm.Parameter('phi0', value=phi0_best)
			phi0.init_value = ResultLS.params['xOffset'].init_value
			phi0.stderr     = ResultLS.params['xOffset'].stderr

			gExp_best       = phi0_best/(self.kSign[iax][ik]*self.Seff[iax])
			gExp            = lm.Parameter('gExp', value=gExp_best)
			gExp.init_value	= (phi0.init_value % (self.kSign[iax][ik]*2.*np.pi) + self.kSign[iax][ik]*n2pi*(2.*np.pi))/(self.kSign[iax][ik]*self.Seff[iax])
			gExp.stderr	    = phi0.stderr/self.Seff[iax]

		sigX = lm.Parameter('sigX', value=np.exp(ResultML.params['log_sigX']))
		sigX.init_value = np.exp(ResultML.params['log_sigX'].init_value)
		sigX.stderr = 0.

		sigY = lm.Parameter('sigY', value=np.exp(ResultML.params['log_sigY']))
		sigY.init_value = np.exp(ResultML.params['log_sigY'].init_value)
		sigY.stderr = 0.

		sigC = lm.Parameter('sigC', value=np.exp(ResultML.params['log_sigC'].value))
		sigC.init_value = np.exp(ResultML.params['log_sigC'].init_value)
		sigC.stderr = 0.

		Pars.add_many(resdev, chisqr, redchi, SNR, phi0, gExp, sigX, sigY, sigC)

		return Pars

	################## End of Raman.AddAuxPars() ####################
	#################################################################

	def UpdateFitDicts(self, iax, ik, Pars):
		"""Update Raman fit dictionaries."""

		pName  = [val.name for val in Pars.values()]
		pBest  = [val.value for val in Pars.values()]

		try:
			pInit  = [val.init_value for val in Pars.values()]
			pError = [val.stderr for val in Pars.values()]
		except:
			pInit  = copy.copy(pBest)
			pError = [0. for val in Pars.values()]

		if self.RawData:
			self.RawFitDict[iax][ik]['Init']   = {key:val for key, val in zip(pName, pInit)}
			self.RawFitDict[iax][ik]['Best']   = {key:val for key, val in zip(pName, pBest)}
			self.RawFitDict[iax][ik]['Error']  = {key:val for key, val in zip(pName, pError)}
		else:
			self.PostFitDict[iax][ik]['Init']  = {key:val for key, val in zip(pName, pInit)}
			self.PostFitDict[iax][ik]['Best']  = {key:val for key, val in zip(pName, pBest)}
			self.PostFitDict[iax][ik]['Error'] = {key:val for key, val in zip(pName, pError)}

	################### End of Raman.UpdateFitDicts() ###################
	#####################################################################

	def AnalyzeRamanData(self):
		"""Analyze Raman data (raw or post-processed)."""

		if self.RawData:
			label = 'raw'
			dfList = self.RawDataDF
		else:
			label = 'post-processed'
			dfList = self.PostDataDF

		logging.info('iXC_Raman::Analyzing {} Raman data for {}...'.format(label, self.RunString))

		if self.RamanOptions['FitData']:
			## Load and fit Raman data, and store fit results in data frames
			for iax in self.iaxList:
				for ik in self.ikList:
					self.FitModel = self.ConstructFitModel(iax, ik)
					self.FitPars  = self.FitModel.make_params()

					[xData, yData, yErr] = self.ParseRamanData(iax, ik)
					[resultLS, yRes, sRes, dsRes] = self.FitRamanData(iax, xData, yData, yErr)

					if self.RamanOptions['RemoveOutliers']:
						iPoint = -1
						for dy in yRes:
							iPoint += 1
							if abs(dy) > self.RamanOptions['OutlierThreshold']*sRes:
								self.Outliers[iax][ik].append(iPoint)
								logging.info('iXC_Raman::Removing outlier at (iax,ik,iPoint) = ({},{},{}), nOutlier = {}...'.format(iax,ik,iPoint,len(self.Outliers[iax][ik])))
								logging.info('iXC_Raman::  Outlier: (x,y,yRes) = ({:5.3e}, {:5.3e}, {:4.2f}*sigma)'.format(xData[iPoint],yData[iPoint],dy/sRes))
						if len(self.Outliers[iax][ik]) > 0:
							xData  = np.delete(xData, self.Outliers[iax][ik])
							yData  = np.delete(yData, self.Outliers[iax][ik])
							yErr   = np.delete(yErr,  self.Outliers[iax][ik])
							[resultLS, yRes, sRes, dsRes] = self.FitRamanData(iax, xData, yData, yErr)

					## Estimate Raman noise parameters
					resultML = self.EstimateRamanNoise(xData, yData, resultLS)

					if self.RawData:
						self.RawFitResult[iax][ik]  = resultLS
					else:
						self.PostFitResult[iax][ik] = resultLS

					auxData = {'ResDev': sRes, 'ResErr': dsRes, 'ChiSqr': resultLS.chisqr, 'RedChiSqr': resultLS.redchi}
					pars    = self.AddAuxPars(iax, ik, resultLS, resultML, auxData)
					self.UpdateFitDicts(iax, ik, pars)
		else:
			for iax in self.iaxList:
				pars = self.RamanOptions['FitParameters'][iax]
				for ik in self.ikList:
					self.UpdateFitDicts(iax, ik, pars)

	################# End of Raman.AnalyzeRamanData() ###################
	#####################################################################

	def PlotRamanAxisData(self, RamanAxs, iRun, iax, CustomPlotOpts, Labels):
		"""Plot Raman data (raw or post-processed) and associated fit for given axis.
		ARGUMENTS:
		\t RamanAxs       (list) - Raman figure axes corresponding to a given Raman axis
		\t iax             (int) - Index corresponding to a given axis
		\t iRun            (int) - Index corresponding to run number in RunList
		\t CustomPlotOpts (dict) - Key:value pairs controlling custom plot options
		\t Labels         (list) - Plot labels
		"""

		[xLabel, yLabel] = Labels

		for ik in self.ikList:
			[xData, yData, yErr] = self.ParseRamanData(iax, ik)

			if self.RamanOptions['RemoveOutliers'] and len(self.Outliers[iax][ik]) > 0:
				xData = np.delete(xData, self.Outliers[iax][ik])
				yData = np.delete(yData, self.Outliers[iax][ik])
				yErr  = np.delete(yErr,  self.Outliers[iax][ik])

			if np.dot(yErr, yErr) == 0.:
				yErr = np.array([])

			## Main Raman plot
			if self.PlotOptions['OverlayRunPlots']:
				nColors = len(self.RamanOptions['RunPlotColors'])
				CustomPlotOpts['Color'] = self.RamanOptions['RunPlotColors'][iRun%(nColors)][ik]
				if self.RamanOptions['RunPlotVariable'] == 'RamanT':
					CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik] + ', {:5.2f} ms'.format(self.RamanT*1.E+3)
				elif self.RamanOptions['RunPlotVariable'] == 'Run':
					CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik] + ', Run {:02d}'.format(self.Run)
				elif self.RamanOptions['RunPlotVariable'] == 'RunTime':
					runTimeStamp = dt.datetime.fromtimestamp(self.RunTime, tz=pytz.timezone('Europe/Paris')).strftime('%H:%M:%S')
					CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik] + ', {}'.format(runTimeStamp)
				else:
					CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik] + ', {:5.2e}'.format(getattr(self, self.RamanOptions['RunPlotVariable']))
			else:
				CustomPlotOpts['Color']    = self.DefaultPlotColors[iax][ik]
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik]

			CustomPlotOpts['Linestyle'] = 'None'
			CustomPlotOpts['Marker']    = '.'
			iXUtils.CustomPlot(RamanAxs[1], CustomPlotOpts, xData, yData, yErr)

			if self.PlotOptions['ShowFit'] and self.RamanOptions['FitData']:
				## Plot fringe fit
				xData = np.sort(xData)
				if self.RamanOptions['SetFitPlotXLimits']:
					[xMin, xMax] = self.RamanOptions['FitPlotXLimits']
					nPoints = 2*round(abs(xMax - xMin)/abs(xData[1] - xData[0]))
				else:
					## Assumes data are sorted
					xMin = xData[0]
					xMax = xData[-1]
					nPoints = 2*len(xData)

				xFit = np.linspace(xMin, xMax, num=nPoints, endpoint=True)
				yFit = np.array([])

				if self.RawData:
					fitResult = self.RawFitResult[iax][ik]
				else:
					fitResult = self.PostFitResult[iax][ik]

				if self.PlotOptions['OverlayRunPlots']:
					CustomPlotOpts['Color'] = self.RamanOptions['RunPlotColors'][iRun%(nColors)][ik]
				else:
					CustomPlotOpts['Color'] = self.DefaultPlotColors[iax][ik]
				CustomPlotOpts['Linestyle'] = '-'
				CustomPlotOpts['Marker']    = None
				CustomPlotOpts['LegLabel']  = None

				yFit = fitResult.eval(x=xFit)
				iXUtils.CustomPlot(RamanAxs[1], CustomPlotOpts, xFit, yFit)

				# dyFit = fitResult.eval_uncertainty(x=xFit)
				# RamanAxs[1].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

				CustomPlotOpts['Linestyle'] = '-'
				CustomPlotOpts['Marker']    = '.'

				## Note that fitResult.residual is divided by the errors
				yRes = yData - fitResult.best_fit ## Already has outliers deleted
				iXUtils.CustomPlot(RamanAxs[0], CustomPlotOpts, xData, yRes, yErr)

		if self.PlotOptions['ShowPlotLabels'][0]:
			RamanAxs[1].set_xlabel(xLabel)
		if self.PlotOptions['ShowPlotLabels'][1]:
			RamanAxs[0].set_ylabel('Residue')
			RamanAxs[1].set_ylabel(yLabel)
		if self.PlotOptions['ShowPlotTitle']:
			RamanAxs[0].set_title(self.RunString + ', T = {:4.2f} ms'.format(self.RamanT*1.0E+3))
		if self.PlotOptions['ShowPlotLegend']:
			if self.PlotOptions['FixLegLocation']:
				## Fix legend location outside upper right of plot
				RamanAxs[1].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
			else:
				## Let matlibplot find best legend location
				RamanAxs[1].legend(loc='best')

	############### End of Raman.PlotRamanAxisData() ################
	#################################################################

	def PlotRamanData(self, RamanAxs, iRun):
		"""Plot Raman data (raw or post-processed) and associated fit (if requested).
		ARGUMENTS:
		\t RamanAx (fig ax) - Current Raman figure axis
		\t iRun       (int) - Index corresponding to run number in RunList   
		"""

		if self.RawData:
			label = 'raw'
			dfList = self.RawDataDF
		else:
			label = 'post-processed'
			dfList = self.PostDataDF

		logging.info('iXC_Raman::Plotting {} Raman data for {}...'.format(label, self.RunString))

		if self.ChirpedData:
			xLabel = r'$|\alpha|$  (Hz/s)'
		else:
			xLabel = r'$\phi_{\rm las}$  (rad)'
		yLabel = r'$N_2/N_{\rm total}$'

		customPlotOpts = {'Color': 'red', 'Linestyle': 'None', 'Marker': '.', 'Title': 'None', 
			'xLabel': 'None', 'yLabel': 'None', 'Legend': False, 'LegLabel': None}

		iRow = -1
		for iax in self.iaxList:
			iRow += 1
			if self.PlotOptions['OverlayRunPlots']:
				## Set options for sharing x axes
				showXLabel = True if iax == self.iaxList[-1] else False
				# plt.setp(RamanAxs[iRow].get_xticklabels(), visible=showXLabel)
				self.PlotOptions['ShowPlotLabels'][0] = showXLabel

				self.PlotRamanAxisData(RamanAxs[iRow], iRun, iax, customPlotOpts, [xLabel, yLabel])
			else:
				self.PlotRamanAxisData(RamanAxs[0], iRun, iax, customPlotOpts, [xLabel, yLabel])

		if self.PlotOptions['SavePlot']:
			plt.savefig(self.PlotPath, dpi=150)
			logging.info('iXC_Raman::Raman plot saved to:')
			logging.info('iXC_Raman::  {}'.format(self.PlotPath))
		elif self.PlotOptions['ShowPlot']:
			plt.show()

	################# End of Raman.PlotRamanData() ##################
	#################################################################

	def PostProcessRamanData(self, DetectOpts):
		"""Post-process Raman detection data and write to file (if requested).
		ARGUMENTS:
		\t DetectOpts (dict) - Key:value pairs controlling detection options
		"""

		logging.info('iXC_Raman::Post-processing Raman data for {}...'.format(self.RunString))

		# Declare detector object
		det = iXC_Detect.Detector(self.WorkDir, self.Folder, self.Run, DetectOpts, self.PlotOptions, False, self.__dict__.items())

		meanHeaders = ['RatioMean_L', 'RatioMean_M', 'RatioMean_U']
		sdevHeaders = ['RatioSDev_L', 'RatioSDev_M', 'RatioSDev_U']

		if det.nIterations > 0:
			self.LoadRawRamanData()

			## Store copy of raw Raman data in post Raman dataframe
			self.PostDataDF = [[self.RawDataDF[iax][ik].copy() for ik in range(2)] for iax in range(3)]

			## Reshape PostDataDF
			for iax in self.iaxList:
				for ik in self.ikList:
					self.PostDataDF[iax][ik].drop(columns=['RamanAxis'], inplace=True)
					self.PostDataDF[iax][ik].rename(columns={'RatioMean':   meanHeaders[0]}, inplace=True)
					self.PostDataDF[iax][ik].rename(columns={'RatioStdDev': sdevHeaders[0]}, inplace=True)
					for id in range(3):
						self.PostDataDF[iax][ik].loc[:,meanHeaders[id]] = 0.
						self.PostDataDF[iax][ik].loc[:,sdevHeaders[id]] = 0.
	
			avgNum = 1
			for iterNum in range(1,det.nIterations+1):

				if iterNum == 1:
					initPars = det.InitializeDetectPars()

				det.LoadDetectTrace(avgNum, iterNum)
				det.AnalyzeDetectTrace(avgNum, iterNum, initPars)

				if 'Accel Axis' in det.DetectTags.keys():
					iax = int(det.DetectTags['Accel Axis'])				
				else:
					iax = int(list(det.DetectTags.values())[13])
				if iax not in range(3):
					logging.error('iXC_Raman::PostProcessRamanData::Raman axis index out of range (iax = {})...'.format(iax))
					logging.error('iXC_Raman::PostProcessRamanData::Aborting...')
					quit()

				if 'k Index' in det.DetectTags.keys():
					ik = int(det.DetectTags['k Index'])				
				else:
					ik = int(list(det.DetectTags.values())[0])
				if ik not in range(2):
					logging.error('iXC_Raman::PostProcessRamanData::k-direction index out of range (ik = {})...'.format(ik))
					logging.error('iXC_Raman::PostProcessRamanData::Aborting...')
					quit()

				mask = (self.PostDataDF[iax][ik]['#Iteration'] == iterNum)
				for id in det.idList:
					for ic in range(4):
						initPars[id][ic] = det.DetectResult[id][det.DetectKeys[ic]]['BestPars']
					self.PostDataDF[iax][ik].loc[mask, meanHeaders[id]] = det.DetectResult[id]['Ratio']['Best']
					self.PostDataDF[iax][ik].loc[mask, sdevHeaders[id]] = det.DetectResult[id]['Ratio']['Error']

			if self.RamanOptions['SaveAnalysisLevel1']:
				showHeaders = True
				showIndices = False
				if self.ChirpedData:
					# Output 9-digits of precision
					floatFormat = '%10.8E'
				else:
					# Output 6-digits of precision
					floatFormat = '%7.5E'

				for ik in self.ikList:
					for iax in self.iaxList:
						iXUtils.WriteDataFrameToFile(self.PostDataDF[iax][ik], self.PostFolderPath, self.PostFilePaths[iax][ik],
							showHeaders, showIndices, floatFormat)

				logging.info('iXC_Raman::Post-processed Raman data saved to:')
				logging.info('iXC_Raman::  {}'.format(self.PostFolderPath))
		else:
			logging.error('iXC_Raman::PostProcessRamanData::Aborting detection analysis on {}...'.format(self.RunString))
			quit()

	############## End of Raman.PostProcessRamanData() ##############
	#################################################################

	def LoadPostRamanData(self):
		"""Load post-processed Raman data into a list of data frames."""

		logging.info('iXC_Raman::Loading post-processed Raman data for {}...'.format(self.RunString))

		nFilesFound = 0
		for iax in self.iaxList:
			for ik in self.ikList:
				if os.path.exists(self.PostFilePaths[iax][ik]):
					nFilesFound += 1
					self.PostDataDF[iax][ik] = pd.read_csv(self.PostFilePaths[iax][ik], sep='\t')

		if nFilesFound == self.nax*self.nk:
			self.PostDataFound = True
			self.GetRunTiming(self.RamanOptions['PrintRunTiming'], RawDF=False)
		else:
			self.PostDataFound = False
			logging.warning('iXC_Raman::LoadPostRamanData::Post-processed Raman data not found in: {}'.format(self.PostFolderPath))

	############## End of Raman.LoadPostRamanData() #################
	#################################################################

	def WriteRamanAnalysisResults(self):
		"""Write Raman analysis results to file."""

		if self.RawData:
			label = 'raw'
		else:
			label = 'post-processed'

		logging.info('iXC_Raman::Writing {} Raman analysis results to:'.format(label))
		logging.info('iXC_Raman::  {}'.format(self.PostFolderPath))	

		for iax in self.iaxList:
			for ik in self.ikList:
				if self.RawData:
					fileName  = 'Raman-Run{:02}-Results-Raw-'.format(self.Run)  + self.AxisFileLabels[iax][ik] + '.txt'
					dataFrame = pd.DataFrame.from_dict(self.RawFitDict[iax][ik])
				else:
					fileName  = 'Raman-Run{:02}-Results-Post-'.format(self.Run) + self.AxisFileLabels[iax][ik] + '.txt'
					dataFrame = pd.DataFrame.from_dict(self.PostFitDict[iax][ik])

				filePath = os.path.join(self.PostFolderPath, fileName)
				iXUtils.WriteDataFrameToFile(dataFrame, self.PostFolderPath, filePath, True, True, '%10.8E')

	########### End of Raman.WriteRamanAnalysisResults() ############
	#################################################################

	def ReadRamanAnalysisResults(self):
		"""Read Raman analysis results to file."""

		if self.RawData:
			label1 = 'raw'
			label2 = 'Raw'
		else:
			label1 = 'post-processed'
			label2 = 'Post'

		logging.info('iXC_Raman::Reading {} Raman analysis results from:'.format(label1))
		logging.info('iXC_Raman::  {}'.format(self.PostFolderPath))

		for iax in self.iaxList:
			for ik in self.ikList:
				fileName = 'Raman-Run{:02}-Results-{}-'.format(self.Run, label2) + self.AxisFileLabels[iax][ik] + '.txt'
				filePath = os.path.join(self.PostFolderPath, fileName)
				if os.path.exists(filePath):
					if self.RawData:
						self.RawFitDict[iax][ik] = pd.read_csv(filePath, sep='\t', header=0, index_col=0).to_dict()
					else:
						self.PostFitDict[iax][ik] = pd.read_csv(filePath, sep='\t', header=0, index_col=0).to_dict()
				else:
					logging.error('iXC_Raman::ReadRamanAnalysisResults::Analysis file not found: {}'.format(filePath))
					logging.error('iXC_Raman::ReadRamanAnalysisResults::Aborting...')
					quit()

	############ End of Raman.ReadRamanAnalysisResults() ############
	#################################################################

	def LoadAndAnalyzeRamanData(self, RamanAxs, iRun, nRuns, PlotOpts, ProcessLevel):
		"""Load and analyze raw and/or post-processed Raman data; 
		plot the results and write to file (if requested).
		ARGUMENTS:
		\t RamanAxs (fig axs) - List of Raman figure axes for overlaying plots
		\t iRun	        (int) - Index corresponding to current run number (starts from 0)
		\t nRuns        (int) - Number of runs contained within RunList
		\t PlotOpts    (dict) - Copy of key:value pairs controlling plot options
		\t ProcessLevel (int) - Secondary control for Raman analysis type
		"""

		if ProcessLevel == 0 or ProcessLevel == 2:

			self.RawData = True
			self.LoadRawRamanData()

			if self.RawDataFound:
				self.AnalyzeRamanData()
				if self.RamanOptions['SaveAnalysisLevel2'] and self.RamanOptions['FitData']:
					self.WriteRamanAnalysisResults()

				if nRuns <= self.PlotOptions['MaxPlotsToDisplay']:
					for iax in self.iaxList:
						for ik in self.ikList:
							print('------------------------------------------------')
							print(' '+self.AxisFileLabels[iax][ik]+' - Raw Data - Fit Results:')
							print('------------------------------------------------')
							print(pd.DataFrame.from_dict(self.RawFitDict[iax][ik]))
			else:
				logging.warning('iXC_Raman::LoadAndAnalyzeRamanData::Aborting Raman raw data analysis for {}...'.format(self.RunString))

		if ProcessLevel == 1 or ProcessLevel == 2:

			self.RawData = False
			self.LoadPostRamanData()

			if self.PostDataFound:
				self.AnalyzeRamanData()
				if self.RamanOptions['SaveAnalysisLevel2'] and self.RamanOptions['FitData']:
					self.WriteRamanAnalysisResults()

				if nRuns <= self.PlotOptions['MaxPlotsToDisplay']:
					for iax in self.iaxList:
						for ik in self.ikList:
							if ProcessLevel == 1 or ProcessLevel == 2:
								print('------------------------------------------------')
								print(' '+self.AxisFileLabels[iax][ik]+' - Post Data - Fit Results:')
								print('------------------------------------------------')
								print(pd.DataFrame.from_dict(self.PostFitDict[iax][ik]))
			else:
				logging.warning('iXC_Raman::LoadAndAnalyzeRamanData::Aborting Raman post-processed data analysis for {}...'.format(self.RunString))

		if (self.PlotOptions['ShowPlot'] and not self.PlotOptions['SavePlot'] and not self.PlotOptions['OverlayRunPlots']) and nRuns > self.PlotOptions['MaxPlotsToDisplay']:
			self.PlotOptions['PlotData'] = False

		if self.PlotOptions['PlotData'] and nRuns <= self.PlotOptions['MaxPlotsToDisplay']:
			if ProcessLevel < 2 and (self.RawDataFound or self.PostDataFound):
				if ProcessLevel == 0:
					self.RawData = True
					self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
						'Raman-Run{:02d}-RawData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])
				else:
					self.RawData = False
					self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
						'Raman-Run{:02d}-PostData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

				if self.PlotOptions['OverlayRunPlots']:
					self.PlotOptions['ShowPlotTitle']  = False
					self.PlotOptions['ShowPlotLegend'] = True
					if iRun < nRuns-1:
						self.PlotOptions['ShowPlot'] = False
						self.PlotOptions['SavePlot'] = False
					else:
						self.PlotOptions['ShowPlot'] = PlotOpts['ShowPlot']
						self.PlotOptions['SavePlot'] = PlotOpts['SavePlot']

				self.PlotRamanData(RamanAxs[0], iRun)

			if ProcessLevel == 2 and (self.RawDataFound and self.PostDataFound):
				self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
					'Raman-Run{:02d}-Raw+PostData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

				self.RawData = True
				self.PlotOptions['ShowPlot'] = False
				self.PlotOptions['SavePlot'] = False
				self.PlotOptions['ShowPlotLabels'] = [True, True]
				if self.PlotOptions['OverlayRunPlots']:
					self.PlotOptions['ShowPlotTitle']  = False
					self.PlotOptions['ShowPlotLegend'] = False
				else:
					self.PlotOptions['ShowPlotTitle']  = PlotOpts['ShowPlotTitle']
					self.PlotOptions['ShowPlotLegend'] = PlotOpts['ShowPlotLegend']
				self.PlotRamanData(RamanAxs[0], iRun)

				self.RawData = False
				self.PlotOptions['ShowPlotLabels'] = [True, False]
				if self.PlotOptions['OverlayRunPlots']:
					self.PlotOptions['ShowPlotLabels'] = [True, False]
					self.PlotOptions['ShowPlotLegend'] = True
					if iRun < nRuns-1:
						self.PlotOptions['ShowPlot'] = False
						self.PlotOptions['SavePlot'] = False
					else:
						self.PlotOptions['ShowPlot'] = PlotOpts['ShowPlot']
						self.PlotOptions['SavePlot'] = PlotOpts['SavePlot']
				else:
					self.PlotOptions['ShowPlot'] = PlotOpts['ShowPlot']
					self.PlotOptions['SavePlot'] = PlotOpts['SavePlot']
				self.PlotRamanData(RamanAxs[1], iRun)

	############ End of Raman.LoadAndAnalyzeRamanData() #############
	#################################################################

#####################################################################
######################## End of Class Raman #########################
#####################################################################

def RamanAnalysisLevel1(AnalysisCtrl, RamanOpts, DetectOpts, PlotOpts, RunPars):
	"""Method for Raman AnalysisLevel = 1:
	Post-process detection data for selected Runs and store results.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamanOpts    (dict) - Copy of key:value pairs controlling Raman options
	\t DetectOpts   (dict) - Copy of key:value pairs controlling Detection options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	for RunNum in RunList:
		if RunNum == RunList[0]:
			Ram = Raman(WorkDir, Folder, RunNum, RamanOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Ram = Raman(WorkDir, Folder, RunNum, RamanOpts, PlotOpts)
		Ram.PostProcessRamanData(DetectOpts)

################### End of RamanAnalysisLevel1() ####################
#####################################################################

def RamanAnalysisLevel2(AnalysisCtrl, RamanOpts, PlotOpts):
	"""Method for Raman AnalysisLevel = 2:
	Load raw/post-processed Raman data for selected Runs, analyze and store results.
	Plot the data and fits in individual figures or overlayed in a single figure.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamanOpts    (dict) - Copy of key:value pairs controlling Raman options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	runList = AnalysisCtrl['RunList']
	nRuns   = len(runList)

	runFitPars = copy.deepcopy(RamanOpts['FitParameters'])
	runVarKey  = RamanOpts['RunPlotVariable']
	runVarList = np.zeros(nRuns)
	xOffsets   = np.zeros((3,2,nRuns)) ## [iax,ik,iRun]

	if not RamanOpts['FitData'] or nRuns == 1: 
		RamanOpts['TrackRunFitPars'] = False

	if RamanOpts['SortVariableOrder'] != 'None':
		logging.info('iXC_Raman::Sorting list of Run variables "{}"...'.format(runVarKey))
		logging.disable(level=logging.INFO) # Disable logger for info & debug levels

		## Load run variables from parameter files and find sort order
		iRun = -1
		for runNum in runList:
			iRun += 1
			Ram = Raman(WorkDir, Folder, runNum, RamanOpts, PlotOpts)
			if runVarKey not in vars(Ram):
				## Raman data must be loaded to extract certain run variables
				Ram.LoadRawRamanData()

			runVarList[iRun] = getattr(Ram, runVarKey)

		## Sort run variables and run list
		orderList  = np.argsort(runVarList)
		runVarList = runVarList[orderList]
		runList    = [runList[iRun] for iRun in orderList]

		if RamanOpts['SortVariableOrder'] == 'Descending':
			runVarList = runVarList[::-1]
			runList    = runList[::-1]

		logging.disable(level=logging.NOTSET) # Re-enable logger at pre-configured level

	RamanAxs = None
	iRun = -1
	for runNum in runList:
		iRun += 1

		Ram = Raman(WorkDir, Folder, runNum, RamanOpts, PlotOpts)
		## Assumes nax is identical for all runs in RunList
		RamanAxs = Ram.CreatePlotAxes(RamanAxs, iRun, nRuns, Ram.nax, PlotOpts, AnalysisCtrl['ProcessLevel'])

		Ram.RamanOptions['FitParameters'] = runFitPars
		Ram.LoadAndAnalyzeRamanData(RamanAxs, iRun, nRuns, PlotOpts, AnalysisCtrl['ProcessLevel'])

		if RamanOpts['TrackRunFitPars'] and iRun < nRuns-1:
			for iax in Ram.iaxList:
				for ik in Ram.ikList:
					if Ram.RawData:
						valDict = Ram.RawFitDict[iax][ik]['Best']
					else:
						valDict = Ram.PostFitDict[iax][ik]['Best']

					xOffsets[iax,ik,iRun] = valDict['xOffset']

					if iRun == 0:
						nextOffset = xOffsets[iax,ik,iRun]
					elif iRun >= 1:
						if runVarList[iRun] != runVarList[iRun-1]:
							slope = (xOffsets[iax,ik,iRun] - xOffsets[iax,ik,iRun-1])/(runVarList[iRun] - runVarList[iRun-1])
						else:
							slope = 0.
						nextOffset = xOffsets[iax,ik,iRun] + slope*(runVarList[iRun+1] - runVarList[iRun])
	
					if Ram.ChirpedData:
						## Compensate addition shift added from central fringe in class method
						nextOffset -= RamanOpts['CentralFringeChirp'][iax]

					## Update initial fit parameters for next iteration
					if ik == 0:
						runFitPars[iax]['xOffset_kU'].value = nextOffset
					else:
						runFitPars[iax]['xOffset_kD'].value = nextOffset
					runFitPars[iax]['yOffset'].value  = valDict['yOffset']
					runFitPars[iax]['Contrast'].value = valDict['Contrast']
					runFitPars[iax]['xScale'].value   = valDict['xScale']

################### End of RamanAnalysisLevel2() ####################
#####################################################################

def RamanAnalysisLevel3(AnalysisCtrl, RamanOpts, PlotOpts, RunPars):
	"""Method for Raman AnalysisLevel = 3:
	Reload and process Raman analysis level 2 results for selected runs and plot a summary.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamanOpts    (dict) - Copy of key:value pairs controlling Raman options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir   = AnalysisCtrl['WorkDir']
	Folder    = AnalysisCtrl['Folder']
	RunList   = AnalysisCtrl['RunList']
	nRuns  	  = len(RunList)

	iaxSet    = set()
	ikSet     = set()

	SummaryDF = [[pd.DataFrame(columns=[
		'Run', 'RunTime', RamanOpts['RunPlotVariable'], 'Seff', 'xOffset', 'xOffset_Err', 'yOffset', 'yOffset_Err',
		'Contrast', 'Contrast_Err', 'xScale', 'xScale_Err', 'SNR', 'SNR_Err', 'sigX', 'sigY', 'phi0', 'phi0_Err',
		'gExp', 'gExp_Err']) for ik in range(2)] for iax in range(3)]

	SummaryDF = [[pd.DataFrame([]) for ik in range(2)] for iax in range(3)]

	iRun = -1
	for RunNum in RunList:
		iRun += 1

		if RunNum == RunList[0]:
			Ram = Raman(WorkDir, Folder, RunNum, RamanOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Ram = Raman(WorkDir, Folder, RunNum, RamanOpts, PlotOpts)

		## Load raw data to extract Run timing
		Ram.LoadRawRamanData()
	
		if AnalysisCtrl['ProcessLevel'] == 0:
			Ram.RawData = True
		else:
			Ram.RawData = False

		Ram.ReadRamanAnalysisResults()

		if AnalysisCtrl['ProcessLevel'] == 0:
			fitList = getattr(Ram, 'RawFitDict')
		else:
			fitList = getattr(Ram, 'PostFitDict')

		iaxSet.update(Ram.iaxList)
		ikSet.update(Ram.ikList)

		for iax in Ram.iaxList:
			for ik in Ram.ikList:
				d = {'Run': iRun, 'RunTime': Ram.RunTime, RamanOpts['RunPlotVariable']: getattr(Ram, RamanOpts['RunPlotVariable']),
					'Seff': Ram.kSign[iax][ik]*Ram.Seff[iax],
					'xOffset': fitList[iax][ik]['Best']['xOffset'], 'xOffset_Err': fitList[iax][ik]['Error']['xOffset'],
					'yOffset': fitList[iax][ik]['Best']['yOffset'], 'yOffset_Err': fitList[iax][ik]['Error']['yOffset'],
					'Contrast': fitList[iax][ik]['Best']['Contrast'], 'Contrast_Err': fitList[iax][ik]['Error']['Contrast'],
					'xScale': fitList[iax][ik]['Best']['xScale'], 'xScale_Err': fitList[iax][ik]['Error']['xScale'],
					'SNR': fitList[iax][ik]['Best']['SNR'], 'SNR_Err': fitList[iax][ik]['Error']['SNR'],
					'sigX': fitList[iax][ik]['Best']['sigX'], 'sigY': fitList[iax][ik]['Best']['sigY'],
					'sigC': fitList[iax][ik]['Best']['sigC'],
					'phi0': fitList[iax][ik]['Best']['phi0'], 'phi0_Err': fitList[iax][ik]['Error']['phi0'],
					'gExp': fitList[iax][ik]['Best']['gExp'], 'gExp_Err': fitList[iax][ik]['Error']['gExp']}
				SummaryDF[iax][ik] = SummaryDF[iax][ik].append(d, ignore_index=True)

	if RamanOpts['SaveAnalysisLevel3']:
		WriteRamanAnalysisSummary(RunList, SummaryDF, PlotOpts['PlotFolderPath'], Ram.AxisFileLabels, list(d.keys()))

	if RunPars.ChirpedData:
		(nRows, nCols) = (3,3)
	else:
		(nRows, nCols) = (2,3)
	fig, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*4.5,nRows*2.), sharex='col', constrained_layout=True)

	if PlotOpts['PlotData']:
		if RamanOpts['RunPlotVariable'] == 'RamanT':
			## Special operations for RamanT
			xScale = 1.0E3
			xLabel = r'$T$  (ms)'
		elif RamanOpts['RunPlotVariable'] == 'RamanTOF':
			## Special operations for RamanT
			xScale = 1.0E3
			xLabel = 'TOF  (ms)'
		elif RamanOpts['RunPlotVariable'] == 'RamanpiX' or RamanOpts['RunPlotVariable'] == 'RamanpiY' or RamanOpts['RunPlotVariable'] == 'RamanpiZ':
			## Special operations for RamanpiX, RamanpiY, RamanpiZ
			xScale = 1.0E6
			xLabel = r'$\tau_{\pi}$  ($\mu$s)'
		elif RamanOpts['RunPlotVariable'] == 'RamanPower':
			## Special operations for RamanPower
			xScale = 1.
			xLabel = r'$P_{\rm Raman}$  (V)'
		elif RamanOpts['RunPlotVariable'] == 'RunTime':
			t0     = SummaryDF[list(iaxSet)[0]][list(ikSet)[0]]['RunTime'].iloc[0]
			dt0    = dt.datetime.fromtimestamp(t0, tz=pytz.timezone('Europe/Paris'))
			xScale = 1/60.
			xLabel = 'Run Time - {}  (min)'.format(dt0.strftime('%H:%M:%S'))
		else:
			xScale = 1.
			xLabel = RamanOpts['RunPlotVariable']

		for iax in iaxSet:
			for ik in ikSet:
				if SummaryDF[iax][ik].shape[0] > 0:
					x   = SummaryDF[iax][ik][RamanOpts['RunPlotVariable']].to_numpy()*xScale
					if RamanOpts['RunPlotVariable'] == 'RunTime':
						x -= t0*xScale
					Seff  = SummaryDF[iax][ik]['Seff'].to_numpy()
					x0    = SummaryDF[iax][ik]['xOffset'].to_numpy()
					dx0   = SummaryDF[iax][ik]['xOffset_Err'].to_numpy()
					y0    = SummaryDF[iax][ik]['yOffset'].to_numpy()
					dy0   = SummaryDF[iax][ik]['yOffset_Err'].to_numpy()
					c     = SummaryDF[iax][ik]['Contrast'].to_numpy()
					dc    = SummaryDF[iax][ik]['Contrast_Err'].to_numpy()
					SNR   = SummaryDF[iax][ik]['SNR'].to_numpy()
					dSNR  = SummaryDF[iax][ik]['SNR_Err'].to_numpy()
					sigX  = SummaryDF[iax][ik]['sigX'].to_numpy()
					sigY  = SummaryDF[iax][ik]['sigY'].to_numpy()
					sigC  = SummaryDF[iax][ik]['sigC'].to_numpy()
					phi0  = SummaryDF[iax][ik]['phi0'].to_numpy()
					dphi0 = SummaryDF[iax][ik]['phi0_Err'].to_numpy()
					gExp  = SummaryDF[iax][ik]['gExp'].to_numpy()
					dgExp = SummaryDF[iax][ik]['gExp_Err'].to_numpy()

					chirpsInterlaced = (SummaryDF[iax][0].shape[0] > 0 and SummaryDF[iax][1].shape[1] > 0)

					customPlotOpts = {'Color': Ram.DefaultPlotColors[iax][ik], 'Linestyle': 'None',
						'Marker': '.', 'Title': 'None', 'xLabel': 'None', 'yLabel': r'$y_0$', 
						'Legend': False, 'LegLabel': Ram.AxisLegLabels[iax][ik]}
					iXUtils.CustomPlot(axs[0][0], customPlotOpts, x, y0, yErr=dy0)

					customPlotOpts['yLabel'] = r'$C$'
					iXUtils.CustomPlot(axs[0][1], customPlotOpts, x, c, yErr=dc)

					if Ram.ChirpedData:
						## Assumes all data are chirped
						customPlotOpts['yLabel'] = r'$2\pi(\alpha_0 - \alpha_c) T_{eff}^2$  (rad)'
						customPlotOpts['Title']  = r'$\alpha_c = ${:8.6e}  Hz/s'.format(RamanOpts['CentralFringeChirp'][iax])

						if chirpsInterlaced and ik == 0:
							phikInd = 0.5*(SummaryDF[iax][0]['phi0'].to_numpy() + SummaryDF[iax][1]['phi0'].to_numpy())
							phikDep = 0.5*(SummaryDF[iax][0]['phi0'].to_numpy() - SummaryDF[iax][1]['phi0'].to_numpy())
							dphi    = 0.5*np.sqrt(SummaryDF[iax][0]['phi0_Err'].to_numpy()**2 + SummaryDF[iax][1]['phi0_Err'].to_numpy()**2)

							customPlotOpts['Color']    = Ram.DefaultPlotColors[iax][2]
							customPlotOpts['LegLabel'] = Ram.AxisLegLabels[iax][2]
							iXUtils.CustomPlot(axs[0][2], customPlotOpts, x, phikInd, dphi)
							customPlotOpts['Color']    = Ram.DefaultPlotColors[iax][3]
							customPlotOpts['LegLabel'] = Ram.AxisLegLabels[iax][3]
							iXUtils.CustomPlot(axs[0][2], customPlotOpts, x, phikDep, dphi)
							customPlotOpts['Color']    = Ram.DefaultPlotColors[iax][ik]
							customPlotOpts['LegLabel'] = Ram.AxisLegLabels[iax][ik]
						elif not chirpsInterlaced:
							iXUtils.CustomPlot(axs[0][2], customPlotOpts, x, phi0, yErr=dphi0)

						customPlotOpts['Title']  = 'None'
					else:
						## Assumes all data are not chirped
						customPlotOpts['yLabel'] = r'$\phi_0 - S_{\rm eff} g$  (rad)'
						iXUtils.CustomPlot(axs[0][2], customPlotOpts, x, phi0 - Seff*Ram.gLocal, yErr=dphi0)

					if nRows == 2:
						customPlotOpts['xLabel'] = xLabel
					customPlotOpts['yLabel']   = 'SNR'
					iXUtils.CustomPlot(axs[1][0], customPlotOpts, x, SNR, yErr=dSNR)

					customPlotOpts['yLabel']   = r'$\sigma_{\phi}$  (rad)'
					iXUtils.CustomPlot(axs[1][1], customPlotOpts, x, sigX)

					customPlotOpts['yLabel']   = r'$\sigma_{Y}$, $\sigma_{C}$'
					customPlotOpts['LegLabel'] = Ram.AxisLegLabels[iax][ik]
					iXUtils.CustomPlot(axs[1][2], customPlotOpts, x, sigY)
					customPlotOpts['Color']    = 'darkviolet' if ik == 0 else 'purple'
					customPlotOpts['LegLabel'] = None
					iXUtils.CustomPlot(axs[1][2], customPlotOpts, x, sigC)
					customPlotOpts['Color']    = Ram.DefaultPlotColors[iax][ik]

					if nRows > 2:
						customPlotOpts['xLabel'] = xLabel
						customPlotOpts['yLabel'] = r'$|\alpha_0|$  (Hz/s)'
						iXUtils.CustomPlot(axs[2][0], customPlotOpts, x, x0, yErr=dx0)

						if chirpsInterlaced:
							customPlotOpts['yLabel'] = r'$a_{\uparrow}, a_{\downarrow}$  (m/s$^s$)'
						elif ik == 0 and SummaryDF[iax][1].shape[1] == 0:
							customPlotOpts['yLabel'] = r'$a_{\uparrow}$  (m/s$^s$)'
						else:
							customPlotOpts['yLabel'] = r'$a_{\downarrow}$  (m/s$^s$)'
						iXUtils.CustomPlot(axs[2][1], customPlotOpts, x, gExp, yErr=dgExp)

						if chirpsInterlaced and ik == 0:
							akInd  = 0.5*(gExp - SummaryDF[iax][1]['gExp'].to_numpy())
							akDep  = 0.5*(gExp + SummaryDF[iax][1]['gExp'].to_numpy())
							da     = 0.5*np.sqrt(dgExp**2 + SummaryDF[iax][1]['gExp_Err'].to_numpy()**2)
							akDep_Avg = np.average(akDep, weights=1./da)
							akInd  = (akInd/akDep_Avg)*1.E6
							akDep  = (akDep/akDep_Avg - 1)*1.E6
							da     = (da/akDep_Avg)*1.E6

							customPlotOpts['yLabel']   = r'$\Delta a / \bar{a}$  (ppm)'
							customPlotOpts['Color']    = Ram.DefaultPlotColors[iax][2]
							customPlotOpts['LegLabel'] = Ram.AxisLegLabels[iax][2]
							iXUtils.CustomPlot(axs[2][2], customPlotOpts, x, akInd, yErr=da)

							customPlotOpts['Color']    = Ram.DefaultPlotColors[iax][3]
							customPlotOpts['LegLabel'] = Ram.AxisLegLabels[iax][3]
							iXUtils.CustomPlot(axs[2][2], customPlotOpts, x, akDep, yErr=da)

	if PlotOpts['ShowPlotLegend']:
		if PlotOpts['FixLegLocation']:
			## Fix legend location outside upper right of plot
			axs[0][2].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
			axs[1][2].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
			if nRows == 3:
				axs[2][2].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		else:
			## Let matlibplot find best legend location
			axs[0][2].legend(loc='best')
			axs[1][2].legend(loc='best')
			if nRows == 3:
				axs[2][2].legend(loc='best')

	if PlotOpts['SavePlot']:
		plt.savefig(Ram.PlotPath, dpi=150)
		logging.info('iXC_Raman::Raman plot saved to:')
		logging.info('iXC_Raman::  {}'.format(Ram.PlotPath))
	elif PlotOpts['ShowPlot']:
		plt.show()

################### End of RamanAnalysisLevel3() ####################
#####################################################################

def WriteRamanAnalysisSummary(RunList, SummaryDF, Folder, Labels, Columns=None):
	"""Method for writing Raman AnalysisLevel = 3 results.
	ARGUMENTS:
	\t RunList   (list) - List of Runs contained in SummaryDF
	\t SummaryDF (list) - List of dataframes containing analysis summary for each axis and k-direction
	\t Folder    (str)  - Path to folder in which to store analysis summary file
	\t Labels    (list) - Raman axis file labels
	\t Columns   (list) - Ordered subset of columns to write to file
	"""

	fileName    = 'Raman-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))
	floatFormat = '%11.9E'

	for iax in range(3):
		for ik in range(2):
			if SummaryDF[iax][ik].shape[0] > 0:
				filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][ik]+'.txt')
				iXUtils.WriteDataFrameToFile(SummaryDF[iax][ik], Folder, filePath, True, False, floatFormat, Columns)

################ End of WriteRamanAnalysisSummary() #################
#####################################################################

def ReadRamanAnalysisSummary(iaxList, ikList, RunList, Folder, Labels):
	"""Method for reading Raman AnalysisLevel = 3 results.
	ARGUMENTS:
	\t iaxList  (list) - List of Raman axis indices
	\t ikList   (list) - List of k-direction indices 
	\t RunList  (list) - List of Runs contained in summary file
	\t Folder   (str)  - Path to folder containing analysis summary file
	\t Labels   (list) - Raman axis file labels
	"""

	SummaryDF = [[pd.DataFrame([]) for ik in range(2)] for iax in range(3)]
	fileName  = 'Raman-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))

	for iax in iaxList:
		for ik in ikList:
			filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][ik]+'.txt')
			if os.path.exists(filePath):
				SummaryDF[iax][ik] = pd.read_csv(filePath, sep='\t')
			else:
				logging.error('iXC_Raman::ReadRamanAnalysisSummary::File not found specified path: {}'.format(filePath))
				logging.error('iXC_Raman::ReadRamanAnalysisSummary::Aborting...')
				quit()

	return SummaryDF

################ End of ReadRamanAnalysisSummary() ##################
#####################################################################

def RamanAnalysisLevel4(AnalysisCtrl, RamanOpts, PlotOpts, RunPars):
	"""Method for Raman AnalysisLevel = 4:
	Reload Raman analysis level 3 results for selected runs and perform time series analysis.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamanOpts    (dict) - Copy of key:value pairs controlling Raman options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	RunNum  = RunList[0]
	nRuns  	= len(RunList)

	## Load first run to extract basic parameters
	Ram = Raman(WorkDir, Folder, RunNum, RamanOpts, PlotOpts, False, RunPars.__dict__.items())
	## Load summary data from Analysis Level = 3
	SummaryDF = ReadRamanAnalysisSummary(Ram.iaxList, Ram.ikList, RunList, PlotOpts['PlotFolderPath'], Ram.AxisFileLabels)

	Location = {
		'Latitude':		44.804,	## [deg]
		'Longitude': 	-0.605,	## [deg]
		'Height':		21.		## [m] 
		}

	for iax in Ram.iaxList:

		nData  = SummaryDF[iax][Ram.ikList[ 0]]['RunTime'].shape[0]
		tStart = SummaryDF[iax][Ram.ikList[ 0]]['RunTime'].iloc[ 0]
		tStop  = SummaryDF[iax][Ram.ikList[-1]]['RunTime'].iloc[-1]
		tStep  = (tStop - tStart)/(nData-1)

		t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
		xLabel = 'Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))

		tRange = np.array([0., tStop - tStart, tStep])
		tData  = np.linspace(tStart, tStop, num=nData, endpoint=True)

		if RamanOpts['TimeSeriesQuantity'] == 'Gravity':
			## Get tidal anomaly and rescale relative to g
			gTide = Ram.Phys.GetTideModel(tData, Location, True)/Ram.gLocal

			if Ram.kInterlaced:
				if RamanOpts['ComputeMovingAvg']:
					winSize = max(int(round(RamanOpts['MovingAvgWindow']/tStep)), 1)
					akU  = SummaryDF[iax][0]['gExp'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					akD  = SummaryDF[iax][1]['gExp'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					dakU = SummaryDF[iax][0]['gExp'].rolling(winSize, center=True, min_periods=1).std().to_numpy()/np.sqrt(winSize)
					dakD = SummaryDF[iax][1]['gExp'].rolling(winSize, center=True, min_periods=1).std().to_numpy()/np.sqrt(winSize)
				else:
					akU  = SummaryDF[iax][0]['gExp'].to_numpy()
					akD  = SummaryDF[iax][1]['gExp'].to_numpy()
					dakU = SummaryDF[iax][0]['gExp_Err'].to_numpy()
					dakD = SummaryDF[iax][1]['gExp_Err'].to_numpy()

				## Rescale accelerations relative to gLocal
				akU     = (akU/Ram.gLocal - 1.)
				akD     = (akD/Ram.gLocal - 1.)
				dakU	= dakU/Ram.gLocal
				dakD	= dakD/Ram.gLocal
				akInd   = 0.5*(akU - akD)
				akDep   = 0.5*(akU + akD)
				dakInd  = 0.5*np.sqrt(dakU**2 + dakD**2)
				yData   = [[ akU,  akD], [ akInd], [ akDep, akDep-gTide, gTide]]
				yErr    = [[dakU, dakD], [dakInd], [dakInd, dakInd, 0.*gTide]]
				yScales = [[1.E6, 1.E6], [1.E6], [1.E6, 1.E6, 1.E6]]
				colors  = [['red', 'blue'], ['gray'], ['black', 'green', 'darkorange']]
				xLabels = ['None', 'None', xLabel]
				yLabels = [r'$a_{\uparrow, \downarrow} - g$  ($\mu$g)', r'$a_{\rm ind}$  ($\mu$g)', r'$a_{\rm dep} - g$  ($\mu$g)',
					r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)']
				lLabels = [[r'$a_{\uparrow}$', r'$a_{\downarrow}$'], [r'$a_{\rm ind}$'], 
					[r'$a_{\rm dep}$', r'$a_{\rm dep} - a_{\rm tide}$', r'$a_{\rm tide}$']]

				ADev_Subsets      = [[True, True], [True], [True, True, False]]
				ADev_Fit          = [[RamanOpts['ADev_Fit'], RamanOpts['ADev_Fit']], [RamanOpts['ADev_Fit']], [RamanOpts['ADev_Fit'], RamanOpts['ADev_Fit'], False]]
				ADev_Fit_FitExp   = [[True, True], [True], [True, True, False]]
				ADev_Fit_SetRange = [[False, False], [False], [False, False, False]]
				ADev_Fit_Range    = [[[1.5E2, 2.E3], [1.5E2, 2.E3]], [[1.5E2, 2.E3]], [[1.5E2, 2.E3], [1.5E2, 2.E3], [1.5E2, 2.E3]]]
			else:
				if RamanOpts['ComputeMovingAvg']:
					winSize = max(int(round(RamanOpts['MovingAvgWindow']/tStep)), 1)
					ak  = SummaryDF[iax][Ram.ikList[0]]['gExp'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					dak = SummaryDF[iax][Ram.ikList[0]]['gExp'].rolling(winSize, center=True, min_periods=1).std().to_numpy()
				else:
					ak  = SummaryDF[iax][Ram.ikList[0]]['gExp'].to_numpy()
					dak = SummaryDF[iax][Ram.ikList[0]]['gExp_Err'].to_numpy()

				## Rescale accelerations relative to gLocal
				ak      = (ak/Ram.gLocal - 1.)
				dak		= dakU/Ram.gLocal
				yData   = [[ak, ak-gTide, gTide]]
				yErr    = [[dak, dak, 0.*gTide]]
				yScales = [[1.E6, 1.E6, 1.E6]]
				colors  = [['red' if Ram.ikList[0] == 0 else 'blue', 'green', 'darkorange']]
				xLabels = [xLabel]
				yLabels = [r'$a_{\uparrow} - g$  ($\mu$g)' if Ram.ikList[0] == 0 else r'$a_{\downarrow} - g$  ($\mu$g)', 
					r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)']
				lLabels = [[r'$a_{\uparrow}$' if Ram.ikList[0] == 0 else r'$a_{\downarrow}$', 
					r'$a_{\uparrow} - a_{\rm tide}$' if Ram.ikList[0] == 0 else r'$a_{\downarrow} - a_{\rm tide}$', r'$a_{\rm tide}$']]

				ADev_Subsets      = [[True, True, False]]
				ADev_Fit          = [[RamanOpts['ADev_Fit'], RamanOpts['ADev_Fit'], False]]
				ADev_Fit_FitExp   = [[False, False, False]]
				ADev_Fit_SetRange = [[False, False, False]]
				ADev_Fit_Range    = [[[1.5E2, 2.E3], [1.5E2, 2.E3], [1.5E2, 2.E3]]]

		else:
			## Perform time-series analysis on interferometer phase

			if Ram.kInterlaced:
				if RamanOpts['ComputeMovingAvg']:
					winSize = max(int(round(RamanOpts['MovingAvgWindow']/tStep)), 1)
					pkU  = SummaryDF[iax][0]['phi0'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					pkD  = SummaryDF[iax][1]['phi0'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					dpkU = SummaryDF[iax][0]['phi0'].rolling(winSize, center=True, min_periods=1).std().to_numpy()/np.sqrt(winSize)
					dpkD = SummaryDF[iax][1]['phi0'].rolling(winSize, center=True, min_periods=1).std().to_numpy()/np.sqrt(winSize)
				else:
					pkU  = SummaryDF[iax][0]['phi0'].to_numpy()
					pkD  = SummaryDF[iax][1]['phi0'].to_numpy()
					dpkU = SummaryDF[iax][0]['phi0_Err'].to_numpy()
					dpkD = SummaryDF[iax][1]['phi0_Err'].to_numpy()

				pkInd   = 0.5*(pkU + pkD)
				pkDep   = 0.5*(pkU - pkD)
				dpk     = 0.5*np.sqrt(dpkU**2 + dpkD**2)
				yData   = [[pkU, pkD], [pkInd, pkDep]]
				yErr    = [[dpkU, dpkD], [dpk, dpk]]
				yScales = [[1., 1.], [1., 1.]]
				colors  = [[Ram.DefaultPlotColors[iax][0], Ram.DefaultPlotColors[iax][1]], [Ram.DefaultPlotColors[iax][2], Ram.DefaultPlotColors[iax][3]]]
				xLabels = ['None', xLabel]
				yLabels = [r'$\phi_{\!\uparrow\!\!\downarrow}$  (rad)', r'$\phi_{\rm ind, dep}$  (rad)',
					r'$\sqrt{\rm PSD}$  (rad$/\sqrt{\rm Hz}$)', r'Allan Deviation  (rad)']
				lLabels = [[r'$\phi_{\!\uparrow}$', r'$\phi_{\!\downarrow}$'], [r'$\phi_{\rm ind}$', r'$\phi_{\rm dep}$']]

				ADev_Subsets      = [[True, True], [True, True]]
				ADev_Fit          = [[RamanOpts['ADev_Fit'], RamanOpts['ADev_Fit']], [RamanOpts['ADev_Fit'], RamanOpts['ADev_Fit']]]
				ADev_Fit_FitExp   = [[True, True], [True, True]]
				ADev_Fit_SetRange = [[False, False], [False, False]]
				ADev_Fit_Range    = [[[1.5E2, 2.E3], [1.5E2, 2.E3]], [[1.5E2, 2.E3], [1.5E2, 2.E3]]]
			else:
				if RamanOpts['ComputeMovingAvg']:
					winSize = max(int(round(RamanOpts['MovingAvgWindow']/tStep)), 1)
					pk  = SummaryDF[iax][Ram.ikList[0]]['phi0'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					dpk = SummaryDF[iax][Ram.ikList[0]]['phi0'].rolling(winSize, center=True, min_periods=1).std().to_numpy()
				else:
					pk  = SummaryDF[iax][Ram.ikList[0]]['phi0'].to_numpy()
					dpk = SummaryDF[iax][Ram.ikList[0]]['phi0_Err'].to_numpy()

				yData   = [[pk]]
				yErr    = [[dak]]
				yScales = [[1.E6]]
				colors  = [[Ram.DefaultPlotColors[iax][ik]]]
				xLabels = [xLabel]
				yLabels = [r'$\phi_{\!\uparrow}$  (rad)' if Ram.ikList[0] == 0 else r'$\phi_{\!\downarrow}$  (rad)', 
					r'$\sqrt{\rm PSD}$  (rad$/\sqrt{\rm Hz}$)', r'Allan Deviation  (rad)']
				lLabels = [[r'$\phi_{\!\uparrow}$' if Ram.ikList[0] == 0 else r'$\phi_{\!\downarrow}$']]

				ADev_Subsets      = [[True]]
				ADev_Fit          = [[RamanOpts['ADev_Fit']]]
				ADev_Fit_FitExp   = [[False]]
				ADev_Fit_SetRange = [[False]]
				ADev_Fit_Range    = [[[1.5E2, 2.E3]]]

		Options = {
			'SavePlot'			: PlotOpts['SavePlot'],
			'PlotFolderPath'	: PlotOpts['PlotFolderPath'],
			'PlotFileName'		: 'Raman-Runs{:02d}-{:02d}-TimeSeriesAnalysis'.format(RunList[0],RunList[-1]) + PlotOpts['PlotExtension'],
			'ColumnDim'			: (5, 6),
			'Colors'			: colors,
			'Linestyle'			: '-',
			'Linewidth'			: 1.,
			'Marker'			: 'None',
			'Markersize'		: 6, 
			'ShowErrors'		: True,
			'SampleRate'		: 1./tStep,
			'xLabels'			: xLabels,
			'yLabels'			: yLabels,
			'yScales'			: yScales,
			'ShowFigureLabels'  : PlotOpts['ShowFigureLabels'],
			'FigureLabels'		: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
			'ShowLegend'		: [True, False, True],
			'LegendLabels'		: lLabels,
			'LegendLocations'	: ['best', 'best', 'best'],
			'LegendFontSize'	: 10,
			'SetPlotLimits'		: [False, False],
			'PlotXLimits'		: [-2500., 1.2*tRange[1]],
			'PlotYLimits'		: [[0.5,0.9], [0.5,0.9]],
			'PSD_Plot'			: RamanOpts['PSD_Plot'],
			'PSD_PlotSubSets'	: ADev_Subsets,
			'PSD_Method'		: RamanOpts['PSD_Method'],
			'ADev_Plot'			: RamanOpts['ADev_Plot'],
			'ADev_PlotSubSets'  : ADev_Subsets,
			'ADev_Type'			: 'Total',
			'ADev_taus'			: 'all',
			'ADev_ShowErrors'	: RamanOpts['ADev_ShowErrors'],
			'ADev_Errorstyle'	: 'Shaded', # 'Bar' or 'Shaded'
			'ADev_Linestyle' 	: '-',
			'ADev_Marker'    	: 'None',
			'ADev_SetLimits'	: [False, False],
			'ADev_XLimits'		: [1.E2, 4.E4],
			'ADev_YLimits'		: [1.E-8, 1.E-6],
			'ADev_Fit'			: ADev_Fit,
			'ADev_Fit_XLimits'	: [0.9*tRange[2], 1.1*(0.4*tRange[1])],
			'ADev_Fit_SetRange'	: ADev_Fit_SetRange,
			'ADev_Fit_Range'	: ADev_Fit_Range,
			'ADev_Fit_FixExp'	: ADev_Fit_FitExp
			}

		iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)

################### End of RamanAnalysisLevel4() ####################
#####################################################################

def RamanAnalysisLevel5(AnalysisCtrl, RamanOpts, MonitorOpts, PlotOpts, RunPars):
	"""Method for Raman AnalysisLevel = 5:
	Reload Raman analysis level 3 results for selected runs and correlate with monitor data.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamanOpts    (dict) - Copy of key:value pairs controlling Raman options
	\t MonitorOpts  (dict) - Copy of key:value pairs controlling monitor options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	RunNum  = RunList[0]
	nRuns  	= len(RunList)

	## Load first run to extract basic parameters
	Ram = Raman(WorkDir, Folder, RunNum, RamanOpts, PlotOpts, False, RunPars.__dict__.items())
	## Load monitor data
	Mon = iXC_Monitor.Monitor(WorkDir, Folder, RunList, MonitorOpts, PlotOpts, False, Ram.__dict__.items())
	Mon.ProcessMonitorData()
	## Load summary data from Analysis Level = 3
	SummaryDF = ReadRamanAnalysisSummary(Ram.iaxList, Ram.ikList, RunList, PlotOpts['PlotFolderPath'], Ram.AxisFileLabels)

	for iax in Ram.iaxList:

		nData  = SummaryDF[iax][Ram.ikList[ 0]]['RunTime'].shape[0]
		tStart = SummaryDF[iax][Ram.ikList[ 0]]['RunTime'].iloc[ 0]
		tStop  = SummaryDF[iax][Ram.ikList[-1]]['RunTime'].iloc[-1]
		tStep  = (tStop - tStart)/(nData-1)

		t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
		xLabel = 'Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))

		tRange = np.array([0., tStop - tStart, tStep])
		tData  = np.linspace(tStart, tStop, num=nData, endpoint=True)

		if Ram.kInterlaced:
			if RamanOpts['ComputeMovingAvg']:
				winSize = max(int(round(RamanOpts['MovingAvgWindow']/tStep)), 1)
				akU = SummaryDF[iax][0]['gExp'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
				akD = SummaryDF[iax][1]['gExp'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
			else:
				akU = SummaryDF[iax][0]['gExp'].to_numpy()
				akD = SummaryDF[iax][1]['gExp'].to_numpy()
			## Rescale accelerations to micro-g relative to gLocal
			akU     = (akU/Ram.gLocal - 1.)*1.E6
			akD     = (akD/Ram.gLocal - 1.)*1.E6
			akInd   = 0.5*(akU - akD)
			akDep   = 0.5*(akU + akD)
			yData   = [akU, akD, akInd, akDep]
			yLabels = [r'$a_{\rm kU}$  ($\mu$g)', r'$a_{\rm kD}$  ($\mu$g)', r'$a_{\rm kInd}$  ($\mu$g)', r'$a_{\rm kDep}$  ($\mu$g)']
			ikRange = range(2,3+1)
		else:
			if RamanOpts['ComputeMovingAvg']:
				winSize = max(int(round(RamanOpts['MovingAvgWindow']/tStep)), 1)
				ak = SummaryDF[iax][Ram.ikList[0]]['gExp'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
			else:
				ak = SummaryDF[iax][Ram.ikList[0]]['gExp'].to_numpy()
			## Rescale accelerations to micro-g relative to gLocal
			ak      = (ak/Ram.gLocal - 1.)*1.E6
			yData   = [ak, ak]
			yLabels = [r'$a_{\rm kU}$  ($\mu$g)', r'$a_{\rm kD}$  ($\mu$g)']
			ikRange = Ram.ikList

		for ik in ikRange:
			Mon.PlotMonitorCorrelations(iax, ik, yData[ik], yLabels[ik], MonDFType='Mean', iStart=0, iSkip=1)

################### End of RamanAnalysisLevel5() ####################
#####################################################################