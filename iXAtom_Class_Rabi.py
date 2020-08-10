#####################################################################
## Filename:	iXAtom_Class_Rabi.py
## Author:		B. Barrett
## Description: Rabi class definition for iXAtom analysis package
## Version:		3.2.4
## Last Mod:	03/07/2020
##===================================================================
## Change Log:
## 03/02/2020 - Rabi class defined based on Rabi class. Basic plotting
##				functionality only.
## 23/02/2020 - Implemented working fit function for Rabi class
##				(valid for pulse-duration-type Rabi oscillations only).
##			  - Completed methods for Rabi Analysis Levels 2 and 3.
## 03/07/2020 - Added capability to analyze N2, NTotal and Ratio data in Rabi
##				analysis. In the future, this feature could be made a general
##				option for all analysis types.
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
from   scipy.special import erf, erfc

import iXAtom_Utilities 		  as iXUtils
import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Class_Detector	  as iXC_Detect
import iXAtom_Class_Monitor		  as iXC_Monitor

class Rabi(iXC_RunPars.RunParameters):
	#################################################################
	## Class for storing and processing Rabi data
	## Inherits all attributes and methods from class: RunParameters
	#################################################################

	def __init__(self, WorkDir, Folder, RunNum, RabiOpts, PlotOpts, LoadRunParsFlag=True, RunPars=[]):
		"""Initialize Rabi variables.
		ARGUMENTS:
		\t WorkDir    (str)  - Path to the top-level directory where dataset is located
		\t Folder     (str)  - Name of folder within WorkDir where dataset is located
		\t RunNum     (int)  - Run number of requested dataset
		\t RabiOpts   (dict) - Key:value pairs controlling Rabi options
		\t PlotOpts   (dict) - Key:value pairs controlling plot options
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

		self.RabiOptions   = copy.deepcopy(RabiOpts)
		self.PlotOptions   = copy.deepcopy(PlotOpts)
		self.idCoeffs      = np.array([0.,0.,0.])
		self.SetDetectCoeffs(self.RabiOptions['DetectCoeffs'])
		self.DetectVariable = self.RabiOptions['DetectVariable']

		self.RawData       = False
		self.RawDataFound  = False
		self.PostDataFound = False
		self.RawDataFiles  = ['Rabi-Run{:02d}-Ratios.txt'.format(self.Run)]
		self.PlotPath  	   = os.path.join(self.PlotOptions['PlotFolderPath'], 'Rabi-Run{:02d}-RawData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

		self.RawDataDF     = [pd.DataFrame([]) for iax in range(3)]
		self.PostDataDF    = [pd.DataFrame([]) for iax in range(3)]
		self.Outliers      = [[] for iax in range(3)]
		self.RawFitResult  = [{} for iax in range(3)]
		self.PostFitResult = [{} for iax in range(3)]
		self.RawFitDict    = [{} for iax in range(3)]
		self.PostFitDict   = [{} for iax in range(3)]

		# self.Fit_method    = 'lm' ## 'lm': Levenberg-Marquardt, 'trf': Trust Region Reflective, 'dogbox': dogleg algorithm with rectangular trust regions
		self.Fit_ftol      = 1.E-5
		self.Fit_xtol      = 1.E-5
		self.Fit_maxfev    = 20000

		self.nFitPars      = len(self.RabiOptions['FitParameters'][0].valuesdict().keys())

	################### End of Rabi.__init__() ######################
	#################################################################

	def LoadRawRabiData(self):
		"""Load raw Rabi data from data file into a Pandas dataframe."""

		logging.info('iXC_Rabi::Loading   raw Rabi data for {}...'.format(self.RunString))

		ik = 0
		dataPath = os.path.join(self.RawFolderPath, self.RawDataFiles[ik])
		if os.path.exists(dataPath):
			self.RawDataFound = True
			df = pd.read_csv(dataPath, sep='\t')
			for iax in self.iaxList:
				# Select rows of the imported data frame corresponding to each Raman axis
				self.RawDataDF[iax] = df[df['AccelAxis'] == iax]

			self.GetRunTiming(self.RabiOptions['PrintRunTiming'], RawDF=True)
		else:
			self.RawDataFound = False
			logging.warning('iXC_Rabi::LoadRawRabiData::Raw Rabi data not found!')

	################ End of Rabi.LoadRawRabiData() ##################
	#################################################################

	def ParseRabiData(self, iax):
		"""Parse Rabi data from pandas dataframe.
		ARGUMENTS:
		\t iax (int) - Index representing Raman axis (X,Y,Z = 0,1,2)
		RETURN:
		\t xData (np array) - 
		\t yData (np array) - 
		\t yErr  (np array) - 
		"""

		if self.RawData:
			df = self.RawDataDF[iax]
		else:
			df = self.PostDataDF[iax]

		xData   = df['CurrentXValue'].to_numpy()
		nPoints = len(xData)

		if self.RawData:
			if self.DetectVariable == 'NT':
				yData = df['NTotal'].to_numpy()
			else:
				yData = df[self.DetectVariable].to_numpy()				
			yErr = np.zeros(nPoints)
		else:
			yData = np.zeros(nPoints)
			yErr  = np.zeros(nPoints)
			for id in self.idList:
				if self.idCoeffs[id] > 0:
					label = self.DetectNames[id][0]
					yData +=  self.idCoeffs[id]*df[self.DetectVariable+'_Mean_'+label].to_numpy()
					yErr  += (self.idCoeffs[id]*df[self.DetectVariable+'_SDev_'+label].to_numpy())**2

			yErr = np.sqrt(yErr)

		return [xData, yData, yErr]

	################# End of Rabi.ParseRabiData() ###################
	#################################################################

	@staticmethod
	def fRabi(x, Omega, xOffset, yOffset, Amplitude, alpha, beta, gamma):
		"""Fit model for Rabi oscillations."""

		return yOffset + alpha*np.tanh(beta*(x-xOffset)) + Amplitude*np.exp(-gamma*(x-xOffset))*np.sin(0.5*abs(Omega)*(x-xOffset))**2

	##################### End of Rabi.fRabi() #######################
	#################################################################

	@staticmethod
	def fDetect(x, xOffset, yOffset, Amplitude, vInitial, rDetect, sigmaR, sigmaV):
		"""Fit model for detection profile based on TOF signal theory."""

		def erf2(x, y):
			"""Two-argument generalized Error function. Avoids catastrophic cancellation when x, y become large."""
			if (x <= 0. and y >= 0.) or (y <= 0. and x >= 0.):
				return erf(y) - erf(x)
			elif x >= 0. and y >= 0.:
				return erfc(x) - erfc(y)
			elif x <= 0. and y <= 0.:
				return erfc(-y) - erfc(-x)

		g   = 9.805642E-3							## mm/ms^2
		dz  = 0.5*g*(x**2 - xOffset**2) 			## mm
		sig = np.sqrt(sigmaR**2 + (sigmaV*x)**2) 	## mm
		a   = sig*(vInitial*x + dz)/(sigmaR*sigmaV*x)
		b   = rDetect*sigmaV*x/(sig*sigmaR)
		z1  = a + b
		z2  = a - b
		n   = len(z1)
		erf2List = np.zeros(n)
		for i in range(n):
			erf2List[i] = erf2(z1[i], z2[i])

		S0  = 0.5#/(np.sqrt(np.pi)*sigmaV)
		S1  = S0*sig/(sigmaV*x)*np.exp(-rDetect**2/sig**2 + (dz**2 + 2.*vInitial*x*dz)/(sigmaV*x)**2)*erf2List
		S2  = S0*np.exp(-(vInitial/sigmaV)**2)*(erf((rDetect - vInitial*x - dz)/sigmaR) + erf((rDetect + vInitial*x + dz)/sigmaR))

		return Amplitude*(S1 + S2) + yOffset

	#################### End of Rabi.fDetect() ######################
	#################################################################

	def ConstructFitModel(self, iax):
		"""Construct fit model from lmfit Model class."""

		pInit = self.RabiOptions['FitParameters'][iax].copy()

		if self.ScanQuantity == 'Detection TOF (ms)':
			model = lm.Model(self.fDetect)
			model.set_param_hint('xOffset',   value=pInit['xOffset'].value,   min=pInit['xOffset'].min,   vary=pInit['xOffset'].vary)
			model.set_param_hint('yOffset',   value=pInit['yOffset'].value,   min=pInit['yOffset'].min,   vary=pInit['yOffset'].vary)
			model.set_param_hint('Amplitude', value=pInit['Amplitude'].value, min=pInit['Amplitude'].min, vary=pInit['Amplitude'].vary)
			model.set_param_hint('vInitial',  value=pInit['vInitial'].value,  min=pInit['vInitial'].min,  vary=pInit['vInitial'].vary)
			model.set_param_hint('rDetect',   value=pInit['rDetect'].value,   min=pInit['rDetect'].min,   vary=pInit['rDetect'].vary)
			model.set_param_hint('sigmaR',    value=pInit['sigmaR'].value,    min=pInit['sigmaR'].min,    vary=pInit['sigmaR'].vary)
			model.set_param_hint('sigmaV',    value=pInit['sigmaV'].value,    min=pInit['sigmaV'].min,    vary=pInit['sigmaV'].vary)
		else:
			model = lm.Model(self.fRabi)
			model.set_param_hint('Omega',     value=pInit['Omega'].value,     min=pInit['Omega'].min,     vary=pInit['Omega'].vary)
			model.set_param_hint('xOffset',   value=pInit['xOffset'].value,   min=pInit['xOffset'].min,   vary=pInit['xOffset'].vary)
			model.set_param_hint('yOffset',   value=pInit['yOffset'].value,   min=pInit['yOffset'].min,   vary=pInit['yOffset'].vary)
			model.set_param_hint('Amplitude', value=pInit['Amplitude'].value, min=pInit['Amplitude'].min, vary=pInit['Amplitude'].vary)
			model.set_param_hint('alpha',     value=pInit['alpha'].value,     min=pInit['alpha'].min,     vary=pInit['alpha'].vary)
			model.set_param_hint('beta',      value=pInit['beta'].value,      min=pInit['beta'].min,      vary=pInit['beta'].vary)
			model.set_param_hint('gamma',     value=pInit['gamma'].value,     min=pInit['gamma'].min,     vary=pInit['gamma'].vary)

		return model

	################ End of Rabi.ConstructFitModel() ################
	#################################################################

	def FitRabiData(self, xData, yData, yErr):
		"""Fit Rabi data using non-linear least squares module.
		ARGUMENTS:
		xData (np.array) - Independent variable values
		yData (np.array) - Dependent variable values
		yErr  (np.array) - Errors on dependent values
		"""

		if np.dot(yErr, yErr) > 0.:
			weights = 1./yErr
		else:
			weights = np.ones(len(yErr))

		fit_kws = {'xtol': self.Fit_xtol, 'ftol': self.Fit_ftol, 'maxfev': self.Fit_maxfev} 
		result  = self.FitModel.fit(yData, self.FitPars, x=xData, weights=weights, method='leastsq', fit_kws=fit_kws)

		## Note that result.residual is divided by the errors
		# yRes    = result.residual
		yRes    = yData - result.best_fit
		sRes    = np.std(yRes)
		dsRes   = sRes/np.sqrt(2*result.nfree)
		message = 'iXC_Rabi::{} (ier = {}, nfev = {}, redchi = {:5.3E})'.format(result.message, result.ier, result.nfev, result.redchi)

		if result.success and result.ier <= 3:
			logging.info(message)
		else:
			logging.warning(message)
			logging.warning(result.lmdif_message)

		return [result, yRes, sRes, dsRes]

	################## End of Rabi.FitRabiData() ####################
	#################################################################

	def AddAuxPars(self, iax, Result, AuxData):
		"""Add auxiliary parameters to a Parameters object, including peak type, chi**2,
		reduced-chi**2, std dev. of residuals, fit SNR, and temperature if appropriate.
		ARGUMENTS:
		iax     (int)            - Index representing Raman axis.
		Result  (lm.ModelResult) - Fit result.
		AuxData (dict)			 - Key:value pairs of auxiliary data to add.
		RETURN:
		Pars    (lm.Parameters)  - Modified parameters.
		"""

		Pars  = lm.Parameters()

		## Add initial value
		for par in Result.params.values():
			par.init_value = Result.init_values[par.name]
			Pars.add(par)

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

		amp    = Result.params['Amplitude']
		SNR    = lm.Parameter('SNR', value=amp.value/resdev)
		SNR.init_value = 0.
		SNR.stderr = (resdev.stderr/resdev.value)*SNR.value

		Pars.add_many(resdev, chisqr, redchi, SNR)

		if self.ScanQuantity != 'Detection TOF (ms)':
			omega = Result.params['Omega']
			x0    = Result.params['Offset']
			taupi = lm.Parameter('taupi', value=np.pi/abs(omega.value) + x0.value)
			taupi.init_value = np.pi/omega.init_value
			try:
				taupi.stderr = np.sqrt((np.pi/omega.value**2*omega.stderr)**2 + (x0.stderr/x0.value)**2)
			except:
				taupi.stderr = 0.

			Pars.add(taupi)

		return Pars

	################### End of Rabi.AddAuxPars() ####################
	#################################################################

	def UpdateFitDicts(self, iax, Pars):
		"""Update Rabi fit dictionaries."""

		pName  = [val.name for val in Pars.values()]
		pBest  = [val.value for val in Pars.values()]

		try:
			pError = [val.stderr for val in Pars.values()]
			pInit  = [val.init_value for val in Pars.values()]
		except:
			pError = [0. for val in Pars.values()]
			pInit  = copy.copy(pBest)

		if self.RawData:
			self.RawFitDict[iax]['Init']   = {key:val for key, val in zip(pName, pInit)}
			self.RawFitDict[iax]['Best']   = {key:val for key, val in zip(pName, pBest)}
			self.RawFitDict[iax]['Error']  = {key:val for key, val in zip(pName, pError)}
		else:
			self.PostFitDict[iax]['Init']  = {key:val for key, val in zip(pName, pInit)}
			self.PostFitDict[iax]['Best']  = {key:val for key, val in zip(pName, pBest)}
			self.PostFitDict[iax]['Error'] = {key:val for key, val in zip(pName, pError)}

	################### End of Rabi.UpdateFitDicts() ####################
	#####################################################################

	def AnalyzeRabiData(self):
		"""Analyze Rabi data (raw or post-processed)."""

		if self.RawData:
			label = 'raw'
			dfList = self.RawDataDF
		else:
			label = 'post-processed'
			dfList = self.PostDataDF

		logging.info('iXC_Rabi::Analyzing {} Rabi data for {}...'.format(label, self.RunString))

		if self.RabiOptions['FitData']:
			## Load and fit Rabi data, and store fit results in data frames
			for iax in self.iaxList:
				## Construct fit model and parameters
				self.FitModel = self.ConstructFitModel(iax)
				self.FitPars  = self.FitModel.make_params()

				[xData, yData, yErr] = self.ParseRabiData(iax)
				[result, yRes, sRes, dsRes] = self.FitRabiData(xData, yData, yErr)

				if self.RabiOptions['RemoveOutliers']:
					iPoint = -1
					for dy in yRes:
						iPoint += 1
						if abs(dy) > self.RabiOptions['OutlierThreshold']*sRes:
							self.Outliers[iax].append(iPoint)
							logging.info('iXC_Rabi::Removing outlier at (iax,iPoint) = ({},{}), nOutlier = {}...'.format(iax,iPoint,len(self.Outliers[iax])))
							logging.info('iXC_Rabi::  Outlier: (x,y,yRes) = ({:5.3e}, {:5.3e}, {:4.2f}*sigma)'.format(xData[iPoint],yData[iPoint],dy/sRes))
					if len(self.Outliers[iax]) > 0:
						xData  = np.delete(xData, self.Outliers[iax])
						yData  = np.delete(yData, self.Outliers[iax])
						yErr   = np.delete(yErr,  self.Outliers[iax])
						[result, yRes, sRes, dsRes] = self.FitRabiData(iax, xData, yData, yErr)

				if self.RawData:
					self.RawFitResult[iax]  = result
				else:
					self.PostFitResult[iax] = result

				auxData = {'ResDev': sRes, 'ResErr': dsRes, 'ChiSqr': result.chisqr, 'RedChiSqr': result.redchi}
				pars    = self.AddAuxPars(iax, result, auxData)
				self.UpdateFitDicts(iax, pars)
		else:
			for iax in self.iaxList:
				pars = self.RabiOptions['FitParameters'][iax]
				self.UpdateFitDicts(iax, pars)

	################## End of Rabi.AnalyzeRabiData() ####################
	#####################################################################

	def PlotRabiAxisData(self, RabiAxs, iRun, iax, CustomPlotOpts, Labels):
		"""Plot Rabi data (raw or post-processed) and associated fit for given axis.
		ARGUMENTS:
		\t RabiAxs      (list) - Rabi figure axes corresponding to a given Raman axis
		\t iax             (int) - Index corresponding to a given axis
		\t iRun            (int) - Index corresponding to run number in RunList
		\t CustomPlotOpts (dict) - Key:value pairs controlling custom plot options
		\t Labels         (list) - Plot labels
		"""

		[xLabel, yLabel] = Labels
		[xData, yData, yErr] = self.ParseRabiData(iax)

		if self.RabiOptions['RemoveOutliers'] and len(self.Outliers[iax]) > 0:
			xData = np.delete(xData, self.Outliers[iax])
			yData = np.delete(yData, self.Outliers[iax])
			yErr  = np.delete(yErr,  self.Outliers[iax])

		if np.dot(yErr, yErr) == 0.:
			yErr = []

		## Main Rabi plot
		if self.PlotOptions['OverlayRunPlots']:
			nColors = len(self.RabiOptions['RunPlotColors'])
			CustomPlotOpts['Color'] = self.RabiOptions['RunPlotColors'][iRun%(nColors)]
			if self.RabiOptions['RunPlotVariable'] == 'Run':
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', Run {:02d}'.format(self.Run)
			elif self.RabiOptions['RunPlotVariable'] == 'RamanTOF':
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', {:5.2f} ms'.format(self.RamanTOF*1.E+3)
			elif self.RabiOptions['RunPlotVariable'] == 'RunTime':
				runTimeStamp = dt.datetime.fromtimestamp(self.RunTime, tz=pytz.timezone('Europe/Paris')).strftime('%H:%M:%S')
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', {}'.format(runTimeStamp)
			else:
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', {:5.2e}'.format(getattr(self, self.RabiOptions['RunPlotVariable']))
		else:
			CustomPlotOpts['Color']    = self.DefaultPlotColors[iax][0]
			CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax]

		CustomPlotOpts['Linestyle'] = 'None'
		CustomPlotOpts['Marker']    = '.'
		iXUtils.CustomPlot(RabiAxs[1], CustomPlotOpts, xData, yData, yErr)

		if self.PlotOptions['ShowFit'] and self.RabiOptions['FitData']:
			## Plot spectra fit
			xData = np.sort(xData)
			if self.RabiOptions['SetFitPlotXLimits']:
				[xMin, xMax] = self.RabiOptions['FitPlotXLimits']
				nPoints = 2*round(abs(xMax - xMin)/abs(xData[1] - xData[0]))
			else:
				## Assumes data are sorted
				xMin = xData[0]
				xMax = xData[-1]
				nPoints = 2*len(xData)

			xFit = np.linspace(xMin, xMax, num=nPoints, endpoint=True)
			yFit = np.array([])

			if self.RawData:
				fitResult = self.RawFitResult[iax]
			else:
				fitResult = self.PostFitResult[iax]

			if self.PlotOptions['OverlayRunPlots']:
				CustomPlotOpts['Color'] = self.RabiOptions['RunPlotColors'][iRun%(nColors)]
			else:
				CustomPlotOpts['Color'] = self.DefaultPlotColors[iax][0]
			CustomPlotOpts['Linestyle'] = '-'
			CustomPlotOpts['Marker']    = None
			CustomPlotOpts['LegLabel']  = None

			yFit = fitResult.eval(x=xFit)
			iXUtils.CustomPlot(RabiAxs[1], CustomPlotOpts, xFit, yFit)

			# dyFit = fitResult.eval_uncertainty(x=xFit)
			# RabiAxs[1].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

			CustomPlotOpts['Linestyle'] = '-'
			CustomPlotOpts['Marker']    = '.'

			# yRes = fitResult.residual ## Already has outliers deleted
			## Note that fitResult.residual is divided by the errors
			yRes = yData - fitResult.best_fit
			iXUtils.CustomPlot(RabiAxs[0], CustomPlotOpts, xData, yRes)

		if self.RabiOptions['SetPlotLimits'][0]:
			RabiAxs[0].set_xlim(self.RabiOptions['PlotLimits'][0])
			RabiAxs[1].set_xlim(self.RabiOptions['PlotLimits'][0])
		if self.RabiOptions['SetPlotLimits'][1]:
			RabiAxs[1].set_ylim(self.RabiOptions['PlotLimits'][1])
		if self.PlotOptions['ShowPlotLabels'][0]:
			RabiAxs[1].set_xlabel(xLabel)
		if self.PlotOptions['ShowPlotLabels'][1]:
			RabiAxs[1].set_ylabel(yLabel)
			RabiAxs[0].set_ylabel('Residue')
		if self.PlotOptions['ShowPlotTitle']:
			RabiAxs[0].set_title(self.RunString + ', TOF = {:4.2f} ms'.format(self.RamanTOF*1.0E+3))
		if self.PlotOptions['ShowPlotLegend']:
			if self.PlotOptions['FixLegLocation']:
				## Fix legend location outside upper right of plot
				RabiAxs[1].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
			else:
				## Let matlibplot find best legend location
				RabiAxs[1].legend(loc='best')

	################ End of Rabi.PlotRabiAxisData() #################
	#################################################################

	def PlotRabiData(self, RabiAxs, iRun):
		"""Plot Rabi data (raw or post-processed) and associated fit (if requested).
		ARGUMENTS:
		\t RabiAxs (axs list) - Current Rabi figure axes
		\t iRun           (int) - Index corresponding to run number in RunList   
		"""

		if self.RawData:
			label = 'raw'
			dfList = self.RawDataDF
		else:
			label = 'post-processed'
			dfList = self.PostDataDF

		logging.info('iXC_Rabi::Plotting {} Rabi data for {}...'.format(label, self.RunString))

		if self.ScanQuantity == 'Raman Pulse Duration (us)':
			xLabel = r'$\tau_{\rm Raman}$  ($\mu$s)'
		elif self.ScanQuantity == 'Raman Power (V)':
			xLabel = r'$P_{\rm Raman}$  (V)'
		elif self.ScanQuantity == 'Selection Pulse Duration (us)':
			xLabel = r'$\tau_{\rm Select}$  ($\mu$s)'
		elif self.ScanQuantity == 'Selection Power (V)':
			xLabel = r'$P_{\rm Select}$  (V)'
		elif self.ScanQuantity == 'Detection TOF (ms)':
			xLabel = 'Detect TOF  (ms)'
		else:
			xLabel = 'Scan Quantity'

		if self.DetectVariable == 'Ratio':
			yLabel = r'$N_2/N_{\rm total}$'
		elif self.DetectVariable == 'N2':
			yLabel = r'$N_2$ (arb. units)'
		else:
			yLabel = r'$N_{\rm total}$ (arb. units)'

		customPlotOpts = {'Color': 'red', 'Linestyle': 'None', 'Marker': '.', 'Title': 'None', 
			'xLabel': 'None', 'yLabel': 'None', 'Legend': False, 'LegLabel': None}

		iRow = -1
		for iax in self.iaxList:
			iRow += 1
			if self.PlotOptions['OverlayRunPlots']:
				## Set options for sharing x axes
				showXLabel = True if iax == self.iaxList[-1] else False
				# plt.setp(RabiAxs[iRow].get_xticklabels(), visible=showXLabel)
				self.PlotOptions['ShowPlotLabels'][0] = showXLabel

				self.PlotRabiAxisData(RabiAxs[iRow], iRun, iax, customPlotOpts, [xLabel, yLabel])
			else:
				self.PlotRabiAxisData(RabiAxs[0], iRun, iax, customPlotOpts, [xLabel, yLabel])

		if self.PlotOptions['SavePlot']:
			plt.savefig(self.PlotPath, dpi=150)
			logging.info('iXC_Rabi::Rabi plot saved to:')
			logging.info('iXC_Rabi::  {}'.format(self.PlotPath))
		elif self.PlotOptions['ShowPlot']:
			plt.show()

	################## End of Rabi.PlotRabiData() ###################
	#################################################################

	def PostProcessRabiData(self, DetectOpts):
		"""Post-process Rabi detection data and write to file (if requested).
		ARGUMENTS:
		\t DetectOpts (dict) - Key:value pairs controlling detection options
		"""

		logging.info('iXC_Rabi::Post-processing Rabi data for {}...'.format(self.RunString))

		# Declare detector object
		det = iXC_Detect.Detector(self.WorkDir, self.Folder, self.Run, DetectOpts, self.PlotOptions, False, self.__dict__.items())

		prefixes    = ['N2', 'NT', 'Ratio']
		meanHeaders = ['_Mean_L', '_Mean_M', '_Mean_U']
		sdevHeaders = ['_SDev_L', '_SDev_M', '_SDev_U']

		if det.nIterations > 0:
			self.LoadRawRabiData()

			## Store copy of raw Rabi data in post Rabi dataframe
			self.PostDataDF = [self.RawDataDF[iax].copy() for iax in range(3)]

			## Reshape PostDataDF
			for iax in self.iaxList:
				self.PostDataDF[iax].drop(columns=['AccelAxis','N2','NTotal','Ratio'], inplace=True)
				# self.PostDataDF[iax].rename(columns={'RatioMean':   'RatioMean'}, inplace=True)
				# self.PostDataDF[iax].rename(columns={'RatioStdDev': 'RatioSDev'}, inplace=True)
				for id in range(3):
					for prefix in prefixes:
						self.PostDataDF[iax].loc[:,prefix+meanHeaders[id]] = 0.
						self.PostDataDF[iax].loc[:,prefix+sdevHeaders[id]] = 0.
	
			avgNum = 1
			for iterNum in range(1,det.nIterations+1):

				if iterNum == 1:
					initPars = det.InitializeDetectPars()

				det.LoadDetectTrace(avgNum, iterNum)
				det.AnalyzeDetectTrace(avgNum, iterNum, initPars)

				if 'Rabi Axis' in det.DetectTags.keys():
					iax = int(det.DetectTags['Rabi Axis'])
				else:
					iax = int(list(det.DetectTags.values())[1])

				if iax not in range(3):
					logging.warning('iXC_Rabi::PostProcessRabiData::Raman axis index out of range. Setting iax = 2 (Z)...')
					iax = 2

				mask = (self.PostDataDF[iax]['#Iteration'] == iterNum)
				for id in det.idList:
					for ic in range(4):
						initPars[id][ic] = det.DetectResult[id][det.DetectKeys[ic]]['BestPars']
					for prefix in prefixes:
						self.PostDataDF[iax].loc[mask, prefix+meanHeaders[id]] = det.DetectResult[id][prefix]['Best']
						self.PostDataDF[iax].loc[mask, prefix+sdevHeaders[id]] = det.DetectResult[id][prefix]['Error']

			if self.RabiOptions['SaveAnalysisLevel1']:
				showHeaders = True
				showIndices = False
				floatFormat = '%11.9E' ## Output 10-digits of precision (needed for the frequency)

				for iax in self.iaxList:
					iXUtils.WriteDataFrameToFile(self.PostDataDF[iax], self.PostFolderPath, self.PostFilePaths[iax],
						showHeaders, showIndices, floatFormat)

				logging.info('iXC_Rabi::Post-processed Rabi data saved to:')
				logging.info('iXC_Rabi::  {}'.format(self.PostFolderPath))
		else:
			logging.error('iXC_Rabi::PostProcessRabiData::Aborting detection analysis on {}...'.format(self.RunString))
			quit()

	############### End of Rabi.PostProcessRabiData() ###############
	#################################################################

	def LoadPostRabiData(self):
		"""Load post-processed Rabi data into a list of data frames."""

		logging.info('iXC_Rabi::Loading   post-processed Rabi data for {}...'.format(self.RunString))

		nFilesFound = 0
		for iax in self.iaxList:
			if os.path.exists(self.PostFilePaths[iax]):
				nFilesFound += 1
				self.PostDataDF[iax] = pd.read_csv(self.PostFilePaths[iax], sep='\t')

		if nFilesFound == self.nax:
			self.PostDataFound = True
			self.GetRunTiming(self.RabiOptions['PrintRunTiming'], RawDF=False)
		else:
			self.PostDataFound = False
			logging.warning('iXC_Rabi::LoadPostRabiData::Post-processed Rabi data not found in: {}'.format(self.PostFolderPath))

	############### End of Rabi.LoadPostRabiData() ##################
	#################################################################

	def WriteRabiAnalysisResults(self):
		"""Write Rabi analysis results to file."""

		if self.RawData:
			label = 'raw'
		else:
			label = 'post-processed'

		logging.info('iXC_Rabi::Writing {} Rabi analysis results to:'.format(label))
		logging.info('iXC_Rabi::  {}'.format(self.PostFolderPath))	

		for iax in self.iaxList:
			if self.RawData:
				fileName  = 'Rabi-Run{:02}-Results-Raw-'.format(self.Run)  + self.AxisFileLabels[iax] + '.txt'
				dataFrame = pd.DataFrame.from_dict(self.RawFitDict[iax])
			else:
				fileName  = 'Rabi-Run{:02}-Results-Post-'.format(self.Run) + self.AxisFileLabels[iax] + '.txt'
				dataFrame = pd.DataFrame.from_dict(self.PostFitDict[iax])

			filePath = os.path.join(self.PostFolderPath, fileName)
			iXUtils.WriteDataFrameToFile(dataFrame, self.PostFolderPath, filePath, True, True, '%10.8E')

	############ End of Rabi.WriteRabiAnalysisResults() #############
	#################################################################

	def ReadRabiAnalysisResults(self):
		"""Read Rabi analysis results to file."""

		if self.RawData:
			label1 = 'raw'
			label2 = 'Raw'
		else:
			label1 = 'post-processed'
			label2 = 'Post'

		logging.info('iXC_Rabi::Reading {} Rabi analysis results from:'.format(label1))
		logging.info('iXC_Rabi::  {}'.format(self.PostFolderPath))

		for iax in self.iaxList:
			fileName = 'Rabi-Run{:02}-Results-{}-'.format(self.Run, label2) + self.AxisFileLabels[iax] + '.txt'
			filePath = os.path.join(self.PostFolderPath, fileName)
			if os.path.exists(filePath):
				if self.RawData:
					self.RawFitDict[iax] = pd.read_csv(filePath, sep='\t', header=0, index_col=0).to_dict()
				else:
					self.PostFitDict[iax] = pd.read_csv(filePath, sep='\t', header=0, index_col=0).to_dict()
			else:
				logging.error('iXC_Rabi::ReadRabiAnalysisResults::Analysis file not found: {}'.format(filePath))
				logging.error('iXC_Rabi::ReadRabiAnalysisResults::Aborting...')
				quit()

	############# End of Rabi.ReadRabiAnalysisResults() #############
	#################################################################

	def LoadAndAnalyzeRabiData(self, RabiAxs, iRun, nRuns, PlotOpts, ProcessLevel):
		"""Load and analyze raw and/or post-processed Rabi data; 
		plot the results and write to file (if requested).
		ARGUMENTS:
		\t RabiAxs (fig axs) - List of Rabi figure axes for overlaying plots
		\t iRun	         (int) - Index corresponding to current run number (starts from 0)
		\t nRuns         (int) - Number of runs contained within RunList
		\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
		\t ProcessLevel  (int) - Secondary control for Rabi analysis type
		"""

		if ProcessLevel == 0 or ProcessLevel == 2:

			self.RawData = True
			self.LoadRawRabiData()

			if self.RawDataFound:
				self.AnalyzeRabiData()
				if self.RabiOptions['SaveAnalysisLevel2'] and self.RabiOptions['FitData']:
					self.WriteRabiAnalysisResults()

				if nRuns <= self.PlotOptions['MaxPlotsToDisplay'] and self.RabiOptions['ShowFitResults']:
					for iax in self.iaxList:
						print('---------------------------------------------------')
						print(self.AxisFileLabels[iax] + ' - Raw Data - Fit Results:')
						print('---------------------------------------------------')
						print(pd.DataFrame.from_dict(self.RawFitDict[iax]))
			else:
				logging.warning('iXC_Rabi::LoadAndAnalyzeRabiData::Aborting Rabi raw data analysis for {}...'.format(self.RunString))

		if ProcessLevel == 1 or ProcessLevel == 2:

			self.RawData = False
			self.LoadPostRabiData()

			if self.PostDataFound:
				self.AnalyzeRabiData()
				if self.RabiOptions['SaveAnalysisLevel2'] and self.RabiOptions['FitData']:
					self.WriteRabiAnalysisResults()

				if nRuns <= self.PlotOptions['MaxPlotsToDisplay'] and self.RabiOptions['ShowFitResults']:
					for iax in self.iaxList:
						if ProcessLevel == 1 or ProcessLevel == 2:
							print('---------------------------------------------------')
							print(self.AxisFileLabels[iax] + ' - Post Data - Fit Results:')
							print('---------------------------------------------------')
							print(pd.DataFrame.from_dict(self.PostFitDict[iax]))
			else:
				logging.warning('iXC_Rabi::LoadAndAnalyzeRabiData::Aborting Rabi post-processed data analysis for {}...'.format(self.RunString))

		if (self.PlotOptions['ShowPlot'] and not self.PlotOptions['SavePlot'] and not self.PlotOptions['OverlayRunPlots']) and nRuns > self.PlotOptions['MaxPlotsToDisplay']:
			self.PlotOptions['PlotData'] = False

		if self.PlotOptions['PlotData'] and nRuns <= self.PlotOptions['MaxPlotsToDisplay']:
			if ProcessLevel < 2 and (self.RawDataFound or self.PostDataFound):
				if ProcessLevel == 0:
					self.RawData = True
					self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
						'Rabi-Run{:02d}-RawData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])
				else:
					self.RawData = False
					self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
						'Rabi-Run{:02d}-PostData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

				if self.PlotOptions['OverlayRunPlots']:
					self.PlotOptions['ShowPlotTitle']  = False
					self.PlotOptions['ShowPlotLegend'] = True
					if iRun < nRuns-1:
						self.PlotOptions['ShowPlot'] = False
						self.PlotOptions['SavePlot'] = False
					else:
						self.PlotOptions['ShowPlot'] = PlotOpts['ShowPlot']
						self.PlotOptions['SavePlot'] = PlotOpts['SavePlot']

				self.PlotRabiData(RabiAxs[0], iRun)

			if ProcessLevel == 2 and (self.RawDataFound and self.PostDataFound):
				self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
					'Rabi-Run{:02d}-Raw+PostData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

				self.RawData = True
				self.PlotOptions['ShowPlot'] = False
				self.PlotOptions['SavePlot'] = False
				self.PlotOptions['ShowPlotLabels'] = [False, True]
				if self.PlotOptions['OverlayRunPlots']:
					self.PlotOptions['ShowPlotTitle']  = False
					self.PlotOptions['ShowPlotLegend'] = False
				else:
					self.PlotOptions['ShowPlotTitle']  = PlotOpts['ShowPlotTitle']
					self.PlotOptions['ShowPlotLegend'] = PlotOpts['ShowPlotLegend']
				self.PlotRabiData(RabiAxs[0], iRun)

				self.RawData = False
				self.PlotOptions['ShowPlotLabels'] = [True, True]
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
				self.PlotRabiData(RabiAxs[1], iRun)

	############# End of Rabi.LoadAndAnalyzeRabiData() ##############
	#################################################################

#####################################################################
######################## End of Class Rabi ##########################
#####################################################################

def RabiAnalysisLevel1(AnalysisCtrl, RabiOpts, DetectOpts, PlotOpts, RunPars):
	"""Method for Rabi AnalysisLevel = 1:
	Post-process detection data for selected Runs and store results.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RabiOpts   (dict) - Copy of key:value pairs controlling Rabi options
	\t DetectOpts   (dict) - Copy of key:value pairs controlling Detection options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	for RunNum in RunList:
		if RunNum == RunList[0]:
			Rab = Rabi(WorkDir, Folder, RunNum, RabiOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Rab = Rabi(WorkDir, Folder, RunNum, RabiOpts, PlotOpts)
		Rab.PostProcessRabiData(DetectOpts)

################### End of RabiAnalysisLevel1() #####################
#####################################################################

def RabiAnalysisLevel2(AnalysisCtrl, RabiOpts, PlotOpts):
	"""Method for Rabi AnalysisLevel = 2:
	Load raw/post-processed Rabi data for selected Runs, analyze and store results.
	Plot the data and fits in individual figures or overlayed in a single figure.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RabiOpts   (dict) - Copy of key:value pairs controlling Rabi options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	runList = AnalysisCtrl['RunList']
	nRuns   = len(runList)

	runFitPars = copy.deepcopy(RabiOpts['FitParameters'])
	runVarKey  = RabiOpts['RunPlotVariable']
	runVarList = np.zeros(nRuns)
	Omegas     = np.zeros((3,nRuns)) ## [iax,iRun]

	if not RabiOpts['FitData'] or nRuns == 1: 
		RabiOpts['TrackRunFitPars'] = False

	if RabiOpts['TrackRunFitPars']:
		logging.info('iXC_Raman::Getting list of Run variables "{}"...'.format(runVarKey))
		logging.disable(level=logging.INFO) # Disable logger for info & debug levels

		## Load run variables from parameter files and find sort order
		iRun = -1
		for runNum in runList:
			iRun += 1
			Rab = Rabi(WorkDir, Folder, runNum, RabiOpts, PlotOpts)
			if runVarKey not in vars(Rab):
				## Rabi data must be loaded to extract certain run variables
				Rab.LoadRawRabiData()

			runVarList[iRun] = getattr(Rab, runVarKey)

		## Sort run variables and run list
		orderList  = np.argsort(runVarList)
		runVarList = runVarList[orderList]
		runList    = [runList[iRun] for iRun in orderList]

		if RabiOpts['SortVariableOrder'] == 'Descending':
			runVarList = runVarList[::-1]
			runList    = runList[::-1]

		logging.disable(level=logging.NOTSET) # Re-enable logger at configured level

	RabiAxs = None
	iRun = -1
	for runNum in runList:
		iRun += 1

		Rab = Rabi(WorkDir, Folder, runNum, RabiOpts, PlotOpts)
		## Assumes number of Raman axes present is identical for all runs in RunList
		RabiAxs = Rab.CreatePlotAxes(RabiAxs, iRun, nRuns, Rab.nax, PlotOpts, AnalysisCtrl['ProcessLevel'])

		Rab.RabiOptions['FitParameters'] = runFitPars
		Rab.LoadAndAnalyzeRabiData(RabiAxs, iRun, nRuns, PlotOpts, AnalysisCtrl['ProcessLevel'])

		if RabiOpts['TrackRunFitPars'] and iRun < nRuns-1:
			for iax in Rab.iaxList:
				if Rab.RawData:
					valDict = Rab.RawFitDict[iax]['Best']
				else:
					valDict = Rab.PostFitDict[iax]['Best']

					Omegas[iax,iRun] = abs(valDict['Omega'])

					if iRun == 0:
						nextOmega = Omegas[iax,iRun]
					elif iRun >= 1:
						if runVarList[iRun] != runVarList[iRun-1]:
							slope = (Omegas[iax,iRun] - Omegas[iax,iRun-1])/(runVarList[iRun] - runVarList[iRun-1])
						else:
							slope = 0.
						nextOmega = Omegas[iax,iRun] + slope*(runVarList[iRun+1] - runVarList[iRun])
	
					## Update initial fit parameters for next iteration
					runFitPars[iax]['Omega'].value     = nextOmega
					runFitPars[iax]['xOffset'].value   = valDict['xOffset']
					runFitPars[iax]['yOffset'].value   = valDict['yOffset']
					runFitPars[iax]['Amplitude'].value = valDict['Amplitude']
					runFitPars[iax]['alpha'].value     = valDict['alpha']
					runFitPars[iax]['beta'].value      = valDict['beta']
					runFitPars[iax]['gamma'].value     = valDict['gamma']

################### End of RabiAnalysisLevel2() #####################
#####################################################################

def RabiAnalysisLevel3(AnalysisCtrl, RabiOpts, PlotOpts, RunPars):
	"""Method for Rabi AnalysisLevel = 3:
	Reload and process Rabi analysis results for selected runs and plot a summary.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RabiOpts     (dict) - Copy of key:value pairs controlling Rabi options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir   = AnalysisCtrl['WorkDir']
	Folder    = AnalysisCtrl['Folder']
	RunList   = AnalysisCtrl['RunList']
	nRuns  	  = len(RunList)

	tRun   = np.zeros(nRuns)
	x      = np.zeros(nRuns)
	Omega  = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	taupi  = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	x0     = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	y0     = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	amp    = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	alpha  = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	beta   = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	gamma  = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	SNR    = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]

	SummaryDF = [pd.DataFrame([]) for iax in range(3)]

	(nRows, nCols) = (2,3)

	Fig, Axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*4,nRows*3), sharex=True, constrained_layout=True)

	iRun = -1
	for RunNum in RunList:
		iRun += 1

		if RunNum == RunList[0]:
			Rab = Rabi(WorkDir, Folder, RunNum, RabiOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Rab = Rabi(WorkDir, Folder, RunNum, RabiOpts, PlotOpts)

		## Load raw data to extract Run timing
		Rab.LoadRawRabiData()
		tRun[iRun] = Rab.RunTime

		if AnalysisCtrl['ProcessLevel'] == 0:
			Rab.RawData = True
		else:
			Rab.RawData = False

		Rab.ReadRabiAnalysisResults()

		x[iRun] = getattr(Rab, RabiOpts['RunPlotVariable'])

		if AnalysisCtrl['ProcessLevel'] == 0:
			fitList = getattr(Rab, 'RawFitDict')
		else:
			fitList = getattr(Rab, 'PostFitDict')

		for iax in Rab.iaxList:
			Omega[iax,0,iRun] = abs(fitList[iax]['Best']['Omega'])
			Omega[iax,1,iRun] = fitList[iax]['Error']['Omega']
			taupi[iax,0,iRun] = fitList[iax]['Best' ]['taupi']
			taupi[iax,1,iRun] = fitList[iax]['Error']['taupi']
			x0[iax,0,iRun]    = fitList[iax]['Best' ]['xOffset']
			x0[iax,1,iRun]    = fitList[iax]['Error']['xOffset']
			y0[iax,0,iRun]    = fitList[iax]['Best' ]['yOffset']
			y0[iax,1,iRun]    = fitList[iax]['Error']['yOffset']
			amp[iax,0,iRun]   = fitList[iax]['Best' ]['Amplitude']
			amp[iax,1,iRun]   = fitList[iax]['Error']['Amplitude']
			alpha[iax,0,iRun] = fitList[iax]['Best' ]['alpha']
			alpha[iax,1,iRun] = fitList[iax]['Error']['alpha']
			beta[iax,0,iRun]  = fitList[iax]['Best' ]['beta']
			beta[iax,1,iRun]  = fitList[iax]['Error']['beta']
			gamma[iax,0,iRun] = fitList[iax]['Best' ]['gamma']
			gamma[iax,1,iRun] = fitList[iax]['Error']['gamma']
			SNR[iax,0,iRun]   = fitList[iax]['Best' ]['SNR']
			SNR[iax,1,iRun]   = fitList[iax]['Error']['SNR']

	for iax in Rab.iaxList:
		d = {'Run': RunList, 'RunTime': tRun, RabiOpts['RunPlotVariable']: x,
			'Omega': Omega[iax,0], 'Omega_Err': Omega[iax,1], 
			'taupi': taupi[iax,0], 'taupi_Err': taupi[iax,1],
			'xOffset': x0[iax,0], 'xOffset_Err': x0[iax,1], 
			'yOffset': y0[iax,0], 'yOffset_Err': y0[iax,1], 
			'amp': amp[iax,0], 'amp_Err': amp[iax,1], 
			'alpha': alpha[iax,0], 'alpha_Err': alpha[iax,1], 
			'beta': beta[iax,0], 'beta_Err': beta[iax,1], 
			'gamma': gamma[iax,0], 'gamma_Err': gamma[iax,1], 
			'SNR': SNR[iax,0], 'SNR_Err': SNR[iax,1]}
		SummaryDF[iax] = pd.DataFrame(data=d)

	if RabiOpts['SaveAnalysisLevel3']:
		WriteRabiAnalysisSummary(Rab.iaxList, RunList, SummaryDF, PlotOpts['PlotFolderPath'], Rab.AxisFileLabels)

	if PlotOpts['PlotData']:
		if RabiOpts['RunPlotVariable'] == 'RamanTOF':
			## Special operations for RamanTOF
			x *= 1.0E3
			xLabel = 'TOF  (ms)'
		elif RabiOpts['RunPlotVariable'] == 'RamanPower':
			## Special operations for RamanPower
			xLabel = 'Raman Power  (V)'
		elif RabiOpts['RunPlotVariable'] == 'RamanpiX' or RabiOpts['RunPlotVariable'] == 'RamanpiY' or RabiOpts['RunPlotVariable'] == 'RamanpiZ':
			## Special operations for RamanpiX, RamanpiY, RamanpiZ
			x *= 1.0E6
			xLabel = r'$\tau_{\pi}$  ($\mu$s)'
		elif RabiOpts['RunPlotVariable'] == 'RunTime':
			t0 = dt.datetime.fromtimestamp(x[0], tz=pytz.timezone('Europe/Paris'))
			x  = (x - x[0])/60.
			xLabel = 'Run Time - {}  (min)'.format(t0.strftime('%H:%M:%S'))
		else:
			xLabel = RabiOpts['RunPlotVariable']

		customPlotOpts = {'Color': 'black', 'Linestyle': 'None',
			'Marker': '.', 'Title': 'None', 'xLabel': 'None', 'yLabel': 'None', 
			'Legend': False, 'LegLabel': 'None'}

		for iax in Rab.iaxList:
			customPlotOpts = {'Color': Rab.DefaultPlotColors[iax][0], 'Linestyle': 'None',
				'Marker': '.', 'Title': 'None', 'xLabel': 'None', 'yLabel': r'$A$', 
				'Legend': False, 'LegLabel': Rab.AxisLegLabels[iax]}
			iXUtils.CustomPlot(Axs[0][0], customPlotOpts, x, amp[iax,0], amp[iax,1])

			customPlotOpts['yLabel'] = r'$\Omega_{\rm eff}$  (rad/$\mu$s)'
			iXUtils.CustomPlot(Axs[0][1], customPlotOpts, x, Omega[iax,0], Omega[iax,1])

			customPlotOpts['yLabel'] = r'$\tau_{\pi}$  ($\mu$s)'
			iXUtils.CustomPlot(Axs[0][2], customPlotOpts, x, taupi[iax,0], taupi[iax,1])

			customPlotOpts['xLabel'] = xLabel
			# customPlotOpts['yLabel'] = 'SNR'
			# iXUtils.CustomPlot(Axs[1][0], customPlotOpts, x, SNR[iax,0], SNR[iax,1])

			customPlotOpts['yLabel'] = r'$\alpha$'
			iXUtils.CustomPlot(Axs[1][0], customPlotOpts, x, alpha[iax,0], alpha[iax,1])

			customPlotOpts['yLabel'] = r'$\beta$ ($\mu$s$^{-1}$)'
			iXUtils.CustomPlot(Axs[1][1], customPlotOpts, x, beta[iax,0], beta[iax,1])

			customPlotOpts['yLabel'] = r'$\gamma$ ($\mu$s$^{-1}$)'
			iXUtils.CustomPlot(Axs[1][2], customPlotOpts, x, gamma[iax,0], gamma[iax,1])

	if PlotOpts['ShowPlotLegend']:
		if PlotOpts['FixLegLocation']:
			## Fix legend location outside upper right of plot
			Axs[0][2].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		else:
			## Let matlibplot find best legend location
			Axs[0][2].legend(loc='best')
	if PlotOpts['SavePlot']:
		plt.savefig(Rab.PlotPath, dpi=150)
		logging.info('iXC_Rabi::Rabi plot saved to:')
		logging.info('iXC_Rabi::  {}'.format(Rab.PlotPath))
	elif PlotOpts['ShowPlot']:
		plt.show()

################### End of RabiAnalysisLevel3() #####################
#####################################################################

def WriteRabiAnalysisSummary(iaxList, RunList, SummaryDF, Folder, Labels):
	"""Method for writing Rabi AnalysisLevel = 3 results.
	ARGUMENTS:
	\t iaxList   (list) - List of Raman axis indices
	\t RunList   (list) - List of Runs contained in SummaryDF
	\t SummaryDF (list) - List of dataframes containing analysis summary for each axis and k-direction
	\t Folder    (str)  - Path to folder in which to store analysis summary file
	\t Labels    (list) - Raman axis file labels
	"""

	fileName    = 'Rabi-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))
	floatFormat = '%11.9E'

	for iax in iaxList:
		filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][0]+'.txt')
		iXUtils.WriteDataFrameToFile(SummaryDF[iax], Folder, filePath, True, False, floatFormat)

################# End of WriteRabiAnalysisSummary() #################
#####################################################################

def ReadRabiAnalysisSummary(iaxList, RunList, Folder, Labels):
	"""Method for reading Raman AnalysisLevel = 3 results.
	ARGUMENTS:
	\t iaxList  (list) - List of Raman axis indices
	\t RunList  (list) - List of Runs contained in summary file
	\t Folder   (str)  - Path to folder containing analysis summary file
	\t Labels   (list) - Raman axis file labels
	"""

	SummaryDF = [pd.DataFrame([]) for iax in range(3)]
	fileName  = 'Rabi-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))

	for iax in iaxList:
		filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][0]+'.txt')
		if os.path.exists(filePath):
			SummaryDF[iax] = pd.read_csv(filePath, sep='\t')
		else:
			logging.error('iXC_Rabi::ReadRamanAnalysisSummary::File not found specified path: {}'.format(filePath))
			logging.error('iXC_Rabi::ReadRamanAnalysisSummary::Aborting...')
			quit()

	return SummaryDF

################# End of ReadRabiAnalysisSummary() ##################
#####################################################################

def RabiAnalysisLevel4(AnalysisCtrl, RabiOpts, PlotOpts, RunPars):
	"""Method for Rabi AnalysisLevel = 4:
	Reload and process Rabi analysis level 3 results for selected runs and perform time series analysis.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RabiOpts     (dict) - Copy of key:value pairs controlling Rabi options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	# WorkDir = AnalysisCtrl['WorkDir']
	# Folder  = AnalysisCtrl['Folder']
	# RunList = AnalysisCtrl['RunList']

	# RunNum  = RunList[0]
	# nRuns  	= len(RunList)

	# ## Load first run to extract basic parameters
	# Rab = Rabi(WorkDir, Folder, RunNum, RabiOpts, PlotOpts, False, RunPars.__dict__.items())
	# ## Load summary data from Analysis Level = 3
	# SummaryDF = ReadRabiAnalysisSummary(Rab.iaxList, RunList, PlotOpts['PlotFolderPath'], Rab.AxisFileLabels)

	# for iax in Rab.iaxList:

	# 	nData  = SummaryDF[iax]['RunTime'].shape[0]
	# 	tStart = SummaryDF[iax]['RunTime'].iloc[ 0]
	# 	tStop  = SummaryDF[iax]['RunTime'].iloc[-1]
	# 	tStep  = (tStop - tStart)/(nData-1)

	# 	t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
	# 	tRange = np.array([0., tStop - tStart, tStep])

	# 	fmFm1  = SummaryDF[iax]['p01_center'].to_numpy()*1.E3 ## [Hz]
	# 	fmF0   = SummaryDF[iax]['p02_center'].to_numpy()*1.E3 ## [Hz]
	# 	fmFp1  = SummaryDF[iax]['p03_center'].to_numpy()*1.E3 ## [Hz]
	# 	Deltaf = 0.5*(fmFp1 - fmFm1)
	# 	Sigmaf = 0.5*(fmFp1 + fmFm1)

	# 	if RabiOpts['ConvertToBField']:
	# 		BmFm1   = Rab.Phys.BBreitRabi(1., -1., 2., -1., fmFm1)
	# 		BmFp1   = Rab.Phys.BBreitRabi(1., +1., 2., +1., fmFp1)
	# 		DeltaB  = 0.5*(BmFp1 + BmFm1)

	# 		yData   = [[BmFm1, BmFp1, DeltaB]]
	# 		colors  = [['red', 'blue', 'black']]
	# 		xLabels = ['Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))]
	# 		yLabels = [r'$B_{|m_F\rangle}$, $\Delta B$  (G)', r'$\sqrt{\rm PSD}$  (G$/\sqrt{\rm Hz}$)', r'Allan Deviation  (G)']
	# 		lLabels = [[r'$B_{|+1\rangle}$', r'$B_{|-1\rangle}$', r'$\Delta B$']]
	# 		xTicks  = [True]
	# 	else:
	# 		yData   = [[fmF0, Sigmaf], [abs(fmFm1), fmFp1, Deltaf]]
	# 		colors  = [['darkgreen', 'gray'], ['red', 'blue', 'black']]
	# 		xLabels = ['None', 'Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))]
	# 		yLabels = [r'$\delta_{|m_F \rangle}$, $\Sigma \delta$  (Hz)', r'$|\delta_{|m_F\rangle}|$, $\Delta \delta$  (Hz)', r'$\sqrt{\rm PSD}$  (Hz$/\sqrt{\rm Hz}$)', r'Allan Deviation  (Hz)']
	# 		lLabels = [[r'$\delta_{|0\rangle}$', r'$\Sigma \delta$'], [r'$\delta_{|+1\rangle}$', r'$\delta_{|-1\rangle}$', r'$\Delta \delta$']]
	# 		xTicks  = [False, True]

	# 	Options = {
	# 		'SavePlot'       : PlotOpts['SavePlot'],
	# 		'PlotFolderPath' : PlotOpts['PlotFolderPath'],
	# 		'PlotFileName'   : 'Rabi-Runs{:02d}-{:02d}-TimeSeriesAnalysis'.format(RunList[0],RunList[-1]) + PlotOpts['PlotExtension'],
	# 		'ColumnDim'      : (6, 8),
	# 		'Colors'         : colors,
	# 		'xLabels'        : xLabels,
	# 		'yLabels'        : yLabels,
	# 		'LegLabels'      : lLabels,
	# 		'ShowLegend'     : [True, True],
	# 		'FixLegLocation' : False,
	# 		'LegendFontSize' : 13,
	# 		'ShowXTickLabels': xTicks,
	# 		'SetPlotXLimits' : False,
	# 		'PlotXLimits'    : [tRange[0], tRange[1]],
	# 		'PSD_Plot'       : RabiOpts['PSD_Plot'],
	# 		'PSD_Method'     : RabiOpts['PSD_Method'],
	# 		'PSD_Scale'      : 1.,
	# 		'PSD_MaxSubSets' : 3,
	# 		'ADev_Plot'      : RabiOpts['ADev_Plot'],
	# 		'ADev_Scale'     : 1.,
	# 		'ADev_MaxSubSets': 3,
	# 		'ADev_ntauStep'	 : 1,
	# 		'ADev_ShowErrors': RabiOpts['ADev_ComputeErrors']
	# 		}

	# 	iXUtils.AnalyzeTimeSeries(tRange, yData, Options)

################### End of RabiAnalysisLevel4() #####################
#####################################################################

def RabiAnalysisLevel5(AnalysisCtrl, RabiOpts, PlotOpts, RunPars):
	"""Method for Rabi AnalysisLevel = 5:
	Reload Rabi analysis level 3 results for selected runs and correlate with monitor data.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RabiOpts   (dict) - Copy of key:value pairs controlling Raman options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	# WorkDir = AnalysisCtrl['WorkDir']
	# Folder  = AnalysisCtrl['Folder']
	# RunList = AnalysisCtrl['RunList']

	# RunNum  = RunList[0]
	# nRuns  	= len(RunList)

	# ## Load first run to extract basic parameters
	# Rab = Rabi(WorkDir, Folder, RunNum, RabiOpts, PlotOpts, False, RunPars.__dict__.items())
	# ## Load monitor data
	# Mon = iXC_Monitor.Monitor(WorkDir, Folder, RunList, PlotOpts, False, Rab.__dict__.items())
	# Mon.ProcessMonitorData()
	# ## Load summary data from Analysis Level = 3
	# SummaryDF = ReadRabiAnalysisSummary(Rab.iaxList, RunList, PlotOpts['PlotFolderPath'], Rab.AxisFileLabels)

	# for iax in Rab.iaxList:

	# 	nData  = SummaryDF[iax]['RunTime'].shape[0]
	# 	tStart = SummaryDF[iax]['RunTime'].iloc[ 0]
	# 	tStop  = SummaryDF[iax]['RunTime'].iloc[-1]
	# 	tStep  = (tStop - tStart)/(nData-1)

	# 	t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
	# 	xLabel = 'Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))

	# 	tRange = np.array([0., tStop - tStart, tStep])
	# 	tData  = np.linspace(tStart, tStop, num=nData, endpoint=True)

	# 	parName = RabiOpts['CorrelParameter']
	# 	yData	= SummaryDF[iax][parName].to_numpy()
	# 	yLabels = parName

	# 	Mon.PlotMonitorCorrelations(iax, 0, yData, yLabels, MonDFType='Mean', iStart=0, iSkip=1)

################### End of RabiAnalysisLevel5() #####################
#####################################################################
