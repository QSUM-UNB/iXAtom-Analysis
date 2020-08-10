#####################################################################
## Filename:	iXAtom_Class_Ramsey.py
## Author:		B. Barrett
## Description: Ramsey class definition for iXAtom analysis package
## Version:		3.2.4
## Last Mod:	29/07/2020
##===================================================================
## Change Log:
## 29/11/2019 - Ramsey class defined based on Raman class.
##			  - Basic bug testing of AnalysisLevels = 0-3.
## 			  - Separated global plot options into its own dictionary
##				'PlotOpts' to facilite easy sharing with other classes.
## 30/11/2019 - Minor modifications and bug fixes
## 04/01/2020 - Completed overhaul of Ramsey class to use lmfit module
##				and built-in models. Now this analysis code supports
##				fitting to an arbitrary number of peaks using peak-like
##				models such as Gaussian, Lorentzian, Sinc2, SincB, 
##				Voigt, Pseudo-Voigt, and Moffat (i.e. Lorentzian-B).
## 03/02/2020 - Implemented method RamseyAnalysisLevel5 for correlating
##				analysis level = 3 results with monitor data.
## 30/05/2020 - Implemented 'ConvertToBField' option in RamseyAnalysisLevel3
## 29/07/2020 - Added parameter constraint functionality to Ramsey fits by passing
##				all initial parameter properties ('min', 'max', 'vary', 'expr')
##				to the lmfit.model.set_param_hint() method.
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

import iXAtom_Utilities 		  as iXUtils
import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Class_Detector	  as iXC_Detect
import iXAtom_Class_Monitor		  as iXC_Monitor

class Ramsey(iXC_RunPars.RunParameters):
	#################################################################
	## Class for storing and processing Ramsey data
	## Inherits all attributes and methods from class: RunParameters
	#################################################################

	def __init__(self, WorkDir, Folder, RunNum, RamseyOpts, PlotOpts, LoadRunParsFlag=True, RunPars=[]):
		"""Initialize Ramsey variables.
		ARGUMENTS:
		\t WorkDir    (str)  - Path to the top-level directory where dataset is located
		\t Folder     (str)  - Name of folder within WorkDir where dataset is located
		\t RunNum     (int)  - Run number of requested dataset
		\t RamseyOpts (dict) - Key:value pairs controlling Ramsey options
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

		self.RamseyOptions = copy.deepcopy(RamseyOpts)
		self.PlotOptions   = copy.deepcopy(PlotOpts)
		self.idCoeffs      = np.array([0.,0.,0.])
		self.SetDetectCoeffs(self.RamseyOptions['DetectCoeffs'])

		self.RawData       = False
		self.RawDataFound  = False
		self.PostDataFound = False
		self.RawDataFiles  = ['Ramsey-Run{:02d}-AvgRatios.txt'.format(self.Run)]
		self.PlotPath  	   = os.path.join(self.PlotOptions['PlotFolderPath'], 'Ramsey-Run{:02d}-RawData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

		self.RawDataDF     = [pd.DataFrame([]) for iax in range(3)]
		self.PostDataDF    = [pd.DataFrame([]) for iax in range(3)]
		self.Outliers      = [[] for iax in range(3)]
		self.RawFitResult  = [{} for iax in range(3)]
		self.PostFitResult = [{} for iax in range(3)]
		self.RawFitDict    = [{} for iax in range(3)]
		self.PostFitDict   = [{} for iax in range(3)]

		# self.Fit_method    = 'lm' ## 'lm': Levenberg-Marquardt, 'trf': Trust Region Reflective, 'dogbox': dogleg algorithm with rectangular trust regions
		self.Fit_ftol      = 4.E-6
		self.Fit_xtol      = 4.E-6
		self.Fit_maxfev    = 20000

		self.nFitPars  = [len(self.RamseyOptions['FitParameters'][iax].valuesdict().keys()) for iax in range(3)]
		self.nFitPeaks = [(self.nFitPars[iax] - 1)//3 for iax in range(3)]
		for iax in range(3):
			if self.nFitPars[iax] != 3*self.nFitPeaks[iax] + 1:
				logging.warning('iXC_Ramsey::__Init__::Number of fit pars ({}) for iax = {} is inconsistent with number of peaks ({}) in RamseyOptions...'.format(self.nFitPars[iax], iax, self.nFitPeaks[iax]))

	################## End of Ramsey.__init__() #####################
	#################################################################

	def LoadRawRamseyData(self):
		"""Load raw Ramsey data from data file into a Pandas dataframe."""

		logging.info('iXC_Ramsey::Loading   raw Ramsey data for {}...'.format(self.RunString))

		ik = 0
		dataPath = os.path.join(self.RawFolderPath, self.RawDataFiles[ik])
		if os.path.exists(dataPath):
			self.RawDataFound = True
			df = pd.read_csv(dataPath, sep='\t')
			for iax in self.iaxList:
				# Select rows of the imported data frame corresponding to each Raman axis
				self.RawDataDF[iax] = df[df['RamseyAxis'] == iax]

			self.GetRunTiming(self.RamseyOptions['PrintRunTiming'], RawDF=True)
		else:
			self.RawDataFound = False
			logging.warning('iXC_Ramsey::LoadRawRamseyData::Raw Ramsey data not found!')

	############## End of Ramsey.LoadRawRamseyData() ################
	#################################################################

	def ParseRamseyData(self, iax):
		"""Parse Ramsey data from pandas dataframe.
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

		xData = (df['CurrentFreq'].to_numpy() - self.omegaHF/(2*np.pi))*1.E-3 ## (kHz)

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

	############### End of Ramsey.ParseRamseyData() #################
	#################################################################

	def ConstructFitModel(self, iax):
		"""Construct fit model from lmfit Model class"""

		initPars = self.RamseyOptions['FitParameters'][iax]

		model = lm.models.ExpressionModel('yOffset + 0*x')
		model.set_param_hint('yOffset', value=initPars['yOffset'].value, vary=initPars['yOffset'].vary, min=initPars['yOffset'].min, max=initPars['yOffset'].max)

		for iPeak in range(1,self.nFitPeaks[iax]+1):
			prefix = 'p{:02d}_'.format(iPeak)

			amp    = prefix+'amplitude'
			cen    = prefix+'center'
			sig    = prefix+'sigma'
			height = prefix+'height'
			fwhm   = prefix+'fwhm'
			gamma  = prefix+'gamma'
			alpha  = prefix+'fraction'
			beta   = prefix+'beta'

			if initPars[height].user_data['Model'] == 'Gauss':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/2.35482, min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value*2.50663*initPars[sig].value, min=0.)
				model += lm.models.GaussianModel(prefix=prefix)

			elif initPars[height].user_data['Model'] == 'Lorentz':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/2., min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value*3.14159*initPars[sig].value, min=0.)
				model += lm.models.LorentzianModel(prefix=prefix)

			elif initPars[height].user_data['Model'] == 'Moffat':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/2., min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value, min=0.)
				model += lm.models.MoffatModel(prefix=prefix)
				if beta in initPars.keys():
					model.set_param_hint(beta, value=initPars[beta].value, min=0., max=5.)
				else:
					model.set_param_hint(beta, value=1., min=0., max=5.)

			elif initPars[height].user_data['Model'] == 'Sinc2':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/2.78332, min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value*3.14159*initPars[sig].value, min=0.)
				expression = amp+'/(pi*'+sig+')*(sin((x-'+cen+')/'+sig+')*'+sig+'/(x-'+cen+'))**2'
				model += lm.models.ExpressionModel(expression)
				model.set_param_hint(height, value=initPars[height].value, min=0., expr='0.3183099*'+amp+'/max(2.220446e-16,'+sig+')')
				model.set_param_hint(fwhm, value=initPars[fwhm].value, min=0., expr='2.7833200*'+sig)

			elif initPars[height].user_data['Model'] == 'SincB':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/2.78332, min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value*3.14159*initPars[sig].value, min=0.)
				expression = amp+'/(pi*'+sig+')*abs(sin((x-'+cen+')/'+sig+')*'+sig+'/(x-'+cen+'))**'+beta
				model += lm.models.ExpressionModel(expression)
				model.set_param_hint(height, value=initPars[height].value, min=0., expr='0.3183099*'+amp+'/max(2.220446e-16,'+sig+')')
				model.set_param_hint(fwhm, value=initPars[fwhm].value, min=0., expr='15.4919/sqrt(2.+15.*'+beta+')*'+sig)
				if beta in initPars.keys():
					model.set_param_hint(beta, value=initPars[beta].value, min=0., max=10.)
				else:
					model.set_param_hint(beta, value=2., min=0., max=10.)

			elif initPars[height].user_data['Model'] == 'Voigt':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/3.6013, min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value*4.13273*initPars[sig].value, min=0.)
				model += lm.models.VoigtModel(prefix=prefix)
				model.set_param_hint(gamma, value=initPars[sig].value, min=0., vary=True, expr='')
				if gamma in initPars.keys():
					model.set_param_hint(gamma, value=initPars[gamma].value, min=0., max=10.)
				else:
					model.set_param_hint(gamma, value=2., min=0., max=10.)

			elif initPars[height].user_data['Model'] == 'PseudoVoigt':
				initPars[sig] = lm.Parameter(name=sig, value=initPars[fwhm].value/2., min=0.)
				initPars[amp] = lm.Parameter(name=amp, value=initPars[height].value*2.50663*initPars[sig].value, min=0.)
				model += lm.models.PseudoVoigtModel(prefix=prefix)
				if alpha in initPars.keys():
					model.set_param_hint(alpha, value=initPars[alpha].value, min=0., max=10.)
				else:
					model.set_param_hint(alpha, value=2., min=0., max=10.)

			model.set_param_hint(cen, value=initPars[cen].value, vary=initPars[cen].vary, min=initPars[cen].min, max=initPars[cen].max, expr=initPars[cen].expr)
			model.set_param_hint(sig, value=initPars[sig].value, vary=initPars[sig].vary, min=initPars[sig].min, max=initPars[sig].max, expr=initPars[sig].expr)
			model.set_param_hint(amp, value=initPars[amp].value, vary=initPars[amp].vary, min=initPars[amp].min, max=initPars[amp].max, expr=initPars[amp].expr)

		return model

	############### End of Ramsey.ConstructFitModel() ###############
	#################################################################

	def FitRamseyData(self, xData, yData, yErr):
		"""Fit Ramsey data using non-linear least squares module.
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
		message = 'iXC_Ramsey::{} (nfev = {}, redchi = {:5.3E})'.format(result.message, result.nfev, result.redchi)

		if result.success:
			logging.info(message)
		else:
			logging.warning(message)

		return [result, yRes, sRes, dsRes]

	################ End of Ramsey.FitRamseyData() ##################
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
		hmax  = -1.
		dhmax = 0.
		ref_prefix  = 'p00_'
		computeTemp = [False for ip in range(self.nFitPeaks[iax])]

		## Initialize user_data keys
		null_data = copy.deepcopy(self.RamseyOptions['FitParameters'][iax]['p01_height'].user_data)
		for key in null_data.keys():
			null_data[key] = 'None'

		## Add user_data from initial fit parameters indicating peak type, and initial value
		for par in Result.params.values():
			par.init_value = Result.init_values[par.name]
			par.user_data  = null_data
			par.expr       = '' ## Clear expressions since they cause an error in Pars.add(par) below

			for ip in range(self.nFitPeaks[iax]):
				prefix = 'p{:02d}_'.format(ip+1)

				## Add user_data to fit parameter objects
				if len(par.name.split(prefix)) > 1:
					user_data = self.RamseyOptions['FitParameters'][iax][prefix+'height'].user_data
					par.user_data = user_data

				## Find max peak height for SNR calculation
				if par.name == prefix+'height' and par.value > hmax:
					hmax = par.value
					try:
						dhmax = par.stderr
					except:
						dhmax = 0.

				if par.name == prefix+'fwhm':
					## Store peak indices that are appropriate for temperature calculation
					if (par.user_data['Peak'] == 'kU' or par.user_data['Peak'] == 'kD') and par.user_data['State'] == 0:
						computeTemp[ip] = True

					## Store prefix for reference peak
					if par.user_data['Peak'] == 'kCo' and par.user_data['State'] == 0:
						ref_prefix = prefix

			Pars.add(par)

		## Add auxiliary data
		resdev = lm.Parameter('ResDev', value=AuxData['ResDev'])
		resdev.init_value = 0.
		resdev.stderr = AuxData['ResErr']
		resdev.user_data = null_data

		chisqr = lm.Parameter('ChiSqr', value=AuxData['ChiSqr'])
		chisqr.init_value = 0.
		chisqr.stderr = 2.*(resdev.stderr/resdev.value)*chisqr.value
		chisqr.user_data = null_data

		redchi = lm.Parameter('RedChiSqr', value=AuxData['RedChiSqr'])
		redchi.init_value = 0.
		redchi.stderr = 2.*(resdev.stderr/resdev.value)*redchi.value
		redchi.user_data = null_data

		SNR = lm.Parameter('SNR', value=hmax/resdev)
		SNR.init_value = 0.
		try:
			SNR.stderr = np.sqrt((resdev.stderr/resdev.value)**2 + (dhmax/hmax)**2)*SNR.value
		except:
			SNR.stderr = (resdev.stderr/resdev.value)*SNR.value
		SNR.user_data = null_data

		Pars.add_many(resdev, chisqr, redchi, SNR)

		## Estimate temperature when appropriate
		for ip in range(self.nFitPeaks[iax]):
			if computeTemp[ip]:
				prefix = 'p{:02d}_'.format(ip+1)
				par    = Pars[prefix+'fwhm']
				refpar = Pars[ref_prefix+'fwhm']
				fwhm   = np.sqrt(abs(par.value**2 - refpar.value**2))
				try:
					dfwhm = np.sqrt((par.stderr/par.value)**2 + (refpar.stderr/refpar.value)**2)*fwhm
				except:
					dfwhm = 0.
				name   = 'Temp_'+par.user_data['Peak']
				temp   = lm.Parameter(name, value=self.TempScale*fwhm**2)
				temp.init_value = 0.
				temp.stderr     = 2*(dfwhm/fwhm)*temp.value
				temp.user_data  = par.user_data
				Pars.add(temp)

		return Pars

	################## End of Ramsey.AddAuxPars() ###################
	#################################################################

	def UpdateFitDicts(self, iax, Pars):
		"""Update Ramsey fit dictionaries."""

		pName  = [val.name for val in Pars.values()]
		pBest  = [val.value for val in Pars.values()]

		try:
			pError = [val.stderr for val in Pars.values()]
			pInit  = [val.init_value for val in Pars.values()]
			pModel = [val.user_data['Model'] for val in Pars.values()]
			pPeak  = [val.user_data['Peak'] for val in Pars.values()]
			pState = [val.user_data['State'] for val in Pars.values()]
		except:
			pError = [0. for val in Pars.values()]
			pInit  = copy.copy(pBest)
			pModel = ['None' for val in Pars.values()]
			pPeak  = ['None' for val in Pars.values()]
			pState = ['None' for val in Pars.values()]

		if self.RawData:
			self.RawFitDict[iax]['Init']   = {key:val for key, val in zip(pName, pInit)}
			self.RawFitDict[iax]['Best']   = {key:val for key, val in zip(pName, pBest)}
			self.RawFitDict[iax]['Error']  = {key:val for key, val in zip(pName, pError)}
			self.RawFitDict[iax]['Model']  = {key:val for key, val in zip(pName, pModel)}
			self.RawFitDict[iax]['Peak']   = {key:val for key, val in zip(pName, pPeak)}
			self.RawFitDict[iax]['State']  = {key:val for key, val in zip(pName, pState)}
		else:
			self.PostFitDict[iax]['Init']  = {key:val for key, val in zip(pName, pInit)}
			self.PostFitDict[iax]['Best']  = {key:val for key, val in zip(pName, pBest)}
			self.PostFitDict[iax]['Error'] = {key:val for key, val in zip(pName, pError)}
			self.PostFitDict[iax]['Model'] = {key:val for key, val in zip(pName, pModel)}
			self.PostFitDict[iax]['Peak']  = {key:val for key, val in zip(pName, pPeak)}
			self.PostFitDict[iax]['State'] = {key:val for key, val in zip(pName, pState)}

	################## End of Ramsey.UpdateFitDicts() ###################
	#####################################################################

	def AnalyzeRamseyData(self):
		"""Analyze Ramsey data (raw or post-processed)."""

		if self.RawData:
			label = 'raw'
			dfList = self.RawDataDF
		else:
			label = 'post-processed'
			dfList = self.PostDataDF

		logging.info('iXC_Ramsey::Analyzing {} Ramsey data for {}...'.format(label, self.RunString))

		if self.RamseyOptions['FitData']:
			## Load and fit Ramsey data, and store fit results in data frames
			for iax in self.iaxList:
				## Construct fit model and parameters
				self.FitModel = self.ConstructFitModel(iax)
				self.FitPars  = self.FitModel.make_params()

				[xData, yData, yErr] = self.ParseRamseyData(iax)
				[result, yRes, sRes, dsRes] = self.FitRamseyData(xData, yData, yErr)

				if self.RamseyOptions['RemoveOutliers']:
					iPoint = -1
					for dy in yRes:
						iPoint += 1
						if abs(dy) > self.RamseyOptions['OutlierThreshold']*sRes:
							self.Outliers[iax].append(iPoint)
							logging.info('iXC_Ramsey::Removing outlier at (iax,iPoint) = ({},{}), nOutlier = {}...'.format(iax,iPoint,len(self.Outliers[iax])))
							logging.info('iXC_Ramsey::  Outlier: (x,y,yRes) = ({:5.3e}, {:5.3e}, {:4.2f}*sigma)'.format(xData[iPoint],yData[iPoint],dy/sRes))
					if len(self.Outliers[iax]) > 0:
						xData  = np.delete(xData, self.Outliers[iax])
						yData  = np.delete(yData, self.Outliers[iax])
						yErr   = np.delete(yErr,  self.Outliers[iax])
						[result, yRes, sRes, dsRes] = self.FitRamseyData(iax, xData, yData, yErr)

				if self.RawData:
					self.RawFitResult[iax]  = result
				else:
					self.PostFitResult[iax] = result

				auxData = {'ResDev': sRes, 'ResErr': dsRes, 'ChiSqr': result.chisqr, 'RedChiSqr': result.redchi}
				pars    = self.AddAuxPars(iax, result, auxData)
				self.UpdateFitDicts(iax, pars)
		else:
			for iax in self.iaxList:
				pars = self.RamseyOptions['FitParameters'][iax]
				self.UpdateFitDicts(iax, pars)

	################ End of Ramsey.AnalyzeRamseyData() ##################
	#####################################################################

	def PlotRamseyAxisData(self, RamseyAxs, iRun, iax, CustomPlotOpts, Labels):
		"""Plot Ramsey data (raw or post-processed) and associated fit for given axis.
		ARGUMENTS:
		\t RamseyAxs      (list) - Ramsey figure axes corresponding to a given Raman axis
		\t iax             (int) - Index corresponding to a given axis
		\t iRun            (int) - Index corresponding to run number in RunList
		\t CustomPlotOpts (dict) - Key:value pairs controlling custom plot options
		\t Labels         (list) - Plot labels
		"""

		[xLabel, yLabel] = Labels
		[xData, yData, yErr] = self.ParseRamseyData(iax)

		if self.RamseyOptions['RemoveOutliers'] and len(self.Outliers[iax]) > 0:
			xData = np.delete(xData, self.Outliers[iax])
			yData = np.delete(yData, self.Outliers[iax])
			yErr  = np.delete(yErr,  self.Outliers[iax])

		if np.dot(yErr, yErr) == 0.:
			yErr = []

		## Main Ramsey plot
		if self.PlotOptions['OverlayRunPlots']:
			nColors = len(self.RamseyOptions['RunPlotColors'])
			CustomPlotOpts['Color'] = self.RamseyOptions['RunPlotColors'][iRun%(nColors)]
			if self.RamseyOptions['RunPlotVariable'] == 'Run':
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', Run {:02d}'.format(self.Run)
			elif self.RamseyOptions['RunPlotVariable'] == 'RamanTOF':
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', {:5.2f} ms'.format(getattr(self, 'RamanTOF')*1.E+3)
			elif self.RamseyOptions['RunPlotVariable'] == 'RunTime':
				runTimeStamp = dt.datetime.fromtimestamp(self.RunTime, tz=pytz.timezone('Europe/Paris')).strftime('%H:%M:%S')
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', {}'.format(runTimeStamp)
			else:
				CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax] + ', {:5.2e}'.format(getattr(self, self.RamseyOptions['RunPlotVariable']))
		else:
			CustomPlotOpts['Color']    = self.DefaultPlotColors[iax][0]
			CustomPlotOpts['LegLabel'] = self.AxisLegLabels[iax]

		CustomPlotOpts['Linestyle'] = 'None'
		CustomPlotOpts['Marker']    = '.'
		iXUtils.CustomPlot(RamseyAxs[1], CustomPlotOpts, xData, yData, yErr)

		if self.PlotOptions['ShowFit'] and self.RamseyOptions['FitData']:
			## Plot spectra fit
			xData = np.sort(xData)
			if self.RamseyOptions['SetFitPlotXLimits']:
				[xMin, xMax] = self.RamseyOptions['FitPlotXLimits']
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
				CustomPlotOpts['Color'] = self.RamseyOptions['RunPlotColors'][iRun%(nColors)]
			else:
				CustomPlotOpts['Color'] = self.DefaultPlotColors[iax][0]
			CustomPlotOpts['Linestyle'] = '-'
			CustomPlotOpts['Marker']    = None
			CustomPlotOpts['LegLabel']  = None

			yFit = fitResult.eval(x=xFit)
			iXUtils.CustomPlot(RamseyAxs[1], CustomPlotOpts, xFit, yFit)

			# dyFit = fitResult.eval_uncertainty(x=xFit)
			# RamseyAxs[1].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

			CustomPlotOpts['Linestyle'] = '-'
			CustomPlotOpts['Marker']    = '.'

			# yRes = fitResult.residual ## Already has outliers deleted
			## Note that fitResult.residual is divided by the errors
			yRes = yData - fitResult.best_fit
			iXUtils.CustomPlot(RamseyAxs[0], CustomPlotOpts, xData, yRes)

		if self.PlotOptions['ShowPlotLabels'][0]:
			RamseyAxs[1].set_xlabel(xLabel)
		if self.PlotOptions['ShowPlotLabels'][1]:
			RamseyAxs[1].set_ylabel(yLabel)
			RamseyAxs[0].set_ylabel('Residue')
		if self.PlotOptions['ShowPlotTitle']:
			RamseyAxs[0].set_title(self.RunString + ', TOF = {:4.2f} ms'.format(self.RamanTOF*1.0E+3))
		if self.PlotOptions['ShowPlotLegend']:
			if self.PlotOptions['FixLegLocation']:
				## Fix legend location outside upper right of plot
				RamseyAxs[1].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
			else:
				## Let matlibplot find best legend location
				RamseyAxs[1].legend(loc='best')

	############## End of Ramsey.PlotRamseyAxisData() ###############
	#################################################################

	def PlotRamseyData(self, RamseyAxs, iRun):
		"""Plot Ramsey data (raw or post-processed) and associated fit (if requested).
		ARGUMENTS:
		\t RamseyAxs (axs list) - Current Ramsey figure axes
		\t iRun           (int) - Index corresponding to run number in RunList   
		"""

		if self.RawData:
			label = 'raw'
			dfList = self.RawDataDF
		else:
			label = 'post-processed'
			dfList = self.PostDataDF

		logging.info('iXC_Ramsey::Plotting {} Ramsey data for {}...'.format(label, self.RunString))

		xLabel = r'$\delta$  (kHz)'
		yLabel = r'$N_2/N_{\rm total}$'

		customPlotOpts = {'Color': 'red', 'Linestyle': 'None', 'Marker': '.', 'Title': 'None', 
			'xLabel': 'None', 'yLabel': 'None', 'Legend': False, 'LegLabel': None}

		iRow = -1
		for iax in self.iaxList:
			iRow += 1
			if self.PlotOptions['OverlayRunPlots']:
				## Set options for sharing x axes
				showXLabel = True if iax == self.iaxList[-1] else False
				# plt.setp(RamseyAxs[iRow].get_xticklabels(), visible=showXLabel)
				self.PlotOptions['ShowPlotLabels'][0] = showXLabel

				self.PlotRamseyAxisData(RamseyAxs[iRow], iRun, iax, customPlotOpts, [xLabel, yLabel])
			else:
				self.PlotRamseyAxisData(RamseyAxs[0], iRun, iax, customPlotOpts, [xLabel, yLabel])

		if self.PlotOptions['SavePlot']:
			plt.savefig(self.PlotPath, dpi=150)
			logging.info('iXC_Ramsey::Ramsey plot saved to:')
			logging.info('iXC_Ramsey::  {}'.format(self.PlotPath))
		elif self.PlotOptions['ShowPlot']:
			plt.show()

	################ End of Ramsey.PlotRamseyData() #################
	#################################################################

	def PostProcessRamseyData(self, DetectOpts):
		"""Post-process Ramsey detection data and write to file (if requested).
		ARGUMENTS:
		\t DetectOpts (dict) - Key:value pairs controlling detection options
		"""

		logging.info('iXC_Ramsey::Post-processing Ramsey data for {}...'.format(self.RunString))

		# Declare detector object
		det = iXC_Detect.Detector(self.WorkDir, self.Folder, self.Run, DetectOpts, self.PlotOptions, False, self.__dict__.items())

		meanHeaders = ['RatioMean_L', 'RatioMean_M', 'RatioMean_U']
		sdevHeaders = ['RatioSDev_L', 'RatioSDev_M', 'RatioSDev_U']

		if det.nIterations > 0:
			self.LoadRawRamseyData()

			## Store copy of raw Ramsey data in post Ramsey dataframe
			self.PostDataDF = [self.RawDataDF[iax].copy() for iax in range(3)]

			## Reshape PostDataDF
			for iax in self.iaxList:
				self.PostDataDF[iax].drop(columns=['RamseyAxis'], inplace=True)
				self.PostDataDF[iax].rename(columns={'RatioMean':   meanHeaders[0]}, inplace=True)
				self.PostDataDF[iax].rename(columns={'RatioStdDev': sdevHeaders[0]}, inplace=True)
				for id in range(3):
					self.PostDataDF[iax].loc[:,meanHeaders[id]] = 0.
					self.PostDataDF[iax].loc[:,sdevHeaders[id]] = 0.
	
			avgNum = 1
			for iterNum in range(1,det.nIterations+1):

				if iterNum == 1:
					initPars = det.InitializeDetectPars()

				det.LoadDetectTrace(avgNum, iterNum)
				det.AnalyzeDetectTrace(avgNum, iterNum, initPars)

				if 'Ramsey Axis' in det.DetectTags.keys():
					iax = int(det.DetectTags['Ramsey Axis'])				
				else:
					iax = int(list(det.DetectTags.values())[1])

				if iax not in range(3):
					logging.warning('iXC_Ramsey::PostProcessRamseyData::Raman axis index out of range. Setting iax = 2 (Z)...')
					iax = 2

				mask = (self.PostDataDF[iax]['#Iteration'] == iterNum)
				for id in det.idList:
					for ic in range(4):
						initPars[id][ic] = det.DetectResult[id][det.DetectKeys[ic]]['BestPars']
					self.PostDataDF[iax].loc[mask, meanHeaders[id]] = det.DetectResult[id]['Ratio']['Best']
					self.PostDataDF[iax].loc[mask, sdevHeaders[id]] = det.DetectResult[id]['Ratio']['Error']

			if self.RamseyOptions['SaveAnalysisLevel1']:
				showHeaders = True
				showIndices = False
				floatFormat = '%11.9E' ## Output 10-digits of precision (needed for the frequency)

				for iax in self.iaxList:
					iXUtils.WriteDataFrameToFile(self.PostDataDF[iax], self.PostFolderPath, self.PostFilePaths[iax],
						showHeaders, showIndices, floatFormat)

				logging.info('iXC_Ramsey::Post-processed Ramsey data saved to:')
				logging.info('iXC_Ramsey::  {}'.format(self.PostFolderPath))
		else:
			logging.error('iXC_Ramsey::PostProcessRamseyData::Aborting detection analysis on {}...'.format(self.RunString))
			quit()

	############# End of Ramsey.PostProcessRamseyData() #############
	#################################################################

	def LoadPostRamseyData(self):
		"""Load post-processed Ramsey data into a list of data frames."""

		logging.info('iXC_Ramsey::Loading   post-processed Ramsey data for {}...'.format(self.RunString))

		nFilesFound = 0
		for iax in self.iaxList:
			if os.path.exists(self.PostFilePaths[iax]):
				nFilesFound += 1
				self.PostDataDF[iax] = pd.read_csv(self.PostFilePaths[iax], sep='\t')

		if nFilesFound == self.nax:
			self.PostDataFound = True
			self.GetRunTiming(self.RamseyOptions['PrintRunTiming'], RawDF=False)
		else:
			self.PostDataFound = False
			logging.warning('iXC_Ramsey::LoadPostRamseyData::Post-processed Ramsey data not found in: {}'.format(self.PostFolderPath))

	############# End of Ramsey.LoadPostRamseyData() ################
	#################################################################

	def WriteRamseyAnalysisResults(self):
		"""Write Ramsey analysis results to file."""

		if self.RawData:
			label = 'raw'
		else:
			label = 'post-processed'

		logging.info('iXC_Ramsey::Writing {} Ramsey analysis results to:'.format(label))
		logging.info('iXC_Ramsey::  {}'.format(self.PostFolderPath))	

		for iax in self.iaxList:
			if self.RawData:
				fileName  = 'Ramsey-Run{:02}-Results-Raw-'.format(self.Run)  + self.AxisFileLabels[iax] + '.txt'
				dataFrame = pd.DataFrame.from_dict(self.RawFitDict[iax])
			else:
				fileName  = 'Ramsey-Run{:02}-Results-Post-'.format(self.Run) + self.AxisFileLabels[iax] + '.txt'
				dataFrame = pd.DataFrame.from_dict(self.PostFitDict[iax])

			filePath = os.path.join(self.PostFolderPath, fileName)
			iXUtils.WriteDataFrameToFile(dataFrame, self.PostFolderPath, filePath, True, True, '%10.8E')

	########## End of Ramsey.WriteRamseyAnalysisResults() ###########
	#################################################################

	def ReadRamseyAnalysisResults(self):
		"""Read Ramsey analysis results to file."""

		if self.RawData:
			label1 = 'raw'
			label2 = 'Raw'
		else:
			label1 = 'post-processed'
			label2 = 'Post'

		logging.info('iXC_Ramsey::Reading {} Ramsey analysis results from:'.format(label1))
		logging.info('iXC_Ramsey::  {}'.format(self.PostFolderPath))

		for iax in self.iaxList:
			fileName = 'Ramsey-Run{:02}-Results-{}-'.format(self.Run, label2) + self.AxisFileLabels[iax] + '.txt'
			filePath = os.path.join(self.PostFolderPath, fileName)
			if os.path.exists(filePath):
				if self.RawData:
					self.RawFitDict[iax] = pd.read_csv(filePath, sep='\t', header=0, index_col=0).to_dict()
				else:
					self.PostFitDict[iax] = pd.read_csv(filePath, sep='\t', header=0, index_col=0).to_dict()
			else:
				logging.error('iXC_Ramsey::ReadRamseyAnalysisResults::Analysis file not found: {}'.format(filePath))
				logging.error('iXC_Ramsey::ReadRamseyAnalysisResults::Aborting...')
				quit()

	########### End of Ramsey.ReadRamseyAnalysisResults() ###########
	#################################################################

	def LoadAndAnalyzeRamseyData(self, RamseyAxs, iRun, nRuns, PlotOpts, ProcessLevel):
		"""Load and analyze raw and/or post-processed Ramsey data; 
		plot the results and write to file (if requested).
		ARGUMENTS:
		\t RamseyAxs (fig axs) - List of Ramsey figure axes for overlaying plots
		\t iRun	         (int) - Index corresponding to current run number (starts from 0)
		\t nRuns         (int) - Number of runs contained within RunList
		\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
		\t ProcessLevel  (int) - Secondary control for Ramsey analysis type
		"""

		if ProcessLevel == 0 or ProcessLevel == 2:

			self.RawData = True
			self.LoadRawRamseyData()

			if self.RawDataFound:
				self.AnalyzeRamseyData()
				if self.RamseyOptions['SaveAnalysisLevel2'] and self.RamseyOptions['FitData']:
					self.WriteRamseyAnalysisResults()

				if nRuns <= self.PlotOptions['MaxPlotsToDisplay'] and self.RamseyOptions['ShowFitResults']:
					for iax in self.iaxList:
						print('---------------------------------------------------')
						print(self.AxisFileLabels[iax] + ' - Raw Data - Fit Results:')
						print('---------------------------------------------------')
						print(pd.DataFrame.from_dict(self.RawFitDict[iax]))
			else:
				logging.warning('iXC_Ramsey::LoadAndAnalyzeRamseyData::Aborting Ramsey raw data analysis for {}...'.format(self.RunString))

		if ProcessLevel == 1 or ProcessLevel == 2:

			self.RawData = False
			self.LoadPostRamseyData()

			if self.PostDataFound:
				self.AnalyzeRamseyData()
				if self.RamseyOptions['SaveAnalysisLevel2'] and self.RamseyOptions['FitData']:
					self.WriteRamseyAnalysisResults()

				if nRuns <= self.PlotOptions['MaxPlotsToDisplay'] and self.RamseyOptions['ShowFitResults']:
					for iax in self.iaxList:
						if ProcessLevel == 1 or ProcessLevel == 2:
							print('---------------------------------------------------')
							print(self.AxisFileLabels[iax] + ' - Post Data - Fit Results:')
							print('---------------------------------------------------')
							print(pd.DataFrame.from_dict(self.PostFitDict[iax]))
			else:
				logging.warning('iXC_Ramsey::LoadAndAnalyzeRamseyData::Aborting Ramsey post-processed data analysis for {}...'.format(self.RunString))

		if (self.PlotOptions['ShowPlot'] and not self.PlotOptions['SavePlot'] and not self.PlotOptions['OverlayRunPlots']) and nRuns > self.PlotOptions['MaxPlotsToDisplay']:
			self.PlotOptions['PlotData'] = False

		if self.PlotOptions['PlotData'] and nRuns <= self.PlotOptions['MaxPlotsToDisplay']:
			if ProcessLevel < 2 and (self.RawDataFound or self.PostDataFound):
				if ProcessLevel == 0:
					self.RawData = True
					self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
						'Ramsey-Run{:02d}-RawData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])
				else:
					self.RawData = False
					self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
						'Ramsey-Run{:02d}-PostData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

				if self.PlotOptions['OverlayRunPlots']:
					self.PlotOptions['ShowPlotTitle']  = False
					self.PlotOptions['ShowPlotLegend'] = True
					if iRun < nRuns-1:
						self.PlotOptions['ShowPlot'] = False
						self.PlotOptions['SavePlot'] = False
					else:
						self.PlotOptions['ShowPlot'] = PlotOpts['ShowPlot']
						self.PlotOptions['SavePlot'] = PlotOpts['SavePlot']

				self.PlotRamseyData(RamseyAxs[0], iRun)

			if ProcessLevel == 2 and (self.RawDataFound and self.PostDataFound):
				self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
					'Ramsey-Run{:02d}-Raw+PostData-Fit.'.format(self.Run) + self.PlotOptions['PlotExtension'])

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
				self.PlotRamseyData(RamseyAxs[0], iRun)

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
				self.PlotRamseyData(RamseyAxs[1], iRun)

	########### End of Ramsey.LoadAndAnalyzeRamseyData() ############
	#################################################################

#####################################################################
####################### End of Class Ramsey #########################
#####################################################################

def RamseyAnalysisLevel1(AnalysisCtrl, RamseyOpts, DetectOpts, PlotOpts, RunPars):
	"""Method for Ramsey AnalysisLevel = 1:
	Post-process detection data for selected Runs and store results.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamseyOpts   (dict) - Copy of key:value pairs controlling Ramsey options
	\t DetectOpts   (dict) - Copy of key:value pairs controlling Detection options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	for RunNum in RunList:
		if RunNum == RunList[0]:
			Rams = Ramsey(WorkDir, Folder, RunNum, RamseyOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Rams = Ramsey(WorkDir, Folder, RunNum, RamseyOpts, PlotOpts)
		Rams.PostProcessRamseyData(DetectOpts)

################## End of RamseyAnalysisLevel1() ####################
#####################################################################

def RamseyAnalysisLevel2(AnalysisCtrl, RamseyOpts, PlotOpts):
	"""Method for Ramsey AnalysisLevel = 2:
	Load raw/post-processed Ramsey data for selected Runs, analyze and store results.
	Plot the data and fits in individual figures or overlayed in a single figure.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamseyOpts   (dict) - Copy of key:value pairs controlling Ramsey options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	runList = AnalysisCtrl['RunList']
	nRuns   = len(runList)

	runFitPars = copy.deepcopy(RamseyOpts['FitParameters'])
	runVarKey  = RamseyOpts['RunPlotVariable']
	runVarList = np.zeros(nRuns)
	nFitPars   = len(runFitPars[0].valuesdict().keys())
	nFitPeaks  = (nFitPars - 1) // 3
	fCenters   = np.zeros((3,nFitPeaks,nRuns)) ## [iax,iPeak,iRun]

	if not RamseyOpts['FitData'] or nRuns == 1: 
		RamseyOpts['TrackRunFitPars'] = False

	if RamseyOpts['TrackRunFitPars']:
		logging.info('iXC_Raman::Getting list of Run variables "{}"...'.format(runVarKey))
		logging.disable(level=logging.INFO) # Disable logger for info & debug levels

		## Load run variables from parameter files and find sort order
		iRun = -1
		for runNum in runList:
			iRun += 1
			Rams = Ramsey(WorkDir, Folder, runNum, RamseyOpts, PlotOpts)
			if runVarKey not in vars(Rams):
				## Ramsey data must be loaded to extract certain run variables
				Rams.LoadRawRamseyData()

			runVarList[iRun] = getattr(Rams, runVarKey)

		## Sort run variables and run list
		orderList  = np.argsort(runVarList)
		runVarList = runVarList[orderList]
		runList    = [runList[iRun] for iRun in orderList]

		if RamseyOpts['SortVariableOrder'] == 'Descending':
			runVarList = runVarList[::-1]
			runList    = runList[::-1]

		logging.disable(level=logging.NOTSET) # Re-enable logger at configured level

	RamseyAxs = None
	iRun = -1
	for runNum in runList:
		iRun += 1

		Rams = Ramsey(WorkDir, Folder, runNum, RamseyOpts, PlotOpts)
		## Assumes number of Raman axes present is identical for all runs in RunList
		RamseyAxs = Rams.CreatePlotAxes(RamseyAxs, iRun, nRuns, Rams.nax, PlotOpts, AnalysisCtrl['ProcessLevel'])

		Rams.RamseyOptions['FitParameters'] = runFitPars
		Rams.LoadAndAnalyzeRamseyData(RamseyAxs, iRun, nRuns, PlotOpts, AnalysisCtrl['ProcessLevel'])

		if RamseyOpts['TrackRunFitPars'] and iRun < nRuns-1:
			for iax in Rams.iaxList:
				if Rams.RawData:
					valDict = Rams.RawFitDict[iax]['Best']
				else:
					valDict = Rams.PostFitDict[iax]['Best']

				## Update initial fit parameters for next iteration
				runFitPars[iax]['yOffset'].value = valDict['yOffset']

				for iPeak in range(nFitPeaks):
					ip = iPeak + 1
					prefix = 'p{:02d}_'.format(ip)
					height = valDict[prefix+'height']
					fwhm   = valDict[prefix+'fwhm']
					fCenters[iax,iPeak,iRun] = valDict[prefix+'center']

					user_data = runFitPars[iax][prefix+'height'].user_data

					if user_data['Model'] == 'SincB' or user_data['Model'] == 'Moffat':
						runFitPars[iax][prefix+'beta'] = lm.Parameter(prefix+'beta', value=valDict[prefix+'beta'], min=0.)
					elif user_data['Model'] == 'Voigt':
						runFitPars[iax][prefix+'gamma'] = lm.Parameter(prefix+'gamma', value=valDict[prefix+'gamma'], min=0.)
					elif user_data['Model'] == 'PseudoVoigt':
						runFitPars[iax][prefix+'fraction'] = lm.Parameter(prefix+'fraction', value=valDict[prefix+'fraction'], min=0., max=1.)

					if iRun == 0:
						nextfCenter = fCenters[iax,iPeak,iRun]
					elif iRun >= 1:
						if runVarList[iRun] != runVarList[iRun-1]:
							slope = (fCenters[iax,iPeak,iRun] - fCenters[iax,iPeak,iRun-1])/(runVarList[iRun] - runVarList[iRun-1])
						else:
							slope = 0.
						nextfCenter = fCenters[iax,iPeak,iRun] + slope*(runVarList[iRun+1] - runVarList[iRun])

					## Update guess for next peak heights, centers, and fwhms
					runFitPars[iax][prefix+'height'].value = height
					runFitPars[iax][prefix+'fwhm'].value   = fwhm
					runFitPars[iax][prefix+'center'].value = nextfCenter

################## End of RamseyAnalysisLevel2() ####################
#####################################################################

def RamseyAnalysisLevel3(AnalysisCtrl, RamseyOpts, PlotOpts, RunPars):
	"""Method for Ramsey AnalysisLevel = 3:
	Reload and process Ramsey analysis results for selected runs and plot a summary.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamseyOpts   (dict) - Copy of key:value pairs controlling Ramsey options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir   = AnalysisCtrl['WorkDir']
	Folder    = AnalysisCtrl['Folder']
	RunList   = AnalysisCtrl['RunList']
	nRuns  	  = len(RunList)
	nFitPars  = [len(RamseyOpts['FitParameters'][iax].valuesdict().keys()) for iax in range(3)]
	nFitPeaks = [(nFitPars[iax] - 1)//3 for iax in range(3)]

	peak   = [['None' for ip in range(nFitPeaks[iax])] for iax in range(3)]
	state  = [['None' for ip in range(nFitPeaks[iax])] for iax in range(3)]
	keys   = [['None' for ip in range(nFitPeaks[iax])] for iax in range(3)]

	tRun   = np.zeros(nRuns)
	x      = np.zeros(nRuns)
	tof    = np.zeros(nRuns)
	y0     = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	SNR    = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	temp   = np.zeros((3,2,nRuns))		 	 ## [iax,(0,1=BestFit,FitErr),iRun]
	height = np.zeros((3,max(nFitPeaks),2,nRuns)) ## [iax,iPeak,(0,1=BestFit,FitErr),iRun]
	center = np.zeros((3,max(nFitPeaks),2,nRuns)) ## [iax,iPeak,(0,1=BestFit,FitErr),iRun]
	fwhm   = np.zeros((3,max(nFitPeaks),2,nRuns)) ## [iax,iPeak,(0,1=BestFit,FitErr),iRun]

	SummaryDF = [pd.DataFrame([]) for iax in range(3)]

	(nRows, nCols) = (2,3)

	Fig, Axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*4,nRows*3), sharex=True, constrained_layout=True)

	iRun = -1
	for RunNum in RunList:
		iRun += 1

		if RunNum == RunList[0]:
			Rams = Ramsey(WorkDir, Folder, RunNum, RamseyOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Rams = Ramsey(WorkDir, Folder, RunNum, RamseyOpts, PlotOpts)

		## Load raw data to extract Run timing
		Rams.LoadRawRamseyData()
		tRun[iRun] = Rams.RunTime

		if AnalysisCtrl['ProcessLevel'] == 0:
			Rams.RawData = True
		else:
			Rams.RawData = False

		Rams.ReadRamseyAnalysisResults()

		x[iRun]   = getattr(Rams, RamseyOpts['RunPlotVariable'])
		tof[iRun] = getattr(Rams, 'RamanTOF')

		if AnalysisCtrl['ProcessLevel'] == 0:
			fitList = getattr(Rams, 'RawFitDict')
		else:
			fitList = getattr(Rams, 'PostFitDict')

		for iax in Rams.iaxList:
			y0[iax,0,iRun]  = fitList[iax]['Best' ]['yOffset']
			y0[iax,1,iRun]  = fitList[iax]['Error']['yOffset']
			SNR[iax,0,iRun] = fitList[iax]['Best' ]['SNR']
			SNR[iax,1,iRun] = fitList[iax]['Error']['SNR']

			if 'Temp_kU' in fitList[iax]['Best'].keys() and 'Temp_kD' in fitList[iax]['Best'].keys():
				temp[iax,0,iRun] = 0.5*(fitList[iax]['Best']['Temp_kU'] + fitList[iax]['Best']['Temp_kD'])
				temp[iax,1,iRun] = 0.5*np.sqrt(fitList[iax]['Error']['Temp_kU']**2 + fitList[iax]['Error']['Temp_kD']**2)

			for ip in range(nFitPeaks[iax]):
				prefix = 'p{:02d}_'.format(ip+1)

				if iRun == 0:
					state[iax][ip] = int(fitList[iax]['State'][prefix+'height'])
					peak[iax][ip]  = fitList[iax]['Peak'][prefix+'height']
					keys[iax][ip]  = Rams.AxisLegLabels[iax]+'-{}, mF = {}'.format(peak[iax][ip], state[iax][ip])

				height[iax,ip,0,iRun] = fitList[iax]['Best' ][prefix+'height']
				height[iax,ip,1,iRun] = fitList[iax]['Error'][prefix+'height']
				center[iax,ip,0,iRun] = fitList[iax]['Best' ][prefix+'center']
				center[iax,ip,1,iRun] = fitList[iax]['Error'][prefix+'center']
				fwhm[iax,ip,0,iRun]   = fitList[iax]['Best' ][prefix+'fwhm']
				fwhm[iax,ip,1,iRun]   = fitList[iax]['Error'][prefix+'fwhm']

	for iax in Rams.iaxList:
		d = {'Run': RunList, 'RunTime': tRun, RamseyOpts['RunPlotVariable']: x,
			'yOffset': y0[iax,0], 'yOffset_Err': y0[iax,1], 
			'SNR': SNR[iax,0], 'SNR_Err': SNR[iax,1],
			'Temp': temp[iax,0], 'Temp_Err': temp[iax,1]}
		for ip in range(nFitPeaks[iax]):
			prefix = 'p{:02d}_'.format(ip+1)
			d.update({prefix+'height': height[iax,ip,0], prefix+'height_Err': height[iax,ip,1],
				prefix+'center': center[iax,ip,0], prefix+'center_Err': center[iax,ip,1],
				prefix+'fwhm': fwhm[iax,ip,0], prefix+'fwhm_Err': fwhm[iax,ip,1]})
		SummaryDF[iax] = pd.DataFrame(data=d)

	if RamseyOpts['SaveAnalysisLevel3']:
		WriteRamseyAnalysisSummary(Rams.iaxList, RunList, SummaryDF, PlotOpts['PlotFolderPath'], Rams.AxisFileLabels)

	if PlotOpts['PlotData']:
		if RamseyOpts['RunPlotVariable'] == 'RamanTOF':
			## Special operations for RamanTOF
			x *= 1.0E3
			xLabel = 'Raman TOF  (ms)'
		elif RamseyOpts['RunPlotVariable'] == 'DetectTOF':
			## Special operations for DetectTOF
			x *= 1.0E3
			xLabel = 'Detect TOF  (ms)'
		elif RamseyOpts['RunPlotVariable'] == 'RamanPower':
			## Special operations for RamanPower
			xLabel = 'Raman Power  (V)'
		elif RamseyOpts['RunPlotVariable'] == 'RamanpiX' or RamseyOpts['RunPlotVariable'] == 'RamanpiY' or RamseyOpts['RunPlotVariable'] == 'RamanpiZ':
			## Special operations for RamanpiX, RamanpiY, RamanpiZ
			x *= 1.0E6
			xLabel = r'$\tau_{\pi}$  ($\mu$s)'
		elif RamseyOpts['RunPlotVariable'] == 'RunTime':
			t0 = dt.datetime.fromtimestamp(x[0], tz=pytz.timezone('Europe/Paris'))
			x  = (x - x[0])/60.
			xLabel = 'Run Time - {}  (min)'.format(t0.strftime('%H:%M:%S'))
		else:
			xLabel = RamseyOpts['RunPlotVariable']

		customPlotOpts = {'Color': 'black', 'Linestyle': 'None',
			'Marker': '.', 'Title': 'None', 'xLabel': 'None', 'yLabel': 'None', 
			'Legend': False, 'LegLabel': 'None'}

		# legLabels = [r'$|m_F = -1\rangle$', r'$|m_F = 0\rangle$', r'$|m_F = +1\rangle$']
		# colors = ['darkgreen', 'limegreen', 'green', 'blue', 'royalblue', 'lightblue', 'darkred', 'red', 'crimson']

		colorDict = {
			'X-kD, mF = -1': 'lightcoral', 'X-kCo, mF = -1': 'indianred', 'X-kU, mF = -1': 'brown',
			'X-kD, mF = 0': 'darkred', 'X-kCo, mF = 0': 'firebrick', 'X-kU, mF = 0': 'red',
			'X-kD, mF = 1': 'orangered', 'X-kCo, mF = 1': 'darkorange', 'X-kU, mF = 1': 'orange',

			'Y-kD, mF = -1': 'goldenrod', 'Y-kCo, mF = -1': 'olive', 'Y-kU, mF = -1': 'olivedrab',
			'Y-kD, mF = 0': 'darkgreen', 'Y-kCo, mF = 0': 'forestgreen', 'Y-kU, mF = 0': 'green',
			'Y-kD, mF = 1': 'seagreen', 'Y-kCo, mF = 1': 'mediumseagreen', 'Y-kU, mF = 1': 'springgreen',

			'Z-kD, mF = -1': 'lightseagreen', 'Z-kCo, mF = -1': 'teal', 'Z-kU, mF = -1': 'steelblue',
			'Z-kD, mF = 0': 'darkblue', 'Z-kCo, mF = 0': 'royalblue', 'Z-kU, mF = 0': 'blue',
			'Z-kD, mF = 1': 'blueviolet', 'Z-kCo, mF = 1': 'darkviolet', 'Z-kU, mF = 1': 'purple'}

		ipkU  = [-1,-1,-1]
		ipkD  = [-1,-1,-1]
		ipkCo = [-1,-1,-1]

		for iax in Rams.iaxList:
			for ip in range(nFitPeaks[iax]):
				if state[iax][ip] == -1:
					if peak[iax][ip] == 'kU':
						ipkU[0] = ip
					elif peak[iax][ip] == 'kD':
						ipkD[0] = ip
					elif peak[iax][ip] == 'kCo':
						ipkCo[0] = ip
				elif state[iax][ip] == 0:
					if peak[iax][ip] == 'kU':
						ipkU[1] = ip
					elif peak[iax][ip] == 'kD':
						ipkD[1] = ip
					elif peak[iax][ip] == 'kCo':
						ipkCo[1] = ip
				elif state[iax][ip] == 1:
					if peak[iax][ip] == 'kU':
						ipkU[2] = ip
					elif peak[iax][ip] == 'kD':
						ipkD[2] = ip
					elif peak[iax][ip] == 'kCo':
						ipkCo[2] = ip

				customPlotOpts['Color']    = colorDict[keys[iax][ip]]
				customPlotOpts['LegLabel'] = keys[iax][ip]
				customPlotOpts['xLabel']   = 'None'
				customPlotOpts['yLabel']   = 'Height'
				iXUtils.CustomPlot(Axs[0][0], customPlotOpts, x, height[iax,ip,0], height[iax,ip,1])

				customPlotOpts['yLabel'] = 'FWHM  (kHz)'
				iXUtils.CustomPlot(Axs[0][1], customPlotOpts, x, fwhm[iax,ip,0], fwhm[iax,ip,1])

				customPlotOpts['yLabel'] = r'$\delta$  (kHz)'
				iXUtils.CustomPlot(Axs[0][2], customPlotOpts, x, center[iax,ip,0], center[iax,ip,1])

			customPlotOpts['xLabel']   = xLabel
			customPlotOpts['yLabel']   = 'SNR'
			customPlotOpts['Color']    = 'grey' if iax == 0 else ('dimgrey' if iax == 1 else 'black')
			customPlotOpts['LegLabel'] = Rams.AxisLegLabels[iax][0]
			iXUtils.CustomPlot(Axs[1][0], customPlotOpts, x, SNR[iax,0], SNR[iax,1])

			if RamseyOpts['CopropagatingSpectrum'] and ipkCo[0] != -1 and ipkCo[2] != -1:
				if RamseyOpts['ConvertToBField']:
					BmFm1   = Rams.Phys.BBreitRabi(1., -1., 2., -1., center[iax,ipkCo[0],0]*1.E3)
					dBmFm1  = abs(Rams.Phys.BBreitRabi(1., -1., 2., -1., (center[iax,ipkCo[0],0] + center[iax,ipkCo[0],1])*1.E3) - BmFm1)
					BmFp1   = Rams.Phys.BBreitRabi(1., +1., 2., +1., center[iax,ipkCo[2],0]*1.E3)
					dBmFp1  = abs(Rams.Phys.BBreitRabi(1., +1., 2., +1., (center[iax,ipkCo[2],0] + center[iax,ipkCo[2],1])*1.E3) - BmFp1)

					customPlotOpts['yLabel'] = r'$B$  (G)'
					iXUtils.CustomPlot(Axs[1][1], customPlotOpts, x, 0.5*(BmFp1 + BmFm1), 0.5*np.sqrt(dBmFp1**2 + dBmFm1**2))
				else:
					customPlotOpts['yLabel'] = r'$(\delta_{|1\rangle} - \delta_{|-1\rangle})/2$  (kHz)'
					iXUtils.CustomPlot(Axs[1][1], customPlotOpts, x, 0.5*(center[iax,ipkCo[2],0] - center[iax,ipkCo[0],0]), 0.5*np.sqrt(center[iax,ipkCo[2],1]**2 + center[iax,ipkCo[0],1]**2))
			elif not RamseyOpts['CopropagatingSpectrum'] and ipkU[1] != -1 and ipkD[1] != -1:
				customPlotOpts['yLabel'] = r'$(\delta_{\uparrow} - \delta_{\downarrow})/2 - \delta_D$  (kHz)'
				fDoppler = Rams.keff*Rams.gLocal*tof*1.E-3/(2*np.pi)
				iXUtils.CustomPlot(Axs[1][1], customPlotOpts, x, 0.5*(center[iax,ipkU[1],0] - center[iax,ipkD[1],0]) - fDoppler, 0.5*np.sqrt(center[iax,ipkU[1],1]**2 + center[iax,ipkD[1],1]**2))

				customPlotOpts['yLabel'] = r'$(\delta_{\uparrow} + \delta_{\downarrow})/2 - \delta_R$  (kHz)'
				fRecoil = Rams.omegaR*1.E-3/(2*np.pi)
				iXUtils.CustomPlot(Axs[1][2], customPlotOpts, x, 0.5*(center[iax,ipkU[1],0] + center[iax,ipkD[1],0]) - fRecoil, 0.5*np.sqrt(center[iax,ipkU[1],1]**2 + center[iax,ipkD[1],1]**2))

	if PlotOpts['ShowPlotLegend']:
		if PlotOpts['FixLegLocation']:
			## Fix legend location outside upper right of plot
			Axs[0][2].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
			Axs[1][2].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		else:
			## Let matlibplot find best legend location
			Axs[0][2].legend(loc='best')
			Axs[1][2].legend(loc='best')
	if PlotOpts['SavePlot']:
		plt.savefig(Rams.PlotPath, dpi=150)
		logging.info('iXC_Ramsey::Ramsey plot saved to:')
		logging.info('iXC_Ramsey::  {}'.format(Rams.PlotPath))
	elif PlotOpts['ShowPlot']:
		plt.show()

################## End of RamseyAnalysisLevel3() ####################
#####################################################################

def WriteRamseyAnalysisSummary(iaxList, RunList, SummaryDF, Folder, Labels):
	"""Method for writing Ramsey AnalysisLevel = 3 results.
	ARGUMENTS:
	\t iaxList   (list) - List of Raman axis indices
	\t RunList   (list) - List of Runs contained in SummaryDF
	\t SummaryDF (list) - List of dataframes containing analysis summary for each axis and k-direction
	\t Folder    (str)  - Path to folder in which to store analysis summary file
	\t Labels    (list) - Raman axis file labels
	"""

	fileName    = 'Ramsey-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))
	floatFormat = '%11.9E'

	for iax in iaxList:
		filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][0]+'.txt')
		iXUtils.WriteDataFrameToFile(SummaryDF[iax], Folder, filePath, True, False, floatFormat)

################ End of WriteRamseyAnalysisSummary() ################
#####################################################################

def ReadRamseyAnalysisSummary(iaxList, RunList, Folder, Labels):
	"""Method for reading Raman AnalysisLevel = 3 results.
	ARGUMENTS:
	\t iaxList  (list) - List of Raman axis indices
	\t RunList  (list) - List of Runs contained in summary file
	\t Folder   (str)  - Path to folder containing analysis summary file
	\t Labels   (list) - Raman axis file labels
	"""

	SummaryDF = [pd.DataFrame([]) for iax in range(3)]
	fileName  = 'Ramsey-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))

	for iax in iaxList:
		filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][0]+'.txt')
		if os.path.exists(filePath):
			SummaryDF[iax] = pd.read_csv(filePath, sep='\t')
		else:
			logging.error('iXC_Ramsey::ReadRamanAnalysisSummary::File not found specified path: {}'.format(filePath))
			logging.error('iXC_Ramsey::ReadRamanAnalysisSummary::Aborting...')
			quit()

	return SummaryDF

################ End of ReadRamseyAnalysisSummary() #################
#####################################################################

def RamseyAnalysisLevel4(AnalysisCtrl, RamseyOpts, PlotOpts, RunPars):
	"""Method for Ramsey AnalysisLevel = 4:
	Reload and process Ramsey analysis level 3 results for selected runs and perform time series analysis.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamseyOpts   (dict) - Copy of key:value pairs controlling Ramsey options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	RunNum  = RunList[0]
	nRuns  	= len(RunList)

	## Load first run to extract basic parameters
	Rams = Ramsey(WorkDir, Folder, RunNum, RamseyOpts, PlotOpts, False, RunPars.__dict__.items())
	## Load summary data from Analysis Level = 3
	SummaryDF = ReadRamseyAnalysisSummary(Rams.iaxList, RunList, PlotOpts['PlotFolderPath'], Rams.AxisFileLabels)

	for iax in Rams.iaxList:

		nData   = SummaryDF[iax]['RunTime'].shape[0]
		tStart  = SummaryDF[iax]['RunTime'].iloc[ 0]
		tStop   = SummaryDF[iax]['RunTime'].iloc[-1]
		tStep   = (tStop - tStart)/(nData-1)

		t0      = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
		tRange  = np.array([0., tStop - tStart, tStep])

		fmFm1   = SummaryDF[iax]['p01_center'].to_numpy()*1.E3 ## [Hz]
		dfmFm1  = SummaryDF[iax]['p01_center_Err'].to_numpy()*1.E3 ## [Hz]
		fmF0    = SummaryDF[iax]['p02_center'].to_numpy()*1.E3 ## [Hz]
		dfmF0   = SummaryDF[iax]['p02_center_Err'].to_numpy()*1.E3 ## [Hz]
		fmFp1   = SummaryDF[iax]['p03_center'].to_numpy()*1.E3 ## [Hz]
		dfmFp1  = SummaryDF[iax]['p03_center_Err'].to_numpy()*1.E3 ## [Hz]

		Deltaf  = 0.5*(fmFp1 - fmFm1)
		dDeltaf = 0.5*np.sqrt(dfmFp1**2 + dfmFm1**2)
		Sigmaf  = 0.5*(fmFp1 + fmFm1)
		dSigmaf = dDeltaf

		if RamseyOpts['ConvertToBField']:
			BmFm1      = Rams.Phys.BBreitRabi(1., -1., 2., -1., fmFm1)
			dBmFm1     = abs(Rams.Phys.BBreitRabi(1., -1., 2., -1., fmFm1 + dfmFm1) - BmFm1)
			BmFp1      = Rams.Phys.BBreitRabi(1., +1., 2., +1., fmFp1)
			dBmFp1     = abs(Rams.Phys.BBreitRabi(1., +1., 2., +1., fmFp1 + dfmFp1) - BmFp1)
			DeltaB     = 0.5*(BmFp1 + BmFm1)
			dDeltaB    = 0.5*np.sqrt(dBmFp1**2 + dBmFm1**2)

			yData      = [[BmFm1, BmFp1, DeltaB]]
			yErr       = [[dBmFm1, dBmFp1, dDeltaB]]
			yScales    = [[1., 1., 1.]]
			colors     = [['red', 'blue', 'black']]
			xLabels    = ['Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))]
			yLabels    = [r'$B_{|m_F\rangle}$, $\Delta B$  (G)', r'$\sqrt{\rm PSD}$  (G$/\sqrt{\rm Hz}$)', r'Allan Deviation  (G)']
			lLabels    = [[r'$B_{|+1\rangle}$', r'$B_{|-1\rangle}$', r'$\Delta B$']]
			ADev_Subsets      = [[True, True, True]]
			ADev_Fit          = [[False, False, False]]
			ADev_Fit_FitExp   = [[False, False, False]]
			ADev_Fit_SetRange = [[False, False, False]]
			ADev_Fit_Range    = [[[1.5E2, 2.E3], [1.5E2, 2.E3], [1.5E2, 2.E3]]]
		else:
			yData      = [[fmF0, Sigmaf], [abs(fmFm1), fmFp1, Deltaf]]
			yErr       = [[dfmF0, dSigmaf], [dfmFm1, dfmFp1, dDeltaf]]
			yScales    = [[1., 1.], [1.E-3, 1.E-3, 1.E-3]]
			colors     = [['darkgreen', 'gray'], ['red', 'blue', 'black']]
			xLabels    = ['None', 'Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))]
			yLabels    = [r'$\delta_{|m_F \rangle}$, $\Sigma \delta$  (Hz)', r'$|\delta_{|m_F\rangle}|$, $\Delta \delta$  (kHz)', r'$\sqrt{\rm PSD}$  (Hz$/\sqrt{\rm Hz}$)', r'Allan Deviation  (Hz)']
			lLabels    = [[r'$\delta_{|0\rangle}$', r'$\Sigma \delta$'], [r'$\delta_{|+1\rangle}$', r'$\delta_{|-1\rangle}$', r'$\Delta \delta$']]
			ADev_Subsets      = [[True, True], [True, True, True]]
			ADev_Fit          = [[False, False], [False, False, False]]
			ADev_Fit_FitExp   = [[False, False], [False, False, False]]
			ADev_Fit_SetRange = [[False, False], [False, False, False]]
			ADev_Fit_Range    = [[[1.5E2, 2.E3], [1.5E2, 2.E3]], [[1.5E2, 2.E3], [1.5E2, 2.E3], [1.5E2, 2.E3]]]

		Options = {
			'SavePlot'			: PlotOpts['SavePlot'],
			'PlotFolderPath'	: PlotOpts['PlotFolderPath'],
			'PlotFileName'		: 'Ramsey-Runs{:02d}-{:02d}-TimeSeriesAnalysis'.format(RunList[0],RunList[-1]) + PlotOpts['PlotExtension'],
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
			'FigLabels'			: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
			'ShowLegend'		: [True, False, True],
			'LegendLabels'		: lLabels,
			'LegendLocations'	: ['best', 'best', 'best'],
			'LegendFontSize'	: 10,
			'SetPlotLimits'		: [False, False],
			'PlotXLimits'		: [-2500., 1.2*tRange[1]],
			'PlotYLimits'		: [[0.5,0.9], [0.5,0.9]],
			'PSD_Plot'			: RamseyOpts['PSD_Plot'],
			'PSD_PlotSubSets'	: ADev_Subsets,
			'PSD_Method'		: RamseyOpts['PSD_Method'],
			'ADev_Plot'			: RamseyOpts['ADev_Plot'],
			'ADev_PlotSubSets'  : ADev_Subsets,
			'ADev_Type'			: 'Total',
			'ADev_taus'			: 'all',
			'ADev_ShowErrors'	: True,
			'ADev_Errorstyle'	: 'Shaded', # 'Bar' or 'Shaded'
			'ADev_Linestyle' 	: '-',
			'ADev_Marker'    	: 'None',
			'ADev_SetLimits'	: [False, False],
			'ADev_XLimits'		: [1.E2, 4.E4],
			'ADev_YLimits'		: [1.E-8, 1.E-6],
			'ADev_Fit'			: ADev_Fit,
			'ADev_Fit_XLimits'	: [2.E2, 4.E4],
			'ADev_Fit_SetRange'	: ADev_Fit_SetRange,
			'ADev_Fit_Range'	: ADev_Fit_Range,
			'ADev_Fit_FixExp'	: ADev_Fit_FitExp
			}

		iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)

################## End of RamseyAnalysisLevel4() ####################
#####################################################################

def RamseyAnalysisLevel5(AnalysisCtrl, RamseyOpts, PlotOpts, RunPars):
	"""Method for Ramsey AnalysisLevel = 5:
	Reload Ramsey analysis level 3 results for selected runs and correlate with monitor data.
	ARGUMENTS:
	\t AnalysisCtrl (dict) - Key:value pairs controlling main analysis options
	\t RamseyOpts   (dict) - Copy of key:value pairs controlling Raman options
	\t PlotOpts     (dict) - Copy of key:value pairs controlling plot options
	\t RunPars    (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	RunNum  = RunList[0]
	nRuns  	= len(RunList)

	## Load first run to extract basic parameters
	Rams = Ramsey(WorkDir, Folder, RunNum, RamseyOpts, PlotOpts, False, RunPars.__dict__.items())
	## Load monitor data
	Mon = iXC_Monitor.Monitor(WorkDir, Folder, RunList, PlotOpts, False, Rams.__dict__.items())
	Mon.ProcessMonitorData()
	## Load summary data from Analysis Level = 3
	SummaryDF = ReadRamseyAnalysisSummary(Rams.iaxList, RunList, PlotOpts['PlotFolderPath'], Rams.AxisFileLabels)

	for iax in Rams.iaxList:

		nData  = SummaryDF[iax]['RunTime'].shape[0]
		tStart = SummaryDF[iax]['RunTime'].iloc[ 0]
		tStop  = SummaryDF[iax]['RunTime'].iloc[-1]
		tStep  = (tStop - tStart)/(nData-1)

		t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
		xLabel = 'Run Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))

		tRange = np.array([0., tStop - tStart, tStep])
		tData  = np.linspace(tStart, tStop, num=nData, endpoint=True)

		parName = RamseyOpts['CorrelParameter']
		yData	= SummaryDF[iax][parName].to_numpy()
		yLabels = parName

		Mon.PlotMonitorCorrelations(iax, 0, yData, yLabels, MonDFType='Mean', iStart=0, iSkip=1)

################## End of RamseyAnalysisLevel5() ####################
#####################################################################
