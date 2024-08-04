#####################################################################
## Filename:	iXAtom_Class_Detector.py
## Author:		B. Barrett
## Description: Detector class definition for iXAtom analysis package
## Version:		3.2.5
## Last Mod:	14/01/2019
#####################################################################

import copy
import glob
import lmfit as lm
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import iXAtom_Utilities 			as iXUtils
import iXAtom_Class_RunParameters 	as iXC_RunPars

class Detector(iXC_RunPars.RunParameters):
	#################################################################
	## Class for processing and storing data from detection traces
	## Inherits all attributes and methods from class: RunParameters
	#################################################################

	def __init__(self, WorkDir, Folder, RunNum, DetectOpts, PlotOpts, LoadRunParsFlag=True, RunPars=[]):
		"""Initialize detection variables.
		ARGUMENTS:
		\t WorkDir    (str)  - Path to the top-level directory where dataset is located
		\t Folder     (str)  - Name of folder within WorkDir where dataset is located
		\t RunNum     (int)  - Run number of requested dataset
		\t DetectOpts (dict) - Key:value pairs controlling detection options, including:
		\t    'OverrideCursors'  (bool) - Flag for overriding raw detection cursors
		\t 	  'NewDetectCursors' (list) - List of new detection cursors (len = 8)
		\t PlotOpts   (dict) - Key:value pairs controlling plot options, including:
		\t    'PlotData'         (bool) - Flag for plotting detection traces
		\t    'ShowPlot'         (bool) - Flag for showing plot
		\t    'ShowFit'          (bool) - Flag for overlaying fits on plot
		\t    'SavePlot'         (bool) - Flag for saving plot (overrides ShowPlot)
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

		self.DetectOptions = copy.deepcopy(DetectOpts)
		self.PlotOptions   = copy.deepcopy(PlotOpts)
		self.nIterations   = 0
		self.nAverages     = 0
		self.nTags         = 0
		self.nTrace        = 0
		self.DetectTags    = {}
		self.DetectKeys    = ['N2Raw', 'NTRaw', 'N2BG', 'NTBG']
		self.DetectTrace   = np.array([[] for id in range(3)])
		self.DetectResult  = [{} for id in range(3)]

		self.Fit_method    = 'lm' ## 'lm': Levenberg-Marquardt, 'trf': Trust Region Reflective, 'dogbox': dogleg algorithm with rectangular trust regions
		self.Fit_ftol      = 1.E-6
		self.Fit_xtol      = 1.E-6
		self.Fit_maxfev    = 5000

		self.GetDetectFileFormat()
		self.SetDetectOptions()
		self.GetDetectFiles()

	################## End of Detector.__init__() ###################
	#################################################################

	def GetDetectFileFormat(self):
		"""Get the generic filename format for detection files of a given DataType."""

		if self.SoftwareVersion < 3.1:
			logging.warning('iXC_Detect::GetDetectionPath::Using pre-v3.1 data format... (NOT IMPLEMENTED)')

		if self.DataType == 'Ramsey' or self.DataType == 'Raman':
			self.RawFileFormat = self.FilePrefix+'-DetectData-Avg{:02d}-{:04d}.txt'
		elif self.DataType == 'Detection' or self.DataType == 'Rabi' or self.DataType == 'Correlation':
			self.RawFileFormat = self.FilePrefix+'-DetectData-{:04d}.txt'
		elif self.DataType == 'Tracking':
			self.RawFileFormat = self.FilePrefix+'-DetectData-{:05d}.txt'
		else:
			logging.error('iXC_Detect::GetDetectionPath::DataType {} not recognized...'.format(self.DataType))
			logging.error('iXC_Detect::GetDetectionPath::Could not generate detection filename. Aborting...')
			quit()

	############ End of Detector.GetDetectFileFormat() ##############
	#################################################################

	def SetDetectOptions(self):
		"""Set detection options."""

		logging.info('iXC_Detect::Setting detection options for {}...'.format(self.RunString))

		if self.DetectOptions['OverrideCursors'] and len(self.DetectOptions['NewDetectCursors']) == 8:
			self.DetectCursors = self.DetectOptions['NewDetectCursors']

	############### End of Detector.GetDetectConfig() ###############
	#################################################################

	def GetDetectFiles(self):
		"""Get the number of detection files and averages present in the specified path."""

		logging.info('iXC_Detect::Getting number of detection files for {}...'.format(self.RunString))

		if self.DataType == 'Ramsey' or self.DataType == 'Raman':
			searchIter = os.path.join(self.RawFolderPath, self.RawFileFormat.format(1,-1)).split('--')[0] + '*'
			searchAll = os.path.join(self.RawFolderPath, self.RawFileFormat).split('{')[0] + '*'
			self.nIterations = len(glob.glob(searchIter))
			if self.nIterations > 0:
				self.nAverages = int(len(glob.glob(searchAll))/self.nIterations)
			else:
				self.nAverages = 1

		else:
			searchIter = os.path.join(self.RawFolderPath, self.RawFileFormat).split('{')[0] + '*'
			self.nIterations = len(glob.glob(searchIter))
			self.nAverages = 1

		if self.nIterations == 0:
			logging.warning('iXC_Detect::GetDetectionFiles::No detection files found in: {}'.format(self.RawFolderPath))

	############### End of Detector.GetDetectFiles() ################
	#################################################################

	def LoadDetectTrace(self, AvgNum, IterNum):
		"""Load detection trace data and associated tags.
		ARGUMENTS:
		\t AvgNum (int)  - Requested average number
		\t IterNum (int) - Requested iteration number
		"""

		logging.info('iXC_Detect::Loading   detection trace (run, avg, iter = {:02d}, {:02d}, {:04d})...'.format(self.Run, AvgNum, IterNum))

		if self.DataType == 'Ramsey' or self.DataType == 'Raman':
			self.RawFileName = self.RawFileFormat.format(AvgNum, IterNum)
		else:
			self.RawFileName = self.RawFileFormat.format(IterNum)

		self.RawFilePath = os.path.join(self.RawFolderPath, self.RawFileName)

		if os.path.exists(self.RawFilePath):
			self.ParseDetectTrace()
		else:
			logging.error('iXC_Detect::LoadDetectTrace::Detection file not found: {}'.format(self.RawFilePath))
			logging.error('iXC_Detect::LoadDetectTrace::Aborting...')
			quit()

	############## End of Detector.LoadDetectTrace() ################
	#################################################################

	def ParseDetectTrace(self):
		"""Parse detection trace data and associated tags."""

		if self.SoftwareVersion >= 3.1:

			nRows = 25
			# Read first nRows of detect file to get Tag info
			strData = pd.read_csv(self.RawFilePath, delimiter='\t', header=None, engine='python', 
				nrows=nRows, error_bad_lines=False, warn_bad_lines=False).values
			self.nTags = len(strData)

			for i in range(self.nTags):
				key = strData[i,0][1:-1]
				if key == 'Date Stamp' or key == 'Time Stamp':
					value = strData[i,1]
				else:
					value = float(strData[i,1])
				self.DetectTags.update({key:value})

			if self.DataType == 'Ramsey':
				self.DetectTags.update({'k Index': 0})

			# Skip first 'nTags' rows of file to get Detection Trace
			self.DetectTrace = pd.read_csv(self.RawFilePath, delimiter='\t', header=None, engine='python',
				skiprows=self.nTags, error_bad_lines=False, warn_bad_lines=False).to_numpy().transpose()
			self.nTrace = len(self.DetectTrace[0])

		else:

			with open(self.RawFilePath, 'r') as f:
				index  = 0
				while index >= 0:
					self.nTags += 1
					fContents = f.readline()
					index = fContents.find('#')
					if index >= 0:
						key = fContents[index+1:].strip()
						if key == 'Date Stamp' or key == 'Time Stamp':
							value = fContents[:index-1]
						else:
							value = float(fContents[:index-1])
						self.DetectTags.update({key:value})

				strTrace = f.readlines()
				strTrace.insert(0,fContents) # prepend the first value (i.e. last line read by while loop)
				self.nTrace = len(strTrace)
				if self.nTrace > 0:
					for i in range(self.nTrace):
						self.DetectTrace[0].append(float(strTrace[i]))

	############# End of Detector.ParseDetectTrace() ################
	#################################################################

	@staticmethod
	def fDetect(x, Amplitude, Gamma, Offset, Slope):
		"""Function for fitting detection data."""
		return Amplitude*np.exp(-Gamma*x) + Offset + Slope*x

	####################### End of fDetect() ########################
	#################################################################

	@staticmethod
	def InitializeDetectPars():
		"""Initialize detection fit parameters."""
		
		InitPars = [[] for id in range(3)]
		## Lower detector (N2Raw, NTRaw, N2BG, NTBG)
		InitPars[0] = [
			{'Amplitude': 0.20, 'Gamma': 0.04, 'Offset': 0.10, 'Slope': 0.},
			{'Amplitude': 0.70, 'Gamma': 0.03, 'Offset': 0.10, 'Slope': 0.},
			{'Amplitude': 0.04, 'Gamma': 0.06, 'Offset': 0.10, 'Slope': 0.},
			{'Amplitude': 0.04, 'Gamma': 0.04, 'Offset': 0.10, 'Slope': 0.}]
		## Middle detector
		InitPars[1] = [
			{'Amplitude': 0.15, 'Gamma': 0.04, 'Offset': 0.20, 'Slope': 0.},
			{'Amplitude': 0.50, 'Gamma': 0.03, 'Offset': 0.20, 'Slope': 0.},
			{'Amplitude': 0.04, 'Gamma': 0.06, 'Offset': 0.20, 'Slope': 0.},
			{'Amplitude': 0.04, 'Gamma': 0.02, 'Offset': 0.20, 'Slope': 0.}]
		## Upper detector
		InitPars[2] = [
			{'Amplitude': 0.01, 'Gamma': 0.07, 'Offset': 0.03, 'Slope': 0.},
			{'Amplitude': 0.03, 'Gamma': 0.03, 'Offset': 0.03, 'Slope': 0.},
			{'Amplitude': 0.01, 'Gamma': 0.05, 'Offset': 0.03, 'Slope': 0.},
			{'Amplitude': 0.01, 'Gamma': 0.02, 'Offset': 0.03, 'Slope': 0.}]

		return InitPars

	############ End of Detector.InitializeDetectPars() #############
	#################################################################

	def SliceDetectTrace(self):
		"""Slice detection trace into four zones determined by detection cursors."""

		cursor = self.DetectCursors
		xData  = [np.array([]) for ic in range(4)]
		yData  = [[np.array([]) for ic in range(4)] for id in range(3)]
		for ic in range(4):
			xData[ic] = np.linspace(0, cursor[2*ic+1]-cursor[2*ic], num=cursor[2*ic+1]-cursor[2*ic]+1, endpoint=True)
			for id in self.idList:
				yData[id][ic] = np.array(self.DetectTrace[id][cursor[2*ic]:cursor[2*ic+1]+1])

		return [xData, yData]

	####################### End of fDetect() ########################
	#################################################################

	def AnalyzeDetectTrace(self, AvgNum, IterNum, InitPars):
		"""Analyze detection trace to obtain N2, NTotal and Ratio for each available detector."""

		logging.info('iXC_Detect::Analyzing detection trace (run, avg, iter = {:02d}, {:02d}, {:04d})...'.format(self.Run, AvgNum, IterNum))

		[xData, yData] = self.SliceDetectTrace()

		## Construct fit model and initial parameters objects
		model = lm.Model(self.fDetect)
		pInit = model.make_params()
		pInit['Amplitude'].min = 0.
		pInit['Gamma'].min = 0.

		fit_kws = {'xtol': self.Fit_xtol, 'ftol': self.Fit_ftol, 'maxfev': self.Fit_maxfev}

		for id in self.idList:
			self.DetectResult[id]['Detector'] = self.DetectNames[id]

			## Fit N2, NT, N2BG, and NTBG data
			for ic in range(4):
				pInit['Amplitude'].value = InitPars[id][ic]['Amplitude']
				pInit['Gamma'].value     = InitPars[id][ic]['Gamma']
				pInit['Offset'].value    = InitPars[id][ic]['Offset']
				pInit['Slope'].value     = InitPars[id][ic]['Slope']

				if ic <= 1:
					pInit['Slope'].vary = self.DetectOptions['FitLinearSlope']
				else:
					## Don't vary slope in BG fits, they may not work
					pInit['Slope'].vary = False

				result = model.fit(yData[id][ic], pInit, x=xData[ic], method='leastsq', fit_kws=fit_kws)
				yFit   = result.best_fit
				yRes   = yData[id][ic] - result.best_fit
				sRes   = np.std(yRes)
				dsRes  = sRes/np.sqrt(2*result.nfree)
				try:
					dyFit = result.eval_uncertainty(x=xData[ic])
					yErr  = np.sqrt(result.params['Amplitude'].stderr**2 + result.params['Offset'].stderr**2)
				except:
					dyFit = np.zeros(len(yFit))
					yErr  = np.sqrt(2.)*dsRes

				message = 'iXC_Detect::{} ({},\t{},\tnfev = {}, redchi = {:5.3E})'.format(result.message, self.DetectNames[id], self.DetectKeys[ic], result.nfev, result.redchi)
				if result.success and self.DetectOptions['ShowFitMessages']:
					logging.info(message)
				elif not result.success:
					logging.warning(message)

				if self.DetectOptions['PrintFitReport']:
					print('---------------- Detect Fit Result ({}, {}) ----------------'.format(self.DetectNames[id], self.DetectKeys[ic]))
					print(result.fit_report())

				self.DetectResult[id][self.DetectKeys[ic]] = {'Best': yFit[0], 'Error': yErr, 'BestPars': result.best_values,
					'InitPars': result.init_values, 'xData': xData[ic], 'yData': yData[id][ic], 'BestFit': yFit, 'FitError': dyFit,
					'Residuals': yRes, 'ResDev': sRes, 'ResDev_Err': dsRes, 'ChiSqr': result.chisqr, 'RedChi': result.redchi}

			self.DetectResult[id]['N2'] = {
				'Best': self.DetectResult[id]['N2Raw']['Best'] - self.DetectResult[id]['N2BG']['Best'],
				'Error': np.sqrt(self.DetectResult[id]['N2Raw']['Error']**2 + self.DetectResult[id]['N2BG']['Error']**2)}
			self.DetectResult[id]['NT'] = {
				'Best': self.DetectResult[id]['NTRaw']['Best'] - self.DetectResult[id]['NTBG']['Best'],
				'Error': np.sqrt(self.DetectResult[id]['NTRaw']['Error']**2 + self.DetectResult[id]['NTBG']['Error']**2)}
			self.DetectResult[id]['Ratio'] = {
				'Best': self.DetectResult[id]['N2']['Best']/self.DetectResult[id]['NT']['Best'],
				'Error': abs(self.DetectResult[id]['N2']['Best']/self.DetectResult[id]['NT']['Best']) * \
					np.sqrt((self.DetectResult[id]['N2']['Error']/self.DetectResult[id]['N2']['Best'])**2 +
					(self.DetectResult[id]['NT']['Error']/self.DetectResult[id]['NT']['Best'])**2)}

			# print(self.DetectResult[0]['N2Raw']['Best'])
			# print(self.DetectResult[0]['NTRaw']['Best'])
			# print(self.DetectResult[0]['N2BG']['Best'])
			# print(self.DetectResult[0]['NTBG']['Best'])
			# print(self.DetectResult[0]['N2']['Best'])
			# print(self.DetectResult[0]['NT']['Best'])
			# print(self.DetectResult[0]['Ratio']['Best'])

	############ End of Detector.AnalyzeDetectionTrace() ############
	#################################################################

	def PlotDetectTrace(self, PlotPath, PlotTitle):
		"""Plot and save detection trace.
		ARGUMENTS:
		\t PlotPath  (str) - Output path for detection plot
		\t PlotTitle (str) - Title for detection plot
		"""

		logging.info('iXC_Detect::Plotting detection trace for {}...'.format(self.RunString))

		## Construct figure
		(nRows, nCols) = (2,2)
		fig = plt.figure(figsize=(9,5), constrained_layout=True)
		gs  = fig.add_gridspec(nRows, nCols)
		axs = []
		axs.append(fig.add_subplot(gs[0,:])) # Top row, spans both columns
		axs.append(fig.add_subplot(gs[1,0])) # Bottom row, left column
		axs.append(fig.add_subplot(gs[1,1])) # Bottom row, right column
		for ax in axs:
			ax.grid(b=True, which='major', axis='both', color='0.75', linestyle='-')

		# iXUtils.SetDefaultPlotOptions()

		## Set default custom plot options
		customPlotOpts = {'Color': 'gold', 'Linestyle': '-', 'Marker': '.',
			'Title': 'None', 'xLabel': 'Time (ms)', 'yLabel': 'Signal (V)',
			'LegLabel': 'None', 'Legend': True, 'LegLocation': 'best'}

		colors = [['gold','darkorange','indianred'], ['black', 'black', 'black']] ## [data=0,fit=1; id=0,1,2]

		[iN2L, _, iNTL, _, iN2BGL, _, iNTBGL, _] = self.DetectCursors

		nPoints = len(self.DetectTrace[self.idList[0]])
		iRatioL = 0
		iRatioR = min(260, nPoints - 1)
		iBGL    = min(975, nPoints - 1)
		iBGR    = min(1235, nPoints - 1)

		xData0 = np.linspace(0, (nPoints-1)*self.DetectTags['Time Step']*1.0e3, num=nPoints, endpoint=True)
		xData1 = np.linspace(iRatioL, iRatioR, num=iRatioR-iRatioL+1, endpoint=True)
		xData2 = np.linspace(iBGL, iBGR, num=iBGR-iBGL+1, endpoint=True)

		for id in self.idList:
			customPlotOpts['Color']    = colors[0][id]
			customPlotOpts['xLabel']   = 'Time (ms)'
			customPlotOpts['yLabel']   = 'Signal (V)'
			customPlotOpts['Title']    = PlotTitle
			customPlotOpts['LegLabel'] = self.DetectResult[id]['Detector']
			customPlotOpts['Legend']   = True
			## Top row: full detection trace
			iXUtils.CustomPlot(axs[0], customPlotOpts, xData0, self.DetectTrace[id])

			customPlotOpts['xLabel']   = 'Index'
			customPlotOpts['Title']    = None
			customPlotOpts['Legend']   = False
			## Bottom row, left col: N2 + NT
			iXUtils.CustomPlot(axs[1], customPlotOpts, xData1, self.DetectTrace[id][iRatioL:iRatioR+1])

			customPlotOpts['yLabel']   = None
			## Bottom row, right col: N2BG + NTBG
			iXUtils.CustomPlot(axs[2], customPlotOpts, xData2, self.DetectTrace[id][iBGL:iBGR+1])

		## Plot N2 + NT cursors
		for ic in self.DetectCursors[:4]:
			axs[1].axvline(x=ic, color='0.25', linestyle='-', linewidth=1, ymin=0., ymax=1.)

		## Plot N2BG + NTBG cursors
		for ic in self.DetectCursors[4:]:
			axs[2].axvline(x=ic, color='0.25', linestyle='-', linewidth=1, ymin=0., ymax=1.)

		if self.PlotOptions['ShowFit']:
			customPlotOpts['xLabel']   = 'Index'
			customPlotOpts['Title']    = None
			customPlotOpts['Marker']   = 'None'
			customPlotOpts['LegLabel'] = 'None'
			customPlotOpts['Legend']   = False

			for id in self.idList:
				customPlotOpts['Color']  = colors[1][id]
				customPlotOpts['yLabel'] = 'Signal (V)'

				## N2 fit
				xFit = self.DetectResult[id]['N2Raw']['xData'] + iN2L
				yFit = self.DetectResult[id]['N2Raw']['BestFit']
				iXUtils.CustomPlot(axs[1], customPlotOpts, xFit, yFit)

				dyFit = self.DetectResult[id]['N2Raw']['FitError']
				axs[1].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

				## NT fit
				xFit = self.DetectResult[id]['NTRaw']['xData'] + iNTL
				yFit = self.DetectResult[id]['NTRaw']['BestFit']
				iXUtils.CustomPlot(axs[1], customPlotOpts, xFit, yFit)

				dyFit = self.DetectResult[id]['NTRaw']['FitError']
				axs[1].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

				customPlotOpts['yLabel'] = None

				## N2BG fit
				xFit = self.DetectResult[id]['N2BG']['xData'] + iN2BGL
				yFit = self.DetectResult[id]['N2BG']['BestFit']
				iXUtils.CustomPlot(axs[2], customPlotOpts, xFit, yFit)

				dyFit = self.DetectResult[id]['N2BG']['FitError']
				axs[2].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

				## NTBG fit
				xFit = self.DetectResult[id]['NTBG']['xData'] + iNTBGL
				yFit = self.DetectResult[id]['NTBG']['BestFit']
				iXUtils.CustomPlot(axs[2], customPlotOpts, xFit, yFit)

				dyFit = self.DetectResult[id]['NTBG']['FitError']
				axs[2].fill_between(xFit, yFit-dyFit, yFit+dyFit, color='blue', alpha=0.5)

		if self.PlotOptions['SavePlot']:
			plt.savefig(PlotPath)
			logging.info('iXC_Detect::Detect plot saved to:')
			logging.info('iXC_Detect::  {}'.format(PlotPath))
		elif self.PlotOptions['ShowPlot']:
			plt.show()

	############### End of Detector.PlotDetectTrace() ###############
	#################################################################

	def AnalyzeSingleDetectTrace(self, AvgNum, IterNum):
		"""Analyze and plot a single detection trace."""

		initPars = self.InitializeDetectPars()

		self.LoadDetectTrace(AvgNum, IterNum)
		self.AnalyzeDetectTrace(AvgNum, IterNum, initPars)

		if 'Accel Axis' in self.DetectTags.keys() and 'k Index' in self.DetectTags.keys():
			[iax, ik] = [int(self.DetectTags['Accel Axis']), int(self.DetectTags['k Index'])]
			iax = max(0,min(iax,2))
			ik  = max(0,min(ik,1))
		else:
			logging.warning('iXC_Detect::AnalyzeSingleDetectTrace::iax or ik index not found in detection tags.')
			logging.warning('iXC_Detect::AnalyzeSingleDetectTrace::Setting iax = 2 (Z), ik = 0 (kU)...')
			iax = 2
			ik  = 0

		if self.PlotOptions['PlotData']:
			PlotPath = os.path.join(self.WorkDir, self.Folder, 'DetectTrace-Run{:02d}-Avg{:02d}-{:04d}.png'.format(self.Run, AvgNum, IterNum))
			PlotTitle = 'Iter = {:03d}, '.format(IterNum) + self.AxisLegLabels[iax][ik]
			if self.DataType == 'Raman':
				if self.ChirpedData:
					PlotTitle += ', Chirp = {:8.6e} Hz/s'.format(self.DetectTags['Current Chirp'])
				else:
					PlotTitle += ', Phase = {:5.3f} rad'.format(self.DetectTags['Current Phase'])
			elif self.DataType == 'Ramsey':
				if 'Current Frequency' in self.DetectTags.keys():
					freq = self.DetectTags['Current Frequency']
				else:
					freq = list(self.DetectTags.values())[0]
				PlotTitle += ', Freq = {:10.8e} Hz'.format(freq)
			elif self.DataType == 'Rabi':
				PlotTitle += ', X = {:6.4e}'.format(self.DetectTags['Current XValue'])

			self.PlotDetectTrace(PlotPath, PlotTitle)

	########## End of Detector.AnalyzeSingleDetectTrace() ###########
	#################################################################