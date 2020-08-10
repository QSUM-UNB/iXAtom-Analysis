#####################################################################
## Filename:	iXAtom_Class_Track.py
## Author:		B. Barrett
## Description: Track class definition for iXAtom analysis package
## Version:		3.2.4
## Last Mod:	23/07/2020
##===================================================================
## Change Log:
## 22/10/2019 - Track class defined and bug tested
##				(for LabVIEW v3.1 data only)
## 06/11/2019 - Created stream class, and moved over related methods
##				in order to broaden the scope it stream data usage to
##				all classes
## 10/12/2019 - Added features for overlaying and subtracting tides.
## 14/02/2019 - Modified PlotTrackingData method for LabVIEW software
##				version 3.3. This version involved changes to the type
##				of data saved in the ratio files (phase correction
##				replaced with error signal integral), and the rate at
##				which the error signal was updated (increased x 2).
## 01/07/2020 - Added method for printing Tracking statistics
## 03/07/2020 - Minor mods and bug fixes to PlotPhaseData method
## 23/07/2020 - Modified PlotPhaseData method to be compatible with older
##				data sets. Also added acceleration plot unit, ADev fit
##				controls in the TrackingOptions dictionary, and re-implemented
##				moving average functionality.
#####################################################################

import copy
import datetime          as dt
import glob
import logging
import matplotlib.pyplot as plt
import numpy             as np
import os
import pandas            as pd
import pytz

from scipy.signal        import periodogram
from scipy.signal        import welch
from scipy.interpolate   import interp1d

import iXAtom_Utilities           as iXUtils
import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Class_Monitor       as iXC_Monitor
# import iXAtom_Class_Physics       as iXC_Physics

#####################################################################

class Track(iXC_RunPars.RunParameters):
	#################################################################
	## Class for storing and processing Tracking data
	## Inherits all attributes and methods from class: RunParameters
	#################################################################

	def __init__(self, WorkDir, Folder, RunNum, TrackOpts, MonitorOpts, PlotOpts, LoadRunParsFlag=True, RunPars=[]):
		"""Initialize tracking variables.
		ARGUMENTS:
		\t WorkDir         (str)  - Path to the top-level directory where dataset is located
		\t Folder          (str)  - Name of folder within WorkDir where dataset is located
		\t RunNum          (int)  - Run number of requested dataset
		\t TrackOpts       (dict) - Key:value pairs controlling tracking options
		\t MonitorOpts     (dict) - Key:value pairs controlling monitor options
		\t PlotOpts        (dict) - Key:value pairs controlling plot options
		\t LoadRunParsFlag (bool) - (Optional) Flag for loading run parameters from file (True) or setting them from input (False).
		\t RunPars 		   (list) - (Optional) Key:value pairs containing run parameters
		"""
		super().__init__(WorkDir, Folder, RunNum)
		if LoadRunParsFlag:
			super().LoadRunParameters()
		else:
			# Take parameters from input RunPars dictionary
			for key, val in RunPars:
				setattr(self, key, val)

		self.TrackOptions   = copy.deepcopy(TrackOpts)
		self.MonitorOptions = copy.deepcopy(MonitorOpts)
		self.PlotOptions    = copy.deepcopy(PlotOpts)
		self.GetTrackConfig()

		self.RawDataFiles = [
			'Track-Run{:02d}-Ratios-kU.txt'.format(self.Run), 
			'Track-Run{:02d}-Ratios-kD.txt'.format(self.Run),
			'Track-Run{:02d}-Phases.txt'.format(self.Run),
			'Track-Run{:02d}-RTAccelData.txt'.format(self.Run),
			'Track-Run{:02d}-RTFreqData.txt'.format(self.Run),
			'Track-Run{:02d}-MonitorData.txt'.format(self.Run)]
		self.RatioFiles  = [self.RawDataFiles[0], self.RawDataFiles[1]]
		self.PhaseFile   = self.RawDataFiles[2]
		self.RTAccelFile = self.RawDataFiles[3]
		self.RTFreqFile  = self.RawDataFiles[4]
		self.MonitorFile = self.RawDataFiles[5]

		self.RawDataFound     = False
		self.RatioFilesFound  = False
		self.PhaseFileFound   = False
		self.RTAccelFileFound = False
		self.RTFreqFileFound  = False
		self.MonitorFileFound = False

		self.RatioDF   = [[pd.DataFrame() for ik in range(2)] for iax in range(3)]
		self.PhaseDF   = [pd.DataFrame() for iax in range(3)]
		self.RTFreqDF  = pd.DataFrame()
		self.RTAccelDF = pd.DataFrame()
		self.MonitorDF = pd.DataFrame()

		self.PrintSubSeqTiming = True

	#################### End of Track.__init__() ####################
	#################################################################

	def GetTrackConfig(self):
		"""Get tracking configuration."""

		logging.info('iXC_Track::Getting tracking config for {}...'.format(self.RunString))

		## Get list of indices representing modulation phases (-1,0,1 = -pi/2, 0, +pi/2)
		if self.TrackProtocol == 'Two-Point kUp' or self.TrackProtocol == 'Two-Point kDown':
			imList = [-1,1] 
		elif self.TrackProtocol == 'Two-Point kInterlaced':
			imList = [-1,1,-1,1]
		elif self.TrackProtocol == 'Three-Point kInterlaced':
			imList = [-1,1,0,-1,1,0]
		else:
			logging.error('iXC_Track::GetTrackConfig::Tracking protocol not recognized: {}'.format(self.TrackProtocol))
			logging.error('iXC_Track::GetTrackConfig::Aborting...')
			quit()

	################# End of Track.GetTrackConfig() #################
	#################################################################

	def LoadAnalysisData(self, analysisLevel):
		"""Load specific tracking data according to analysisLevel."""

		logging.info('iXC_Track::Loading raw tracking data from:')
		logging.info('iXC_Track::  {}'.format(self.RawFolderPath))

		if analysisLevel <= 2:
			## Load ratio data
			for ik in range(len(self.RatioFiles)):
				[self.RatioFilesFound, df] = self.LoadData(self.RawFolderPath, self.RatioFiles[ik])
				if self.RatioFilesFound:
					for iax in self.iaxList:
						self.RatioDF[iax][ik] = df[df['AxisIndex'] == iax]

			## Load phase data
			[self.PhaseFileFound, df] = self.LoadData(self.RawFolderPath, self.PhaseFile)
			if self.PhaseFileFound:
				for iax in self.iaxList:
					self.PhaseDF[iax] = df[df['AccelAxis'] == iax]

			if self.RatioFilesFound and self.PhaseFileFound:
				self.RawDataFound = True

		elif analysisLevel == 3:
			## Load RT accel data
			[self.RTAccelFileFound, self.RTAccelDF] = self.LoadData(self.RawFolderPath, self.RTAccelFile)
			if self.RTAccelFileFound:
				self.RawDataFound = True

			## Load RT freq data
			# [self.RTFreqFileFound, self.RTFreqDF] = self.LoadData(self.RawFolderPath, self.RTFreqFile)
			# if self.RTFreqFileFound:
			# 	self.RawDataFound = True

		# if analysisLevel == 2 or analysisLevel == 3:
		# 	## Load monitor data
		# 	[self.MonitorFileFound, self.MonitorDF] = self.LoadData(self.RawFolderPath, self.MonitorFile)
		# 	if self.MonitorFileFound:
		# 		self.RawDataFound = True

		if self.RawDataFound:
			self.GetSequenceTiming(analysisLevel)
		else:
			logging.error('iXC_Track::LoadAnalysisData::Requested data not found in: {}'.format(self.RawFolderPath))
			logging.error('iXC_Track::LoadAnalysisData::Aborting...')
			quit()

	################ End of Track.LoadAnalysisData() ################
	#################################################################

	def GetSequenceTiming(self, analysisLevel):
		"""Get experimental sequence timing parameters."""

		logging.info('iXC_Track::Getting sequence timing for {}...'.format(self.RunString))

		if analysisLevel <= 2:
			iaxMin = self.iaxList[0]
			ikMin  = self.ikList[0]
			self.nSamplesFull = 1000000000

			df = self.RatioDF
			## Find the minimum number of samples among all axes and directions
			for iax in self.iaxList:
				for ik in self.ikList:
					nSamps = df[iax][ik].shape[0]
					if nSamps < self.nSamplesFull:
						iaxMin = iax
						ikMin  = ik
						self.nSamplesFull = nSamps

			self.dtStartFull = iXUtils.TimestampToDatetime(df[self.iaxList[0]][self.ikList[0]]['Date'].iloc[0], df[self.iaxList[0]][self.ikList[0]]['Time'].iloc[0])
			self.dtStopFull  = iXUtils.TimestampToDatetime(df[iaxMin][ikMin]['Date'].iloc[-1], df[iaxMin][ikMin]['Time'].iloc[-1])
			self.nIterFull   = self.nSamplesFull*self.nk*self.nax

		elif analysisLevel == 3:
			df = self.RTAccelDF

			self.nSamplesFull = df['#Iteration'].shape[0]
			self.dtStartFull  = iXUtils.TimestampToDatetime(df['Date'].iloc[0], df['Time'].iloc[0])
			self.dtStopFull   = iXUtils.TimestampToDatetime(df['Date'].iloc[-1], df['Time'].iloc[-1])
			self.nIterFull    = df['#Iteration'].iloc[-1]

		self.DurationFull = (self.dtStopFull - self.dtStartFull).total_seconds()
		self.tCycMeanFull = self.DurationFull/(self.nIterFull - 1)
		self.fCycMeanFull = 1./self.tCycMeanFull

		print('---------------- Full Sequence Timing ----------------')
		print(' Number of Samples: {}'.format(self.nSamplesFull))
		print(' Number Iterations: {}'.format(self.nIterFull))
		print(' Start Time (Full): {}'.format(self.dtStartFull))
		print(' Stop  Time (Full): {}'.format(self.dtStopFull))
		print(' Duration   (Full): {:05d} s ({:5.2f} hrs)'.format(round(self.DurationFull), self.DurationFull/3600.))
		print(' Cycle Time (Full): {:5.3f} s'.format(self.tCycMeanFull))
		print(' Cycle Rate (Full): {:5.3f} Hz'.format(self.fCycMeanFull))
		print('-------------------------------------------------------')

	############### End of Track.GetSequenceTiming() ################
	#################################################################

	def SetTrackTimeRange(self, DFIn):
		"""Set subset of time series to plot.
		ARGUMENTS:
		DFIn  (dataframe) - Input data frame.
		RETURN FORMAT:
		DFOut (dataframe) - Output data frame.
		"""

		DFOut = DFIn.copy()
		self.nSamples = DFIn.shape[0]
		self.IterMin  = int(DFIn['#Iteration'].iloc[0])
		self.IterMax  = int(DFIn['#Iteration'].iloc[-1])
		self.IterStep = int((DFIn['#Iteration'].iloc[2] - DFIn['#Iteration'].iloc[0])/2.)
		self.nIter    = self.IterMax
		self.tStep    = self.tCycMeanFull*self.IterStep

		if self.TrackOptions['SetTimeRange']:
			## Select subset of iteration range
			[self.tStart, self.tStop] = self.TrackOptions['TimeRange']

			self.IndexStart = int(round(self.tStart/self.tStep))
			self.IndexStop  = int(round(self.tStop/self.tStep))
			if self.IndexStart < 0:
				self.IndexStart = 0
			if self.IndexStop > self.nSamples:
				self.IndexStop = self.nSamples

			self.IterStart = self.IndexStart*self.IterStep
			self.IterStop  = self.IndexStop*self.IterStep

			if self.IterStart > self.IterMin:
				DFOut = DFOut.loc[DFOut['#Iteration'] >= self.IterStart]
			if self.IterStop < self.IterMax:
				DFOut = DFOut.loc[DFOut['#Iteration'] <= self.IterStop]

			self.nSamples = DFOut.shape[0]
			self.nIter    = self.IterStop - self.IterStart + 1
			self.dtStart  = iXUtils.TimestampToDatetime(DFOut['Date'].iloc[0], DFOut['Time'].iloc[0])
			self.dtStop   = iXUtils.TimestampToDatetime(DFOut['Date'].iloc[-1], DFOut['Time'].iloc[-1])
			self.Duration = (self.dtStop - self.dtStart).total_seconds()
			self.tCycMean = self.Duration/(self.nIter - 1)
			self.fCycMean = 1./self.tCycMean

			if self.PrintSubSeqTiming:
				self.PrintSubSeqTiming = False
				print('----------------- Sub-Sequence Timing ----------------')
				print(' Number of Samples: {}'.format(self.nSamples))
				print(' Number Iterations: {}'.format(self.nIter))
				print(' Iteration Step   : {}'.format(self.IterStep))
				print(' Start Time  (Sub): {}'.format(self.dtStart))
				print(' Stop  Time  (Sub): {}'.format(self.dtStop))
				print(' Duration    (Sub): {:05d} s ({:5.2f} hrs)'.format(round(self.Duration), self.Duration/3600.))
				print(' Cycle Time  (Sub): {:5.3f} s'.format(self.tCycMean))
				print(' Cycle Rate  (Sub): {:5.3f} Hz'.format(self.fCycMean))
				print('------------------------------------------------------')

		else:
			## Select whole iteration range
			self.IndexStart = 0
			self.IndexStop  = self.nSamples
			self.IterStart  = self.IterMin
			self.IterStop   = self.IterMax
			self.tStart     = self.IndexStart*self.tStep
			self.tStop      = self.IndexStop*self.tStep
			self.dtStart    = self.dtStartFull
			self.dtStop     = self.dtStopFull
			self.Duration	= self.DurationFull

		return DFOut

	############### End of Track.SetTrackTimeRange() ################
	#################################################################

	# def PlotTrackData(self):
	# 	"""Plot summary of raw tracking data."""

	# 	logging.info('iXC_Track::Plotting tracking data for {}...'.format(self.RunString))

	# 	plt.rc('legend', fontsize=10)
	# 	# plt.rc('axes', labelsize=16)
	# 	plt.rc('lines', linewidth=1.0)

	# 	if self.TrackOptions['ADev_Plot'] and self.TrackOptions['PSD_Plot']:
	# 		## Plot time series + ADev + PSD
	# 		nCol = 3
	# 		[cPSD, cADev] = [nCol-2,nCol-1]
	# 	elif self.TrackOptions['ADev_Plot'] or self.TrackOptions['PSD_Plot']:
	# 		## Plot time series + ADev or PSD
	# 		nCol = 3
	# 		if self.TrackOptions['PSD_Plot']:
	# 			cPSD = nCol-1
	# 		else:
	# 			cADev = nCol-1
	# 	else:
	# 		## Plot time series only
	# 		nCol = 2

	# 	(colW, colH) = (5,6)
	# 	fig = plt.figure(figsize=(nCol*colW, colH), constrained_layout=True)
	# 	gs  = fig.add_gridspec(3, nCol)
	# 	axs = [[] for iCol in range(nCol)]

	# 	if self.TrackOptions['ADev_Plot'] and self.TrackOptions['PSD_Plot']:
	# 		for i in range(3):
	# 			axs[0].append(fig.add_subplot(gs[i, 0])) # 1st column, independent rows
	# 		axs[1].append(fig.add_subplot(gs[:, 1])) # 2nd column spans all rows
	# 		axs[2].append(fig.add_subplot(gs[:, 2])) # 3rd column spans all rows
	# 	elif self.TrackOptions['ADev_Plot'] or self.TrackOptions['PSD_Plot']:
	# 		for i in range(3):
	# 			axs[0].append(fig.add_subplot(gs[i, 0])) # 1st column, independent rows
	# 			axs[1].append(fig.add_subplot(gs[i, 1])) # 2nd column, independent rows
	# 		axs[2].append(fig.add_subplot(gs[:, 2])) # 3rd column spans all rows
	# 	else:
	# 		for i in range(3):
	# 			axs[0].append(fig.add_subplot(gs[i, 0])) # 1st column, independent rows
	# 			axs[1].append(fig.add_subplot(gs[i, 1])) # 2nd column, independent rows

	# 	customPlotOpts = {'Color': 'red', 'eColor': 'red', 'Linestyle': '-', 'Marker': None,
	# 		'Title': 'None', 'xLabel': 'None', 'yLabel': r'$N_2/N_{\rm total}$',
	# 		'Legend': False, 'LegLabel': 'None'}

	# 	phaseLabels = [[r'X-$k_{\rm ind}$',r'X-$k_{\rm dep}$'], [r'Y-$k_{\rm ind}$',r'Y-$k_{\rm dep}$'], [r'Z-$k_{\rm ind}$',r'Z-$k_{\rm dep}$']]
	# 	phaseColors = [['gold', 'darkorange'], ['magenta', 'darkmagenta'], ['gray', 'black']]

	# 	if self.SoftwareVersion <= 3.2:
	# 		iSkipRatio = self.nax
	# 		iSkipError = self.nk*self.nax
	# 		iSkipPhase = 2*self.nk*self.nax
	# 	else:
	# 		iSkipRatio = self.nax
	# 		iSkipError = self.nax
	# 		iSkipPhase = self.nk*self.nax

	# 	rateError = self.fCycMeanFull/iSkipError
	# 	ratePhase = self.fCycMeanFull/iSkipPhase

	# 	for iax in self.iaxList:

	# 		dFPhase = self.SetTrackTimeRange(self.PhaseDF[iax])

	# 		for ik in self.ikList:

	# 			dFRatio = self.SetTrackTimeRange(self.RatioDF[iax][ik])

	# 			plt.rc('lines', linewidth=1.0)
	# 			customPlotOpts['Color']     = self.DefaultPlotColors[iax][ik]
	# 			customPlotOpts['eColor']    = self.DefaultPlotColors[iax][ik]
	# 			customPlotOpts['LegLabel']  = self.AxisLegLabels[iax][ik]
	# 			customPlotOpts['Linestyle'] = '-'
	# 			customPlotOpts['Marker']    = 'None'

	# 			if self.tStop > 10000:
	# 				xScale = 1.E-3
	# 				xLabel = r'Time  ($\times 10^3$ s)'
	# 			else:
	# 				xScale = 1.
	# 				xLabel = 'Time  (s)'

	# 			iCol   = 0
	# 			xData1 = dFRatio['#Iteration']*self.tCycMeanFull*xScale
	# 			xData2 = dFPhase['#Iteration']*self.tCycMeanFull*xScale

	# 			if not (self.TrackOptions['PSD_Plot'] and self.TrackOptions['ADev_Plot']):
	# 				##--------------------------------------------
	# 				## Plots for 1st column (skip if plotting both PSD and ADev)
	# 				customPlotOpts['xLabel'] = 'None'
	# 				customPlotOpts['yLabel'] = r'$N_2/N_{\rm total}$'
	# 				customPlotOpts['Legend'] = False

	# 				if self.TrackOptions['ComputeMovingAvg']:
	# 					winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkipRatio)))
	# 					logging.info('iXC_Track::Computing moving average for window size {} ({} s)...'.format(winSize, self.TrackOptions['MovingAvgWindow']))
	# 					yData = dFRatio['Ratio'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
	# 				else:
	# 					yData = dFRatio['Ratio'].to_numpy()

	# 				iXUtils.CustomPlot(axs[iCol][0], customPlotOpts, xData1, yData)
	# 				plt.setp(axs[iCol][0].get_xticklabels(), visible=False)

	# 				customPlotOpts['yLabel'] = r'$y_0$'
	# 				if ik == 0:
	# 					colName = 'kUOffset'
	# 				else:
	# 					colName = 'kDOffset'

	# 				if self.TrackOptions['ComputeMovingAvg']:
	# 					winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkipError)))
	# 					logging.info('iXC_Track::Computing moving average for window size {} ({} s)...'.format(winSize, self.TrackOptions['MovingAvgWindow']))
	# 					yData = dFPhase[colName].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
	# 				else:
	# 					yData = dFPhase[colName].to_numpy()

	# 				iXUtils.CustomPlot(axs[iCol][1], customPlotOpts, xData2, yData)
	# 				plt.setp(axs[iCol][1].get_xticklabels(), visible=False)

	# 				customPlotOpts['xLabel'] = xLabel

	# 				if self.SoftwareVersion <= 3.2:
	# 					customPlotOpts['yLabel'] = 'Correction (rad)'
	# 					colName = 'CorrectPhase'
	# 					iSkip   = 2
	# 				else:
	# 					customPlotOpts['yLabel'] = 'Error Integral'
	# 					colName = 'ErrorInt'
	# 					iSkip   = 1

	# 				if self.TrackOptions['ComputeMovingAvg']:
	# 					winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkipError)))
	# 					logging.info('iXC_Track::Computing moving average for window size {} ({} s)...'.format(winSize, self.TrackOptions['MovingAvgWindow']))
	# 					yData = dFRatio[colName].iloc[::iSkip].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
	# 				else:
	# 					yData = dFRatio[colName].iloc[::iSkip].to_numpy()

	# 				iXUtils.CustomPlot(axs[iCol][2], customPlotOpts, xData1[::iSkip], yData)
	# 				iCol = 1

	# 			##--------------------------------------------
	# 			## Plots for 1st/2nd column
	# 			customPlotOpts['xLabel'] = 'None'
	# 			customPlotOpts['yLabel'] = 'Error Signal'
	# 			if self.TrackOptions['PSD_Plot'] or self.TrackOptions['ADev_Plot']:
	# 				customPlotOpts['Legend'] = False
	# 			else:
	# 				customPlotOpts['Legend'] = True

	# 			if self.SoftwareVersion <= 3.2:
	# 				iSkip   = 2
	# 				colName = 'ErrorSignal'
	# 			else:
	# 				iSkip   = 1
	# 				colName = 'Error'

	# 			if self.TrackOptions['ComputeMovingAvg']:
	# 				winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkipError)))
	# 				logging.info('iXC_Track::Computing moving average for window size {} ({} s)...'.format(winSize, self.TrackOptions['MovingAvgWindow']))
	# 				yErrSig = dFRatio[colName].iloc[::iSkip].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
	# 			else:
	# 				yErrSig = dFRatio[colName].iloc[::iSkip].to_numpy()

	# 			iXUtils.CustomPlot(axs[iCol][0], customPlotOpts, xData1[::iSkip], yErrSig)
	# 			plt.setp(axs[iCol][0].get_xticklabels(), visible=False)

	# 			customPlotOpts['yLabel']   = r'$\phi$ (rad)'
	# 			customPlotOpts['Color']    = phaseColors[iax][ik]
	# 			customPlotOpts['eColor']   = phaseColors[iax][ik]
	# 			customPlotOpts['LegLabel'] = phaseLabels[iax][ik]

	# 			if self.SoftwareVersion <= 3.2:
	# 				if ik == 0:
	# 					colName = 'kIndPhaseCent'
	# 				else:
	# 					colName = 'kDepPhaseCent'
	# 			else:
	# 				if ik == 0:
	# 					colName = 'kIndPhase'
	# 				else:
	# 					colName = 'kDepPhase'

	# 			if self.TrackOptions['ComputeMovingAvg']:
	# 				winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkipPhase)))
	# 				logging.info('iXC_Track::Computing moving average for window size {} ({} s)...'.format(winSize, self.TrackOptions['MovingAvgWindow']))
	# 				yData = dFPhase[colName].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
	# 			else:
	# 				yData = dFPhase[colName].to_numpy()

	# 			iXUtils.CustomPlot(axs[iCol][1], customPlotOpts, xData2, yData)
	# 			plt.setp(axs[iCol][1].get_xticklabels(), visible=False)

	# 			if self.TrackMode == 'Open Loop, Fixed Chirp' or self.TrackMode == 'Open Loop, No Chirp':
	# 				yScale = 1.E6/self.gLocal
	# 				yLabel = r'$a_{\rm bias}$ ($\mu$g)'
	# 				customPlotOpts['yLabel'] = yLabel
	# 			else:
	# 				yScale = 1.
	# 				yLabel = r'$a_{\rm bias}$ (m/s$^2$)'
	# 				customPlotOpts['yLabel'] = yLabel

	# 			if self.SoftwareVersion <= 3.2:
	# 				if ik == 0:
	# 					colName = 'kIndAccelBias'
	# 				else:
	# 					colName = 'kDepAccelBias'
	# 			else:
	# 				if ik == 0:
	# 					colName = 'kIndAccel'
	# 				else:
	# 					colName = 'kDepAccel'

	# 			if self.TrackOptions['ComputeMovingAvg']:
	# 				winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkipPhase)))
	# 				logging.info('iXC_Track::Computing moving average for window size {} ({} s)...'.format(winSize, self.TrackOptions['MovingAvgWindow']))
	# 				yData = dFPhase[colName].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
	# 			else:
	# 				yData = dFPhase[colName].to_numpy()

	# 			customPlotOpts['xLabel'] = xLabel
	# 			iXUtils.CustomPlot(axs[iCol][2], customPlotOpts, xData2, yData*yScale)

	# 			##--------------------------------------------
	# 			## Plots for 2nd/3rd column
	# 			if self.TrackOptions['PSD_Plot']:

	# 				plt.rc('lines', linewidth=1.0)
	# 				customPlotOpts['xLabel']      = r'$f$  (Hz)'
	# 				customPlotOpts['yLabel']      = r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)'
	# 				customPlotOpts['Legend']      = True
	# 				customPlotOpts['LegLocation'] = 'lower left'
	# 				customPlotOpts['Linestyle']   = '-'
	# 				customPlotOpts['Marker']      = 'None'
	# 				logScale = [True, True]

	# 				if self.TrackOptions['PSD_Method'] == 'welch':
	# 					if self.TrackOptions['PSD_ShowErrorSignal']:
	# 						fData1, psdData1 = welch(yErrSig/(self.Seff[iax]*self.gLocal), fs=rateError, return_onesided=True, scaling='density')

	# 					fData2, psdData2 = welch(yData/self.gLocal, fs=ratePhase, return_onesided=True, scaling='density')
	# 				else:
	# 					if self.TrackOptions['PSD_ShowErrorSignal']:
	# 						fData1, psdData1 = periodogram(yErrSig/(self.Seff[iax]*self.gLocal), fs=rateError, return_onesided=True, scaling='density')

	# 					fData2, psdData2 = periodogram(yData/self.gLocal, fs=ratePhase, return_onesided=True, scaling='density')

	# 				fData2 = fData2[1:-1]
	# 				psdData2 = np.sqrt(psdData2[1:-1])

	# 				if self.TrackOptions['PSD_ShowErrorSignal']:
	# 					fData1 = fData1[1:-1]
	# 					psdData1 = np.sqrt(psdData1[1:-1])

	# 					customPlotOpts['Color']    = self.DefaultPlotColors[iax][ik]
	# 					customPlotOpts['eColor']   = self.DefaultPlotColors[iax][ik]
	# 					customPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik]+'-Err'

	# 					iXUtils.CustomPlot(axs[cPSD][0], customPlotOpts, fData1, psdData1, [], logScale)
	# 					# axs[cPSD][0].set_xlim(0.75*min(fData2), 1.75*max(fData2))
	# 					# axs[cPSD][0].set_ylim(0.5*min(psdData2), 1.5*max(psdData2))

	# 				customPlotOpts['Color']    = phaseColors[iax][ik]
	# 				customPlotOpts['eColor']   = phaseColors[iax][ik]
	# 				customPlotOpts['LegLabel'] = phaseLabels[iax][ik]
	# 				iXUtils.CustomPlot(axs[cPSD][0], customPlotOpts, fData2, psdData2, [], logScale)

	# 			if self.TrackOptions['ADev_Plot']:

	# 				plt.rc('lines', linewidth=1.5)
	# 				customPlotOpts['xLabel']      = r'$\tau$  (s)'
	# 				customPlotOpts['yLabel']      = r'Allan Deviation  ($g$)'
	# 				customPlotOpts['Legend']      = True
	# 				customPlotOpts['LegLocation'] = 'upper right'
	# 				logScale = [True, True]

	# 				taus    = self.TrackOptions['ADev_taus']
	# 				tauStep = self.TrackOptions['ADev_tauStep']
	# 				if taus == 'octave' or taus == 'decade':
	# 					customPlotOpts['Linestyle'] = 'None'
	# 					customPlotOpts['Marker']    = '.'
	# 				else:
	# 					customPlotOpts['Linestyle'] = '-'
	# 					customPlotOpts['Marker']    = 'None'

	# 				(tau, aDev, aDevSigL, aDevSigU) = iXUtils.AllanDev(yData/self.gLocal, taus=taus, rate=ratePhase, 
	# 					ADevType='Total', ComputeErr=self.TrackOptions['ADev_ShowErrors'])
	# 				x, y, dy = tau[::tauStep], aDev[::tauStep], np.array([aDevSigL[::tauStep], aDevSigU[::tauStep]])
	# 				if self.TrackOptions['ADev_ShowErrors']:
	# 					if self.TrackOptions['ADev_Errorstyle'] == 'Bar':
	# 						iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=dy, LogScale=logScale)
	# 					else:
	# 						iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=[], LogScale=logScale)
	# 						axs[cADev][0].fill_between(x, y - dy[0], y + dy[1], color=customPlotOpts['eColor'], alpha=0.3)
	# 				else:
	# 					iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=[], LogScale=logScale)

	# 				if self.TrackOptions['ADev_ShowErrorSignal']:
	# 					customPlotOpts['Color']    = self.DefaultPlotColors[iax][ik]
	# 					customPlotOpts['eColor']   = self.DefaultPlotColors[iax][ik]
	# 					customPlotOpts['LegLabel'] = self.AxisLegLabels[iax][ik]+'-Err'

	# 					(tau, aDev, aDevSigL, aDevSigU) = iXUtils.AllanDev(yErrSig/self.gLocal, taus=taus, rate=rateError, 
	# 							ADevType='Total', ComputeErr=self.TrackOptions['ADev_ShowErrors'])
	# 					x, y, dy = tau[::tauStep], aDev[::tauStep], np.array([aDevSigL[::tauStep], aDevSigU[::tauStep]])
	# 					if self.TrackOptions['ADev_ShowErrors']:
	# 						if self.TrackOptions['ADev_Errorstyle'] == 'Bar':
	# 							iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=dy, LogScale=logScale)
	# 						else:
	# 							iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=[], LogScale=logScale)
	# 							axs[cADev][0].fill_between(x, y - dy[0], y + dy[1], color=customPlotOpts['eColor'], alpha=0.3)
	# 					else:
	# 						iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=[], LogScale=logScale)

	# 	if self.TrackOptions['SubtractTideModel']:

	# 		# Location = {
	# 		# 	'Latitude':		44.804,	## [deg]
	# 		# 	'Longitude': 	-0.605,	## [deg]
	# 		# 	'Height':		21.		## [m] 
	# 		# 	}

	# 		Location = {
	# 			'Latitude':		self.Phys.Latitude,
	# 			'Longitude': 	self.Phys.Longitude,
	# 			'Height':		self.Phys.Height
	# 			}

	# 		dateStamp, timeStamp = dFPhase[['Date', 'Time']].iloc[0]
	# 		tStart = iXUtils.TimestampToDatetime(dateStamp, timeStamp).timestamp()

	# 		dateStamp, timeStamp = dFPhase[['Date', 'Time']].iloc[-1]
	# 		tStop  = iXUtils.TimestampToDatetime(dateStamp, timeStamp).timestamp()

	# 		tData  = np.linspace(tStart, tStop, num=dFPhase.shape[0], endpoint=True)
	# 		gTide  = self.Phys.GetTideModel(tData, Location, Recompute=self.TrackOptions['RecomputeTideModel'])
	# 		yData -= gTide

	# 		plt.rc('lines', linewidth=1.0)
	# 		customPlotOpts = {'Color': 'blue', 'eColor': 'blue', 'Linestyle': '-', 'Marker': 'None',
	# 			'Title': 'None', 'xLabel': xLabel, 'yLabel': yLabel, 'Legend': True, 'LegLabel': r'Z-$k_{\rm dep}$-Tides'}
	# 		iXUtils.CustomPlot(axs[iCol][2], customPlotOpts, xData2, yData*yScale)

	# 		plt.rc('lines', linewidth=1.5)
	# 		customPlotOpts['Color']    = 'red'
	# 		customPlotOpts['eColor']   = 'red'
	# 		customPlotOpts['LegLabel'] = 'Tides'
	# 		iXUtils.CustomPlot(axs[iCol][2], customPlotOpts, xData2, gTide*yScale)

	# 		if self.TrackOptions['ADev_Plot']:
	# 			customPlotOpts['Color']       = 'blue'
	# 			customPlotOpts['eColor']      = 'blue'
	# 			customPlotOpts['xLabel']      = r'$\tau$  (s)'
	# 			customPlotOpts['yLabel']      = r'Allan Deviation  ($g$)'
	# 			customPlotOpts['LegLabel']    = r'Z-$k_{\rm dep}$-Tides'
	# 			customPlotOpts['Legend']      = True
	# 			customPlotOpts['LegLocation'] = 'upper right'
	# 			logScale = [True, True]

	# 			taus    = self.TrackOptions['ADev_taus']
	# 			tauStep = self.TrackOptions['ADev_tauStep']
	# 			if taus == 'octave' or taus == 'decade':
	# 				customPlotOpts['Linestyle'] = 'None'
	# 				customPlotOpts['Marker']    = '.'
	# 			else:
	# 				customPlotOpts['Linestyle'] = '-'
	# 				customPlotOpts['Marker']    = 'None'

	# 			(tau, aDev, aDevSigL, aDevSigU) = iXUtils.AllanDev(yData/self.gLocal, taus=taus, rate=ratePhase, 
	# 					ADevType='Total', ComputeErr=self.TrackOptions['ADev_ShowErrors'])
	# 			x, y, dy = tau[::tauStep], aDev[::tauStep], np.array([aDevSigL[::tauStep], aDevSigU[::tauStep]])
	# 			if self.TrackOptions['ADev_ShowErrors']:
	# 				if self.TrackOptions['ADev_Errorstyle'] == 'Bar':
	# 					iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=dy, LogScale=logScale)
	# 				else:
	# 					iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=[], LogScale=logScale)
	# 					axs[cADev][0].fill_between(x, y - dy[0], y + dy[1], color=customPlotOpts['eColor'], alpha=0.3)
	# 			else:
	# 				iXUtils.CustomPlot(axs[cADev][0], customPlotOpts, x, y, yErr=[], LogScale=logScale)

	# 	if self.TrackOptions['SetPlotXLimits']:
	# 		## Set xlimits
	# 		[xMin, xMax] = self.TrackOptions['PlotXLimits']
	# 		for row in range(3):
	# 			for col in range (2):
	# 				axs[col][row].set_xlim(xMin,xMax)

	# 	if self.PlotOptions['ShowPlotLegend'] and self.PlotOptions['FixLegLocation']:
	# 		## Fix legend location to upper right of last plot
	# 		cLast = len(axs)-1
	# 		nRows = len(axs[cLast])
	# 		for row in range(nRows):
	# 			axs[cLast][row].legend(loc='upper left', bbox_to_anchor=(1.05,1.0))

	# 	if self.PlotOptions['SavePlot']:
	# 		if not os.path.exists(self.PlotOptions['PlotFolderPath']):
	# 			os.makedirs(self.PlotOptions['PlotFolderPath'])

	# 		self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
	# 			'Track-Run{:02d}-TrackSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'])

	# 		plt.savefig(self.PlotPath, dpi=150)
	# 		logging.info('iXC_Track::Tracking data plot saved to:')
	# 		logging.info('iXC_Track::  {}'.format(self.PlotPath))
	# 	elif self.PlotOptions['ShowPlot']:
	# 		plt.show()

	# ################# End of Track.PlotTrackData() ##################
	# #################################################################

	def PlotRatioData(self):
		"""Plot time-series analysis of ratio data."""

		if self.RatioFilesFound:
			logging.info('iXC_Track::Plotting ratio data for {}...'.format(self.RunString))

			if self.SoftwareVersion <= 3.2:
				iSkip = 2*self.nk*self.nax
			else:
				iSkip = self.nk*self.nax

			ratio  = [[[] for ik in range(2)] for iax in range(3)]
			phase  = [[[] for ik in range(2)] for iax in range(3)]
			errSig = [[[] for ik in range(2)] for iax in range(3)]
			errInt = [[[] for ik in range(2)] for iax in range(3)]

			nData  = int(1E10)
			for iax in self.iaxList:
				for ik in self.ikList:
					df = self.SetTrackTimeRange(self.RatioDF[iax][ik])
					nData  = min(nData, df.shape[0])

					if iax == self.iaxList[0] and ik == self.ikList[0]:
						iStart = self.IterStart
						if self.tStop > 10000:
							xScale = 1.E-3
							xLabel = r'Time  ($\times 10^3$ s)'
						else:
							xScale = 1.
							xLabel = 'Time  (s)'

					if self.TrackOptions['ComputeMovingAvg']:
						winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkip)))
						logging.info('iXC_Track::Computing moving average for window size {} ({:.1f} s)...'.format(winSize, winSize*self.tCycMeanFull*iSkip))
						ratio[iax][ik] = df['Ratio'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
						phase[iax][ik] = (df['CurrentPhase'] - df['ModPhase']).rolling(winSize, center=True, min_periods=1).mean().to_numpy()

						if 'Error' not in df.columns:
							## Use old column header ('ErrorSignal') and compute ErrorInt from cumulative sum
							errSig[iax][ik] = df['ErrorSignal'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
							errInt[iax][ik] = df['ErrorSignal'].cumsum().rolling(winSize, center=True, min_periods=1).mean().to_numpy()
						else:
							errSig[iax][ik] = df['Error'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
							errInt[iax][ik] = df['ErrorInt'].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					else:
						ratio[iax][ik] = dFRatio['Ratio'].to_numpy()
						phase[iax][ik] = (df['CurrentPhase'] - df['ModPhase']).to_numpy()

						if 'Error' not in df.columns:
							## Use old column header ('ErrorSignal') and compute ErrorInt from cumulative sum
							errSig[iax][ik] = df['ErrorSignal'].to_numpy()
							errInt[iax][ik] = df['ErrorSignal'].cumsum().to_numpy()
						else:
							errSig[iax][ik] = df['Error'].to_numpy()
							errInt[iax][ik] = df['ErrorInt'].to_numpy()

			tRange  = np.array([iStart, iStart + nData*iSkip, iSkip])*self.tCycMeanFull*xScale
			yData   = [
				[ratio[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList],
				[errSig[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList],
				[errInt[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList],
				[phase[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList]]
			yErr    = [[] for _ in range(4)]
			yScales = [np.ones(self.nax*self.nk) for _ in range(4)]
			colors  = [[self.DefaultPlotColors[iax][ik] for iax in self.iaxList for ik in self.ikList] for _ in range(4)]
			lLabels = [[self.AxisLegLabels[iax][ik] for iax in self.iaxList for ik in self.ikList] for _ in range(4)]
			PSD_PlotSubSets = [
				[False for _ in range(self.nax*self.nk)],
				[self.TrackOptions['PSD_ShowErrorSignal'] for _ in range(self.nax*self.nk)],
				[False for _ in range(self.nax*self.nk)],
				[True for _ in range(self.nax*self.nk)]]
			ADev_PlotSubSets = [
				[False for _ in range(self.nax*self.nk)],
				[self.TrackOptions['ADev_ShowErrorSignal'] for _ in range(self.nax*self.nk)],
				[False for _ in range(self.nax*self.nk)],
				[True for _ in range(self.nax*self.nk)]]
			ADev_Fit = [
				[False for _ in range(self.nax*self.nk)],
				[False for _ in range(self.nax*self.nk)],
				[False for _ in range(self.nax*self.nk)],
				[self.TrackOptions['ADev_Fit'] for _ in range(self.nax*self.nk)]]

			Options = {
				'SavePlot'			: self.PlotOptions['SavePlot'],
				'PlotFolderPath'	: self.PlotOptions['PlotFolderPath'],
				'PlotFileName'		: 'Track-Run{:02d}-AIRatioDataSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
				'ColumnDim'			: (6, 6),
				'Colors'			: colors,
				'Linestyle'			: '-',
				'Linewidth'			: 1.,
				'Marker'			: 'None',
				'Markersize'		: 6, 
				'ShowErrors'		: False,
				'SampleRate'		: 1./(iSkip*self.tCycMeanFull),
				'xLabels'			: ['None', 'None', 'None', xLabel],
				'yLabels'        	: [r'$N_2/N_{\rm total}$', 'Error Sig.', 'Error Int.', 'Phase  (rad)', r'$\sqrt{\rm PSD}$  (rad$/\sqrt{\rm Hz}$)', r'Allan Deviation  (rad)'],
				'yScales'			: yScales,
				'ShowFigureLabels'  : self.PlotOptions['ShowFigureLabels'],
				'FigLabels'			: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
				'ShowLegend'		: [False, True, True],
				'LegendLabels'		: lLabels,
				'LegendLocations'	: ['outside', 'best', 'best'],
				'LegendFontSize'	: 10,
				'SetPlotLimits'		: [False, False],
				'PlotXLimits'		: [-2500., 1.2*tRange[1]],
				'PlotYLimits'		: [[0.5,0.9], [0.5,0.9]],
				'PSD_Plot'			: self.TrackOptions['PSD_Plot'],
				'PSD_PlotSubSets'	: PSD_PlotSubSets,
				'PSD_Method'		: self.TrackOptions['PSD_Method'],
				'ADev_Plot'			: self.TrackOptions['ADev_Plot'],
				'ADev_PlotSubSets'  : ADev_PlotSubSets,
				'ADev_Type'			: 'Total',
				'ADev_taus'			: self.TrackOptions['ADev_taus'],
				'ADev_ShowErrors'	: self.TrackOptions['ADev_ShowErrors'],
				'ADev_Errorstyle'	: 'Shaded', # 'Bar' or 'Shaded'
				'ADev_Linestyle' 	: '-',
				'ADev_Marker'    	: 'None',
				'ADev_SetLimits'	: [False, False],
				'ADev_XLimits'		: [1.E2, 4.E4],
				'ADev_YLimits'		: [1.E-8, 1.E-6],
				'ADev_Fit'			: ADev_Fit,
				'ADev_Fit_FixExp'	: ADev_Fit,
				'ADev_Fit_XLimits'	: [0.9*tRange[2], 1.1*(0.4*tRange[1])],
				'ADev_Fit_SetRange'	: [[False for j in range(self.nax*self.nk)] for r in range(4)],
				'ADev_Fit_Range'	: [[[0., 2.E3] for j in range(self.nax*self.nk)] for r in range(4)]
				}

			iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)
		else:
			logging.error('iXC_Track::PlotRatioData::Phase data for {} not found...'.format(self.RunString))
			logging.error('iXC_Track::PlotRatioData::Aborting...')
			quit()

	################# End of Track.PlotRatioData() ##################
	#################################################################

	def PlotPhaseData(self):
		"""Plot time-series analysis of phase data."""

		if self.PhaseFileFound:
			logging.info('iXC_Track::Plotting phase data for {}...'.format(self.RunString))

			if 'kDepPhase' not in self.PhaseDF[self.iaxList[0]].columns:
				## Use old header names:
				headers = ['kDepPhaseCent','kIndPhaseCent','kDepAccelBias','kIndAccelBias']
			else:
				## Use new headers names:
				headers = ['kDepPhase','kIndPhase','kDepAccel','kIndAccel']

			if self.SoftwareVersion <= 3.2:
				## Previous software versions update kDep and kInd phases only every 2 cycles (every 4 shots in k-Interlaced mode)
				iSkip = 2*self.nk*self.nax
			else:
				iSkip = self.nk*self.nax

			ikDepList = self.iaxList.copy()
			ikIndList = self.iaxList.copy()

			if self.nax == 3:
				## Add element for |g|
				ikDepList += [3]

			if self.TrackOptions['SubtractTideModel']:
				## Add elements for atide and az-atide
				ikDepList += [4,5]

			nkDep = len(ikDepList)
			nkInd = len(ikIndList)

			kDepPhase = [[] for iax in range(6)] ## [phix,phiy,phiz,phi|g|,phitide,phiz-phitide]
			kIndPhase = [[] for iax in range(3)] ## [phix,phiy,phiz]
			kDepAccel = [[] for iax in range(6)] ## [ax,ay,az,|g|,atide,az-atide]
			kIndAccel = [[] for iax in range(3)] ## [ax,ay,az]

			nData = int(1E10)
			for iax in self.iaxList:
				df = self.SetTrackTimeRange(self.PhaseDF[iax])
				dateStamp, timeStamp = df[['Date', 'Time']].iloc[0]
				tAbsStart = iXUtils.TimestampToDatetime(dateStamp, timeStamp).timestamp()

				dateStamp, timeStamp = df[['Date', 'Time']].iloc[-1]
				tAbsStop  = iXUtils.TimestampToDatetime(dateStamp, timeStamp).timestamp()

				# if self.SoftwareVersion >= 3.3:
				# 	## New software versions update kDep phase every cycle, this rolling average renders the data similar to
				# 	## previous software formats where kDep phase was updated every 2 cycles.
				# 	df = df[headers].rolling(window=2, min_periods=1).mean().iloc[1::2]

				nData = min(nData, df.shape[0])

				if iax == self.iaxList[0]:
					iStart = self.IterStart
					if self.tStop > 10000:
						xScale = 1.E-3
						xLabel = r'Time  ($\times 10^3$ s)'
					else:
						xScale = 1.
						xLabel = 'Time  (s)'

				if self.TrackOptions['ComputeMovingAvg']:
					winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkip)))
					logging.info('iXC_Track::Computing moving average for window size {} ({:.1f} s)...'.format(winSize, winSize*self.tCycMeanFull*iSkip))
					kDepPhase[iax] = df[headers[0]].rolling(winSize, center=True, min_periods=1).mean().to_numpy() ## rad
					kIndPhase[iax] = df[headers[1]].rolling(winSize, center=True, min_periods=1).mean().to_numpy() ## rad
					kDepAccel[iax] = df[headers[2]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()/self.gLocal ## g
					kIndAccel[iax] = df[headers[3]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()/self.gLocal ## g
				else:
					kDepPhase[iax] = df[headers[0]].to_numpy() ## rad
					kIndPhase[iax] = df[headers[1]].to_numpy() ## rad
					kDepAccel[iax] = df[headers[2]].to_numpy()/self.gLocal ## g
					kIndAccel[iax] = df[headers[3]].to_numpy()/self.gLocal ## g

			for iax in self.iaxList:
				kDepPhase[iax] = kDepPhase[iax][:nData]
				kIndPhase[iax] = kIndPhase[iax][:nData]
				kDepAccel[iax] = kDepAccel[iax][:nData]
				kIndAccel[iax] = kIndAccel[iax][:nData]

			## Misalignment coefficients
			cm = 1. + self.TrackOptions['AxisMisalignments']

			if self.nax == 3:
				kDepPhase[3] = np.sqrt((cm[0]*kDepPhase[0])**2 + (cm[1]*kDepPhase[1])**2 + (cm[2]*kDepPhase[2])**2)
				kDepAccel[3] = np.sqrt((cm[0]*kDepAccel[0])**2 + (cm[1]*kDepAccel[1])**2 + (cm[2]*kDepAccel[2])**2)

			if self.TrackOptions['SubtractTideModel']:
				Location = {
					'Latitude':		self.Phys.Latitude,
					'Longitude': 	self.Phys.Longitude,
					'Height':		self.Phys.Height
					}

				tData 		 = np.linspace(tAbsStart, tAbsStop, num=nData, endpoint=True)
				kDepAccel[4] = self.Phys.GetTideModel(tData, Location, Recompute=self.TrackOptions['RecomputeTideModel'])/self.gLocal ## g
				if self.nax == 3:
					kDepAccel[5] = kDepAccel[3] - kDepAccel[4]
				else:
					kDepAccel[5] = kDepAccel[2] - kDepAccel[4]

				kDepPhase[4] = kDepAccel[4]*self.Seff[2]
				kDepPhase[5] = kDepAccel[5]*self.Seff[2]

			tRange = np.array([iStart, iStart + nData*iSkip, iSkip])*self.tCycMeanFull*xScale

			if self.TrackOptions['AccelUnit'] == 'ug':
				aScale  = 1.E6
				aUnit   = r'($\mu$g)'
			else:
				aScale  = self.gLocal
				aUnit   = r'(m/s$^2$)'

			if self.TrackOptions['PlotQuantity'] == 'AllPhases':
				yData   = [[kDepPhase[iax]] for iax in ikDepList] + [[kIndPhase[iax] for iax in ikIndList]]
				yScales = [[1.] for iax in ikDepList] + [[1. for iax in ikIndList]]
				yLabels = [r'$\phi_x^{\rm dep}$  (rad)', r'$\phi_y^{\rm dep}$  (rad)', r'$\phi_z^{\rm dep}$  (rad)', r'$|\mathbf{\phi}^{\rm dep}|$  (rad)', r'$\phi_z^{\rm dep} - \phi_z^{\rm tide}$  (rad)']
				yLabels = [yLabels[iax] for iax in ikDepList] + [r'$a^{\rm ind}$  (m/s$^2$)', r'$\sqrt{\rm PSD}$  (rad$/\sqrt{\rm Hz}$)', r'Allan Deviation  (rad)']
				kDepLab = [r'$\phi_x^{\rm dep}$', r'$\phi_y^{\rm dep}$', r'$\phi_z^{\rm dep}$', r'$|\mathbf{\phi}^{\rm dep}|$', r'$\phi_z^{\rm tide}$  (rad)', r'$\phi_z^{\rm dep} - \phi_z^{\rm tide}$  (rad)']
				kIndLab = [r'$\phi_x^{\rm ind}$', r'$\phi_y^{\rm ind}$', r'$\phi_z^{\rm ind}$']
				lLabels = [[kDepLab[iax]] for iax in ikDepList] + [[kIndLab[iax] for iax in ikIndList]]
			elif self.TrackOptions['PlotQuantity'] == 'kDepPhases':
				yData   = [[kDepPhase[iax]] for iax in ikDepList]
				yScales = [[1.] for iax in ikDepList]
				yLabels = [r'$\phi_x^{\rm dep}$  (rad)', r'$\phi_y^{\rm dep}$  (rad)', r'$\phi_z^{\rm dep}$  (rad)', r'$|\mathbf{\phi}^{\rm dep}|$  (rad)']
				yLabels = [yLabels[iax] for iax in ikDepList] + [r'$\sqrt{\rm PSD}$  (rad$/\sqrt{\rm Hz}$)', r'Allan Deviation  (rad)']
				lLabels = [r'$\phi_x^{\rm dep}$', r'$\phi_y^{\rm dep}$', r'$\phi_z^{\rm dep}$', r'$|\mathbf{\phi}^{\rm dep}|$']
				lLabels = [[lLabels[iax]] for iax in ikDepList]
			elif self.TrackOptions['PlotQuantity'] == 'AllAccels':
				yData   = [[kDepAccel[iax]] for iax in ikDepList] + [[kIndAccel[iax] for iax in ikIndList]]
				yScales = [[aScale] for iax in ikDepList] + [[aScale for iax in ikIndList]]
				yLabels = [r'$a_x^{\rm dep}$  '+aUnit, r'$a_y^{\rm dep}$  '+aUnit, r'$a_z^{\rm dep}$  '+aUnit, r'$|\mathbf{a}^{\rm dep}|$  '+aUnit, r'$a_z^{\rm tide}$  '+aUnit, r'$|\mathbf{a}^{\rm dep}| - a_z^{\rm tide}$  '+aUnit]
				yLabels = [yLabels[iax] for iax in ikDepList] + [r'$a^{\rm ind}$  '+aUnit, r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)']
				kDepLab = [r'$a_x^{\rm dep}$', r'$a_y^{\rm dep}$', r'$a_z^{\rm dep}$', r'$|\mathbf{a}^{\rm dep}|$', r'$a_z^{\rm tide}$', r'$a_z^{\rm dep} - a_z^{\rm tide}$']
				kIndLab = [r'$a_x^{\rm ind}$', r'$a_y^{\rm ind}$', r'$a_z^{\rm ind}$']
				lLabels = [[kDepLab[iax]] for iax in ikDepList] + [[kIndLab[iax] for iax in ikIndList]]
			else: ## kDepAccels
				if self.TrackOptions['SubtractTideModel']:
					yLabels = [r'$a_x$  '+aUnit, r'$a_y$  '+aUnit, r'$a_z$  '+aUnit, r'$|\mathbf{a}|$  '+aUnit, r'$a^{\rm tide}$  '+aUnit, r'$|\mathbf{a}| - a^{\rm tide}$  '+aUnit]
					lLabels = [r'$a_x$', r'$a_y$', r'$a_z$', r'$|\mathbf{a}|$', r'$a^{\rm tide}$', r'$|\mathbf{a}| - a^{\rm tide}$']
				else:
					yLabels = [r'$a_x$  '+aUnit, r'$a_y$  '+aUnit, r'$a_z$  '+aUnit, r'$|\mathbf{a}|$  '+aUnit]
					lLabels = [r'$a_x$', r'$a_y$', r'$a_z$', r'$|\mathbf{a}|$']
				yData   = [[kDepAccel[iax]] for iax in ikDepList]
				yScales = [[aScale] for iax in ikDepList]
				yLabels = [yLabels[iax] for iax in ikDepList] + [r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)']
				lLabels = [[lLabels[iax]] for iax in ikDepList]

			xLabels = ['None' for r in range(len(yData)-1)] + [xLabel]
			kDepCol = [self.DefaultPlotColors[iax][0] for iax in range(3)] + ['black','orange','blue'] ## [ax,ay,az,|g|,atide,|g|-atide]

			yErr   	= [[[]] for iax in ikDepList]
			colors 	= [[kDepCol[iax]] for iax in ikDepList]

			if self.TrackOptions['SubtractTideModel']:
				mask 			  = [True, True, True, True, False, True] ## [ax,ay,az,|g|,atide,|g|-atide]
				PSD_PlotSubSets   = [[mask[iax]] for iax in ikDepList]
				ADev_PlotSubSets  = [[mask[iax]] for iax in ikDepList]
				mask 			  = [False, False, False, False, False, self.TrackOptions['ADev_Fit']] ## [ax,ay,az,|g|,atide,|g|-atide]				
				ADev_Fit          = [[mask[iax]] for iax in ikDepList]
				mask 			  = [False, False, False, False, False, self.TrackOptions['ADev_Fit_FitExp']] ## [ax,ay,az,|g|,atide,|g|-atide]				
				ADev_Fit_FitExp   = [[mask[iax]] for iax in ikDepList]
				mask 			  = [False, False, False, False, False, self.TrackOptions['ADev_Fit_SetRange']] ## [ax,ay,az,|g|,atide,|g|-atide]				
				ADev_Fit_SetRange = [[mask[iax]] for iax in ikDepList]
				ADev_Fit_Range    = [[self.TrackOptions['ADev_Fit_Range']] for iax in ikDepList]
			else:
				PSD_PlotSubSets   = [[True] for iax in ikDepList]
				ADev_PlotSubSets  = [[True] for iax in ikDepList]
				ADev_Fit          = [[self.TrackOptions['ADev_Fit']] for iax in ikDepList]
				ADev_Fit_FitExp   = [[self.TrackOptions['ADev_Fit_FitExp']] for iax in ikDepList]
				ADev_Fit_SetRange = [[self.TrackOptions['ADev_Fit_SetRange']] for iax in ikDepList]
				ADev_Fit_Range    = [[self.TrackOptions['ADev_Fit_Range']] for iax in ikDepList]

			## Add elements for kInd quantities
			if self.TrackOptions['PlotQuantity'] == 'AllPhases' or self.TrackOptions['PlotQuantity'] == 'AllAccels':
				yErr   			  += [[[] for iax in ikIndList]]
				colors 			  += [[self.DefaultPlotColors[iax][1] for iax in ikIndList]]
				PSD_PlotSubSets   += [[True for iax in ikIndList]]
				ADev_PlotSubSets  += [[True for iax in ikIndList]]
				ADev_Fit          += [[False for iax in ikIndList]]
				ADev_Fit_FitExp   += [[False for iax in ikIndList]]
				ADev_Fit_SetRange += [[False for iax in ikIndList]]
				ADev_Fit_Range    += [[[0., 2.E3] for iax in ikIndList]]

			Options = {
				'SavePlot'			: self.PlotOptions['SavePlot'],
				'PlotFolderPath'	: self.PlotOptions['PlotFolderPath'],
				'PlotFileName'		: 'Track-Run{:02d}-AIPhaseDataSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
				'ColumnDim'			: (6, 6),
				'Colors'			: colors,
				'Linestyle'			: '-',
				'Linewidth'			: 1.,
				'Marker'			: 'None',
				'Markersize'		: 6, 
				'ShowErrors'		: False,
				'SampleRate'		: 1./(iSkip*self.tCycMeanFull),
				'xLabels'			: xLabels,
				'yLabels'        	: yLabels,
				'yScales'			: yScales,
				'ShowFigureLabels'  : self.PlotOptions['ShowFigureLabels'],
				'FigLabels'			: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
				'ShowLegend'		: [False, True, True],
				'LegendLabels'		: lLabels,
				'LegendLocations'	: ['outside', 'best', 'best'],
				'LegendFontSize'	: 14,
				'SetPlotLimits'		: [False, False],
				'PlotXLimits'		: [-2500., 1.2*tRange[1]],
				'PlotYLimits'		: [[0.5,0.9], [0.5,0.9]],
				'PSD_Plot'			: self.TrackOptions['PSD_Plot'],
				'PSD_PlotSubSets'	: PSD_PlotSubSets,
				'PSD_Method'		: self.TrackOptions['PSD_Method'],
				'ADev_Plot'			: self.TrackOptions['ADev_Plot'],
				'ADev_PlotSubSets'  : ADev_PlotSubSets,
				'ADev_Type'			: 'Total',
				'ADev_taus'			: self.TrackOptions['ADev_taus'],
				'ADev_ShowErrors'	: self.TrackOptions['ADev_ShowErrors'],
				'ADev_Errorstyle'	: self.TrackOptions['ADev_Errorstyle'], ## 'Bar' or 'Shaded'
				'ADev_Linestyle' 	: '-' if self.TrackOptions['ADev_Errorstyle'] == 'Shaded' else 'None',
				'ADev_Marker'    	: 'None' if self.TrackOptions['ADev_Errorstyle'] == 'Shaded' else '.',
				'ADev_SetLimits'	: [False, False],
				'ADev_XLimits'		: [1.E2, 4.E4],
				'ADev_YLimits'		: [1.E-7, 1.E-5],
				'ADev_Fit'			: ADev_Fit,
				'ADev_Fit_FixExp'	: ADev_Fit_FitExp,
				'ADev_Fit_XLimits'	: [1.E2, 1.E4],
				'ADev_Fit_SetRange'	: ADev_Fit_SetRange,
				'ADev_Fit_Range'	: ADev_Fit_Range
				}

			self.PrintStats(yData, ikDepList, ikIndList)
			iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)
		else:
			logging.error('iXC_Track::PlotPhaseData::Phase data for {} not found...'.format(self.RunString))
			logging.error('iXC_Track::PlotPhaseData::Aborting...')
			quit()

	################# End of Track.PlotPhaseData() ##################
	#################################################################

	def PrintStats(self, yData, ikDepList, ikIndList):

		if self.TrackOptions['PlotQuantity'] == 'AllPhases' or self.TrackOptions['PlotQuantity'] == 'kDepPhases':
			label = 'Phase'
			unit  = ' rad'
		else:
			label = 'Accel'
			unit  = ' ug'
		axLabels = ['X','Y','Z','|g|','tide','Z - tide']

		nData = len(yData[0][0])
		print('------------------- kDep Statistics ------------------')
		for r in range(len(yData[0])):
			iax = ikDepList[r]
			print(label+' '+axLabels[iax]+' Mean = {:.3E}'.format(np.mean(yData[0][r])*1.E6)+unit)
			print(label+' '+axLabels[iax]+' SDev = {:.3E}'.format(np.std(yData[0][r])*1.E6)+unit)
			# print(label+' '+axLabels[iax]+'-Dep SErr = {:.3E}'.format(np.std(yData[0][r])/np.sqrt(nData)*1.E6)+unit)
		if self.TrackOptions['PlotQuantity'] == 'AllPhases' or self.TrackOptions['PlotQuantity'] == 'AllAccels':
			print('------------------- kInd Statistics ------------------')
			for r in range(len(yData[1])):
				iax = ikIndList[r]
				print(label+' '+axLabels[iax]+' Mean = {:.3E}'.format(np.mean(yData[1][r])*1.E6)+unit)
				print(label+' '+axLabels[iax]+' SDev = {:.3E}'.format(np.std(yData[1][r])*1.E6)+unit)
				# print(label+' '+axLabels[iax]+'-Ind SErr = {:.3E}'.format(np.std(yData[1][r])/np.sqrt(nData)*1.E6)+unit)
			print('------------------------------------------------------')

	################## End of Track.PrintStats() ####################
	#################################################################

	def PlotAccelMean(self):
		"""Plot summary of mean accelerometer data."""

		if self.RTAccelFileFound:
			logging.info('iXC_Track::Plotting mean accel data for {}...'.format(self.RunString))

			dFAccel = self.SetTrackTimeRange(self.RTAccelDF)

			tRange  = np.array([dFAccel['#Iteration'].iloc[0], dFAccel['#Iteration'].iloc[-1], 1.])*self.tCycMeanFull
			axData  = np.array(dFAccel['AccelMean_X'])
			ayData  = np.array(dFAccel['AccelMean_Y'])
			azData  = np.array(dFAccel['AccelMean_Z'])
			gData   = np.sqrt(axData**2 + ayData**2 + azData**2)
			nData   = len(axData)
			yData   = [[axData], [ayData], [azData], [gData]]
			yErr    = [np.zeros(nData) for _ in range(4)]
			yScales = [[1.], [1.], [1.], [1.]]

			Options = {
				'SavePlot'			: self.PlotOptions['SavePlot'],
				'PlotFolderPath'	: self.PlotOptions['PlotFolderPath'],
				'PlotFileName'		: 'Track-Run{:02d}-RTAccelSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
				'ColumnDim'			: (5, 6),
				'Colors'			: [['green'], ['royalblue'], ['red'], ['black']],
				'Linestyle'			: '-',
				'Linewidth'			: 1.,
				'Marker'			: 'None',
				'Markersize'		: 6, 
				'ShowErrors'		: True,
				'xLabels'			: ['None', 'None', 'None', 'Time (s)'],
				'yLabels'        	: [r'$a_x$  (m/s$^2$)', r'$a_y$  (m/s$^2$)', r'$a_z$  (m/s$^2$)',
					r'$|g|$  (m/s$^2$)', r'PSD  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)'],
				'yScales'			: yScales,
				'ShowFigureLabels'  : self.PlotOptions['ShowFigureLabels'],
				'FigLabels'			: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
				'ShowLegend'		: [False, False, True],
				'LegendLabels'		: [[r'$a_x$'], [r'$a_y$'], [r'$a_z$'], [r'$|g|$']],
				'LegendLocations'	: ['best', 'best', 'best'],
				'LegendFontSize'	: 10,
				'SetPlotLimits'		: [False, False],
				'PlotXLimits'		: [-2500., 1.2*tRange[1]],
				'PlotYLimits'		: [[0.5,0.9], [0.5,0.9]],
				'PSD_Plot'			: self.TrackOptions['PSD_Plot'],
				'PSD_PlotSubSets'	: [[True], [True], [True], [True]],
				'PSD_Method'		: self.TrackOptions['PSD_Method'],
				'ADev_Plot'			: self.TrackOptions['ADev_Plot'],
				'ADev_PlotSubSets'  : [[True], [True], [True], [True]],
				'ADev_Type'			: 'Total',
				'ADev_taus'			: self.TrackOptions['ADev_taus'],
				'ADev_Rate'			: 1./self.tCycMeanFull,
				'ADev_ShowErrors'	: self.TrackOptions['ADev_ShowErrors'],
				'ADev_Errorstyle'	: 'Shaded', # 'Bar' or 'Shaded'
				'ADev_Linestyle' 	: '-',
				'ADev_Marker'    	: 'None',
				'ADev_SetLimits'	: [False, False],
				'ADev_XLimits'		: [1.E2, 4.E4],
				'ADev_YLimits'		: [1.E-8, 1.E-6],
				'ADev_Fit'			: [[self.TrackOptions['ADev_Fit']], [self.TrackOptions['ADev_Fit']], [self.TrackOptions['ADev_Fit']], [self.TrackOptions['ADev_Fit']]],
				'ADev_Fit_XLimits'	: [0.9*tRange[2], 1.1*(0.4*tRange[1])],
				'ADev_Fit_SetRange'	: [[False], [False], [False], [False]],
				'ADev_Fit_Range'	: [[0., 2.E3], [0., 2.E3], [0., 2.E3], [0., 2.E3]],
				'ADev_Fit_FixExp'	: [[True], [True], [True], [True]]
				}

			iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)
		else:
			logging.error('iXC_Track::PlotAccelMean::Accel data for {} not found...'.format(self.RunString))
			logging.error('iXC_Track::PlotAccelMean::Aborting...')
			quit()

	################# End of Track.PlotAccelMean() ##################
	#################################################################

	def PlotCorrelations(self):
		"""Plot correlations with monitor data."""

		logging.info('iXC_Track::Plotting correlations with monitor data for {}...'.format(self.RunString))

		## Load monitor data
		Mon = iXC_Monitor.Monitor(self.WorkDir, self.Folder, [self.Run], self.MonitorOptions, self.PlotOptions, False, self.__dict__.items())

		if Mon.MonitorFileFound:
			Mon.ProcessMonitorData(self.TrackOptions['ComputeMovingAvg'], self.TrackOptions['MovingAvgWindow'])

			if self.SoftwareVersion <= 3.2:
				colNames = ['kUPhaseCent', 'kDPhaseCent', 'kIndAccelBias', 'kDepAccelBias']
			else:
				colNames = ['kUPhase', 'kDPhase', 'kIndAccel', 'kDepAccel']

			for iax in self.iaxList:
				fringeDF = self.SetTrackTimeRange(self.PhaseDF[iax])

				nData  = fringeDF['#Iteration'].shape[0]
				tStart = self.dtStartFull.timestamp() + fringeDF['#Iteration'].iloc[ 0]*self.tCycMeanFull
				tStop  = self.dtStartFull.timestamp() + fringeDF['#Iteration'].iloc[-1]*self.tCycMeanFull
				tStep  = (tStop - tStart)/(nData-1)

				t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
				xLabel = 'Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))

				tRange = np.array([0., tStop - tStart, tStep])
				tData  = np.linspace(tStart, tStop, num=nData, endpoint=True)

				if self.kInterlaced:
					if self.TrackOptions['ComputeMovingAvg']:
						winSize = max(int(round(self.TrackOptions['MovingAvgWindow']/tStep)), 1)
						akU   = fringeDF[colNames[0]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
						akD   = fringeDF[colNames[1]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
						akInd = fringeDF[colNames[2]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
						akDep = fringeDF[colNames[3]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
					else:
						akU   = fringeDF[colNames[0]].to_numpy()
						akD   = fringeDF[colNames[1]].to_numpy()
						akInd = fringeDF[colNames[2]].to_numpy()
						akDep = fringeDF[colNames[3]].to_numpy()
					## Rescale phases and accelerations to micro-g
					akU     = akU/(self.keff*self.Teff[iax]**2*self.gLocal)*1.E6
					akD     = akD/(self.keff*self.Teff[iax]**2*self.gLocal)*1.E6
					akInd   = akInd/self.gLocal*1.E6
					akDep   = akDep/self.gLocal*1.E6
					yData   = [akU, akD, akInd, akDep]
					yLabels = [r'$a_{\rm kU}$  ($\mu$g)', r'$a_{\rm kD}$  ($\mu$g)', r'$a_{\rm kInd}$  ($\mu$g)', r'$a_{\rm kDep}$  ($\mu$g)']
					ikRange = range(3,3+1)
				else:
					if self.TrackOptions['ComputeMovingAvg']:
						winSize = max(int(round(self.TrackOptions['MovingAvgWindow']/tStep)), 1)
						if self.ikList[0] == 0:
							ak = fringeDF[colNames[0]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()
						else:
							ak = fringeDF[colNames[1]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()	
					else:
						if self.ikList[0] == 0:
							ak = fringeDF[colNames[0]].to_numpy()
						else:
							ak = fringeDF[colNames[1]].to_numpy()
					## Rescale phases to acceleration in micro-g
					ak      = ak/(self.keff*self.Teff[iax]**2*self.gLocal)*1.E6
					yData   = [ak, ak]
					yLabels = [r'$a_{\rm kU}$  ($\mu$g)', r'$a_{\rm kD}$  ($\mu$g)']
					ikRange = self.ikList

				for ik in ikRange:
					Mon.PlotMonitorCorrelations(iax, ik, yData[ik], yLabels[ik], MonDFType='Raw', iStart=self.IterStart, iSkip=self.IterStep)