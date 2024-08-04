#####################################################################
## Filename:	iXAtom_Class_Track.py
## Author:		B. Barrett
## Description: Track class definition for iXAtom analysis package
## Version:		3.2.5
## Last Mod:	09/05/2020
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
		# self.GetTrackConfig()

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

		self.RatioDF	= [[pd.DataFrame() for ik in range(2)] for iax in range(3)]
		self.PhaseDF	= [pd.DataFrame() for iax in range(3)]
		self.RTFreqDF	= pd.DataFrame()
		self.RTAccelDF	= pd.DataFrame()
		self.MonitorDF	= pd.DataFrame()

		self.TimeSeries_DF = pd.DataFrame()
		self.PSD_DF		= pd.DataFrame()
		self.ADev_DF	= pd.DataFrame()

		self.PrintSubSeqTiming = True

	#################### End of Track.__init__() ####################
	#################################################################

	def LoadAnalysisData(self, AnalysisLevel):
		"""Load specific tracking data according to analysisLevel."""

		logging.info('iXC_Track::Loading raw tracking data from:')
		logging.info('iXC_Track::  {}'.format(self.RawFolderPath))

		if AnalysisLevel <= 2:
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

		elif AnalysisLevel == 3:
			## Load RT accel data
			[self.RTAccelFileFound, self.RTAccelDF] = self.LoadData(self.RawFolderPath, self.RTAccelFile)
			if self.RTAccelFileFound:
				self.RawDataFound = True

			## Load RT freq data
			# [self.RTFreqFileFound, self.RTFreqDF] = self.LoadData(self.RawFolderPath, self.RTFreqFile)
			# if self.RTFreqFileFound:
			# 	self.RawDataFound = True

		elif AnalysisLevel == 4:
			## Load phase data
			[self.PhaseFileFound, df] = self.LoadData(self.RawFolderPath, self.PhaseFile)
			if self.PhaseFileFound:
				self.RawDataFound = True
				for iax in self.iaxList:
					self.PhaseDF[iax] = df[df['AccelAxis'] == iax]

			## Load monitor data
			# [self.MonitorFileFound, self.MonitorDF] = self.LoadData(self.RawFolderPath, self.MonitorFile)
			# if self.MonitorFileFound:
			# 	self.RawDataFound = True

		if self.RawDataFound:
			self.GetSequenceTiming(AnalysisLevel)
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
			df = self.RTAccelDF if analysisLevel == 3 else self.MonitorDF

			self.nSamplesFull = df['#Iteration'].shape[0]
			self.dtStartFull  = iXUtils.TimestampToDatetime(df['Date'].iloc[0], df['Time'].iloc[0])
			self.dtStopFull   = iXUtils.TimestampToDatetime(df['Date'].iloc[-1], df['Time'].iloc[-1])
			self.nIterFull    = df['#Iteration'].iloc[-1]

		if analysisLevel == 4:
			df = self.PhaseDF
			self.nSamplesFull = df[self.iaxList[0]].shape[0]
			self.dtStartFull  = iXUtils.TimestampToDatetime(df[self.iaxList[0]]['Date'].iloc[0], df[self.iaxList[0]]['Time'].iloc[0])
			self.dtStopFull   = iXUtils.TimestampToDatetime(df[self.iaxList[0]]['Date'].iloc[-1], df[self.iaxList[0]]['Time'].iloc[-1])
			self.nIterFull    = self.nSamplesFull*self.nk*self.nax

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

	def PlotRatioData(self):
		"""Plot time-series analysis of ratio data."""

		if self.RatioFilesFound:
			logging.info('iXC_Track::Plotting ratio data for {}...'.format(self.RunString))

			iSkip  = self.nk*self.nax
			# iterat = [[[] for ik in range(2)] for iax in range(3)]
			ratio  = [[[] for ik in range(2)] for iax in range(3)]
			phase  = [[[] for ik in range(2)] for iax in range(3)]
			errSig = [[[] for ik in range(2)] for iax in range(3)]
			errInt = [[[] for ik in range(2)] for iax in range(3)]

			self.SummaryDF = [[pd.DataFrame() for ik in range(2)] for iax in range(3)]

			nData  = int(1E10)
			for iax in self.iaxList:
				for ik in self.ikList:
					df = self.SetTrackTimeRange(self.RatioDF[iax][ik])
					# nData = min(nData, df.shape[0])
					nData = df.shape[0]

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
						ratio[iax][ik] = df['Ratio'].to_numpy()
						phase[iax][ik] = (df['CurrentPhase'] - df['ModPhase']).to_numpy()

						if 'Error' not in df.columns:
							## Use old column header ('ErrorSignal') and compute ErrorInt from cumulative sum
							errSig[iax][ik] = df['ErrorSignal'].to_numpy()
							errInt[iax][ik] = df['ErrorSignal'].cumsum().to_numpy()
						else:
							errSig[iax][ik] = df['Error'].to_numpy()
							errInt[iax][ik] = df['ErrorInt'].to_numpy()

					d = {'Iteration': df['#Iteration'].to_numpy(), 'Date': df['Date'], 'Time': df['Time'], 
						'RelTime': np.linspace(self.tStart, self.tStop, nData), 'Ratio': ratio[iax][ik], 'Error': errSig[iax][ik],
						'ErrorInt': errInt[iax][ik]}
					self.SummaryDF[iax][ik] = pd.DataFrame(data=d)

			tRange  = np.array([iStart, iStart + nData*iSkip, iSkip])*self.tCycMeanFull*xScale
			yData   = [
				[ratio[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList],
				[errSig[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList],
				[errInt[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList],
				[phase[iax][ik][:nData] for iax in self.iaxList for ik in self.ikList]]
			yErr    = [[] for _ in range(4)]
			yScales = [np.ones(self.nax*self.nk) for _ in range(4)]
			eColors = [['turquoise', 'darkcyan'], ['blueviolet', 'indigo'], ['darkorange', 'saddlebrown']]
			colors  = [
				[self.DefaultPlotColors[iax][ik] for iax in self.iaxList for ik in self.ikList],
				[eColors[iax][ik] for iax in self.iaxList for ik in self.ikList],
				[eColors[iax][ik] for iax in self.iaxList for ik in self.ikList],
				[self.DefaultPlotColors[iax][ik] for iax in self.iaxList for ik in self.ikList]]
			lLabels = [
				[self.AxisLegLabels[iax][ik]+'-Ratio' for iax in self.iaxList for ik in self.ikList],
				[self.AxisLegLabels[iax][ik]+'-ErrSig' for iax in self.iaxList for ik in self.ikList],
				[self.AxisLegLabels[iax][ik]+'-ErrInt' for iax in self.iaxList for ik in self.ikList],
				[self.AxisLegLabels[iax][ik] for iax in self.iaxList for ik in self.ikList]]
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
			ADev_Fit_FixExp = [
				[False for _ in range(self.nax*self.nk)],
				[False for _ in range(self.nax*self.nk)],
				[False for _ in range(self.nax*self.nk)],
				[self.TrackOptions['ADev_Fit_FixExp'] for _ in range(self.nax*self.nk)]]

			Options = {
				'SavePlot'			: self.PlotOptions['SavePlot'],
				'PlotFolderPath'	: self.PlotOptions['PlotFolderPath'],
				'PlotFileName'		: 'Track-Run{:02d}-AIRatioDataSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
				'ColumnDim'			: (5.5, 6),
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
				'LegendFontSize'	: 12,
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
				'ADev_Errorstyle'	: self.TrackOptions['ADev_Errorstyle'], # 'Bar' or 'Shaded'
				'ADev_Linestyle' 	: '-' if self.TrackOptions['ADev_Errorstyle'] == 'Shaded' else 'None',
				'ADev_Marker'    	: 'None' if self.TrackOptions['ADev_Errorstyle'] == 'Shaded' else '.',
				'ADev_SetLimits'	: [False, False],
				'ADev_XLimits'		: [1.E2, 4.E4],
				'ADev_YLimits'		: [1.E-8, 1.E-6],
				'ADev_Fit'			: ADev_Fit,
				'ADev_Fit_FixExp'	: ADev_Fit_FixExp,
				'ADev_Fit_XLimits'	: [0.9*tRange[2], 1.1*(0.4*tRange[1])],
				'ADev_Fit_SetRange'	: [[False for j in range(self.nax*self.nk)] for r in range(4)],
				'ADev_Fit_Range'	: [[[0., 2.E3] for j in range(self.nax*self.nk)] for r in range(4)]
				}

			_ = iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)

			# [tData, fData, PSD, tau, ADev, ADevErrL, ADevErrU]
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

			# nkDep = len(ikDepList)
			# nkInd = len(ikIndList)

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

				if self.SoftwareVersion >= 3.4 and self.TrackMode == 'Open Loop, Fixed Chirp':
					aOffset = 2.*np.pi*self.RamankUChirp[iax]/self.keff ## m/s^2
				else:
					aOffset = 0.

				if self.TrackOptions['ComputeMovingAvg']:
					winSize = int(round(self.TrackOptions['MovingAvgWindow']/(self.tCycMeanFull*iSkip)))
					logging.info('iXC_Track::Computing moving average for window size {} ({:.1f} s)...'.format(winSize, winSize*self.tCycMeanFull*iSkip))
					kDepPhase[iax] =  df[headers[0]].rolling(winSize, center=True, min_periods=1).mean().to_numpy() ## rad
					kIndPhase[iax] =  df[headers[1]].rolling(winSize, center=True, min_periods=1).mean().to_numpy() ## rad
					kDepAccel[iax] = (df[headers[2]].rolling(winSize, center=True, min_periods=1).mean().to_numpy() + aOffset)/self.gLocal ## g
					kIndAccel[iax] =  df[headers[3]].rolling(winSize, center=True, min_periods=1).mean().to_numpy()/self.gLocal ## g
				else:
					kDepPhase[iax] =  df[headers[0]].to_numpy() ## rad
					kIndPhase[iax] =  df[headers[1]].to_numpy() ## rad
					kDepAccel[iax] = (df[headers[2]].to_numpy() + aOffset)/self.gLocal ## g
					kIndAccel[iax] =  df[headers[3]].to_numpy()/self.gLocal ## g

			for iax in self.iaxList:
				kDepPhase[iax] = kDepPhase[iax][:nData]
				kIndPhase[iax] = kIndPhase[iax][:nData]
				kDepAccel[iax] = kDepAccel[iax][:nData]
				kIndAccel[iax] = kIndAccel[iax][:nData]

			if self.nax == 3:
				if self.TrackOptions['CorrectAcceleros']:
					## Misalignment coefficients and biases
					[tyx, tzx, tzy, tyy, tzz, p0x, p0y, p0z] = list(self.TrackOptions['QAModelParameters'].values())
					p0 = np.array([p0x, p0y, p0z])
					T  = np.array([[1., 0., 0.], [-tyx/tyy, 1./tyy, 0.], [(-tzx + tyx*tzy)/(tyy*tzz), -tzy/(tyy*tzz), 1./tzz]])
					kDepAccel[:3] = np.matmul(T, kDepAccel[:3] - p0.reshape((3,1)))
					kDepPhase[:3] = np.matmul(T, kDepPhase[:3] - (p0*self.Seff).reshape((3,1)))

				kDepAccel[3]  = np.sqrt(np.sum([kDepAccel[iax]**2 for iax in range(3)], axis=0))
				kDepPhase[3]  = np.sqrt(np.sum([(self.Seff[iax]*kDepAccel[iax])**2 for iax in range(3)], axis=0))*self.gLocal
				kDepPhase[3] -= np.linalg.norm(self.alpha*self.Teff**2)

			if self.TrackOptions['SubtractTideModel']:
				Location = {
					'Latitude':		self.Phys.Latitude,
					'Longitude': 	self.Phys.Longitude,
					'Height':		self.Phys.Height
					}

				tData 		 = np.linspace(tAbsStart, tAbsStop, num=nData, endpoint=True)
				kDepAccel[4] = self.Phys.GetTideModel(tData, Location, Recompute=self.TrackOptions['RecomputeTideModel'])/self.gLocal ## g
				kDepPhase[4] = kDepAccel[4]*self.Seff[2]

				if self.nax == 3:
					kDepAccel[5] = kDepAccel[3] - kDepAccel[4]
					kDepPhase[5] = kDepPhase[3] - kDepPhase[4]
				else:
					kDepAccel[5] = kDepAccel[2] - kDepAccel[4]
					kDepPhase[5] = kDepPhase[2] - kDepPhase[4]

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
				yLabels = [yLabels[iax] for iax in ikDepList] + [r'$a^{\rm ind}$  (rad)', r'$\sqrt{\rm PSD}$  (rad$/\sqrt{\rm Hz}$)', r'Allan Deviation  (rad)']
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
				mask 			  = [False, False, False, False, False, self.TrackOptions['ADev_Fit_FixExp']] ## [ax,ay,az,|g|,atide,|g|-atide]				
				ADev_Fit_FixExp   = [[mask[iax]] for iax in ikDepList]
				mask 			  = [False, False, False, False, False, self.TrackOptions['ADev_Fit_SetRange']] ## [ax,ay,az,|g|,atide,|g|-atide]				
				ADev_Fit_SetRange = [[mask[iax]] for iax in ikDepList]
				ADev_Fit_Range    = [[self.TrackOptions['ADev_Fit_Range']] for iax in ikDepList]
			else:
				PSD_PlotSubSets   = [[True] for iax in ikDepList]
				ADev_PlotSubSets  = [[True] for iax in ikDepList]
				mask 			  = [False, False, False, self.TrackOptions['ADev_Fit']] ## [ax,ay,az,|g|]
				ADev_Fit          = [[mask[iax]] for iax in ikDepList]
				# ADev_Fit          = [[self.TrackOptions['ADev_Fit']] for iax in ikDepList]
				ADev_Fit_FixExp   = [[self.TrackOptions['ADev_Fit_FixExp']] for iax in ikDepList]
				ADev_Fit_SetRange = [[self.TrackOptions['ADev_Fit_SetRange']] for iax in ikDepList]
				ADev_Fit_Range    = [[self.TrackOptions['ADev_Fit_Range']] for iax in ikDepList]

			## Add elements for kInd quantities
			if self.TrackOptions['PlotQuantity'] == 'AllPhases' or self.TrackOptions['PlotQuantity'] == 'AllAccels':
				yErr   			  += [[[] for iax in ikIndList]]
				colors 			  += [[self.DefaultPlotColors[iax][1] for iax in ikIndList]]
				PSD_PlotSubSets   += [[True for iax in ikIndList]]
				ADev_PlotSubSets  += [[True for iax in ikIndList]]
				ADev_Fit          += [[False for iax in ikIndList]]
				ADev_Fit_FixExp   += [[False for iax in ikIndList]]
				ADev_Fit_SetRange += [[False for iax in ikIndList]]
				ADev_Fit_Range    += [[[0., 2.E3] for iax in ikIndList]]

			Options = {
				'SavePlot'			: self.PlotOptions['SavePlot'],
				'PlotFolderPath'	: self.PlotOptions['PlotFolderPath'],
				'PlotFileName'		: 'Track-Run{:02d}-AIPhaseDataSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
				'ColumnDim'			: (5.5, 6),
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
				'LegendFontSize'	: 12,
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
				'ADev_Fit_FixExp'	: ADev_Fit_FixExp,
				'ADev_Fit_XLimits'	: self.TrackOptions['ADev_Fit_Range'],
				'ADev_Fit_SetRange'	: ADev_Fit_SetRange,
				'ADev_Fit_Range'	: ADev_Fit_Range
				}

			self.PrintStats(yData, ikDepList, ikIndList)
			_ = iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)

			# [tData, fData, PSD, tau, ADev, ADevErrL, ADevErrU]
		else:
			logging.error('iXC_Track::PlotPhaseData::Phase data for {} not found...'.format(self.RunString))
			logging.error('iXC_Track::PlotPhaseData::Aborting...')
			quit()

	################# End of Track.PlotPhaseData() ##################
	#################################################################

	def PrintStats(self, yData, ikDepList, ikIndList):

		if self.TrackOptions['PlotQuantity'] == 'AllPhases' or self.TrackOptions['PlotQuantity'] == 'kDepPhases':
			label = 'Phase'
			unit  = 'rad'
			scale = 1.
		else:
			label = 'Accel'
			unit  = self.TrackOptions['AccelUnit']
			scale = 1.E6 if unit == 'ug' else self.gLocal

		axLabels = [' X ',' Y ',' Z ','|g|','tide','Z - tide']

		print('------------------- kDep Statistics ------------------')
		for r in range(len(ikDepList)):
			iax = ikDepList[r]
			print(label+' '+axLabels[iax]+' Mean = {:.5E}'.format(np.mean(yData[r][0])*scale)+' '+unit)
			print(label+' '+axLabels[iax]+' SDev = {:.5E}'.format(np.std(yData[r][0])*scale)+' '+unit)
			# print(label+' '+axLabels[iax]+'-Dep SErr = {:.5E}'.format(np.std(yData[0][r])/np.sqrt(nData)*scale)+' '+unit)
		if self.TrackOptions['PlotQuantity'] == 'AllPhases' or self.TrackOptions['PlotQuantity'] == 'AllAccels':
			print('------------------- kInd Statistics ------------------')
			for r in range(self.nax):
				iax = ikIndList[r]
				print(label+' '+axLabels[iax]+' Mean = {:.5E}'.format(np.mean(yData[-1][r])*scale)+' '+unit)
				print(label+' '+axLabels[iax]+' SDev = {:.5E}'.format(np.std(yData[-1][r])*scale)+' '+unit)
				# print(label+' '+axLabels[iax]+'-Ind SErr = {:.5E}'.format(np.std(yData[1][r])/np.sqrt(nData)*scale)+' '+unit)
		print('------------------------------------------------------')

	################## End of Track.PrintStats() ####################
	#################################################################

	def PlotAccelMean(self):
		"""Plot summary of mean accelerometer data."""

		if self.RTAccelFileFound:
			logging.info('iXC_Track::Plotting mean accel data for {}...'.format(self.RunString))

			df = self.SetTrackTimeRange(self.RTAccelDF)

			if self.tStop > 10000:
				xScale = 1.E-3
				xLabel = r'Time  ($\times 10^3$ s)'
			else:
				xScale = 1.
				xLabel = 'Time  (s)'

			tRange  = np.array([df['#Iteration'].iloc[0], df['#Iteration'].iloc[-1], 1.])*self.tCycMeanFull*xScale

			if self.TrackOptions['ComputeMovingAvg']:
				winSize = int(round(self.TrackOptions['MovingAvgWindow']/self.tCycMeanFull))
				logging.info('iXC_Track::Computing moving average for window size {} ({:.1f} s)...'.format(winSize, winSize*self.tCycMeanFull))
				a = df[['AccelMean_X', 'AccelMean_Y', 'AccelMean_Z']].rolling(winSize, center=True, min_periods=1).mean().to_numpy() ## m/s**2
			else:
				a = df[['AccelMean_X', 'AccelMean_Y', 'AccelMean_Z']].to_numpy() ## m/s**2

			ax = a[:,0]
			ay = a[:,1]
			az = a[:,2]

			if self.TrackOptions['CorrectAcceleros']:
				## Misalignment coefficients and biases
				[tyx, tzx, tzy, tyy, tzz, p0x, p0y, p0z] = list(self.TrackOptions['MAModelParameters'].values())
				p0 = np.array([p0x, p0y, p0z])
				T  = np.array([[1., 0., 0.], [-tyx/tyy, 1./tyy, 0.], [(-tzx + tyx*tzy)/(tyy*tzz), -tzy/(tyy*tzz), 1./tzz]])
				p  = np.array([ax, ay, az])
				a  = np.matmul(T, p - p0.reshape((3,1)))
				[ax, ay, az] = a[:3]

			g       = np.sqrt(ax**2 + ay**2 + az**2)
			nData   = len(ax)
			yData   = np.array([[ax], [ay], [az], [g]])/self.gLocal
			yErr    = [np.zeros(nData) for _ in range(4)]
			yScales = [[self.gLocal], [self.gLocal], [self.gLocal], [self.gLocal]]

			Options = {
				'SavePlot'			: self.PlotOptions['SavePlot'],
				'PlotFolderPath'	: self.PlotOptions['PlotFolderPath'],
				'PlotFileName'		: 'Track-Run{:02d}-RTAccelSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
				'ColumnDim'			: (5.5, 6),
				'Colors'			: [['green'], ['royalblue'], ['red'], ['black']],
				'Linestyle'			: '-',
				'Linewidth'			: 1.,
				'Marker'			: 'None',
				'Markersize'		: 6, 
				'ShowErrors'		: False,
				'SampleRate'		: 1./self.tCycMeanFull,
				'xLabels'			: ['None', 'None', 'None', xLabel],
				'yLabels'        	: [r'$a_x$  (m/s$^2$)', r'$a_y$  (m/s$^2$)', r'$a_z$  (m/s$^2$)',
					r'$|\mathbf{a}|$  (m/s$^2$)', r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)'],
				'yScales'			: yScales,
				'ShowFigureLabels'  : self.PlotOptions['ShowFigureLabels'],
				'FigLabels'			: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
				'ShowLegend'		: [False, False, True],
				'LegendLabels'		: [[r'$a_x$'], [r'$a_y$'], [r'$a_z$'], [r'$|\mathbf{a}|$']],
				'LegendLocations'	: ['best', 'best', 'best'],
				'LegendFontSize'	: 12,
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
				'ADev_Errorstyle'	: self.TrackOptions['ADev_Errorstyle'], # 'Bar' or 'Shaded'
				'ADev_Linestyle' 	: '-' if self.TrackOptions['ADev_Errorstyle'] == 'Shaded' else 'None',
				'ADev_Marker'    	: 'None' if self.TrackOptions['ADev_Errorstyle'] == 'Shaded' else '.',
				'ADev_SetLimits'	: self.TrackOptions['ADev_SetLimits'],
				'ADev_XLimits'		: self.TrackOptions['ADev_XLimits'],
				'ADev_YLimits'		: self.TrackOptions['ADev_YLimits'],
				'ADev_Fit'			: [[False], [False], [False], [self.TrackOptions['ADev_Fit']]],
				'ADev_Fit_XLimits'	: self.TrackOptions['ADev_Fit_Range'],
				'ADev_Fit_SetRange'	: [[False], [False], [False], [self.TrackOptions['ADev_Fit_SetRange']]],
				'ADev_Fit_Range'	: [[self.TrackOptions['ADev_Fit_Range']] for _ in range(4)],
				'ADev_Fit_FixExp'	: [[False], [False], [False], [self.TrackOptions['ADev_Fit_FixExp']]]
				}

			self.TrackOptions['PlotQuantity'] = 'kDepAccels'
			ikDepList = [0,1,2,3]
			ikIndList = []
			self.PrintStats(yData, ikDepList, ikIndList)

			[tData, fData, PSD, tau, ADev, ADevErrL, ADevErrU] = iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)

			if self.TrackOptions['SaveAnalysisLevel3']:
				self.TimeSeries_DF['AbsTime']   = tData/xScale + self.dtStartFull.timestamp()
				self.TimeSeries_DF['RelTime']	= tData/xScale
				self.TimeSeries_DF['AccelX'] 	= ax
				self.TimeSeries_DF['AccelY']	= ay
				self.TimeSeries_DF['AccelZ']	= az
				self.TimeSeries_DF['AccelNorm']	= g

				if self.TrackOptions['PSD_Plot']:
					self.PSD_DF['Freq'] 			= fData[0]
					self.PSD_DF['AccelX_PSD']     	= PSD[0]
					self.PSD_DF['AccelY_PSD']     	= PSD[1]
					self.PSD_DF['AccelZ_PSD']     	= PSD[2]
					self.PSD_DF['AccelNorm_PSD']  	= PSD[3]

				if self.TrackOptions['ADev_Plot']:
					self.ADev_DF['tau']     		= tau[0]
					self.ADev_DF['AccelX_ADev']		= ADev[0]
					self.ADev_DF['AccelX_ADevErrL']	= ADevErrL[0]
					self.ADev_DF['AccelX_ADevErrU']	= ADevErrU[0]
					self.ADev_DF['AccelY_ADev'] 	= ADev[1]
					self.ADev_DF['AccelY_ADevErrL']	= ADevErrL[1]
					self.ADev_DF['AccelY_ADevErrU']	= ADevErrU[1]
					self.ADev_DF['AccelZ_ADev']		= ADev[2]
					self.ADev_DF['AccelZ_ADevErrL']	= ADevErrL[2]
					self.ADev_DF['AccelZ_ADevErrU']	= ADevErrU[2]
					self.ADev_DF['AccelNorm_ADev']	= ADev[3]
					self.ADev_DF['AccelNorm_ADevErrL'] = ADevErrL[3]
					self.ADev_DF['AccelNorm_ADevErrU'] = ADevErrU[3]

				self.WriteAnalysisLevel3()

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
			Mon.MonitorOptions['ComputeMovingAvg'] = self.TrackOptions['ComputeMovingAvg']
			Mon.MonitorOptions['MovingAvgWindow'] = self.TrackOptions['MovingAvgWindow']
			Mon.ProcessMonitorData()

			# if self.TrackOptions['ComputeMovingAvg']:
			# 	monDFType = 'Mean'
			# else:
			# 	monDFType = 'Raw'

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

				# t0     = dt.datetime.fromtimestamp(tStart, tz=pytz.timezone('Europe/Paris'))
				# xLabel = 'Time - {}  (s)'.format(t0.strftime('%H:%M:%S'))
				# tRange = np.array([0., tStop - tStart, tStep])
				# tData  = np.linspace(tStart, tStop, num=nData, endpoint=True)

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

	############### End of Track.PlotCorrelations() #################
	#################################################################

	def WriteAnalysisLevel3(self):
		"""Method for writing Tracking AnalysisLevel = 3 results.
		NEEDS:
		\t self.Run				(int)  - Run number
		\t self.PostFolderPath	(str)  - Path to folder in which to store analysis files
		\t self.TimeSeries_DF 	(list) - Dataframe containing time series of RT accelerations (X,Y,Z,Norm) 
		\t self.PSD_DF 			(list) - Dataframe containing PSD of RT accelerations (X,Y,Z,Norm)
		\t self.ADev_DF 		(list) - Dataframe containing ADev of RT accelerations (X,Y,Z,Norm)
		"""

		logging.info('iXC_Track::Writing AnalysisLevel = 3 results for {}...'.format(self.RunString))

		dfNames = ['TimeSeries', 'PSD', 'ADev']
		formats = ['%.10E', '%.6E', '%.6E']

		idf = -1
		for dfName in dfNames:
			idf += 1
			df = getattr(self, dfName+'_DF')
			if df.shape[0] > 0:
				fileName = 'Track-Run{:02d}-RTAccel-{}.txt'.format(self.Run, dfName)
				filePath = os.path.join(self.PostFolderPath, fileName)
				iXUtils.WriteDataFrameToFile(df, self.PostFolderPath, filePath, True, False, formats[idf], list(df.keys()))

	#################### End of WriteAnalysisLevel3() ###################
	#####################################################################

#####################################################################
###################### End of Class Tracking ########################
#####################################################################

def TrackAnalysis(AnalysisLevel, AnalysisCtrl, TrackOpts, MonitorOpts, PlotOpts, RunPars):
	"""Method for Tracking AnalysisLevels = 1,2,3,4:
	AnalysisLevels = 1: Load tracking data, and perform time series analysis of ratio, phase, and error signal.
	AnalysisLevels = 2: Load tracking data, and perform time series of quantum accelerometers.
	AnalysisLevels = 3: Load tracking data, and perform time series of classical accelerometers.	
	AnalysisLevels = 4: Load tracking and monitor data, plot correlations between monitor and phase data.
	ARGUMENTS:
	\t AnalysisLevel  (int) - Selected analysis level
	\t AnalysisCtrl  (dict) - Key:value pairs controlling main analysis options
	\t TrackOpts     (dict) - Copy of key:value pairs controlling Tracking options
	\t MonitorOpts   (dict) - Copy of key:value pairs controlling Monitor options
	\t PlotOpts      (dict) - Copy of key:value pairs controlling plot options
	\t RunPars     (object) - Instance of Run Parameters class for RunList[0]
	"""

	WorkDir = AnalysisCtrl['WorkDir']
	Folder  = AnalysisCtrl['Folder']
	RunList = AnalysisCtrl['RunList']

	for RunNum in RunList:
		if RunNum == RunList[0]:
			Trk = Track(WorkDir, Folder, RunNum, TrackOpts, MonitorOpts, PlotOpts, False, RunPars.__dict__.items())
		else:
			Trk = Track(WorkDir, Folder, RunNum, TrackOpts, MonitorOpts, PlotOpts)

		Trk.LoadAnalysisData(AnalysisLevel=AnalysisLevel)

		if AnalysisLevel == 1:
			Trk.PlotRatioData()
			print(Trk.SummaryDF[1][0].head())

		elif AnalysisLevel == 2:
			Trk.PlotPhaseData()

		elif AnalysisLevel == 3:
			Trk.PlotAccelMean()

		elif AnalysisLevel == 4:
			Trk.PlotCorrelations()

################## End of TrackAnalysisLevel123() ###################
#####################################################################

# def WriteTrackAnalysisSummary(RunList, SummaryDF, Folder, Labels, Columns=None):
# 	"""Method for writing Tracking AnalysisLevel = 1,2,3 results.
# 	ARGUMENTS:
# 	\t RunList   (list) - List of Runs contained in SummaryDF
# 	\t SummaryDF (list) - List of dataframes containing analysis summary for each axis and k-direction
# 	\t Folder    (str)  - Path to folder in which to store analysis summary file
# 	\t Labels    (list) - Raman axis file labels
# 	\t Columns   (list) - Ordered subset of columns to write to file
# 	"""

# 	fileName    = 'Track-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))
# 	floatFormat = '%11.9E'

# 	for iax in range(3):
# 		for ik in range(2):
# 			if SummaryDF[iax][ik].shape[0] > 0:
# 				filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][ik]+'.txt')
# 				iXUtils.WriteDataFrameToFile(SummaryDF[iax][ik], Folder, filePath, True, False, floatFormat, Columns)

# ################ End of WriteTrackAnalysisSummary() #################
# #####################################################################

# def ReadTrackAnalysisSummary(iaxList, ikList, RunList, Folder, Labels):
# 	"""Method for reading Tracking AnalysisLevel = 1,2,3 results.
# 	ARGUMENTS:
# 	\t iaxList  (list) - List of Raman axis indices
# 	\t ikList   (list) - List of k-direction indices 
# 	\t RunList  (list) - List of Runs contained in summary file
# 	\t Folder   (str)  - Path to folder containing analysis summary file
# 	\t Labels   (list) - Raman axis file labels
# 	"""

# 	SummaryDF = [[pd.DataFrame([]) for ik in range(2)] for iax in range(3)]
# 	fileName  = 'Track-Runs{:02d}-{:02d}-AnalysisSummary.txt'.format(min(RunList), max(RunList))

# 	for iax in iaxList:
# 		for ik in ikList:
# 			filePath = os.path.join(Folder, fileName[:-4]+'-'+Labels[iax][ik]+'.txt')
# 			if os.path.exists(filePath):
# 				SummaryDF[iax][ik] = pd.read_csv(filePath, sep='\t')
# 			else:
# 				logging.error('iXC_Track::ReadTrackAnalysisSummary::File not found specified path: {}'.format(filePath))
# 				logging.error('iXC_Track::ReadTrackAnalysisSummary::Aborting...')
# 				quit()

# 	return SummaryDF

# ################ End of ReadTrackAnalysisSummary() ##################
# #####################################################################