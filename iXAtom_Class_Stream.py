#####################################################################
## Filename:	iXAtom_Class_Stream.py
## Author:		B. Barrett
## Description: Stream class definition for iXAtom analysis package
## Version:		3.2.5
## Last Mod:	13/10/2020
#####################################################################

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import allantools as allan

from scipy.signal import periodogram
from scipy.signal import welch

import iXAtom_Utilities as iXUtils

#####################################################################

class Stream():
	#################################################################
	## Class for storing and processing Streaming data
	#################################################################

	def __init__(self, WorkDir, Folder, SubFolder, RunNum, StreamOpts, PlotOpts):
		"""Initialize detection variables.
		ARGUMENTS:
		\t WorkDir    (str)  - Path to the top-level directory where dataset is located
		\t Folder     (str)  - Name of folder within WorkDir where dataset is located
		\t Subfolder  (str)  - Name of subfolder within Folder where dataset is located
		\t RunNum     (int)  - Run number of requested dataset
		\t StreamOpts (dict) - Key:value pairs controlling plot options, including:
		\t PlotOpts   (dict) - Key:value pairs controlling plot options
		"""

		## Path variables
		self.WorkDir		= WorkDir
		self.Folder			= Folder
		self.SubFolder		= SubFolder
		self.Run 			= RunNum
		self.RunString  	= 'Run {:02d}'.format(RunNum)
		if self.SubFolder != 'None':
			self.StreamFolderPath = os.path.join(WorkDir, Folder, SubFolder, self.RunString)
		else:
			self.StreamFolderPath = os.path.join(WorkDir, Folder, self.RunString)
		self.PostFolderPath = os.path.join(WorkDir, 'PostProcessed', Folder, self.RunString)

		self.StreamOptions	= StreamOpts.copy()
		self.PlotOptions	= PlotOpts.copy()
		self.GetStreamConfig()

		self.AccelStreamDF	= pd.DataFrame()
		self.TimeSeries_DF	= pd.DataFrame()
		self.PSD_DF			= pd.DataFrame()
		self.ADev_DF		= pd.DataFrame()

		self.gLocal 		= 9.805642 ## m/s^2

	#################### End of Stream.__init__() ####################
	#################################################################

	def GetStreamConfig(self):
		"""Get stream configuration."""

		logging.info('iXC_Stream::Getting stream config for {}...'.format(self.RunString))

		self.StreamFilePaths = glob.glob(os.path.join(self.StreamFolderPath,'*.csv'))
		self.nStreamFiles    = len(self.StreamFilePaths)

		self.CompFileName    = 'Stream-Run{:02d}-Compressed.h5'.format(self.Run)
		self.CompFilePath    = os.path.join(self.StreamFolderPath, self.CompFileName)

		if self.nStreamFiles == 0 and not os.path.exists(self.CompFilePath):
			logging.error('iXC_Stream::LoadAccelStream::Accel stream not found in: {}'.format(self.StreamFolderPath))
			logging.error('iXC_Stream::LoadAccelStream::Aborting...')
			quit()

	################ End of Stream.GetStreamConfig() ################
	#################################################################

	def LoadAccelStream(self):
		"""Load raw accel stream into a Pandas dataframe.
		Compress stream contained in CSV files into a Hierarchical Data Format (h5) file for efficient reloading."""

		if not os.path.exists(self.CompFilePath):
			logging.info('iXC_Stream::Loading raw accel stream files from:')
			logging.info('iXC_Stream::  {}'.format(self.StreamFolderPath))

			strmFilePrefix = self.StreamFilePaths[0].split('\\')[-1].split('_')[0]

			for f in range(self.nStreamFiles):
				strmFileName = strmFilePrefix + '_{}.csv'.format(f+1)
				strmFilePath = os.path.join(format(self.StreamFolderPath), strmFileName)

				logging.info('iXC_Stream::Compressing accel stream file {}...'.format(strmFileName))

				dfStrm = pd.read_csv(strmFilePath, sep=';', usecols=['acc X', 'acc Y', 'acc Z'])
				dfStrm.to_hdf(self.CompFilePath, key='df', mode='a', append=True, format='table', complevel=1)

		strmFileName = self.StreamFilePaths[0].split('\\')[-1].split('_')[0] + '_1.csv'
		self.tStartFull = os.path.getmtime(os.path.join(self.StreamFolderPath, strmFileName))

		logging.info('iXC_Stream::Loading compressed accel stream file from:')
		logging.info('iXC_Stream::  {}'.format(self.StreamFolderPath))

		self.StreamSampleRate = self.StreamOptions['StreamRate'] # Hz

		if self.StreamOptions['SetTimeRange']:
			[iStart, iStop, iStep] = np.round(np.array(self.StreamOptions['TimeRange'])*self.StreamSampleRate).astype(int)
			self.StreamSampleRate /= iStep
			self.AccelStreamDF = pd.read_hdf(self.CompFilePath, mode='r')[iStart:iStop:iStep]
		else:
			self.AccelStreamDF = pd.read_hdf(self.CompFilePath, mode='r')
			[iStart, iStop, iStep] = [0, self.AccelStreamDF.shape[0]-1, 1]

		self.StreamSampleTime = 1./self.StreamSampleRate # s
		self.nAccelStream     = self.AccelStreamDF.shape[0]
		self.StreamTotalTime  = self.nAccelStream*self.StreamSampleTime
		self.StreamRange      = np.array([iStart, iStop, iStep])

		print('----------------- Accel Stream -----------------')
		print('Samples loaded: {}'.format(self.nAccelStream))
		print('Total duration: {} s ({:5.3} hrs)'.format(self.StreamTotalTime, self.StreamTotalTime/3600.))
		print('Sample rate:    {:5.3e} Hz'.format(self.StreamSampleRate))
		print('Sample time:    {:5.3e} s'.format(self.StreamSampleTime))
		print('------------------------------------------------')
		if self.StreamOptions['PrintStreamStats']:
			print(self.AccelStreamDF.describe())
			print('------------------------------------------------')

	################ End of Stream.LoadStreamData() #################
	#################################################################

	def PlotAccelStream(self):
		"""Plot summary of accelerometer stream."""

		logging.info('iXC_Stream::Plotting accel stream for {}...'.format(self.RunString))

		# [iStart, iStop, iStep] = self.StreamRange
		# if (iStop-iStart+1)/iStep > self.nAccelStream:
		# 	iStop = iStart + (self.nAccelStream-1)*iStep
		# tRange = np.array([iStart, iStop, iStep])*self.StreamSampleTime

		tRange = self.StreamRange*self.StreamSampleTime

		if self.StreamOptions['ComputeMovingAvg']:
			winSize = int(round(self.StreamOptions['MovingAvgWindow']/tRange[2]))
			logging.info('iXC_Stream::Computing moving average for window size {} ({:.1f} s)...'.format(winSize, winSize*tRange[2]))
			a = self.AccelStreamDF[['acc X', 'acc Y', 'acc Z']].rolling(winSize, center=True, min_periods=1).mean().to_numpy() ## m/s**2
		else:
			a = self.AccelStreamDF[['acc X', 'acc Y', 'acc Z']].to_numpy() ## m/s**2

		ax = a[:,0]
		ay = a[:,1]
		az = a[:,2]

		if self.StreamOptions['CorrectAcceleros']:
			## Misalignment coefficients and biases
			[tyx, tzx, tzy, tyy, tzz, p0x, p0y, p0z] = list(self.StreamOptions['MAModelParameters'].values())
			p0 = np.array([p0x, p0y, p0z])
			T  = np.array([[1., 0., 0.], [-tyx/tyy, 1./tyy, 0.], [(-tzx + tyx*tzy)/(tyy*tzz), -tzy/(tyy*tzz), 1./tzz]])
			p  = np.array([ax, ay, az])
			a  = np.matmul(T, p - p0.reshape((3,1)))
			[ax, ay, az] = a[:3]

		g       = np.sqrt(ax**2 + ay**2 + az**2)
		nData   = len(ax)
		yData   = np.array([[ax], [ay], [az], [g]])/self.gLocal
		yErr    = [[np.zeros(nData)] for _ in range(4)]
		yScales = [[self.gLocal] for _ in range(4)]

		if nData >= 1E6:
			if nData >= 1E7:
				logging.error('iXC_Stream::PlotAccelStream::Too much data to process ({} points >= 1E7). Reduce range or increase timestep.'.format(nData))
				logging.error('iXC_Stream::PlotAccelStream::Aborting...')
				quit()
			logging.warning('iXC_Stream::PlotAccelStream::Lots of data to process ({} points >= 1E6).'.format(nData))
			if self.StreamOptions['ADev_taus'] == 'all':
				logging.warning('iXC_Stream::PlotAccelStream::Setting ADev_taus = octave.')
				self.StreamOptions['ADev_taus'] = 'octave'
			if self.StreamOptions['ADev_ShowErrors'] and self.StreamOptions['ADev_Errorstyle'] == 'Shaded':
				logging.warning('iXC_Stream::PlotAccelStream::Setting ADev_Errorstyle = Bar.')
				self.StreamOptions['ADev_Errorstyle'] = 'Bar'

		if self.StreamOptions['IncludeAccelNorm']:
			colors  = [['green'], ['royalblue'], ['red'], ['black']]
			xLabels = ['None', 'None', 'None', 'Time (s)']
			yLabels = [r'$a_x$  (m/s$^2$)', r'$a_y$  (m/s$^2$)', r'$a_z$  (m/s$^2$)',
				r'$|g|$  (m/s$^2$)', r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)']
		else:
			yData   = yData[:3]
			yErr    = yErr[:3]
			colors  = [['green'], ['royalblue'], ['red']]
			xLabels = ['None', 'None', 'Time (s)']
			yLabels = [r'$a_x$  (m/s$^2$)', r'$a_y$  (m/s$^2$)', r'$a_z$  (m/s$^2$)',
				r'$\sqrt{\rm PSD}$  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)']

		Options = {
			'SavePlot'       	: self.PlotOptions['SavePlot'],
			'PlotFolderPath' 	: self.PlotOptions['PlotFolderPath'],
			'PlotFileName'   	: 'Stream-Run{:02d}-Summary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
			'ColumnDim'      	: (5, 8),
			'Colors'         	: colors,
			'Linestyle'		 	: '-',
			'Linewidth'		 	: 1.,
			'Marker'		 	: 'None',
			'Markersize'	 	: 6, 
			'ShowErrors'	 	: False,
			'SampleRate'	 	: 1./tRange[2],
			'xLabels'        	: xLabels,
			'yLabels'        	: yLabels,
			'yScales'			: yScales,
			'ShowFigureLabels'  : self.PlotOptions['ShowFigureLabels'],
			'FigureLabels'		: ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
			'ShowLegend'     	: [False, True, True],
			'LegendLabels'      : [[r'$a_x$'], [r'$a_y$'], [r'$a_z$'], [r'$|g|$']],
			'LegendLocations' 	: ['best', 'best', 'best'],
			'LegendFontSize' 	: 12,
			'SetPlotLimits'		: [self.StreamOptions['SetPlotXLimits'], False],
			'PlotXLimits'		: [self.StreamOptions['PlotXLimits'], self.StreamOptions['PlotXLimits'], self.StreamOptions['PlotXLimits'], self.StreamOptions['PlotXLimits']],
			'PlotYLimits'		: [[0.,10.], [0.,10.], [0.,10.], [0.,10.]],
			'PSD_Plot'			: self.StreamOptions['PSD_Plot'],
			'PSD_PlotSubSets'	: [[True], [True], [True], [True]],
			'PSD_Method'		: self.StreamOptions['PSD_Method'],
			'ADev_Plot'			: self.StreamOptions['ADev_Plot'],
			'ADev_PlotSubSets'  : [[True], [True], [True], [True]],
			'ADev_Type'			: 'Total',
			'ADev_taus'			: self.StreamOptions['ADev_taus'],
			'ADev_ShowErrors'	: self.StreamOptions['ADev_ShowErrors'],
			'ADev_Errorstyle'	: self.StreamOptions['ADev_Errorstyle'], # 'Bar' or 'Shaded'
			'ADev_Linestyle' 	: 'None' if self.StreamOptions['ADev_Errorstyle'] == 'Bar' else '-',
			'ADev_Marker'    	: '.' if self.StreamOptions['ADev_Errorstyle'] == 'Bar' else 'None',
			'ADev_SetLimits'	: [False, False],
			'ADev_XLimits'		: [1.E2, 4.E4],
			'ADev_YLimits'		: [1.E-8, 1.E-6],
			'ADev_Fit'			: [[self.StreamOptions['ADev_Fit']], [self.StreamOptions['ADev_Fit']], [self.StreamOptions['ADev_Fit']], [self.StreamOptions['ADev_Fit']]],
			'ADev_Fit_XLimits'	: [0.9*tRange[2], 1.1*(0.4*tRange[1])],
			'ADev_Fit_SetRange'	: [[self.StreamOptions['ADev_Fit_SetRange']], [self.StreamOptions['ADev_Fit_SetRange']], [self.StreamOptions['ADev_Fit_SetRange']], [self.StreamOptions['ADev_Fit_SetRange']]],
			'ADev_Fit_Range'	: [[self.StreamOptions['ADev_Fit_Range']], [self.StreamOptions['ADev_Fit_Range']], [self.StreamOptions['ADev_Fit_Range']], [self.StreamOptions['ADev_Fit_Range']]],
			'ADev_Fit_FixExp'	: [[self.StreamOptions['ADev_Fit_FixExp']], [self.StreamOptions['ADev_Fit_FixExp']], [self.StreamOptions['ADev_Fit_FixExp']], [self.StreamOptions['ADev_Fit_FixExp']]]
			}

		[tData, fData, PSD, tau, ADev, ADevErrL, ADevErrU] = iXUtils.AnalyzeTimeSeries(tRange, yData, yErr, Options)

		if self.StreamOptions['SaveAnalysisLevel1']:
			self.TimeSeries_DF['AbsTime']   = tData + self.tStartFull
			self.TimeSeries_DF['RelTime']	= tData
			self.TimeSeries_DF['AccelX'] 	= ax
			self.TimeSeries_DF['AccelY']	= ay
			self.TimeSeries_DF['AccelZ']	= az
			self.TimeSeries_DF['AccelNorm']	= g

			if self.StreamOptions['PSD_Plot']:
				self.PSD_DF['Freq'] 			= fData[0]
				self.PSD_DF['AccelX_PSD']     	= PSD[0]
				self.PSD_DF['AccelY_PSD']     	= PSD[1]
				self.PSD_DF['AccelZ_PSD']     	= PSD[2]
				if self.StreamOptions['IncludeAccelNorm']:
					self.PSD_DF['AccelNorm_PSD'] = PSD[3]

			if self.StreamOptions['ADev_Plot']:
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
				if self.StreamOptions['IncludeAccelNorm']:
					self.ADev_DF['AccelNorm_ADev']     = ADev[3]
					self.ADev_DF['AccelNorm_ADevErrL'] = ADevErrL[3]
					self.ADev_DF['AccelNorm_ADevErrU'] = ADevErrU[3]

			self.WriteAnalysisLevel1()

	################ End of Stream.PlotAccelStream() ################
	#################################################################

	def WriteAnalysisLevel1(self):
		"""Method for writing Streaming AnalysisLevel = 1 results.
		NEEDS:
		\t self.Run				(int)  - Run number
		\t self.PostFolderPath	(str)  - Path to folder in which to store analysis files
		\t self.TimeSeries_DF 	(list) - Dataframe containing time series of RT accelerations (X,Y,Z,Norm) 
		\t self.PSD_DF 			(list) - Dataframe containing PSD of RT accelerations (X,Y,Z,Norm)
		\t self.ADev_DF 		(list) - Dataframe containing ADev of RT accelerations (X,Y,Z,Norm)
		"""

		logging.info('iXC_Stream::Writing AnalysisLevel = 1 results for {}...'.format(self.RunString))

		dfNames = ['TimeSeries', 'PSD', 'ADev']
		formats = ['%.10E', '%.6E', '%.6E']

		idf = -1
		for dfName in dfNames:
			idf += 1
			df = getattr(self, dfName+'_DF')
			if df.shape[0] > 0:
				if dfName == 'TimeSeries':
					fileName = 'Stream-Run{:02d}-{}-Compressed.h5'.format(self.Run, dfName)
					filePath = os.path.join(self.PostFolderPath, fileName)
					df.to_hdf(filePath, key='df', mode='w', format='fixed', complevel=1)
				else:
					fileName = 'Stream-Run{:02d}-{}.txt'.format(self.Run, dfName)
					filePath = os.path.join(self.PostFolderPath, fileName)
					iXUtils.WriteDataFrameToFile(df, self.PostFolderPath, filePath, True, False, formats[idf], list(df.keys()))



	#################### End of WriteAnalysisLevel1() ###################
	#####################################################################
