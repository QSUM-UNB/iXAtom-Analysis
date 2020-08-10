#####################################################################
## Filename:	iXAtom_Class_Stream.py
## Author:		B. Barrett
## Description: Stream class definition for iXAtom analysis package
## Version:		3.2.4
## Last Mod:	16/01/2020
##===================================================================
## Change Log:
## 06/11/2019 - Stream class defined. Moved stream methods from old
##				Track class.
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

	def __init__(self, WorkDir, Folder, SubFolder, RunNum, StreamOpts):
		"""Initialize detection variables.
		ARGUMENTS:
		\t WorkDir            (str)  - Path to the top-level directory where dataset is located
		\t Folder             (str)  - Name of folder within WorkDir where dataset is located
		\t SubFolder		  (str)  - Name of subfolder within Folder where dataset is located
		\t RunNum             (int)  - Run number of requested dataset
		\t StreamOpts         (dict) - Key:value pairs controlling plot options, including:
		\t   'ShowPlot'       (bool) - Flag for showing plot
		\t   'SavePlot'       (bool) - Flag for saving plot (overrides ShowPlot)
		\t   'ShowPlotLabels' (list) - List of flag for showing x- and y-axis labels
		\t   'ShowPlotTitle'  (bool) - Flag for showing title
		\t   'ShowPlotLegend' (bool) - Flag for showing legend
		\t   'PlotColors'     (list) - List of colors for each beam axis and k-vector
		\t   'PlotFolderPath' (str)  - Path of folder in which to save plot
		\t   'PlotExtension'  (str)  - Filename extension for plot
		"""

		## Path variables
		self.WorkDir          = WorkDir
		self.Folder           = Folder
		self.SubFolder        = SubFolder
		self.Run 			  = RunNum
		self.RunString        = 'Run {:02d}'.format(RunNum)
		if self.SubFolder != 'None':
			self.StreamFolderPath = os.path.join(WorkDir, Folder, SubFolder, self.RunString)
		else:
			self.StreamFolderPath = os.path.join(WorkDir, Folder, self.RunString)

		self.StreamOptions    = StreamOpts.copy()
		self.GetStreamConfig()

		self.AccelStreamDF    = pd.DataFrame([])

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
			# dfStrmFull     = pd.DataFrame([])

			for f in range(self.nStreamFiles):
				strmFileName = strmFilePrefix + '_{}.csv'.format(f+1)
				strmFilePath = os.path.join(format(self.StreamFolderPath), strmFileName)

				logging.info('iXC_Stream::Compressing accel stream file {}...'.format(strmFileName))

				dfStrm = pd.read_csv(strmFilePath, sep=';', usecols=['acc X', 'acc Y', 'acc Z'])
				# dfStrmFull = dfStrmFull.append(dfStrm, ignore_index=True)
				dfStrm.to_hdf(self.CompFilePath, key='df', mode='a', append=True, format='table', complevel=1)
				# dfStrmFull.to_pickle(self.CompFilePath, compression='zip', protocol=4)

		logging.info('iXC_Stream::Loading compressed accel stream file from:')
		logging.info('iXC_Stream::  {}'.format(self.StreamFolderPath))

		if self.StreamOptions['SetStreamRange']:
			[iStart, iStop, iStep] = self.StreamOptions['StreamRange']
			self.StreamSampleRate = 1000./iStep # Hz
			self.AccelStreamDF = pd.read_hdf(self.CompFilePath, mode='r')[iStart:iStop:iStep]
		else:
			self.StreamSampleRate = 1000. # Hz
			self.AccelStreamDF = pd.read_hdf(self.CompFilePath, mode='r')
			# self.AccelStreamDF = pd.read_pickle(self.CompFilePath, compression='zip')

		self.StreamSampleTime = 1./self.StreamSampleRate # s
		self.nAccelStream     = self.AccelStreamDF.shape[0]
		self.StreamTotalTime  = self.nAccelStream*self.StreamSampleTime

		print('----------------- Accel Stream -----------------')
		print('Samples loaded: {}'.format(self.nAccelStream))
		print('Total duration: {} s ({:5.3} hrs)'.format(self.StreamTotalTime, self.StreamTotalTime/3600.))
		print('Sample rate:    {:5.3e} Hz'.format(self.StreamSampleRate))
		print('Sample time:    {:5.3e} s'.format(self.StreamSampleTime))
		if self.StreamOptions['PrintStreamStats']:
			print(self.AccelStreamDF.describe())
		print('------------------------------------------------')

	################ End of Stream.LoadStreamData() #################
	#################################################################

	def PlotAccelStream(self):
		"""Plot summary of accelerometer stream."""

		logging.info('iXC_Stream::Plotting accel stream for {}...'.format(self.RunString))

		[iStart, iStop, iStep] = self.StreamOptions['StreamRange']
		if (iStop-iStart+1)/iStep > self.nAccelStream:
			iStop = iStart + self.nAccelStream*iStep
		tRange = np.array([iStart, iStop, iStep])*self.StreamSampleTime

		axData  = self.AccelStreamDF['acc X'].to_numpy()
		ayData  = self.AccelStreamDF['acc Y'].to_numpy()
		azData  = self.AccelStreamDF['acc Z'].to_numpy()
		gData   = np.sqrt(axData**2 + ayData**2 + azData**2)
		yData   = [[axData], [ayData], [azData], [gData]]

		Options = {
			'SavePlot'       : self.PlotOptions['SavePlot'],
			'PlotFolderPath' : self.PlotOptions['PlotFolderPath'],
			'PlotFileName'   : 'Stream-Run{:02d}-Summary.'.format(self.Run) + self.PlotOptions['PlotExtension'],
			'ColumnDim'      : (5, 8),
			'Colors'         : [['green'], ['royalblue'], ['red'], ['black']],
			'xLabels'        : ['None', 'None', 'None', 'Time (s)'],
			'yLabels'        : [r'$a_x$  (m/s$^2$)', r'$a_y$  (m/s$^2$)', r'$a_z$  (m/s$^2$)',
				r'$|g|$  (m/s$^2$)', r'PSD  ($g/\sqrt{\rm Hz}$)', r'Allan Deviation  ($g$)'],
			'LegLabels'      : [[r'$a_x$'], [r'$a_y$'], [r'$a_z$'], [r'$|g|$']],
			'ShowLegend'     : [False, True],
			'FixLegLocation' : self.PlotOptions['FixLegLocation'],
			'LegendFontSize' : 13,
			'ShowXTickLabels': [False, False, False, True],
			'SetPlotXLimits' : self.StreamOptions['SetPlotXLimits'],
			'PlotXLimits'    : self.StreamOptions['PlotXLimits'],
			'PSD_Plot'       : self.StreamOptions['PSD_Plot'],
			'PSD_Method'     : self.StreamOptions['PSD_Method'],
			'PSD_Scale'      : 1./self.gLocal,
			'PSD_MaxSubSets' : 2,
			'ADev_Plot'      : self.StreamOptions['ADev_Plot'],
			'ADev_Scale'     : 1./self.gLocal,
			'ADev_MaxSubSets': 2,
			'ADev_ntauStep'	 : 10,
			'ADev_ShowErrors': self.StreamOptions['ADev_ComputeErrors']
			}

		iXUtils.AnalyzeTimeSeries(tRange, yData, Options)

	################ End of Stream.PlotAccelStream() ################
	#################################################################
