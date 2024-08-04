#####################################################################
## Filename:	iXAtom_Class_Monitor.py
## Author:		B. Barrett
## Description: Monitor class definition for iXAtom analysis package
## Version:		3.2.5
## Last Mod:	15/10/2020
#####################################################################

import copy
import csv
import datetime as dt
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz

import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Utilities 	      as iXUtils

class Monitor(iXC_RunPars.RunParameters):
	#################################################################
	## Class for storing, merging, and manipulating monitor data
	## Inherits all attributes and methods from class: RunParameters
	#################################################################

	def __init__(self, WorkDir, Folder, Runs, MonitorOpts, PlotOpts, LoadRunParsFlag=True, RunPars=[]):
		"""Initialize detection variables.
		ARGUMENTS:
		\t WorkDir     (str)  - Path to the top-level directory where dataset is located
		\t Folder      (str)  - Name of folder within WorkDir where dataset is located
		\t Runs        (list) - Runs in which to search for monitor data
		\t MonitorOpts (dict) - Key:value pairs controlling monitor options
		\t PlotOpts    (dict) - Key:value pairs controlling plot options
		\t LoadRunParsFlag (bool) - (Optional) Flag for loading run parameters from file (True) or setting them from input (False).
		\t RunPars         (list) - (Optional) Key:value pairs containing run parameters
		"""

		super().__init__(WorkDir, Folder, Runs[0])
		if LoadRunParsFlag:
			super().LoadRunParameters()
		else:
			# Take parameters from input list 'RunPars'
			for key, val in RunPars:
				setattr(self, key, val)

		self.PlotOptions       = copy.deepcopy(PlotOpts)
		self.MonitorOptions    = copy.deepcopy(MonitorOpts)
		if Folder == 'Monitor':
			self.MonitorFileFormat = 'Monitor-Run{:02d}-Data.txt'
		else:
			self.MonitorFileFormat = self.FilePrefix+'-Run{:02d}-MonitorData.txt'
		self.MonitorDF         = pd.DataFrame()
		self.MonitorMeanDF     = [[pd.DataFrame() for ik in range(2)] for iax in range(3)]
		self.MonitorSDevDF     = [[pd.DataFrame() for ik in range(2)] for iax in range(3)]

		if self.SoftwareVersion <= 3.1:
			self.Monitors = ['TiltX', 'TiltY', 'MOTPowerX', 'MOTPowerY', 'MOTPowerZ', 
				'RamanBX', 'RamanBY', 'RamanBZ', 'MOTLevel', 'TempChamber', 'NTotalRaw', 'NTotalBG']
			self.Temp_Key    = 'TempChamber'
			self.MOTPX_Key   = 'MOTPowerX'
			self.MOTPY_Key   = 'MOTPowerY'
			self.MOTPZ_Key   = 'MOTPowerZ'
			self.MOTPMon_Key = 'None'
			self.MOTFlu_Key  = 'MOTLevel'
			self.RamanBX_Key = 'RamanBX'
			self.RamanBY_Key = 'RamanBY'
			self.RamanBZ_Key = 'RamanBZ'
			self.RamanBT_Key = 'RamanBT'
		elif self.SoftwareVersion == 3.2:
			self.Monitors = ['TiltX', 'TiltY', 'MOTPowerX', 'MOTPowerY', 'MOTPowerZ', 'MOTMonitor', 
				'MOTFluores', 'RamanBX', 'RamanBY', 'RamanBZ', 'TempChamber', 'NTotalRaw', 'NTotalBG']
			self.Temp_Key    = 'TempChamber'
			self.MOTPX_Key   = 'MOTPowerX'
			self.MOTPY_Key   = 'MOTPowerY'
			self.MOTPZ_Key   = 'MOTPowerZ'
			self.MOTPMon_Key = 'MOTMonitor'
			self.MOTFlu_Key  = 'MOTFluores'
			self.RamanBX_Key = 'RamanBX'
			self.RamanBY_Key = 'RamanBY'
			self.RamanBZ_Key = 'RamanBZ'
			self.RamanBT_Key = 'RamanBT'
		elif self.SoftwareVersion == 3.3:
			self.Monitors = ['MOTMeanPX', 'MOTMeanPY', 'MOTMeanPZ', 'MOTMeanPT', 'MOTMeanBX', 'MOTMeanBY', 'MOTMeanBZ',
				'RamanMeanPX', 'RamanMeanPY', 'RamanMeanPZ', 'RamanDeltaPX', 'RamanDeltaPY', 'RamanDeltaPZ',
				'RamanMeanBX', 'RamanMeanBY', 'RamanMeanBZ', 'RamanDeltaBX', 'RamanDeltaBY', 'RamanDeltaBZ',
				'TiltX', 'TiltY', 'LCRErrorZ', 'Temperature', 'NTotalRaw', 'NTotalBG']
			self.Temp_Key    = 'Temperature'
			self.MOTPX_Key   = 'MOTMeanPX'
			self.MOTPY_Key   = 'MOTMeanPY'
			self.MOTPZ_Key   = 'MOTMeanPZ'
			self.MOTPMon_Key = 'MOTMeanPT'
			self.RamanBX_Key = 'RamanMeanBX'
			self.RamanBY_Key = 'RamanMeanBY'
			self.RamanBZ_Key = 'RamanMeanBZ'
			self.RamanBT_Key = 'RamanMeanBT'
		else: ## self.SoftwareVersion == 3.4:
			self.Monitors = ['MOTMeanPX', 'MOTMeanPY', 'MOTMeanPZ', 'MOTMeanPT', 'MOTMeanBX', 'MOTMeanBY', 'MOTMeanBZ',
				'RamanMeanPX', 'RamanMeanPY', 'RamanMeanPZ', 'RamanDeltaPX', 'RamanDeltaPY', 'RamanDeltaPZ',
				'RamanMeanBX', 'RamanMeanBY', 'RamanMeanBZ', 'RamanDeltaBX', 'RamanDeltaBY', 'RamanDeltaBZ',
				'TiltX', 'TiltY', 'TempLCRX', 'TempLCRY', 'TempLCRZ', 'TempChamber', 'NTotalRaw', 'NTotalBG']
			self.Temp_Key    = 'TempChamber'
			self.MOTPX_Key   = 'MOTMeanPX'
			self.MOTPY_Key   = 'MOTMeanPY'
			self.MOTPZ_Key   = 'MOTMeanPZ'
			self.MOTPMon_Key = 'MOTMeanPT'
			self.RamanBX_Key = 'RamanMeanBX'
			self.RamanBY_Key = 'RamanMeanBY'
			self.RamanBZ_Key = 'RamanMeanBZ'
			self.RamanBT_Key = 'RamanMeanBT'

		self.nMonitors = len(self.Monitors)
		self.GetMonitorData(Runs)

	################### End of Monitor.__init__() ###################
	#################################################################

	def GetMonitorData(self, Runs):
		"""Load individual monitor data file into MonitorDF."""

		logging.info('iXC_Monitor::Loading monitor data for runs {:02d} through {:02d}...'.format(min(Runs), max(Runs)))

		for run in Runs:
			folder = os.path.join(self.WorkDir, self.Folder, 'Run {:02d}'.format(run))
			[self.MonitorFileFound, df] = self.LoadData(folder, self.MonitorFileFormat.format(run))

			if self.MonitorFileFound:
				## Create new column for datetimes by merging 'Date' and 'Time' columns
				df['DateTime'] = df[['Date', 'Time']].apply(lambda x: ' '.join(x), axis=1)
				df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S.%f')
				self.tStep = (df['DateTime'].iloc[-1] - df['DateTime'].iloc[0]).total_seconds()/df.shape[0]

				self.MonitorDF = self.MonitorDF.append(df, ignore_index=True)
				self.GetMonitorMean(run, df)
			else:
				logging.warning('iXC_Monitor::GetMonitorData::Monitor data not found for Run {:02d}...'.format(run))


	################ End of Monitor.GetMonitorData() ################
	#################################################################

	def GetMonitorMean(self, Run, DF):
		"""Parse monitor data for a given run associated with Raman axes and k-directions,
		compute column means and std devs, and append to an existing dataframe."""

		iSkip  = self.nax*self.nk
		iMid   = DF.shape[0] // 2
		series = pd.Series({'Run': Run, 'Date': DF['Date'].iloc[iMid], 'Time': DF['Time'].iloc[iMid], 'DateTime': DF['DateTime'].iloc[iMid]})
		cols   = DF.columns.values[3:]

		iStart = -1
		for iax in self.iaxList:
			for ik in self.ikList:
				iStart += 1
				means = DF[cols].iloc[iStart::iSkip].mean()
				sdevs = DF[cols].iloc[iStart::iSkip].std()
				self.MonitorMeanDF[iax][ik] = self.MonitorMeanDF[iax][ik].append(series.append(means), ignore_index=True)
				self.MonitorSDevDF[iax][ik] = self.MonitorSDevDF[iax][ik].append(series.append(sdevs), ignore_index=True)

	################ End of Monitor.GetMonitorMean() ################
	#################################################################

	def ProcessMonitorData(self):
		"""Perform moving average on raw monitor data (if requested), and convert to physical units."""

		colNames = list(self.MonitorDF.columns.values)
		self.MonitorMask = [(self.Monitors[iMon] in colNames) for iMon in range(self.nMonitors)]

		if self.MonitorOptions['ComputeMovingAvg']:
			winTime = self.MonitorOptions['MovingAvgWindow']
			winSize = max(int(round(winTime/self.tStep)), 1)
			logging.info('iXC_Monitor::Computing moving average for window size {} ({} s)...'.format(winSize, winTime))
			for iMon in range(self.nMonitors):
				if self.MonitorMask[iMon]:
					self.MonitorDF[self.Monitors[iMon]] = self.MonitorDF[self.Monitors[iMon]].rolling(winSize, center=True, min_periods=1).mean()

		self.MonitorDF = self.ConvertMonitorData(self.MonitorDF)

		for iax in self.iaxList:
			for ik in self.ikList:
				self.MonitorSDevDF[iax][ik] = self.ConvertMonitorData(self.MonitorSDevDF[iax][ik], self.MonitorMeanDF[iax][ik], SDev=True)
				self.MonitorMeanDF[iax][ik] = self.ConvertMonitorData(self.MonitorMeanDF[iax][ik])

	############## End of Monitor.ConvertMonitorData() ##############
	#################################################################

	def ConvertMonitorData(self, DF, DFRef=None, SDev=False):
		"""Convert certain monitor data columns to physical units."""

		## Chamber Temperature conversion parameters
		RS   = 9.1E3  		# Ohm
		RT0  = 10.E3  		# Ohm
		VIn  = 14.95  		# V
		T0   = 298.15 		# K
		beta = 3455   		# K

		## LCR Temperature conversion parameters
		IBias = 1.0E-4 		# A

		## Tilt conversion parameters
		TiltOffset = 2.5 	# V
		TiltScale  = 0.5 	# rad/V

		## B-Field conversion parameters
		BScale = 0.5 		# Gauss/V

		if SDev:
			## Create new column for background subtracted NTotal
			DF['NTotal'] = np.sqrt(DF['NTotalRaw'].to_numpy()**2 + DF['NTotalBG'].to_numpy()**2)
			DF['TiltX']  = TiltScale*DF['TiltX'].to_numpy()	## rad
			DF['TiltY']  = TiltScale*DF['TiltY'].to_numpy()	## rad

			VOut  = DFRef[self.Temp_Key].to_numpy()
			dVOut = DF[self.Temp_Key].to_numpy()
			RT    = VOut/(VIn - VOut)*RS
			dRT   = abs((1./(VOut - VIn) + VOut/(VOut - VIn)**2))*dVOut*RS
			if self.MonitorOptions['ConvertToTemperature']:
				T     = 1./(1./T0 + 1./beta*np.log(RT/RT0))
				dT    = dRT/(beta*RT)*T**2
				DF[self.Temp_Key] = dT ## C
			else:
				DF[self.Temp_Key] = dRT*1.E-3 ## kOhm
		else:
			## Create new column for background subtracted NTotal
			DF['NTotal'] = DF['NTotalRaw'].to_numpy() - DF['NTotalBG'].to_numpy()
			DF['TiltX']  = np.arcsin(TiltScale*(DF['TiltX'].to_numpy() - TiltOffset)) ## rad
			DF['TiltY']  = np.arcsin(TiltScale*(DF['TiltY'].to_numpy() - TiltOffset)) ## rad

			if self.MonitorOptions['ConvertToTemperature']:
				VOut = DF[self.Temp_Key].to_numpy()
				RT   = VOut/(VIn - VOut)*RS
				DF[self.Temp_Key] = 1./(1./T0 + 1./beta*np.log(RT/RT0)) - 273.15 ## C
			else:
				VOut = DF[self.Temp_Key].to_numpy()
				DF[self.Temp_Key] = VOut/(VIn - VOut)*RS*1E-3			## kOhm

		DF[self.RamanBX_Key] = BScale*DF[self.RamanBX_Key].to_numpy()	## Gauss
		DF[self.RamanBY_Key] = BScale*DF[self.RamanBY_Key].to_numpy()	## Gauss
		DF[self.RamanBZ_Key] = BScale*DF[self.RamanBZ_Key].to_numpy() 	## Gauss
		## Create new column for total Raman magnetic field
		DF[self.RamanBT_Key] = np.sqrt(DF[self.RamanBX_Key].to_numpy()**2 + \
			DF[self.RamanBY_Key].to_numpy()**2 + DF[self.RamanBZ_Key].to_numpy()**2) ## Gauss

		## Create new columns for MOT monitor and MOT total power
		if self.MOTPMon_Key != 'None':
			DF['MOTMeanPMon'] = DF[self.MOTPMon_Key].to_numpy()			## V
		DF['MOTMeanPT'] = np.sqrt(DF[self.MOTPX_Key].to_numpy()**2 + \
			DF[self.MOTPY_Key].to_numpy()**2 + DF[self.MOTPZ_Key].to_numpy()**2) ## V

		if self.SoftwareVersion >= 3.3:
			DF['MOTMeanBX'] = BScale*DF['MOTMeanBX'].to_numpy() 		## Gauss
			DF['MOTMeanBY'] = BScale*DF['MOTMeanBY'].to_numpy() 		## Gauss
			DF['MOTMeanBZ'] = BScale*DF['MOTMeanBZ'].to_numpy() 		## Gauss
			DF['MOTMeanBT'] = np.sqrt(DF['MOTMeanBX'].to_numpy()**2 + \
				DF['MOTMeanBY'].to_numpy()**2 + DF['MOTMeanBZ'].to_numpy()**2) ## Gauss
			DF['RamanDeltaBX'] = BScale*DF['RamanDeltaBX'].to_numpy() 	## Gauss
			DF['RamanDeltaBY'] = BScale*DF['RamanDeltaBY'].to_numpy() 	## Gauss
			DF['RamanDeltaBZ'] = BScale*DF['RamanDeltaBZ'].to_numpy() 	## Gauss

		if self.SoftwareVersion >= 3.4:
			if self.MonitorOptions['ConvertToTemperature']:
				DF['TempLCRX'] = 1./(1./T0 + 1./beta*np.log(DF['TempLCRX'].to_numpy()/(IBias*RT0))) - 273.15 ## C
				DF['TempLCRY'] = 1./(1./T0 + 1./beta*np.log(DF['TempLCRY'].to_numpy()/(IBias*RT0))) - 273.15 ## C
				DF['TempLCRZ'] = 1./(1./T0 + 1./beta*np.log(DF['TempLCRZ'].to_numpy()/(IBias*RT0))) - 273.15 ## C
			else:
				DF['TempLCRX'] = DF['TempLCRX'].to_numpy()/IBias*1E-3 ## kOhm
				DF['TempLCRY'] = DF['TempLCRY'].to_numpy()/IBias*1E-3 ## kOhm
				DF['TempLCRZ'] = DF['TempLCRZ'].to_numpy()/IBias*1E-3 ## kOhm

		return DF

	############## End of Monitor.ConvertMonitorData() ##############
	#################################################################

	def PlotMonitorData(self):
		"""Plot summary of monitor data."""

		logging.info('iXC_Monitor::Plotting monitor data...')

		# self.MonitorDF = self.SetTrackTimeRange(self.MonitorDF)
		# winSize = max(int(round(Options['MovingAvgWindow']/self.tCycMeanFull)), 1)
		# self.ConvertMonitorData(self.TrackOptions['ComputeMovingAvg'], winSize)

		self.tMon = np.array((self.MonitorDF['DateTime'] - self.MonitorDF['DateTime'].iloc[0]) // pd.Timedelta('1ms'))*1.E-3

		if self.tMon[-1] > 10000:
			self.tMon *= 1.E-3
			xLabel = r'Time  ($\times 10^3$ s)'
		else:
			xLabel = 'Time  (s)'

		plt.rc('font', size=10)
		plt.rc('lines', markersize=3)
		plt.rc('axes', labelsize=11)

		if self.SoftwareVersion >= 3.4:
			(nRows, nCols) = (4, 4)
		else:
			(nRows, nCols) = (4, 3)
		(colW, rowH) = (4, 2)

		_, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*colW,nRows*rowH), sharex='col', constrained_layout=True)

		customPlotOpts = {'Color': 'red', 'Linestyle': 'None', 'Marker': '.',
			'Title': 'None', 'xLabel': 'None', 'yLabel': 'None',
			'Legend': False, 'LegLabel': 'None'}

		## 1st column
		customPlotOpts['xLabel']   = 'None'
		customPlotOpts['yLabel']   = 'Tilt X  (mrad)'
		customPlotOpts['Color']    = 'crimson'
		iXUtils.CustomPlot(axs[0,0], customPlotOpts, self.tMon, self.MonitorDF['TiltX'].to_numpy()*1.E3)

		customPlotOpts['yLabel']   = 'Tilt Y  (mrad)'
		customPlotOpts['Color']    = 'darkorange'
		iXUtils.CustomPlot(axs[1,0], customPlotOpts, self.tMon, self.MonitorDF['TiltY'].to_numpy()*1.E3)

		if self.MonitorOptions['ConvertToTemperature']:
			customPlotOpts['yLabel'] = 'Temperature  (C)'
		else:
			customPlotOpts['yLabel'] = r'Thermistor  (k$\Omega$)'

		customPlotOpts['Color']    = 'goldenrod'
		customPlotOpts['LegLabel'] = 'Chamber'
		customPlotOpts['Legend']   = True
		iXUtils.CustomPlot(axs[2,0], customPlotOpts, self.tMon, self.MonitorDF[self.Temp_Key].to_numpy())

		if self.SoftwareVersion >= 3.4:
			customPlotOpts['Color']    = 'yellowgreen'
			customPlotOpts['LegLabel'] = 'LCR X'
			iXUtils.CustomPlot(axs[2,0], customPlotOpts, self.tMon, self.MonitorDF['TempLCRX'].to_numpy())
			customPlotOpts['Color']    = 'deepskyblue'
			customPlotOpts['LegLabel'] = 'LCR Y'
			iXUtils.CustomPlot(axs[2,0], customPlotOpts, self.tMon, self.MonitorDF['TempLCRY'].to_numpy())
			customPlotOpts['Color']    = 'indianred'
			customPlotOpts['LegLabel'] = 'LCR Z'
			iXUtils.CustomPlot(axs[2,0], customPlotOpts, self.tMon, self.MonitorDF['TempLCRZ'].to_numpy())

		customPlotOpts['xLabel']   = xLabel
		customPlotOpts['yLabel']   = 'Detection  (V)'
		customPlotOpts['Color']    = 'chocolate'
		customPlotOpts['LegLabel'] = 'MOT'

		if self.SoftwareVersion <= 3.2:
			iXUtils.CustomPlot(axs[3,0], customPlotOpts, self.tMon, self.MonitorDF[self.MOTFlu_Key].to_numpy())

		customPlotOpts['Color']    = 'red'
		customPlotOpts['LegLabel'] = r'$N_{\rm total}$ Raw'
		iXUtils.CustomPlot(axs[3,0], customPlotOpts, self.tMon, self.MonitorDF['NTotalRaw'].to_numpy())

		customPlotOpts['Color']    = 'blue'
		customPlotOpts['LegLabel'] = r'$N_{\rm total}$ BG'
		iXUtils.CustomPlot(axs[3,0], customPlotOpts, self.tMon, self.MonitorDF['NTotalBG'].to_numpy())

		customPlotOpts['Color']    = 'black'
		customPlotOpts['LegLabel'] = r'$N_{\rm total}$'
		iXUtils.CustomPlot(axs[3,0], customPlotOpts, self.tMon, self.MonitorDF['NTotal'].to_numpy())

		## 2nd column
		customPlotOpts['xLabel']   = 'None'
		customPlotOpts['yLabel']   = r'MOT $\langle P_X \rangle$  (V)'
		customPlotOpts['Color']    = 'forestgreen'
		customPlotOpts['LegLabel'] = 'None'
		customPlotOpts['Legend']   = False
		iXUtils.CustomPlot(axs[0,1], customPlotOpts, self.tMon, self.MonitorDF[self.MOTPX_Key].to_numpy())

		customPlotOpts['yLabel']   = r'MOT $\langle P_Y \rangle$  (V)'
		customPlotOpts['Color']    = 'lightseagreen'
		iXUtils.CustomPlot(axs[1,1], customPlotOpts, self.tMon, self.MonitorDF[self.MOTPY_Key].to_numpy())

		customPlotOpts['yLabel']   = r'MOT $\langle P_Z \rangle$  (V)'
		customPlotOpts['Color']    = 'dodgerblue'
		iXUtils.CustomPlot(axs[2,1], customPlotOpts, self.tMon, self.MonitorDF[self.MOTPZ_Key].to_numpy())

		customPlotOpts['xLabel']   = xLabel
		customPlotOpts['yLabel']   = r'MOT $\langle P_{\rm total} \rangle$  (V)'
		customPlotOpts['Color']    = 'grey'
		customPlotOpts['LegLabel'] = 'Monitor'
		customPlotOpts['Legend']   = True
		if 'MOTMeanPMon' in self.MonitorDF.columns:
			iXUtils.CustomPlot(axs[3,1], customPlotOpts, self.tMon, self.MonitorDF['MOTMeanPMon'].to_numpy())

		customPlotOpts['Color']    = 'black'
		customPlotOpts['LegLabel'] = 'Total'
		iXUtils.CustomPlot(axs[3,1], customPlotOpts, self.tMon, self.MonitorDF['MOTMeanPT'].to_numpy())

		if self.SoftwareVersion >= 3.4:
			## 3rd column
			customPlotOpts['xLabel']   = 'None'
			customPlotOpts['yLabel']   = r'Raman $\langle P_X \rangle$  (V)'
			customPlotOpts['Color']    = 'yellowgreen'
			customPlotOpts['LegLabel'] = 'None'
			customPlotOpts['Legend']   = False
			iXUtils.CustomPlot(axs[0,2], customPlotOpts, self.tMon, self.MonitorDF['RamanMeanPX'].to_numpy())

			customPlotOpts['yLabel']   = r'Raman $\langle P_Y \rangle$  (V)'
			customPlotOpts['Color']    = 'deepskyblue'
			iXUtils.CustomPlot(axs[1,2], customPlotOpts, self.tMon, self.MonitorDF['RamanMeanPY'].to_numpy())

			customPlotOpts['yLabel']   = r'Raman $\langle P_Z \rangle$  (V)'
			customPlotOpts['Color']    = 'indianred'
			iXUtils.CustomPlot(axs[2,2], customPlotOpts, self.tMon, self.MonitorDF['RamanMeanPZ'].to_numpy())

			customPlotOpts['xLabel']   = xLabel
			customPlotOpts['yLabel']   = r'Raman $\Delta P$  (V)'
			customPlotOpts['Legend']   = True

			customPlotOpts['LegLabel'] = 'X'
			customPlotOpts['Color']    = 'olivedrab'
			iXUtils.CustomPlot(axs[3,2], customPlotOpts, self.tMon, self.MonitorDF['RamanDeltaPX'].to_numpy())

			customPlotOpts['LegLabel'] = 'Y'
			customPlotOpts['Color']    = 'darkcyan'
			iXUtils.CustomPlot(axs[3,2], customPlotOpts, self.tMon, self.MonitorDF['RamanDeltaPY'].to_numpy())

			customPlotOpts['LegLabel'] = 'Z'
			customPlotOpts['Color']    = 'orangered'
			iXUtils.CustomPlot(axs[3,2], customPlotOpts, self.tMon, self.MonitorDF['RamanDeltaPZ'].to_numpy())

		## Last column
		customPlotOpts['xLabel']   = 'None'
		customPlotOpts['yLabel']   = r'Raman $B_X$  (G)'
		customPlotOpts['Color']    = 'mediumblue'
		customPlotOpts['LegLabel'] = 'None'
		customPlotOpts['Legend']   = False
		iXUtils.CustomPlot(axs[0,nCols-1], customPlotOpts, self.tMon, self.MonitorDF[self.RamanBX_Key].to_numpy())

		customPlotOpts['yLabel']   = r'Raman $B_Y$  (G)'
		customPlotOpts['Color']    = 'blueviolet'
		iXUtils.CustomPlot(axs[1,nCols-1], customPlotOpts, self.tMon, self.MonitorDF[self.RamanBY_Key].to_numpy())

		customPlotOpts['yLabel']   = r'Raman $B_Z$  (G)'
		customPlotOpts['Color']    = 'magenta'
		iXUtils.CustomPlot(axs[2,nCols-1], customPlotOpts, self.tMon, self.MonitorDF[self.RamanBZ_Key].to_numpy())

		customPlotOpts['xLabel']   = xLabel
		customPlotOpts['yLabel']   = r'Raman $B_{\rm total}$  (G)'
		customPlotOpts['Color']    = 'black'
		iXUtils.CustomPlot(axs[3,nCols-1], customPlotOpts, self.tMon, self.MonitorDF[self.RamanBT_Key].to_numpy())

		if self.PlotOptions['SavePlot']:
			if not os.path.exists(self.PlotOptions['PlotFolderPath']):
				os.makedirs(self.PlotOptions['PlotFolderPath'])

			self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
				self.FilePrefix + '-Run{:02d}-MonitorSummary.'.format(self.Run) + self.PlotOptions['PlotExtension'])

			plt.savefig(self.PlotPath, dpi=150)
			logging.info('iXC_Monitor::Monitor data plot saved to:')
			logging.info('iXC_Monitor::  {}'.format(self.PlotPath))
		elif self.PlotOptions['ShowPlot']:
			plt.show()

	############### End of Monitor.PlotMonitorData() ################
	#################################################################

	def PlotMonitorCorrelations(self, iax, ik, yData, yLabel, MonDFType='Mean', iStart=0, iSkip=1):
		"""Plot correlations of monitor data with yData.
		ARGUMENTS:
		\t iax       (int) - Index specifying Raman axis
		\t ik        (int) - Index specifying k-direction (0 = kU, 1 = kD, special cases: 2 = kInd, 3 = kDep)
		\t yData   (array) - Data to correlate with Monitor
		\t yLabel    (str) - Y-axis plot label
		\t MonDFType (str) - String indicating which monitor dataframe to correlate yData with ('Raw' or 'Mean')
		\t iStart    (int) - Start index of yData (i.e. yData[iStart::iSkip] is aligned with MonitorMean[iax][ik])
		\t iSkip     (int) - Indices to skip in yData (i.e. yData[iStart::iSkip] is aligned with MonitorMean[iax][ik])
		"""

		logging.info('iXC_Monitor::Plotting monitor correlations...')

		plt.rc('font', size=10)
		plt.rc('lines', markersize=3)
		plt.rc('axes', labelsize=11)

		(nRows, nCols) = (4, 3)
		(colW, rowH)   = (4, 2)
		## returns list axs[row,col]
		_, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*colW,nRows*rowH), sharey='row', constrained_layout=True)
		axs = [ax for sub in list(map(list, zip(*axs))) for ax in sub]

		customPlotOpts = {'Color': 'red', 'Linestyle': 'None', 'Marker': '.',
			'Title': 'None', 'xLabel': 'None', 'yLabel': yLabel,
			'Legend': False, 'LegLabel': 'None'}

		if self.SoftwareVersion <= 3.2:
			correlMons = ['TiltX', 'TiltY', self.Temp_Key, 'NTotal',
				self.MOTPX_Key, self.MOTPY_Key, self.MOTPZ_Key, self.MOTPMon_Key, 
				'RamanBX', 'RamanBY', 'RamanBZ', 'RamanBT']
			xLabels = ['Tilt X  (rad)', 'Tilt Y  (rad)', 'Temperature  (C)', r'$N_{\rm total}$  (V)', 
				r'MOT $P_X$  (V)', r'MOT $P_Y$  (V)', r'MOT $P_Z$  (V)', r'MOT $P_T$  (V)', 
				r'Raman $B_X$  (G)', r'Raman $B_Y$  (G)', r'Raman $B_Z$  (G)', r'Raman $B_T$  (G)']
		else:
			correlMons = ['TiltX', 'TiltY', self.Temp_Key, 'NTotal', 
				self.MOTPX_Key, self.MOTPY_Key, self.MOTPZ_Key, self.MOTPMon_Key, 
				'RamanMeanPX', 'RamanMeanPY', 'RamanMeanPZ', 'RamanDeltaPZ']
			xLabels = ['Tilt X  (rad)', 'Tilt Y  (rad)', 'Temperature  (C)', r'$N_{\rm total}$  (V)', 
				r'MOT $P_X$  (V)', r'MOT $P_Y$  (V)', r'MOT $P_Z$  (V)', r'MOT $P_T$  (V)', 
				r'Raman $P_X$  (V)', r'Raman $P_Y$  (V)', r'Raman $P_Z$  (V)', r'Raman $\Delta P_Z$  (V)']

		colors = ['crimson', 'darkorange', 'goldenrod', 'red', 'forestgreen', 'lightseagreen', 
			'dodgerblue', 'black', 'mediumblue', 'blueviolet', 'magenta', 'black']

		if MonDFType == 'Raw':
			df       = self.MonitorDF[correlMons].iloc[iStart::iSkip]
			nyData   = len(yData)
			nMonData = df.shape[0]
		else:
			nyData = len(yData[iStart::iSkip])
			dfList = self.MonitorMeanDF			
			if ik < 2:
				nMonData = dfList[iax][ik].shape[0]
				df = dfList[iax][ik][correlMons]
			else:
				nMonData = dfList[iax][0].shape[0]
				df = 0.5*(dfList[iax][0][correlMons] + dfList[iax][1][correlMons])

		if nyData != nMonData:
			logging.error('iXC_Monitor::PlotMonitorCorrelations::Lengths of yData ({}) and MonitorMeanDF ({}) do not match...'.format(nyData, nMonData))
			logging.error('iXC_Monitor::PlotMonitorCorrelations::Aborting...')
			quit()

		for iMon in range(len(correlMons)):
			xData = df[correlMons[iMon]].to_numpy()
			customPlotOpts['xLabel'] = xLabels[iMon]
			customPlotOpts['Color']  = colors[iMon]
			iXUtils.CustomPlot(axs[iMon], customPlotOpts, xData, yData)

		# pOpt  = np.polyfit(xData, yData, deg=1)
		# pFit  = np.poly1d(pOpt)
		# customPlotOpts['Color']     = 'black'
		# customPlotOpts['Linestyle'] = '-'
		# customPlotOpts['Marker']    = 'None'
		# iXUtils.CustomPlot(axs[0,0], customPlotOpts, xData, pFit(xData))

		if self.PlotOptions['SavePlot']:
			if not os.path.exists(self.PlotOptions['PlotFolderPath']):
				os.makedirs(self.PlotOptions['PlotFolderPath'])

			self.PlotPath = os.path.join(self.PlotOptions['PlotFolderPath'],
				self.DataType+'-Run{:02d}-Correlations.'.format(self.Run) + self.PlotOptions['PlotExtension'])

			plt.savefig(self.PlotPath, dpi=150)
			logging.info('iXC_Monitor::Correlation data plot saved to:')
			logging.info('iXC_Monitor::  {}'.format(self.PlotPath))
		elif self.PlotOptions['ShowPlot']:
			plt.show()

	########### End of Monitor.PlotMonitorCorrelations() ############
	#################################################################