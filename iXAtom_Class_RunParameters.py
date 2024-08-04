#####################################################################
## Filename:	iXAtom_Class_RunParameters.py
## Author:		B. Barrett
## Description: RunParameters class definition for iXAtom analysis package
## Version:		3.2.5
## Last Mod:	20/11/2020
#####################################################################

import csv
import datetime as dt
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytz

import iXAtom_Class_Physics as iXC_Phys
import iXAtom_Utilities 	as iXUtils

class RunParameters:
	#################################################################
	## Class for parsing and storing run parameters from file
	#################################################################

	## Class variables
	VersionChangeDate = dt.datetime(2019,9,1,0,0,0)

	def __init__(self, WorkDir, Folder, RunNum):
		"""Initialize run parameters. Data types are used to parse the parameter file.
		ARGUMENTS:
		\t WorkDir (str) - Path to the top-level directory where dataset is located
		\t Folder (str)  - Name of folder within WorkDir where dataset is located
		\t RunNum (int)  - Run number of requested dataset
		"""
		## Instance variables: parameters from 'RunParameters.txt'
		self.SoftwareVersion = 0.
		self.Run = 0
		self.DataType = 'Unknown'
		self.TrackProtocol = 'Unknown'
		self.TrackMode = 'Unknown'
		self.AxisMode = 'Unknown'
		self.RamanScanMode = 'Unknown'
		self.ScanQuantity = 'Unknown'

		## Pre-v3.4 keys:
		self.kUpFrequency = 0.
		self.kDownFrequency = 0.
		self.kUpChirpRate = 0.
		self.kDownChirpRate = 0.
		## Post-v3.4 keys:
		self.RamankUFreq = [0., 0., 0.]
		self.RamankDFreq = [0., 0., 0.]
		self.RamankUChirp = [0., 0., 0.]
		self.RamankDChirp = [0., 0., 0.]

		self.RamanDetuning = 0.
		self.RamanPower = 0.
		self.RamanTOF = 0.
		self.RamanT = 0.
		self.RamanpiX = 0.
		self.RamanpiY = 0.
		self.RamanpiZ = 0.
		self.MOTCoils = 0.
		self.MOTLoading = 0.
		self.MolassesTime = 0.
		self.GMolassFrequency = 0.
		self.RepumpFreqs = [0., 0.]
		self.SelectionFreqs = [0., 0.]
		self.SelectionSweepTime = 0.
		self.BiasMOTX = 0.
		self.BiasMOTY = 0.
		self.BiasMOTZ = 0.
		self.BiasRamanX = 0.
		self.BiasRamanY = 0.
		self.BiasRamanZ = 0.
		self.BiasGradientZ = 0.
		self.LCRMOTX = 0.
		self.LCRMOTY = 0.
		self.LCRMOTZ = 0.
		self.LCRRamanX = 0.
		self.LCRRamanY = 0.
		self.LCRRamanZ = 0.
		self.LCRBurstUp = 0.
		self.LCRBurstDownX = 0.
		self.LCRBurstDownY = 0.
		self.LCRBurstDownZ = 0.
		self.LCRBurstUpLength = 0.
		self.LCRBurstDownLength = 0.
		self.TiltX = 0.
		self.TiltZ = 0.
		self.ShieldState = 'Unknown'
		self.UsingRTSequencer = False
		self.RTAccelType = 'Unknown'
		self.RTPhaseContinuous = False
		self.RTPhaseEnabled = False
		self.RTPhaseFilterEnabled = False
		self.RTFreqEnabled = False
		self.RTPositiveFeedback = False
		self.RTModelAccel = False
		self.DetectConfig = 'Unknown'
		self.Detector = 'Unknown'
		self.DetectCursors = [0,0,0,0,0,0,0,0]
		self.DetectTOF = 0.

		## Pre-v3.1 keys:
		# self.RamanAxisMode = 'Unknown'
		# self.Ramanpi = 0.
		# self.Ramanpi/2 = 0. 
		# self.MolassesTOF = 0.
		# self.DetectionTOF = 0.
		# self.MOTPower = 0.
		# self.MOTX = 0.
		# self.MOTY = 0.
		# self.MOTZ = 0.
		# self.BiasX = 0. 
		# self.BiasY = 0.
		# self.BiasZ = 0.
		# self.UsingRTDDS? = False

		## Path variables
		self.WorkDir        = WorkDir
		self.Folder         = Folder
		self.RunString      = 'Run {:02d}'.format(RunNum)
		self.RawFolderPath  = os.path.join(self.WorkDir, self.Folder, self.RunString)
		self.RawFileName    = 'Parameters.txt'
		self.RawFilePath    = os.path.join(self.RawFolderPath, self.RawFileName)
		self.RawDataDF		= []
		self.PostFolderPath = os.path.join(self.WorkDir, 'PostProcessed', self.Folder, self.RunString)
		self.PostFileNames  = [['' for ik in range(2)] for iax in range(3)]
		self.PostFilePaths  = [['' for ik in range(2)] for iax in range(3)]
		self.PostDataDF		= []
		self.StreamFolderPath = os.path.join(self.WorkDir, 'Streaming', self.Folder, self.RunString)

		self.DefaultPlotColors = [
			['limegreen', 'darkgreen', 'darkorange', 'purple'],
			['royalblue', 'blue', 'forestgreen', 'deeppink'],
			['red', 'darkred', 'grey', 'black']]
		self.DetectNames    = ['Lower', 'Middle', 'Upper']
		self.idActive       = -1
		self.idList 		= []
		self.ikList			= []
		self.iaxList		= []
		self.ChirpedData 	= False
		self.kInterlaced 	= False
		self.kSign			= [[+1,-1], [+1,-1], [+1,-1]] ## Sign of keff for [iax=0,1,2][ik=0,1]

	################ End of RunParameters.__init__() ################
	#################################################################

	def GetRunAttributes(self):
		"""Get attributes of parameter file."""

		if os.path.exists(self.RawFilePath):
			self.FileCreationDate = dt.datetime.fromtimestamp(int(os.path.getmtime(self.RawFilePath)))
			if self.FileCreationDate < RunParameters.VersionChangeDate:
				self.SoftwareVersion = 3.0
				logging.warning('iXC_RunPars::GetAttributes::Parameter file created before {}.'.format(RunParameters.VersionChangeDate.strftime('%Y-%m-%d')))
				logging.warning('iXC_RunPars::GetAttributes::Using pre-v3.1 data format...')
			else:
				self.SoftwareVersion = 3.1
		else:
			logging.error('iXC_RunPars::GetAttributes::Parameter file not found in: {}'.format(self.RawFolderPath))
			logging.error('iXC_RunPars::GetAttributes::Aborting...')
			quit()

	########### End of RunParameters.GetRunAttributes() #############
	#################################################################

	def ParseRunParameters(self):
		"""Parse run parameters from parameter file into a dictionary."""

		if self.SoftwareVersion >= 3.1:
			## Use v3.1 formatting
			parDF    = pd.read_csv(self.RawFilePath, delimiter=':', header=None)
			parList  = parDF.values
			parItems = list(self.__dict__.items())
			for par in parList:
				for (key,val) in parItems:
					## Delete all whitespace and '?' in par
					if par[0].replace(' ','').replace('?','') == key: # key found in parList
						# print(key, type(val), par[1][1:])
						if type(val) is int:
							setattr(self, key, int(par[1][1:]))
						elif type(val) is float:
							setattr(self, key, float(par[1][1:]))
						elif type(val) is bool:
							setattr(self, key, bool(int(par[1][1:])))
						elif type(val) is list:
							sList = par[1][1:].split(' ')
							if type(val[0]) is float:
								setattr(self, key, [float(s) for s in sList])
							else:
								setattr(self, key, [int(s) for s in sList])
						else: # type(val) is str
							setattr(self, key, par[1][1:])
						break
		else:
			## Use pre-v3.1 formatting
			with open(self.RawFilePath,'r') as f:
				parText = f.read()

			parList  = parText.split('\n')[:-1] # Omit last element which is an empty line
			parItems = list(self.__dict__.items())

			for par in parList:
				for (key,val) in parItems:
					# Remove whitespace and '?' in par
					index = par.replace(' ','').replace('?','').find(key+' ') # append a space to make key unique
					if index >= 0: # key found in parList
						iValue = index + len(key) + 1 # Add key length + 1 to account for appended space
						if type(val) is int:
							setattr(self, key, int(par[iValue:]))
						elif type(val) is float:
							setattr(self, key, float(par[iValue:]))
						elif type(val) is bool:
							setattr(self, key, bool(int(par[iValue:])))
						elif type(val) is list:
							setattr(self, key, [int(s) for s in par[iValue:].split(' ')])
						else: # type(val) is str:
							setattr(self, key, par[iValue:])
						break

	########### End of RunParameters.ParseRunParameters() ###########
	#################################################################

	def SetPhysicalConstants(self):
		"""Set useful physical constants and AI parameters based on run parameters."""

		self.Phys      = iXC_Phys.Physics(self)	## Instance of iXAtom Physics class

		## Useful physical constants
		self.gLocal    = self.Phys.gLocal	## (float) Local value of g (m/s^2)
		self.aBody     = self.Phys.aBody	## (np.array) Acceleration vector in body frame (m/s^2)
		self.omegaHF   = self.Phys.omegaHF	## (float) Hyperfine splitting of 87Rb 5S_{1/2} (rad/s)
		self.kBoltz    = self.Phys.kBoltz	## (float) Boltzmann's constant (J/K)
		self.MRb	   = self.Phys.MRb		## (float) Atomic mass of 87Rb (kg)

		## Atom interferometer parameters
		self.keff      = self.Phys.keff  	## (float) Effective Raman wavenumber (rad/m)
		self.deltaSum  = self.Phys.deltaSum	## (float) Raman half-sum frequency (rad/s)
		self.deltaDiff = self.Phys.deltaDiff## (float) Raman half-difference frequency (rad/s)
		self.omegaR    = self.Phys.omegaR 	## (float) Recoil frequency (rad/s)
		self.taupi 	   = self.Phys.taupi 	## (list)  Raman pi-pulse durations (s)
		self.Omegaeff  = self.Phys.Omegaeff	## (list)  Effective Rabi frequencies (rad/s)
		self.Teff      = self.Phys.Teff 	## (list)  Effective interrogation times (s)
		self.Seff      = self.Phys.Seff		## (list)  Effective AI scale factors (rad/m/s^2)
		self.alpha     = self.Phys.alphakU	## (list)  Raman chirp rates (rad/s^2)
		self.TempScale = (0.5*np.pi**2/np.log(2))*self.MRb/(self.kBoltz*self.keff**2)*1.E12 ## (float) Conversion factor from peak FWHM to temperature (uK/kHz**2)

	######### End of RunParameters.SetPhysicalConstants() ###########
	#################################################################

	def GetDetectConfig(self):
		"""Parse detection configuration from run parameters."""

		logging.info('iXC_RunPars::Getting detection config for {}...'.format(self.RunString))

		## Set idList based on run parameter: 'DetectConfig'
		if self.DetectConfig == 'All Detectors':
			self.idList = [0,1,2] # 0: Lower detector, 1: Middle, 2: Upper
		elif self.DetectConfig == 'Lower + Middle':
			self.idList = [0,1]
		elif self.DetectConfig == 'Middle + Upper':
			self.idList = [1,2]
		elif self.DetectConfig == 'Lower Only':
			self.idList = [0]
		elif self.DetectConfig == 'Middle Only':
			self.idList = [1]
		elif self.DetectConfig == 'Upper Only':
			self.idList = [2]
		else:
			logging.error('iXC_RunPars::GetDetectConfig::Detection config not recognized: {}'.format(self.DetectConfig))
			logging.error('iXC_RunPars::GetDetectConfig::Aborting...')
			quit()

		for id in range(len(self.DetectNames)):
			if self.Detector == self.DetectNames[id]:
				self.idActive = id
		if self.idActive == -1:
			logging.warning('iXC_RunPars::GetDetectConfig::Detector not recognized: {}'.format(self.Detector))

	############ End of RunParameters.GetDetectConfig() #############
	#################################################################

	def SetDetectCoeffs(self, DetectCoeffs):
		"""Set detector coefficients."""

		self.idCoeffs = np.array([0.,0.,0.])
		if len(self.idList) == 1:
			self.idCoeffs[self.idList[0]] = 1.
		else:
			for id in self.idList:
				self.idCoeffs[id] = abs(DetectCoeffs[id])
			## Renormalize
			self.idCoeffs /= np.sum(self.idCoeffs)

	############ End of RunParameters.SetDetectCoeffs() #############
	#################################################################

	def GetRamanConfig(self):
		"""Get Raman beam configuration."""

		logging.info('iXC_RunPars::Getting Raman config for {}...'.format(self.RunString))

		## Get iaxList based on run parameter 'Axis Mode'
		if self.AxisMode == 'X-Only':
			self.iaxList = [0]
		elif self.AxisMode == 'Y-Only':
			self.iaxList = [1]
		elif self.AxisMode == 'Z-Only':
			self.iaxList = [2]
		elif self.AxisMode == 'X, Y':
			self.iaxList = [0,1]
		elif self.AxisMode == 'Y, Z':
			self.iaxList = [1,2]
		elif self.AxisMode == 'Z, X':
			self.iaxList = [2,0]
		elif self.AxisMode == 'X, Y, Z':
			self.iaxList = [0,1,2]
		else:
			logging.error('iXC_RunPars::GetRamanConfig::Axis Mode not recognized: {}'.format(self.AxisMode))
			logging.error('iXC_RunPars::GetRamanConfig::Aborting...')
			quit()

		if self.SoftwareVersion >= 3.4:
			self.kUpFrequency   = self.RamankUFreq[self.iaxList[0]]
			self.kDownFrequency = self.RamankDFreq[self.iaxList[0]]
			self.kUpChirpRate   = self.RamankUChirp[self.iaxList[0]]
			self.kDownChirpRate = self.RamankDChirp[self.iaxList[0]]
		else:
			self.RamankUFreq[self.iaxList[0]]  = self.kUpFrequency
			self.RamankDFreq[self.iaxList[0]]  = self.kDownFrequency
			self.RamankUChirp[self.iaxList[0]] = self.kUpChirpRate
			self.RamankDChirp[self.iaxList[0]] = self.kDownChirpRate

		if self.DataType == 'Ramsey':
			self.ikList = [0]
			self.ChirpedData = False
			self.kInterlaced = False
		else:
			## Get ik indices, ChirpData and kInterlaced flags based on run parameter 'RamanScanMode'
			if self.RamanScanMode == 'Chirp kInterlaced':
				self.ikList = [0,1]
				self.ChirpedData = True
				self.kInterlaced = True
			elif self.RamanScanMode == 'Chirp kUp':
				self.ikList = [0]
				self.ChirpedData = True
				self.kInterlaced = False
			elif self.RamanScanMode == 'Chirp kDown':
				self.ikList = [1]
				self.ChirpedData = True
				self.kInterlaced = False
			elif self.RamanScanMode == 'Phase kInterlaced' or self.RamanScanMode == 'Phase kInterlaced Fixed Chirp':
				self.ikList = [0,1]
				self.ChirpedData = False
				self.kInterlaced = True
			elif self.RamanScanMode == 'Phase kUp':
				self.ikList = [0]
				self.ChirpedData = False
				self.kInterlaced = False
			elif self.RamanScanMode == 'Phase kDown':
				self.ikList = [1]
				self.ChirpedData = False
				self.kInterlaced = False
			else:
				logging.error('iXC_RunPars::GetRamanConfig::Raman Scan Mode not recognized: {}'.format(self.RamanScanMode))
				logging.error('iXC_RunPars::GetRamanConfig::Aborting...')
				quit()

		self.nk  = len(self.ikList)
		self.nax = len(self.iaxList)

	############ End of RunParameters.GetRamanConfig() ##############
	#################################################################

	def SetFilePaths(self):
		"""Derive names and paths for post-processed data files."""

		if self.DataType == 'Tracking':
			self.FilePrefix = 'Track'
		else:
			self.FilePrefix = self.DataType

		if self.DataType == 'Detection' or self.DataType == 'Rabi' or self.DataType == 'Ramsey':
			self.AxisFileLabels = ['X','Y','Z']
			self.AxisLegLabels  = ['X','Y','Z']
			self.PostFileNames  = [self.FilePrefix+'-Run{:02d}-Ratios'.format(self.Run)+'-'+self.AxisFileLabels[iax]+'.txt' for iax in range(3)]
			self.PostFilePaths  = [os.path.join(self.PostFolderPath, self.PostFileNames[iax]) for iax in range(3)]
		else:
			self.AxisFileLabels = [['X-kU','X-kD'], ['Y-kU','Y-kD'], ['Z-kU','Z-kD']] ## Default values reset in SetPostFileNames()
			self.AxisLegLabels  = [
				[r'X-$k_U$',r'X-$k_D$',r'X-$k_{\rm ind}$',r'X-$k_{\rm dep}$'], 
				[r'Y-$k_U$',r'Y-$k_D$',r'Y-$k_{\rm ind}$',r'Y-$k_{\rm dep}$'], 
				[r'Z-$k_U$',r'Z-$k_D$',r'Z-$k_{\rm ind}$',r'Z-$k_{\rm dep}$']]
			self.PostFileNames  = [[self.FilePrefix+'-Run{:02d}-Ratios'.format(self.Run)+'-'+self.AxisFileLabels[iax][ik]+'.txt' for ik in range(2)] for iax in range(3)]
			self.PostFilePaths  = [[os.path.join(self.PostFolderPath, self.PostFileNames[iax][ik]) for ik in range(2)] for iax in range(3)]

	############## End of RunParameters.SetFilePaths() ##############
	#################################################################

	def LoadRunParameters(self):
		"""Get parameter file attributes and load run parameters from file."""

		self.GetRunAttributes()
		logging.info('iXC_RunPars::===================================================================================')		
		logging.info('iXC_RunPars::Loading run parameters for {}...'.format(self.RunString))
		self.ParseRunParameters()
		self.SetPhysicalConstants()
		self.GetDetectConfig()
		self.GetRamanConfig()
		self.SetFilePaths()

	########### End of RunParameters.LoadRunParameters() ############
	#################################################################

	def GetRunTiming(self, PrintRunTiming=False, RawDF=True):
		"""Get run timing from raw dataset."""

		if self.DataType == 'Raman':
			if RawDF:
				dfList = self.RawDataDF
			else:
				dfList = self.PostDataDF
		else: #elif self.DataType == 'Ramsey':
			if RawDF:
				dfList = [[self.RawDataDF[iax] for ik in range(2)] for iax in range(3)]
			else:
				dfList = [[self.PostDataDF[iax] for ik in range(2)] for iax in range(3)]

		dateStamp, timeStamp = dfList[self.iaxList[0]][self.ikList[0]][['Date', 'Time']].iloc[0]
		self.RunStartTime = iXUtils.TimestampToDatetime(dateStamp, timeStamp)

		dateStamp, timeStamp = dfList[self.iaxList[-1]][self.ikList[-1]][['Date', 'Time']].iloc[-1]
		self.RunStopTime = iXUtils.TimestampToDatetime(dateStamp, timeStamp)

		## Returns timezone-naive mid-run time since epoch (01/01/1970):
		self.RunTime     = (self.RunStartTime.timestamp() + self.RunStopTime.timestamp())/2.
		self.RunDuration = (self.RunStopTime - self.RunStartTime).total_seconds()

		## Count number of shots per k-dir and per axis
		self.RunIters = [[0 for ik in range(2)] for iax in range(3)]
		for iax in self.iaxList:
			for ik in self.ikList:
				self.RunIters[iax][ik] = dfList[iax][ik].shape[0]

		self.RunTotalIters = sum([item for sublist in self.RunIters for item in sublist])
		self.RunCycleTime  = self.RunDuration/(self.RunTotalIters - 1)
		self.RunCycleRate  = 1./self.RunCycleTime

		nIterPerAxis = sum([self.RunIters[self.iaxList[0]][ik] for ik in self.ikList])
		nIterPerkDir = self.RunIters[self.iaxList[0]][self.ikList[0]]

		if PrintRunTiming:
			print('--------------------- Run {:02d} Timing --------------------'.format(self.Run))
			print(' Total  Iterations: {}'.format(self.RunTotalIters))
			print(' Num of Iters/axis: {}'.format(nIterPerAxis))
			print(' Num of Iters/kdir: {}'.format(nIterPerkDir))
			print(' Run Start  Time:   {}'.format(self.RunStartTime))
			print(' Run Middle Time:   {}'.format(dt.datetime.fromtimestamp(self.RunTime, tz=pytz.timezone('Europe/Paris'))))
			print(' Run Stop   Time:   {}'.format(self.RunStopTime))
			print(' Run Duration:       {:04d} s ({:4.2f} min)'.format(int(round(self.RunDuration)), self.RunDuration/60.))
			print(' Run Cycle Time:    {:5.3f} s'.format(self.RunCycleTime))
			print(' Run Cycle Rate:    {:5.3f} Hz'.format(self.RunCycleRate))
			print('-------------------------------------------------------')

	############# End of RunParameters.GetRunTiming() ###############
	#################################################################

	@staticmethod
	def LoadData(FolderPath, FileName, Separator='\t'):
		"""Generic routine for loading a dataframe from tab-separated file.
		ARGUMENTS:
		\t FolderPath (str)  - Path to folder containing data file.
		\t FileName   (list) - Names of data files to load.
		\t Separator  (str)  - Column separator in data file.
		RETURN FORMAT: [DataFound, DFList]
		\t Found      (bool) - Flag indicating if data file was found.
		\t DF         (list) - Pandas dataframe containing data.
		"""

		DF       = pd.DataFrame([])
		filePath = os.path.join(FolderPath, FileName)
		if os.path.exists(filePath):
			Found = True
			DF    = pd.read_csv(filePath, sep=Separator)
		else:
			Found = False
			logging.warning('iXC_RunPars::LoadData::Data file {} not found in: {}'.format(FileName, FolderPath))

		return [Found, DF]

	################ End of RunParameters.LoadData() ################
	#################################################################

	@staticmethod
	def CreatePlotAxes(FigAxsIn, iRun, nRuns, nax, PlotOpts, ProcessLevel):
		"""Create plot axes for Raman/Ramsey/Rabi AnalysisLevel = 2."""

		if PlotOpts['PlotData'] and nRuns <= PlotOpts['MaxPlotsToDisplay']:
			if PlotOpts['OverlayRunPlots']:
				nRows   = 2*nax
				hRatios = [[1,3] for iax in range(nax)]
				hRatios = [hRatio for sub in hRatios for hRatio in sub] ## Flatten list
				if ProcessLevel < 2 and iRun == 0:
					## Create a single figure, with one column and two rows for each Raman axis (fit + residual)
					nCols  = 1
					axs    = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*7,nax*3.), sharex='col', constrained_layout=True, gridspec_kw={'height_ratios': hRatios})[1]
					FigAxs = [[[axs[r] for r in range(2*iax,2*iax+2)] for iax in range(nax)]]
				elif ProcessLevel == 2 and iRun == 0:
					## Create a single figure, with two columns and two rows for each Raman axis (fit + residual)
					nCols  = 2
					axs    = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*7,nax*3.), sharex='col', sharey='row', constrained_layout=True, gridspec_kw={'height_ratios': hRatios})[1]
					FigAxs = [
						[[axs[r,0] for r in range(2*iax,2*iax+2)] for iax in range(nax)], 
						[[axs[r,1] for r in range(2*iax,2*iax+2)] for iax in range(nax)]] 
				else:
					FigAxs   = FigAxsIn
			else: 
				nRows = 2
				if ProcessLevel < 2:
					## A new figure is created on each iteration for all Raman axes, with one column (raw or post) and two rows (fit + residuals)
					nCols  = 1
					axs    = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*7,3.), sharex='col', constrained_layout=True, gridspec_kw={'height_ratios': [1,3]})[1]
					FigAxs = [[[axs[0], axs[1]]]]
				else:
					## A new figure is created on each iteration for all Raman axes, with two columns (raw + post) and two rows (fit + residuals)
					nCols  = 2
					axs    = plt.subplots(nrows=nRows, ncols=nCols, figsize=(nCols*7,3.), sharex='col', sharey='row', constrained_layout=True, gridspec_kw={'height_ratios': [1,3]})[1]
					FigAxs = [[[axs[0,0], axs[1,0]]], [[axs[0,1], axs[1,1]]]]
		else:
			FigAxs = FigAxsIn

		return FigAxs

	############ End of RunParameters.CreatePlotAxes() ##############
	#################################################################
