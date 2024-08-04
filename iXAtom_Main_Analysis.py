#####################################################################
## Filename:	iXAtom_Main.py
## Author:		B. Barrett
## Version:		3.2.5
## Description:	Main implementation of iXAtom analysis package.
## 				Compatible with LabVIEW Data Acquisition v3.2 and later.
## Last Mod:	14/11/2020
#####################################################################

import os
import lmfit  as lm
import logging
import numpy  as np

import iXAtom_Utilities 		  as iXUtils
import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Class_Detector 	  as iXC_Detect
import iXAtom_Class_Rabi 		  as iXC_Rabi
import iXAtom_Class_Ramsey 		  as iXC_Ramsey
import iXAtom_Class_Raman 		  as iXC_Raman
import iXAtom_Class_Track		  as iXC_Track
import iXAtom_Class_Stream		  as iXC_Stream
import iXAtom_Class_Monitor       as iXC_Monitor

##===================================================================
##  Set analysis options
##===================================================================
##----------- Levels for all data types excluding Tracking ----------
## AnalysisLevel = 0: Plot specific detection data
## ProcessLevel  = 0: Post-process a single detection shot (Run = RunList[0], Iter = IterNum)
## ProcessLevel  = 1: Plot aggregated monitor data (Runs = RunList)
##---------------- Levels for Raman/Ramsey/Rabi data ----------------
## AnalysisLevel = 1: Post-process all detection shots for selected Runs (ProcessLevel ignored)
## AnalysisLevel = 2: Reload and analyze data for selected Runs, store analysis results
## AnalysisLevel = 3: Reload analysis results and plot fit parameters for selected Runs
## AnalysisLevel = 4: Reload analysis results and plot time series analysis for selected Runs
## ProcessLevel  = 0: Analyze raw data only
## ProcessLevel  = 1: Analyze post-processed data only
## ProcessLevel  = 2: Analyze both raw and post-processed data
##--------------------- Levels for Tracking data --------------------
## AnalysisLevel = 0: Load monitor data and plot time series
## AnalysisLevel = 1: Load tracking data, perform time series analysis of ratio, phase and error signals
## AnalysisLevel = 2: Load tracking data, perform time series analysis of quantum accelerometers
## AnalysisLevel = 3: Load RT data, perform time series analysis of classical accelerometers
## AnalysisLevel = 4: Load tracking and monitor data, plot correlations
## ProcessLevel		: Ignored
##-------------------- Levels for Streaming data --------------------
## AnalysisLevel = 1: Load streamed accelerometer data and plot time series (RunNum = RunList[0])
## ProcessLevel		: Ignored
##-------------------------------------------------------------------

PathCtrl = {
	'RootDir':				'C:\\Bryns Goodies\\Work-iXAtom', 				## Root directory containing data
	# 'RootDir':				'C:\\iXAtom Desktop', 				## Root directory containing data

	# 'DataDate':				{'Day': 25, 'Month': 'June', 'Year': 2020},	## Date the data was recorded

	# 'DataDate':				{'Day': 29, 'Month': 'July', 'Year': 2020},	## Date the data was recorded

	# 'DataDate':				{'Day': 21, 'Month': 'August', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 28, 'Month': 'August', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 29, 'Month': 'August', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 31, 'Month': 'August', 'Year': 2020},	## Date the data was recorded

	# 'DataDate':				{'Day': 1, 'Month': 'September', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 3, 'Month': 'September', 'Year': 2020},	## Date the data was recorded

	# 'DataDate':				{'Day': 1, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 2, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 13, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 15, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 19, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 22, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	'DataDate':				{'Day': 23, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 26, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 27, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 28, 'Month': 'October', 'Year': 2020},	## Date the data was recorded
	# 'DataDate':				{'Day': 29, 'Month': 'October', 'Year': 2020},	## Date the data was recorded

	# 'DataDate':				{'Day': 6, 'Month': 'November', 'Year': 2020},	## Date the data was recorded
}

workDir = os.path.join(PathCtrl['RootDir'], 'Data {}'.format(PathCtrl['DataDate']['Year']),
	PathCtrl['DataDate']['Month'], '{:02d}'.format(PathCtrl['DataDate']['Day']))

AnalysisCtrl = {
	'AnalysisLevels':		[2,3],			## List of primary data analysis control types
	'ProcessLevel': 		0,				## Secondary control of data analysis processing
	'WorkDir': 				workDir, 		## Working directory
	'Folder': 		 		'Raman',		## Data folder within WorkDir (e.g. 'Raman', 'Ramsey', 'Rabi', 'Tracking', 'Streaming')
	'SubFolder':			'None', 		## Subfolder within Folder (Folder = 'Streaming' only, SubFolder ignored if 'None')
	# 'RunList':				[1,2],			## List of run numbers of loop over

	# 'RunList':				list(range( 1,17+1)),	## (June 25, Raman Fringes vs TOF, Z-Only, Tilt X,Z = 360, 0 deg)
	# 'RunList':				list(range(18,35+1)),	## (June 25, Raman Fringes vs T, Z-Only, Tilt X,Z = 360, 0 deg)
	# 'RunList':				[28,20,19],	## (June 25, Raman Fringes vs T, Z-Only, Tilt X,Z = 360, 0 deg)

	# 'RunList':				list(range(1,36+1)),	## (July 29, kCo Ramsey Spectrum vs TOF, Z-Only, Tilt X,Z = 360, 0 deg)

	# 'RunList':				list(range(10,18+1)),	## (Aug 21, Raman Fringes vs T, X-Only, Tilt X,Z = 340, -30 deg)
	# 'RunList':				list(range(19,27+1)),	## (Aug 21, Raman Fringes vs T, Y-Only, Tilt X,Z = 340, -30 deg)
	# 'RunList':				list(range(28,36+1)),	## (Aug 21, Raman Fringes vs T, Z-Only, Tilt X,Z = 340, -30 deg)
	# 'RunList':				list(range(37,45+1)),	## (Aug 21, Raman Fringes vs T, X-Only, Tilt X,Z = 320, -30 deg) ***Use BadRuns = [38] (kU data missing)
	# 'RunList':				list(range(46,54+1)),	## (Aug 21, Raman Fringes vs T, Y-Only, Tilt X,Z = 320, -30 deg)
	# 'RunList':				list(range(55,63+1)),	## (Aug 21, Raman Fringes vs T, Z-Only, Tilt X,Z = 320, -30 deg)

	# 'RunList':				list(range( 1, 8+1)),	## (Aug 28, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  30, +30 deg)

	# 'RunList':				list(range( 1, 8+1)),	## (Aug 29, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  330, +30 deg)
	# 'RunList':				list(range( 9,16+1)),	## (Aug 29, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z = 300, +30 deg)
	# 'RunList':				list(range(17,24+1)),	## (Aug 29, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  40, +30 deg)

	# 'RunList':				list(range( 1, 8+1)),	## (Aug 31, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  20, +30 deg)
	# 'RunList':				list(range(11,18+1)),	## (Aug 31, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z = 340, +30 deg)

	# 'RunList':				list(range( 1, 8+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  30, -15 deg)
	# 'RunList':				list(range( 9,16+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z = 320, -15 deg)
	# 'RunList':				list(range(17,24+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z = 300, -15 deg)
	# 'RunList':				list(range(25,32+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  50, -15 deg)
	# 'RunList':				list(range(33,40+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  20, -45 deg)
	# 'RunList':				list(range(41,48+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  40, -45 deg)
	# 'RunList':				list(range(49,50+1)),	## (Sep 01, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z =  56, -45 deg)

	# 'RunList':				list(range( 1, 1+1)),	## (Sep 03, Raman Fringes vs T, X,Y,Z-axes, Tilt X,Z = 315, -45 deg)

	# 'RunList':				list(range(83,94+1)),	## (Oct 01, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z = 360, -45 deg)
	# 'RunList':				list(range(83,84+1)),	## (Oct 01, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z = 360, -45 deg)
	# 'RunList':				list(range(98,108+1)),	## (Oct 01, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z = 330, -45 deg)

	# 'RunList':				list(range( 3,11+1)),	## (Oct 02, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z =  30, -45 deg)
	# 'RunList':				list(range(14,22+1)),	## (Oct 02, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z =  45, -45 deg)
	# 'RunList':				list(range(25,33+1)),	## (Oct 02, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z = 316, -45 deg)
	# 'RunList':				list(range(36,44+1)),	## (Oct 02, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z = 300, -45 deg)
	# 'RunList':				list(range(47,55+1)),	## (Oct 02, kCo Ramsey Spectrum vs TOF, X,Y,Z-axes, Tilt X,Z =  60, -45 deg)

	# 'RunList':				list(range(1,3+1)),		## (Oct 15, kU  Rabi Oscillation vs TOF, X,Y,Z-axes, Tilt X,Z = 54.7, -45 deg)

	# 'RunList':				list(range(1,5+1)),		## (Oct 19, kU  Rabi Oscillation vs TOF, X,Y,Z-axes, Tilt X,Z = 54.7, -45 deg)
	# 'RunList':				list(range(6,10+1)),	## (Oct 19, kCo Rabi Oscillation vs TOF, X,Y,Z-axes, Tilt X,Z = 54.7, -45 deg)

	# 'RunList':				list(range(1,2+1)),		## (Oct 22, kU, kD Rabi Oscillation, Y,Z-axes, Tilt X,Z = 30., 0 deg)

	# 'RunList':				list(range(5,35+1,2)),	## (Oct 22, Raman Fringes vs TiltX, Y,Z-axes, T = 2.5 ms)
	# 'RunList':				list(range(6,36+1,2)),	## (Oct 22, Raman Fringes vs TiltX, Y,Z-axes, T = 5.0 ms)

	'RunList':				list(range(3,16+1,1)),	## (Oct 23, Raman Fringes vs TiltX, Y,Z-axes, T = 10.0 ms, BadRuns = [8])
	# 'RunList':				list(range(17,37+1,2)),	## (Oct 23, Raman Fringes vs TiltX, Y,Z-axes, T = 2.5 ms)
	# 'RunList':				list(range(18,38+1,2)),	## (Oct 23, Raman Fringes vs TiltX, Y,Z-axes, T = 5.0 ms)
	# 'RunList':				list(range(39,65+1,3)),	## (Oct 23, Raman Fringes vs TiltX, Z-axis, T = 2.5 ms, RT off)
	# 'RunList':				list(range(40,65+1,3)),	## (Oct 23, Raman Fringes vs TiltX, Z-axis, T = 5.0 ms, RT off)
	# 'RunList':				list(range(41,65+1,3)),	## (Oct 23, Raman Fringes vs TiltX, Z-axis, T = 10. ms, RT off)

	# 'RunList':				[1,2,3,4,5,6,16,17,18,19],	## (Oct 23, Ramsey Spectra vs TiltX, Z-Only)

	# 'RunList':				[1],					## (Oct 23, Tracking, Closed loop, X,Y,Z-axes, Tilt X,Z = 45, -30 deg)

	# 'RunList':				list(range(1,5+1,1)),	## (Oct 26, Ramsey Spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(1,5+1,1)),	## (Oct 26, Counter-prop. Rabi Oscillations vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(6,10+1,1)),	## (Oct 26, Co-prop. Rabi Oscillations vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(6,13+1,1)),	## (Oct 26, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(14,21+1,1)),	## (Oct 26, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(22,29+1,1)),	## (Oct 26, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(30,37+1,1)),	## (Oct 26, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(38,45+1,1)),	## (Oct 26, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)

	# 'RunList':				list(range(1,8+1,1)),	## (Oct 27-28-29, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(9,16+1,1)),	## (Oct 27-28-29, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(17,24+1,1)),	## (Oct 27-28-29, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(25,32+1,1)),	## (Oct 27-28-29, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)
	# 'RunList':				list(range(33,40+1,1)),	## (Oct 27-28-29, Co-prop. Raman spectra vs Raman TOF, X,Y,Z)

	# 'RunList':				list(range( 2, 6+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z = 300, 30 deg)
	# 'RunList':				list(range( 7,11+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z = 315, 30 deg)
	# 'RunList':				list(range(12,16+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z = 330, 30 deg)
	# 'RunList':				list(range(17,21+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z = 345, 30 deg)
	# 'RunList':				list(range(22,26+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z =  15, 30 deg)
	# 'RunList':				list(range(27,31+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z =  30, 30 deg)
	# 'RunList':				list(range(32,36+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z =  45, 30 deg)
	# 'RunList':				list(range(37,41+1,1)),	## (Oct 29, Rabi Oscillations vs Raman TOF, X,Y,Z, Tilt X,Z =  60, 30 deg)

	# 'RunList':				list(range(1,11+1,1)),	## (Nov 06, Contra-prop. Raman spectra vs Detection TOF, Z)

	'BadRuns':				[8], 			## List of run numbers to remove from RunList
	'AvgNum': 		 		1,				## Average number of dataset to analyze (fixed number for Raman data only)
	'IterNum': 		 		15				## Iteration number of dataset to analyze
}

PlotOpts = {
	'PlotData': 			True,			## Flag for plotting data
	'ShowPlot': 			True,			## Flag for showing data plots (PlotData = True and SavePlot = False)
	'ShowFit': 				True,			## Flag for overlaying fits with data (PlotData = True)
	'SavePlot': 			False,			## Flag for saving data plots (PlotData = True)
	'OverlayRunPlots':		True,			## Flag for overlaying plots from different runs (PlotData = True). Set False to overlay data for all Raman axes.
	'MaxPlotsToDisplay':	8,				## Maximum number of individual plots to display (AnalysisLevel = 2, OverlayRunPlots = True)
	'ShowPlotLabels':		[True, True], 	## Plot labels [xLabel, yLabel]
	'ShowPlotTitle': 		True,			## Flag for showing plot title (OverlayRunPlots = False)
	'ShowFigureLabels':		False,			## Flag for showing figure labels ('a','b','c', etc.)
	'ShowPlotLegend':		True,			## Flag for showing plot legend
	'FixLegLocation':		True, 			## Flag for fixing location of legend (ShowPlotLegend = True)
	'PlotFolderPath':		os.path.join(AnalysisCtrl['WorkDir'], 'PostProcessed', AnalysisCtrl['Folder']), ## Path for saving figures
	'PlotExtension':		'png',			## Plot file extension (png, pdf)
}

DetectOpts = {
	'OverrideCursors':		True,			## Flag for overriding detection cursors in run parameters
	'NewDetectCursors':		[8, 76, 161, 230, 985, 1055, 1140, 1210], ## New cursors for detection analysis (OverrideCursors = True)
	'ShowFitMessages':		False,			## Flag for showing detection fit messages
	'PrintFitReport':		False,			## Flag for printing detection fit report
	'FitLinearSlope':		False			## Flag for varying linear slope in detection fit function (not recommended)
}

MonitorOpts = {
	'ConvertToTemperature':	True, 			## Flag for converting certain monitor voltages to temperature
	'ComputeMovingAvg':		True,			## Flag for computing moving average of monitor data (AnalysisLevel = 0)
	'MovingAvgWindow':		1000.			## Time window (sec) for moving average (AnalysisLevel = 0)
}

RamanXFitPars = lm.Parameters()
RamanXFitPars.add_many(
	lm.Parameter('xOffset_kD', value=+0.0),
	lm.Parameter('xOffset_kU', value=-0.0),
	lm.Parameter('yOffset',    value=+0.45),
	lm.Parameter('Contrast',   value=+0.2, min=0.),
	lm.Parameter('xScale',     value=+1.0, min=0., vary=False))
RamanYFitPars = lm.Parameters()
RamanYFitPars.add_many(
	lm.Parameter('xOffset_kD', value=-0.0),
	lm.Parameter('xOffset_kU', value=-0.0),
	lm.Parameter('yOffset',    value=+0.45),
	lm.Parameter('Contrast',   value=+0.2, min=0.),
	lm.Parameter('xScale',     value=+1.0, min=0., vary=False))
RamanZFitPars = lm.Parameters()
RamanZFitPars.add_many(
	lm.Parameter('xOffset_kD', value=-0.0),
	lm.Parameter('xOffset_kU', value=+0.0),
	lm.Parameter('yOffset',    value=+0.45),
	lm.Parameter('Contrast',   value=+0.2, min=0.),
	lm.Parameter('xScale',     value=+1.0, min=0., vary=False))

RamanOpts = {
	'FitData':				True,			## Flag for fitting data
	'RunPlotVariable':		'TiltX',			## Independent variable for run plots (e.g. 'Run', 'RamanT', 'RamanpiZ', 'TiltX' 'deltaDiff', 'RunTime')
	'RunPlotColors':		[['red', 'darkred'], ['royalblue', 'blue'], ['green','darkgreen'], ['darkorange','gold'], ['purple','darkviolet'], ['cyan','darkcyan'], ['pink','deeppink'], ['violet','darkviolet']],
	'SortVariableOrder':	'Ascending',	## Control for sorting RunPlotVariable (AnalysisLevel = 2, 'None', Ascending', 'Descending')
	'TrackRunFitPars':		True,			## Flag for using fit results from one run as the initial parameters for the next
	'CentralFringeChirp':	[ 0.0000E6,  0.0000E6, 25.13488E6], ## List of central fringe chirps (0, 0 deg)
	# 'CentralFringeChirp':	[14.7281E6, 14.3604E6, 14.4466E6], ## List of central fringe chirps (54.7,-45 deg)

	# 'CentralFringeChirp':	[ 4.2320E6, 7.41450E6, 23.6390E6], ## List of central fringe chirps (Aug.21, 340, -30 deg)
	# 'CentralFringeChirp':	[ 8.0810E6, 13.9520E6, 19.2790E6], ## List of central fringe chirps (Aug.21, 320, -30 deg)

	# 'CentralFringeChirp':	[ 6.23038E6, 11.0244E6, 21.7129E6], ## List of central fringe chirps (Aug.28, 30, +30 deg)

	# 'CentralFringeChirp':	[ 6.29996E6, 10.8205E6, 21.7947E6], ## List of central fringe chirps (Aug.29, 330, +30 deg)
	# 'CentralFringeChirp':	[ 10.8613E6, 18.8201E6, 12.6375E6], ## List of central fringe chirps (Aug.29, 300, +30 deg)
	# 'CentralFringeChirp':	[ 8.02417E6, 14.1397E6, 19.1706E6], ## List of central fringe chirps (Aug.29,  40, +30 deg)

	# 'CentralFringeChirp':	[ 4.24759E6, 7.57487E6, 23.5875E6], ## List of central fringe chirps (Aug.31,  20, +30 deg)
	# 'CentralFringeChirp':	[ 4.33105E6, 7.38093E6, 23.6330E6], ## List of central fringe chirps (Aug.31, 340, +30 deg)

	# 'CentralFringeChirp':	[ 3.40245E6, 12.1900E6, 21.7181E6], ## List of central fringe chirps (Sep.01,  30, -15 deg)
	# 'CentralFringeChirp':	[ 4.18299E6, 15.5565E6, 19.2909E6], ## List of central fringe chirps (Sep.01, 320, -15 deg)
	# 'CentralFringeChirp':	[ 5.67383E6, 20.9864E6, 12.6103E6], ## List of central fringe chirps (Sep.01, 300, -15 deg)
	# 'CentralFringeChirp':	[ 5.15661E6, 18.6403E6, 16.0568E6], ## List of central fringe chirps (Sep.01,  50, -15 deg)
	# 'CentralFringeChirp':	[ 6.25803E6, 6.06427E6, 23.5776E6], ## List of central fringe chirps (Sep.01,  20, -45 deg)
	# 'CentralFringeChirp':	[ 11.6069E6, 11.3480E6, 19.1932E6], ## List of central fringe chirps (Sep.01,  40, -45 deg)
	# 'CentralFringeChirp':	[ 14.9484E6, 14.6584E6, 13.9105E6], ## List of central fringe chirps (Sep.01,  56, -45 deg)

	# 'CentralFringeChirp':	[ 12.5814E6, 12.4946E6, 17.8095E6], ## List of central fringe chirps (Sep.03, 315, -45 deg)

	'FitParameters':		[RamanXFitPars, RamanYFitPars, RamanZFitPars],	## Fit parameters for each Raman axis
	'SetFitPlotXLimits':	False, 			## Flag for setting fit plot x-axis limits
	'FitPlotXLimits':		[14.0E6, 15.0E6], ## Fit plot x-axis limits [xMin, xMax]
	'DetectCoeffs': 		[1., 0., 0.], 	## Detector mixing coefficients [Lower, Middle, Upper]
	'RemoveOutliers':		True,			## Flag for removing outliers from sinusoidal fits
	'OutlierThreshold':		3.3,			## Outlier threshold (in standard deviations of fit residuals)
	'PrintRunTiming':		False,			## Flag for printing run timing
	'TimeSeriesQuantity':	'Phase',		## Control for time series analysis quantity ('Gravity' or 'Phase') (AnalysisLevel = 4)
	'ComputeMovingAvg':		False,			## Flag for computing moving average of acceleration data (AnalysisLevel = 4)
	'MovingAvgWindow':		2000.,			## Time window (sec) for moving average (AnalysisLevel = 4)
	'PSD_Plot': 			False,			## Flag for plotting PSD of accelerations (AnalysisLevel = 4)
	'PSD_Method': 			'welch',		## Control for type of PSD algorithm ('periodogram', 'welch')
	'ADev_Plot': 			True,			## Flag for plotting Allan deviation of accelerations (AnalysisLevel = 4)
	'ADev_Fit':				False, 			## Flag for fitting specific Allan deviation curves (AnalysisLevel = 4)
	'ADev_ShowErrors':		True, 			## Flag for showing errors on Allan deviation plot (AnalysisLevel = 4)
	'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file
	'SaveAnalysisLevel2': 	True,			## Flag for saving analysis level 2 results to file (FitData = True)
	'SaveAnalysisLevel3':	True			## Flag for saving analysis level 3 results to file
}

## Peak models: 'Gauss', 'Lorentz', 'Sinc2', 'SincB', 'Raman', 'Voigt', 'PseudoVoigt', 'Moffat'
## Peak labels: 'kD' (-keff counter-prop.), 'kCo' (co-prop.), 'kU' (+keff counter-prop.)
RamseyXFitPars = lm.Parameters()
RamseyXFitPars.add_many(
	lm.Parameter('yOffset',    value=+0.00),
	lm.Parameter('p01_height', value=+0.25, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State': -1}),
	lm.Parameter('p02_height', value=+0.25, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State':  0}),
	lm.Parameter('p03_height', value=+0.25, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State': +1}),
	lm.Parameter('p01_center', value=-200.),
	lm.Parameter('p02_center', value=+0.),
	lm.Parameter('p03_center', value=+200.),
	lm.Parameter('p01_fwhm',   value=+40., min=0.),
	lm.Parameter('p02_fwhm',   value=+40., min=0.),
	lm.Parameter('p03_fwhm',   value=+40., min=0.))
RamseyYFitPars = lm.Parameters()
RamseyYFitPars.add_many(
	lm.Parameter('yOffset',    value=+0.00),
	lm.Parameter('p01_height', value=+0.30, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State': -1}),
	lm.Parameter('p02_height', value=+0.30, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State':  0}),
	lm.Parameter('p03_height', value=+0.30, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State': +1}),
	lm.Parameter('p01_center', value=-200.),
	lm.Parameter('p02_center', value=+0.),
	lm.Parameter('p03_center', value=+200.),
	lm.Parameter('p01_fwhm',   value=+30., min=0.),
	lm.Parameter('p02_fwhm',   value=+30., min=0.),
	lm.Parameter('p03_fwhm',   value=+30., min=0.))
RamseyZFitPars = lm.Parameters()
RamseyZFitPars.add_many(
	lm.Parameter('yOffset',    value=+0.00),
	lm.Parameter('p01_height', value=+0.30, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State': -1}),
	lm.Parameter('p02_height', value=+0.30, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State':  0}),
	lm.Parameter('p03_height', value=+0.30, min=0., user_data={'Model': 'SincB', 'Peak': 'kCo', 'State': +1}),
	lm.Parameter('p01_center', value=-200.),
	lm.Parameter('p02_center', value=+0.),
	lm.Parameter('p03_center', value=+200.),
	lm.Parameter('p01_fwhm',   value=+40., min=0.),
	lm.Parameter('p02_fwhm',   value=+40., min=0.),
	lm.Parameter('p03_fwhm',   value=+40., min=0.))
## Raman TOF = 30 ms
# RamseyXFitPars = lm.Parameters()
# RamseyXFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.18, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.20, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.18, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p02_center', value=-440.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-210.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+220.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+460.),
# 	lm.Parameter('p09_center', expr ='p08_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p01_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p02_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p03_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p04_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p05_fwhm',   value=+30., min=0.),
# 	lm.Parameter('p06_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p07_fwhm',   expr ='p08_fwhm'),
# 	lm.Parameter('p08_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p09_fwhm',   expr ='p08_fwhm'))
# RamseyYFitPars = lm.Parameters()
# RamseyYFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.18, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.20, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.18, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p02_center', value=-440.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-210.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+220.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+460.),
# 	lm.Parameter('p09_center', expr ='p08_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p01_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p02_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p03_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p04_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p05_fwhm',   value=+30., min=0.),
# 	lm.Parameter('p06_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p07_fwhm',   expr ='p08_fwhm'),
# 	lm.Parameter('p08_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p09_fwhm',   expr ='p08_fwhm'))
# RamseyZFitPars = lm.Parameters()
# RamseyZFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.18, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.20, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.18, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p02_center', value=-440.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-210.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+220.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+460.),
# 	lm.Parameter('p09_center', expr ='p08_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p01_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p02_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p03_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p04_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p05_fwhm',   value=+30., min=0.),
# 	lm.Parameter('p06_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p07_fwhm',   expr ='p08_fwhm'),
# 	lm.Parameter('p08_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p09_fwhm',   expr ='p08_fwhm'))

RamseyOpts = {
	'FitData':				True,			## Flag for fitting data
	'ShowFitResults':		True,			## Flag for printing fit results
	'RunPlotVariable':		'RamanTOF',		## Independent variable for run plots (an instance variable; e.g. 'Run', 'RamanT', 'RamanpiZ', 'RunTime')
	'SortVariableOrder':	'Ascending',	## Control for sorting RunPlotVariable (AnalysisLevel = 2, 'Ascending', 'Descending')
	'RunPlotColors':		['red', 'royalblue', 'green', 'darkorange', 'purple', 'darkcyan', 'deeppink', 'gold'],
	'TrackRunFitPars':		True,			## Flag for using fit results from one run as the initial parameters for the next
	'FitParameters':		[RamseyXFitPars, RamseyYFitPars, RamseyZFitPars], ## Fit parameters for each Raman axis
	'SetFitPlotXLimits':	False, 			## Flag for setting fit plot x-axis limits
	'FitPlotXLimits':		[-350., 350.], 	## Fit plot x-axis limits [xMin, xMax]
	'DetectCoeffs': 		[0., 1., 0.], 	## Detector mixing coefficients [Lower, Middle, Upper]
	'RemoveOutliers':		True,			## Flag for removing outliers from sinusoidal fits
	'OutlierThreshold':		3.3,			## Outlier threshold (in standard deviations of fit residuals)
	'PrintRunTiming':		False,			## Flag for printing run timing
	'CopropagatingSpectrum':True, 			## Flag for setting which peaks to analyze (AnalysisLevel = 3)
	'ConvertToBField':		True,			## Flag for converting frequency to magnetic field (AnalysisLevel = 3,4)
	'FitPositionData':		True,			## Flag for fitting data vs position (e.g. estimating dB/dz) (AnalysisLevel = 3)
	'PSD_Plot': 			False,			## Flag for plotting PSD of accelerations (AnalysisLevel = 4)
	'PSD_Method': 			'welch',		## Control for type of PSD algorithm ('periodogram', 'welch')
	'ADev_Plot': 			True,			## Flag for plotting Allan deviation of accelerations (AnalysisLevel = 4)
	'CorrelParameter':		'p06_center',	## Parameter name which to correlate with Monitor data (AnalysisLevel = 5)
	'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file
	'SaveAnalysisLevel2': 	True,			## Flag for saving analysis level 2 results to file (FitData = True)
	'SaveAnalysisLevel3':	True			## Flag for saving analysis level 3 results to file
}

## FitFunction: 'Rabi_Theory'
# RabiXFitPars = lm.Parameters()
# RabiXFitPars.add_many(
# 	lm.Parameter('Omega',     value=+np.pi/12., min=0.),
# 	lm.Parameter('xOffset',   value=+0.100),
# 	lm.Parameter('yOffset',   value=+0.040),
# 	lm.Parameter('Amplitude', value=+0.900, min=0.00, vary=True),
# 	lm.Parameter('alpha',     value=+0.900, min=0.00, vary=True),
# 	lm.Parameter('gammaA',    value=+0.050, min=0.00, vary=True),
# 	lm.Parameter('gammaB',    value=+0.010, min=0.00, vary=True),
# 	lm.Parameter('sigmaV',    value=+0.022, min=0.01, vary=True, max=0.050),
# 	lm.Parameter('vOffset',   value=+0.000, min=0.00, vary=False, max=0.050))
# RabiYFitPars = lm.Parameters()
# RabiYFitPars.add_many(
# 	lm.Parameter('Omega',     value=+np.pi/11., min=0.),
# 	lm.Parameter('xOffset',   value=+0.300),
# 	lm.Parameter('yOffset',   value=+0.000),
# 	lm.Parameter('Amplitude', value=+0.800, min=0.00, vary=True),
# 	lm.Parameter('alpha',     value=+0.300, min=0.00, vary=True),
# 	lm.Parameter('gammaA',    value=+0.050, min=0.00, vary=True),
# 	lm.Parameter('gammaB',    value=+0.050, min=0.00, vary=True),
# 	lm.Parameter('sigmaV',    value=+0.022, min=0.01, vary=True, max=0.050),
# 	lm.Parameter('vOffset',   value=+0.000, min=0.00, vary=False, max=0.050))
# RabiZFitPars = lm.Parameters()
# RabiZFitPars.add_many(
# 	lm.Parameter('Omega',     value=+np.pi/10., min=0.),
# 	lm.Parameter('xOffset',   value=+0.200),
# 	lm.Parameter('yOffset',   value=-0.010),
# 	lm.Parameter('Amplitude', value=+1.000, min=0.00, vary=True),
# 	lm.Parameter('alpha',     value=+1.400, min=0.00, vary=True),
# 	lm.Parameter('gammaA',    value=+0.060, min=0.00, vary=True),
# 	lm.Parameter('gammaB',    value=+0.010, min=0.00, vary=True),
# 	lm.Parameter('sigmaV',    value=+0.022, min=0.01, vary=True, max=0.050),
# 	lm.Parameter('vOffset',   value=+0.000, min=0.00, vary=False, max=0.050))
## FitFunction: 'Rabi_Phenom'
RabiXFitPars = lm.Parameters()
RabiXFitPars.add_many(
	lm.Parameter('Omega',     value=+np.pi/10., min=0.),
	lm.Parameter('xOffset',   value=+0.10),
	lm.Parameter('yOffset',   value=-0.05),
	lm.Parameter('Amplitude', value=+1.00, min=0.),
	lm.Parameter('alpha',     value=+0.50, min=0., vary=True),
	lm.Parameter('beta',      value=+0.03, min=0., vary=True),
	lm.Parameter('gammaA',    value=+0.04, min=0., vary=True))
RabiYFitPars = lm.Parameters()
RabiYFitPars.add_many(
	lm.Parameter('Omega',     value=+np.pi/12., min=0.),
	lm.Parameter('xOffset',   value=+0.10),
	lm.Parameter('yOffset',   value=-0.05),
	lm.Parameter('Amplitude', value=+1.00, min=0.),
	lm.Parameter('alpha',     value=+0.50, min=0., vary=True),
	lm.Parameter('beta',      value=+0.03, min=0., vary=True),
	lm.Parameter('gammaA',    value=+0.04, min=0., vary=True))
RabiZFitPars = lm.Parameters()
RabiZFitPars.add_many(
	lm.Parameter('Omega',     value=+np.pi/12., min=0.),
	lm.Parameter('xOffset',   value=+0.10),
	lm.Parameter('yOffset',   value=-0.05),
	lm.Parameter('Amplitude', value=+1.00, min=0.),
	lm.Parameter('alpha',     value=+0.50, min=0., vary=True),
	lm.Parameter('beta',      value=+0.03, min=0., vary=True),
	lm.Parameter('gammaA',    value=+0.04, min=0., vary=True))
## FitFunction: 'DetectProfile'
# RabiZFitPars.add_many(
# 	lm.Parameter('xOffset',   value=+86.0),
# 	lm.Parameter('yOffset',   value=+0.0),
# 	lm.Parameter('Amplitude', value=+1.2,   min=0.),
# 	lm.Parameter('vInitial',  value=+0.0,   min=-0.05, max=+0.05, vary=False),
# 	lm.Parameter('rDetect',   value=+7.0,   min=0.,    max=10.,   vary=True),
# 	lm.Parameter('sigmaR',    value=+3.0,   min=0.,    max=10.,   vary=True),
# 	lm.Parameter('sigmaV',    value=+0.022, min=0.010, max=0.050, vary=False))

RabiOpts = {
	'FitData':				True,			## Flag for fitting data
	'ShowFitResults':		True,			## Flag for printing fit results
	'DetectVariable':		'Ratio',		## Dependent variable to analyze from detection data ('N2', 'NT', 'Ratio')
	'RunPlotVariable':		'RamanTOF',			## Independent variable for run plots (an instance variable; e.g. 'Run', 'RamanT', 'RamanpiZ', 'RunTime')
	'SortVariableOrder':	'Ascending',	## Control for sorting RunPlotVariable (AnalysisLevel = 2, 'Ascending', 'Descending')
	'RunPlotColors':		['red', 'royalblue', 'green', 'darkorange', 'purple', 'darkcyan', 'deeppink', 'gold'],
	'TrackRunFitPars':		True,			## Flag for using fit results from one run as the initial parameters for the next
	'FitFunction':			'Rabi_Phenom',	## Fit function type ('Rabi_Phenom', 'Rabi_Theory', 'DetectProfile')
	'FitParameters':		[RabiXFitPars, RabiYFitPars, RabiZFitPars], ## Fit parameters for each Raman axis
	'SetPlotLimits':		[False, False],	## Flags for setting plot limits [x-axis, y-axis]
	'PlotLimits':			[[70., 110.], [-0.05, 1.05]], ## Plot axis limits [[xMin, xMax], [yMin, yMax]]
	'SetFitPlotXLimits':	False, 			## Flag for setting fit plot x-axis limits
	'FitPlotXLimits':		[70., 105.], 	## Fit plot x-axis limits [xMin, xMax]
	'DetectCoeffs': 		[1., 0., 0.], 	## Detector mixing coefficients [Lower, Middle, Upper]
	'RemoveOutliers':		False,			## Flag for removing outliers from sinusoidal fits
	'OutlierThreshold':		3.3,			## Outlier threshold (in standard deviations of fit residuals)
	'PrintRunTiming':		False,			## Flag for printing run timing
	'PSD_Plot': 			False,			## Flag for plotting PSD of accelerations (AnalysisLevel = 4)
	'PSD_Method': 			'welch',		## Control for type of PSD algorithm ('periodogram', 'welch')
	'ADev_Plot': 			True,			## Flag for plotting Allan deviation of accelerations (AnalysisLevel = 4)
	'ADev_ShowErrors':		False,			## Flag for showing error on Allan deviation (AnalysisLevel = 4)
	'CorrelParameter':		'Amplitude',	## Parameter name which to correlate with Monitor data (AnalysisLevel = 5)
	'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file
	'SaveAnalysisLevel2': 	True,			## Flag for saving analysis level 2 results to file (FitData = True)
	'SaveAnalysisLevel3':	True			## Flag for saving analysis level 3 results to file
}

QAModelPars = {
	'tyx': +1.63935288e-04,					## Misalignment factor between y & x axes
	'tzx': +1.82378907e-04,					## Misalignment factor between z & x axes
	'tzy': -1.27697393e-04,					## Misalignment factor between z & y axes
	'tyy': +1.00009467e+00,					## Misalignment factor on y axis
	'tzz': +1.00005015e+00,					## Misalignment factor on z axis
	'p0x': +0.00000000e+00,					## Bias on x axis (m/s^2)
	'p0y': +0.00000000e+00,					## Bias on y axis (m/s^2)
	'p0z': +0.00000000e+00					## Bias on z axis (m/s^2)
}
MAModelPars = {
	'tyx': +1.63935288e-04,					## Misalignment factor between y & x axes
	'tzx': +1.82378907e-04,					## Misalignment factor between z & x axes
	'tzy': -1.27697393e-04,					## Misalignment factor between z & y axes
	'tyy': +1.00009467e+00,					## Misalignment factor on y axis
	'tzz': +1.00005015e+00,					## Misalignment factor on z axis
	# 'p0x': +0.05512000e+00,					## Bias on x axis (m/s^2)
	# 'p0y': +0.01281000e+00,					## Bias on y axis (m/s^2)
	# 'p0z': +0.01018000e+00					## Bias on z axis (m/s^2)
	'p0x': +0.00000000e+00,					## Bias on x axis (m/s^2)
	'p0y': +0.00000000e+00,					## Bias on y axis (m/s^2)
	'p0z': +0.00000000e+00					## Bias on z axis (m/s^2)
}

TrackOpts = {
	'PSD_Plot':				True,			## Flag for plotting power spectral density
	'PSD_Method':			'welch', 		## Control for type of PSD algorithm ('periodogram', 'welch')
	'PSD_ShowErrorSignal':	True,			## Flag for computing PSD of error signals (AnalysisLevel = 1 only)
	'ADev_Plot':			True,			## Flag for plotting Allan deviation
	'ADev_Fit':				False,			## Flag for fitting k-dependent Allan deviation curves
	'ADev_Fit_FixExp':		True,			## Flag for fixing exponent in ADev log-log fit (True: fixed exponent = -0.5)
	'ADev_Fit_SetRange':	True,			## Flag for setting time range in ADev fit
	'ADev_Fit_Range':		[4.0E2,3.0E4],	## Time range to use for fitting k-dependent Allan deviation curves 
	'ADev_ShowErrorSignal':	False,			## Flag for computing ADev of error signals (AnalysisLevel = 1 only)
	'ADev_taus':			'octave',		## Control for steps of tau to use ('all', 'octave', 'decade')
	'ADev_ShowErrors':		True,			## Flag for showing ADev uncertainties
	'ADev_Errorstyle':		'Bar',			## Control for ADev error style ('Bar' or 'Shaded')
	'ADev_SetLimits': 		[True, True],	## Flags for setting ADev plot limits
	'ADev_XLimits': 		[1.E+0, 1.E+5],	## Control for ADev plot X limits
	'ADev_YLimits': 		[1.E-7, 2.E-5],	## Control for ADev plot Y limits
	'ComputeMovingAvg':		True,			## Flag for computing moving average of tracking data (AnalysisLevel = 1,2,3)
	'MovingAvgWindow':		200.,			## Time window (sec) for moving average (AnalysisLevel = 1,3)
	'SubtractTideModel':	False,			## Flag for overlaying tidal anomaly model (AnalysisLevel = 1,2)
	'RecomputeTideModel':	True,			## Flag for recomputing tidal anomaly model (should be done whenever TimeRange is changed)
	'SetPlotXLimits':		False, 			## Flag for setting x-axis plot limits for last figure axis
	'PlotXLimits':			[-20,250], 		## X-axis plot limits for last figure axis [xMin, xMax]
	'SetTimeRange':			False, 			## Flag for setting time range for tracking data
	'TimeRange':			[1.0E2,2.21E5],	## Time range for tracking data [tStart, tStop] (s)
	'PlotQuantity':			'kDepAccels',	## Plot quantities for AnalysisLevels >= 2 ('AllPhases', 'AllAccels', 'kDepPhases', 'kDepAccels')
	'AccelUnit':			'm/s^2', 		## Unit for plotting accelerations ('ug', 'm/s^2')
	'CorrectAcceleros': 	True,			## Flag for correcting axis misalignments and biases on accelerometers
	'QAModelParameters':	QAModelPars, 	## Quantum    accelerometer model parameters (AnalysisLevel = 2)
	'MAModelParameters':	MAModelPars, 	## Mechanical accelerometer model parameters (AnalysisLevel = 3)
	# 'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file (NOT OPERATIONAL)
	# 'SaveAnalysisLevel2':	True,			## Flag for saving analysis level 2 results to file (NOT OPERATIONAL)
	'SaveAnalysisLevel3':	True			## Flag for saving analysis level 3 results to file
}

StreamOpts = {
	'PSD_Plot':				True,			## Flag for generating power spectral density plot
	'PSD_Method':			'welch', 		## Flag for type of PSD algorithm ('periodogram' or 'welch')
	'ADev_Plot':			True,			## Flag for generating total Allan deviation plot
	'ADev_Fit':				False,			## Flag for fitting k-dependent Allan deviation curves
	'ADev_Fit_FixExp':		True,			## Flag for varying exponential in ADev fit (False: fixed exponent = -0.5)
	'ADev_Fit_SetRange':	True,			## Flag for setting time range in ADev fit
	'ADev_Fit_Range':		[1.E0, 1.E2],	## Time range to use for fitting k-dependent Allan deviation curves 
	'ADev_taus':			'octave',		## Control for steps of tau to use ('all', 'octave', 'decade')
	'ADev_ShowErrors':		True,			## Flag for showing ADev uncertainties
	'ADev_Errorstyle':		'Bar',			## Control for ADev error style ('Bar' or 'Shaded')
	'ComputeMovingAvg':		False,			## Flag for computing moving average of stream data
	'MovingAvgWindow':		2.E-0,			## Time window (sec) for moving average
	'StreamRate':           1000.,			## Rate for stream data (Hz)
	'SetPlotXLimits':		False, 			## Flag for setting x-axis plot limits for time series plots
	'PlotXLimits':			[-20,420], 		## Plot limits for time series plots [tMin, tMax] (s)
	'SetTimeRange':			True, 			## Flag for setting data range for stream data
	'TimeRange':			[2.0E1,3.6E3,1E-3],	## Stream time range to include in analysis [tStart, tStop, tStep] (s)
	'PrintStreamStats': 	True,			## Flag for printing statistics of stream data
	'IncludeAccelNorm':		True, 			## Flag for including norm of acceleration vector (g) in analysis
	'CorrectAcceleros':		True,			## Flag for correcting axis misalignments and biases on accelerometers
	'MAModelParameters':	MAModelPars, 	## Mechanical accelerometer model parameters
	'SaveAnalysisLevel1':	True			## Flag for saving analysis level 1 results to file
}

##===================================================================
##  Configure logger
##===================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s::%(message)s')
## 	DEBUG:    Detailed information, typically of interest only when diagnosing problems.
##	INFO:     Confirmation that things are working as expected.
##	WARNING:  An indication that something unexpected happened, 
## 	          or indicative of some problem in the near future (e.g. ‘disk space low’). 
## 	          The software is still working as expected.
## 	ERROR:    Due to a more serious problem, the software has not been able to perform some function.
## 	CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
##===================================================================

##===================================================================
## Set default plot options
##===================================================================
iXUtils.SetDefaultPlotOptions()

#####################################################################
def main():

	##----------------------- Initialization ------------------------
	# logging.disable(level=logging.INFO) # Disable logger for info & debug levels

	WorkDir   = AnalysisCtrl['WorkDir']
	Folder    = AnalysisCtrl['Folder']
	SubFolder = AnalysisCtrl['SubFolder']
	RunList   = AnalysisCtrl['RunList'].copy()
	BadRuns   = AnalysisCtrl['BadRuns'].copy()
	AvgNum    = AnalysisCtrl['AvgNum']
	IterNum   = AnalysisCtrl['IterNum']
	## Remove bad runs
	if len(BadRuns) > 0:
		RunList = list(set(RunList) - set(BadRuns))
	RunNum    = RunList[0]
	# nRuns     = len(RunList)
	AnalysisCtrl['RunList'] = RunList

	if Folder == 'Streaming':
		logging.info('iX_Main::#######################################################################################')

		if AnalysisCtrl['AnalysisLevels'][0] == 1:
		##================================================================
		## Load streamed accelerometer data and plot time series
		##================================================================	
			Strm = iXC_Stream.Stream(WorkDir, Folder, SubFolder, RunNum, StreamOpts, PlotOpts)
			Strm.LoadAccelStream()
			Strm.PlotAccelStream()
		else: 
			logging.info('iX_Main::Streaming::AnalysisLevel = {} not implemented. Use AnalysisLevel = 1'.format(AnalysisCtrl['AnalysisLevels'][0]))

	elif Folder == 'Monitor':
		logging.info('iX_Main::#######################################################################################')

		if AnalysisCtrl['AnalysisLevels'][0] == 1:
		##================================================================
		## Load monitor data and plot time series
		##================================================================	
			Mon = iXC_Monitor.Monitor(WorkDir, Folder, RunList, MonitorOpts, PlotOpts)
			Mon.ProcessMonitorData()
			Mon.PlotMonitorData()
		else: 
			logging.info('iX_Main::Monitor::AnalysisLevel = {} not implemented. Use AnalysisLevel = 1.'.format(AnalysisCtrl['AnalysisLevels'][0]))
	else:
		RunPars = iXC_RunPars.RunParameters(WorkDir, Folder, RunNum)
		RunPars.LoadRunParameters()

		# help(RunPars.LoadRunParameters)
		# print(pd.DataFrame(data=RunPars.__dict__.items()).head(100))
		# logging.disable(level=logging.NOTSET) # Re-enable logger at configured level

		for analysisLevel in AnalysisCtrl['AnalysisLevels']:

			logging.info('iX_Main::#######################################################################################')
			logging.info('iX_Main::                         Initiating {} AnalysisLevel = {}'.format(RunPars.DataType, analysisLevel))
			logging.info('iX_Main::#######################################################################################')

			if RunPars.DataType != 'Tracking' and analysisLevel == 0:
				if AnalysisCtrl['ProcessLevel'] == 0:
				##================================================================
				## Load and analyze a single detection shot for a single Run
				##================================================================
					Det = iXC_Detect.Detector(WorkDir, Folder, RunNum, DetectOpts, PlotOpts, False, RunPars.__dict__.items())
					Det.AnalyzeSingleDetectTrace(AvgNum, IterNum)
				elif AnalysisCtrl['ProcessLevel'] == 1:
				##================================================================
				## Load and plot aggregated monitor data for selected runs
				##================================================================
					Mon = iXC_Monitor.Monitor(WorkDir, Folder, RunList, MonitorOpts, PlotOpts, False, RunPars.__dict__.items())
					Mon.ProcessMonitorData()
					Mon.PlotMonitorData()

			elif RunPars.DataType == 'Raman':
				if analysisLevel == 1:
				##================================================================
				## Post-process detection data for selected Runs
				##================================================================
					iXC_Raman.RamanAnalysisLevel1(AnalysisCtrl, RamanOpts, DetectOpts, PlotOpts, RunPars)

				elif analysisLevel == 2:
				##================================================================
				## Load and analyze raw and/or post-processed data for selected Runs
				##================================================================
					iXC_Raman.RamanAnalysisLevel2(AnalysisCtrl, RamanOpts, PlotOpts)

				elif analysisLevel == 3:
				##================================================================
				## Reload analysis level 2 results and plot fringe parameters for selected Runs
				##================================================================
					iXC_Raman.RamanAnalysisLevel3(AnalysisCtrl, RamanOpts, PlotOpts, RunPars)

				elif analysisLevel == 4:
				##================================================================
				## Reload analysis level 3 results and perform time series analysis for selected Runs
				##================================================================
					iXC_Raman.RamanAnalysisLevel4(AnalysisCtrl, RamanOpts, PlotOpts, RunPars)

				elif analysisLevel == 5:
				##================================================================
				## Reload analysis level 3 results and plot monitor correlations for selected Runs
				##================================================================
					iXC_Raman.RamanAnalysisLevel5(AnalysisCtrl, RamanOpts, MonitorOpts, PlotOpts, RunPars)

				else:
					logging.info('iX_Main::{}::AnalysisLevel = {} not implemented. Use AnalysisLevels = 0-5.'.format(RunPars.DataType, analysisLevel))

			elif RunPars.DataType == 'Ramsey':
				if analysisLevel == 1:
				##================================================================
				## Post-process detection data for selected Runs
				##================================================================
					iXC_Ramsey.RamseyAnalysisLevel1(AnalysisCtrl, RamseyOpts, DetectOpts, PlotOpts, RunPars)

				elif analysisLevel == 2:
				##================================================================
				## Load and analyze raw and/or post-processed data for selected Runs
				##================================================================
					iXC_Ramsey.RamseyAnalysisLevel2(AnalysisCtrl, RamseyOpts, PlotOpts)

				elif analysisLevel == 3:
				##================================================================
				## Reload analysis level 2 results and plot spectra parameters for selected Runs
				##================================================================
					iXC_Ramsey.RamseyAnalysisLevel3(AnalysisCtrl, RamseyOpts, PlotOpts, RunPars)

				elif analysisLevel == 4:
				##================================================================
				## Reload analysis level 3 results and perform time series analysis for selected Runs
				##================================================================
					iXC_Ramsey.RamseyAnalysisLevel4(AnalysisCtrl, RamseyOpts, PlotOpts, RunPars)

				elif analysisLevel == 5:
				##================================================================
				## Reload analysis level 3 results and plot monitor correlations for selected Runs
				##================================================================
					iXC_Ramsey.RamseyAnalysisLevel5(AnalysisCtrl, RamseyOpts, PlotOpts, RunPars)

				else:
					logging.info('iX_Main::{}::AnalysisLevel = {} not implemented. Use AnalysisLevels = 0-5.'.format(RunPars.DataType, analysisLevel))

			elif RunPars.DataType == 'Rabi':
				if analysisLevel == 1:
				##================================================================
				## Post-process detection data for selected Runs
				##================================================================
					iXC_Rabi.RabiAnalysisLevel1(AnalysisCtrl, RabiOpts, DetectOpts, PlotOpts, RunPars)

				elif analysisLevel == 2:
				##================================================================
				## Load and analyze raw and/or post-processed data for selected Runs
				##================================================================
					iXC_Rabi.RabiAnalysisLevel2(AnalysisCtrl, RabiOpts, PlotOpts)

				elif analysisLevel == 3:
				##================================================================
				## Reload analysis level 2 results and plot spectra parameters for selected Runs
				##================================================================
					iXC_Rabi.RabiAnalysisLevel3(AnalysisCtrl, RabiOpts, PlotOpts, RunPars)

				elif analysisLevel == 4:
				##================================================================
				## Reload analysis level 3 results and perform time series analysis for selected Runs
				##================================================================
					iXC_Rabi.RabiAnalysisLevel4(AnalysisCtrl, RabiOpts, PlotOpts, RunPars)

				elif analysisLevel == 5:
				##================================================================
				## Reload analysis level 3 results and plot monitor correlations for selected Runs
				##================================================================
					iXC_Rabi.RabiAnalysisLevel5(AnalysisCtrl, RabiOpts, PlotOpts, RunPars)

				else:
					logging.info('iX_Main::{}::AnalysisLevel = {} not implemented. Use AnalysisLevels = 0-5.'.format(RunPars.DataType, analysisLevel))

			elif RunPars.DataType == 'Tracking':
				if analysisLevel <= 0:
				##================================================================
				## Load monitor data and plot time series
				##================================================================	
					Mon = iXC_Monitor.Monitor(WorkDir, Folder, [RunNum], MonitorOpts, PlotOpts)
					Mon.ProcessMonitorData()
					Mon.PlotMonitorData()

				elif analysisLevel <= 4:
				##================================================================
				## Load tracking/accelerometer/monitor data and perform time series analysis
				##================================================================
					iXC_Track.TrackAnalysis(analysisLevel, AnalysisCtrl, TrackOpts, MonitorOpts, PlotOpts, RunPars)

				else:
					logging.info('iX_Main::{}::AnalysisLevel = {} not implemented. Use AnalysisLevels = 0-4.'.format(RunPars.DataType, analysisLevel))

	logging.info('iX_Main::#######################################################################################')
	logging.info('iX_Main::Done!')

######################### End of main() #############################
#####################################################################

if __name__ == '__main__':
	main()