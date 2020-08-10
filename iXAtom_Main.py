#####################################################################
## Filename:	iXAtom_Main.py
## Author:		B. Barrett
## Version:		3.2.4
## Description:	Main implementation of iXAtom analysis package.
## 				Compatible with LabVIEW data acquisition v3.1 and later.
## Last Mod:	29/07/2020
##===================================================================
## Change Log:
## 11/10/2019 - Object-oriented update of iXAtom analysis v3.1
## 17/10/2019 - Minor modifications for LabVIEW v3.2 compatibility
## 19/10/2019 - Added control for combining detector results with
##				different coefficients.
##			  - Separated class definitions into separate files.
## 22/10/2019 - Raman AnalysisLevel = 3: Added this analysis Level for
##				reloading analysis results and plotting a summary
## 06/11/2019 - Added Stream class along with its own analysis levels
## 08/11/2019 - Raman AnalysisLevel = 2: Added run plot overlaying feature
## 10/11/2019 - Raman AnalysisLevel = 2: Added linear tracking correction
##				for initial parameter estimates when fitting fringes for
##				different run variables
## 11/11/2019 - Raman AnalysisLevel = 2: Added sorting feature for RunList
##				that performs analysis for run variables in ascending order 
## 19/11/2019 - Raman AnalysisLevel = 2: Bug fixes to fringe tracking
##				correction and fitting chirped data with fixed scale factor 
##			  - Raman AnalysisLevel = 3: Replaced plot of TExp with
##				fringe scale factor, and added functionality for 
##				RunPlotVariable = 'RamanpiX', 'RamanpiY', 'RamanpiZ'
## 25/11/2019 - Implemented a loop over AnalysisLevels
##			  - Removed 'ConvertChirpToPhase' option. This is now done
##				anytime ChirpedData = True and the phase parameter is 
##				stored when Raman AnalysisLevel = 3 is run.
## 29/11/2019 - Created Ramsey class based on Raman one, performed 
##				basic bug tests and implemented AnalysisLevels = 0-3.
##			  - Separated global plot options into its own dictionary
##				'PlotOpts' to facilite easy sharing with other classes.
## 02/12/2019 - Created Physics class in 'iXAtom_Class_Physics.py' for
##				storing useful physical constants and formulas. Now
##				RunParameters.SetPhysicalConstants() calls the __init__ 
##				method of this class and creates a local instance.
## 10/12/2019 - Added computation of tidal gravity anomaly to Physics
##				class (wrapper for ETGTAB F77 code). Features for overlaying
##				and subtracting tides from Tracking data were also added.
## 21/12/2019 - Created iXAtom analysis v3.2.2 to facilitate overhaul of
##				fitting routines to use the lmfit module.
## 04/01/2020 - Completed overhaul of Ramsey class to use lmfit module
##				and built-in models. Now this analysis code supports
##				fitting to an arbitrary number of peaks using peak-like
##				models such as Gaussian, Lorentzian, Sinc2, SincB, 
##				Voigt, Pseudo-Voigt, and Moffat (i.e. Lorentzian-B).
## 07/01/2020 - Completed overhaul of Raman class to use lmfit module.
## 08/01/2020 - Completed overhaul of Detector class to use lmfit module.
##			  - Implemented recursive initial parameter estimation when
##				post-processsing of Raman/Ramsey detection data.
## 14/01/2020 - Implemented Raman phase and offset noise estimation based
##				on an optimization of the log-likelihood distribution.
##			  - Added method to RunParameters to get run timing parameters
##				from timestamps recorded in the raw data files.
##			  - Added 'BadRuns' to AnalysisCtrl to facilitate removal
##				of bad runs from 'RunList'.
## 16/01/2020 - Implemented Raman analysis level 4 for performing a time
##				series analysis of level 3 results.
## 28/01/2020 - Fixed bug in error estimate of post-processed atomic ratio.
## 30/01/2020 - Added Monitor class for storing, merging and manipulating
##				monitor data.
## 03/02/2020 - Implemented Analysis Level 5 in Raman and Ramsey classes
##				for correlating Analysis Level 3 results with monitor data. 
##			  - Fixed bug when trying to read RunTime attribute from
##				post-processed data.
##            - Created Rabi class based on Ramsey class. Basic plotting
##				functionality only.
## 14/02/2020 - Modified PlotTrackingData method for LabVIEW software
##				version 3.3. This version involved changes to the type
##				of data saved in the ratio files (phase correction
##				replaced with error signal integral), and the rate at
##				which the error signal was updated (increased x 2).
##			  - LabVIEW software version 3.3 also incorporated major
##				changes to the monitor data types, which need to be
##				integrated into the Monitor class.
## 22/02/2020 - Improved robustness of Raman fitting function to avoid
##				zero-contrast results.
## 23/02/2020 - Implemented working fit function for Rabi class
##				(valid for pulse-duration-type Rabi oscillations only).
##			  - Completed methods for Rabi Analysis Levels 2 and 3.
## 31/03/2020 - Fixed bug in calculation of Allan deviation errors in iXUtils
##			  - Updated TimeSeriesAnalysis method with more options for
##				plotting time series, PSD, and ADev. 
## 04/04/2020 - Upgraded Raman AnalysisLevel3 to be handle runs with different
##				sets of Raman axes more intelligently.
## 24/04/2020 - Upgraded Raman noise estimation methods to estimate
##				fringe contrast noise in addition to phase and offset noise.
## 30/05/2020 - Implemented 'ConvertToBField' option in RamseyAnalysisLevel3
## 03/07/2020 - Added capability to analyze N2, NTotal and Ratio data in Rabi
##				analysis. In the future, this feature could be made a general
##				option for all analysis types.
## 03/09/2020 - Created v2.3.4 to accomodate updates to systematics class
## 23/07/2020 - Minor upgrades to Tracking class methods.
## 29/07/2020 - Added mathematical constraint functionality to Ramsey fits
#####################################################################

import os
import lmfit  as lm
import logging
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

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
##-------------------- Levels for all data types --------------------
## AnalysisLevel = 0: Plot specific detection or monitor data
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
## AnalysisLevel = 1: Load tracking data, plot time series of ratio, phase and error signals (RunNum = RunList[0])
## AnalysisLevel = 2: Load tracking data, plot time series of AI acceleration
## AnalysisLevel = 3: Load RT accelerometer data, plot time series of MA accelerations
## AnalysisLevel = 4: Load tracking and monitor data, plot correlations
## ProcessLevel		: Ignored
##-------------------- Levels for Streaming data --------------------
## AnalysisLevel = 1: Load streamed accelerometer data and plot time series (RunNum = RunList[0])
## ProcessLevel		: Ignored
##-------------------------------------------------------------------

PathCtrl = {
	'RootDir':				'C:\\Bryns Goodies\\Work-iXAtom', 				## Root directory containing data
	'DataDate':				{'Day': 2, 'Month': 'July', 'Year': 2020},	## Date the data was recorded
}

workDir = os.path.join(PathCtrl['RootDir'], 'Data {}'.format(PathCtrl['DataDate']['Year']),
	PathCtrl['DataDate']['Month'], '{:02d}'.format(PathCtrl['DataDate']['Day']))

AnalysisCtrl = {
	'AnalysisLevels':		[2,3],			## List of primary data analysis control types
	'ProcessLevel': 		0,				## Secondary control of data analysis processing
	'WorkDir': 				workDir, 		## Working directory
	'Folder': 		 		'Ramsey', 		## Data folder within WorkDir (e.g. 'Raman', 'Ramsey', 'Rabi', 'Tracking', 'Streaming')
	'SubFolder':			'None', 		## Subfolder within Folder (Folder = 'Streaming' only, SubFolder ignored if 'None')
	# 'RunList':				[1],			## List of run numbers of loop over
	# 'RunList':				list(range(1,7+1)),	## List of run numbers of loop over
	# 'RunList':				list(range(1,3+1)),		## (July 01, RamanTOF = 76 ms, tau = 30 us)
	# 'RunList':				list(range(4,6+1)),		## (July 01, RamanTOF = 46 ms, tau = 30 us)
	# 'RunList':				list(range(7,9+1)),		## (July 01, RamanTOF = 16 ms, tau = 30 us)
	# 'RunList':				list(range(19,23+1)),	## (July 01, RamanTOF = 16 ms, tau = 15 us)
	# 'RunList':				list(range(24,28+1)),	## (July 01, RamanTOF = 46 ms, tau = 15 us)
	# 'RunList':				list(range(1,5+1)),		## (July 02, RamanTOF = 76 ms, tau = 15 us)
	'RunList':				list(range(6,7+1)),		## (July 02, RamanTOF = 16 ms, tau = 30 us)
	# 'RunList':				list(range(8,9+1)),		## (July 02, RamanTOF = 46 ms, tau = 30 us)
	# 'RunList':				list(range(10,11+1)),	## (July 02, RamanTOF = 76 ms, tau = 30 us)
	# 'RunList':				list(range(17,24+1)),	## (July 27, RamanTOF = 77 ms, tau = 30 us)
	# 'RunList':				list(range(25,32+1)),	## (July 27, RamanTOF = 47 ms, tau = 30 us)
	# 'RunList':				list(range(33,40+1)),	## (July 27, RamanTOF = 17 ms, tau = 30 us)
	'BadRuns':				[], 			## List of run numbers to remove from RunList
	'AvgNum': 		 		1,				## Average number of dataset to analyze (fixed number for Raman data only)
	'IterNum': 		 		15				## Iteration number of dataset to analyze
}

PlotOpts = {
	'PlotData': 			True,			## Flag for plotting data
	'ShowPlot': 			True,			## Flag for showing data plots (PlotData = True and SavePlot = False)
	'ShowFit': 				True,			## Flag for overlaying fits with data (PlotData = True)
	'SavePlot': 			False,			## Flag for saving data plots (PlotData = True)
	'OverlayRunPlots':		True,			## Flag for overlaying plots from different runs (PlotData = True)
	'MaxPlotsToDisplay':	9,				## Maximum number of individual plots to display (AnalysisLevel = 2, OverlayRunPlots = True)
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
	'ConvertToTemperature':	False, 			## Flag for converting certain monitor voltages to temperature
	'ComputeMovingAvg':		True,			## Flag for computing moving average of monitor data (AnalysisLevel = 0)
	'MovingAvgWindow':		1000.			## Time window (sec) for moving average (AnalysisLevel = 0)
}

RamanXFitPars = lm.Parameters()
RamanXFitPars.add_many(
	lm.Parameter('xOffset_kD', value=+1.5),
	lm.Parameter('xOffset_kU', value=-1.5),
	lm.Parameter('yOffset',    value=+0.45),
	lm.Parameter('Contrast',   value=+0.2, min=0.),
	lm.Parameter('xScale',     value=+1.0, min=0., vary=False))
RamanYFitPars = lm.Parameters()
RamanYFitPars.add_many(
	lm.Parameter('xOffset_kD', value=-0.5),
	lm.Parameter('xOffset_kU', value=-0.5),
	lm.Parameter('yOffset',    value=+0.5),
	lm.Parameter('Contrast',   value=+0.2, min=0.),
	lm.Parameter('xScale',     value=+1.0, min=0., vary=False))
RamanZFitPars = lm.Parameters()
RamanZFitPars.add_many(
	lm.Parameter('xOffset_kD', value=-2.0),
	lm.Parameter('xOffset_kU', value=-2.0),
	lm.Parameter('yOffset',    value=+0.45),
	lm.Parameter('Contrast',   value=+0.3, min=0.),
	lm.Parameter('xScale',     value=+1.0, min=0., vary=False))

RamanOpts = {
	'FitData':				True,			## Flag for fitting data
	'RunPlotVariable':		'RunTime',		## Independent variable for run plots (e.g. 'Run', 'RamanT', 'RamanpiZ', 'RunTime')
	'RunPlotColors':		[['red', 'darkred'], ['royalblue', 'blue'], ['green','darkgreen'], ['darkorange','gold'], ['purple','darkviolet'], ['cyan','darkcyan'], ['pink','deeppink'], ['violet','darkviolet']],
	'SortVariableOrder':	'Ascending',	## Control for sorting RunPlotVariable (AnalysisLevel = 2, 'None', Ascending', 'Descending')
	'TrackRunFitPars':		False,			## Flag for using fit results from one run as the initial parameters for the next
	'CentralFringeChirp':	[0.0000000E7, 0.0000000E7, 2.5134880E7], ## List of central fringe chirps ( 0 deg)
	# 'CentralFringeChirp':	[1.2479733E7, 1.2385175E7, 1.7952328E7], ## List of central fringe chirps (45 deg)
	# 'CentralFringeChirp':	[1.1257936E7, 1.1173756E7, 1.9489458E7], ## List of central fringe chirps (39 deg)
	# 'CentralFringeChirp':	[1.0184188E7, 1.0084955E7, 2.0640457E7], ## List of central fringe chirps (35 deg)
	# 'CentralFringeChirp':	[0.8926609E7, 0.8857384E7, 2.1756684E7], ## List of central fringe chirps (30 deg)
	# 'CentralFringeChirp':	[0.6879514E7, 0.6795192E7, 2.3195896E7], ## List of central fringe chirps (23 deg)
	# 'CentralFringeChirp':	[0.4735331E7, 0.4664925E7, 2.4236231E7], ## List of central fringe chirps (15 deg)
	# 'CentralFringeChirp':	[0.2875202E7, 0.2818202E7, 2.4808811E7], ## List of central fringe chirps (10 deg)
	'FitParameters':		[RamanXFitPars, RamanYFitPars, RamanZFitPars],	## Fit parameters for each Raman axis
	'SetFitPlotXLimits':	False, 			## Flag for setting fit plot x-axis limits
	'FitPlotXLimits':		[2.513E7, 2.514E7], ## Fit plot x-axis limits [xMin, xMax]
	'DetectCoeffs': 		[1., 0., 0.], 	## Detector mixing coefficients [Lower, Middle, Upper]
	'RemoveOutliers':		True,			## Flag for removing outliers from sinusoidal fits
	'OutlierThreshold':		3.3,			## Outlier threshold (in standard deviations of fit residuals)
	'PrintRunTiming':		False,			## Flag for printing run timing
	'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file
	'SaveAnalysisLevel2': 	True,			## Flag for saving analysis level 2 results to file (FitData = True)
	'SaveAnalysisLevel3':	True,			## Flag for saving analysis level 3 results to file
	'TimeSeriesQuantity':	'Phase',		## Control for time series analysis quantity ('Gravity' or 'Phase') (AnalysisLevel = 4)
	'ComputeMovingAvg':		False,			## Flag for computing moving average of acceleration data (AnalysisLevel = 4)
	'MovingAvgWindow':		2000.,			## Time window (sec) for moving average (AnalysisLevel = 4)
	'PSD_Plot': 			False,			## Flag for plotting PSD of accelerations (AnalysisLevel = 4)
	'PSD_Method': 			'welch',		## Control for type of PSD algorithm ('periodogram', 'welch')
	'ADev_Plot': 			True,			## Flag for plotting Allan deviation of accelerations (AnalysisLevel = 4)
	'ADev_Fit':				False, 			## Flag for fitting specific Allan deviation curves (AnalysisLevel = 4)
	'ADev_ShowErrors':		True 			## Flag for showing errors on Allan deviation plot (AnalysisLevel = 4)
}

## Peak models: 'Gauss', 'Lorentz', 'Sinc2', 'SincB', 'Voigt', 'PseudoVoigt', 'Moffat'
## Peak labels: 'kD' (-keff counter-prop.), 'kCo' (co-prop.), 'kU' (+keff counter-prop.)
RamseyXFitPars = lm.Parameters()
RamseyXFitPars.add_many(
	lm.Parameter('yOffset',    value=+0.00),
	lm.Parameter('p01_height', value=+0.10, min=0., user_data={'Model': 'Sinc2', 'Peak': 'kCo', 'State': -1}),
	lm.Parameter('p02_height', value=+0.50, min=0., user_data={'Model': 'Sinc2', 'Peak': 'kCo', 'State':  0}),
	lm.Parameter('p03_height', value=+0.10, min=0., user_data={'Model': 'Sinc2', 'Peak': 'kCo', 'State': +1}),
	lm.Parameter('p01_center', value=-245.),
	lm.Parameter('p02_center', value=+0.),
	lm.Parameter('p03_center', value=+245.),
	lm.Parameter('p01_fwhm',   value=+50., min=0.),
	lm.Parameter('p02_fwhm',   value=+50., min=0.),
	lm.Parameter('p03_fwhm',   value=+50., min=0.))
RamseyYFitPars = lm.Parameters()
RamseyYFitPars.add_many(
	lm.Parameter('yOffset',    value=+0.00),
	lm.Parameter('p01_height', value=+0.10, min=0., user_data={'Model': 'Sinc2', 'Peak': 'kCo', 'State': -1}),
	lm.Parameter('p02_height', value=+0.50, min=0., user_data={'Model': 'Sinc2', 'Peak': 'kCo', 'State':  0}),
	lm.Parameter('p03_height', value=+0.10, min=0., user_data={'Model': 'Sinc2', 'Peak': 'kCo', 'State': +1}),
	lm.Parameter('p01_center', value=-245.),
	lm.Parameter('p02_center', value=+0.),
	lm.Parameter('p03_center', value=+245.),
	lm.Parameter('p01_fwhm',   value=+50., min=0.),
	lm.Parameter('p02_fwhm',   value=+50., min=0.),
	lm.Parameter('p03_fwhm',   value=+50., min=0.))
RamseyZFitPars = lm.Parameters()
## Raman TOF = 16 ms
RamseyZFitPars.add_many(
	lm.Parameter('yOffset',    value=+0.01),
	lm.Parameter('p01_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
	lm.Parameter('p02_height', value=+0.24, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
	lm.Parameter('p03_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
	lm.Parameter('p04_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
	lm.Parameter('p05_height', value=+0.20, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
	lm.Parameter('p06_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
	lm.Parameter('p07_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
	lm.Parameter('p08_height', value=+0.24, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
	lm.Parameter('p09_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
	lm.Parameter('p02_center', value=-400.),
	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
	lm.Parameter('p04_center', value=-200.),
	lm.Parameter('p05_center', value=+0.),
	lm.Parameter('p06_center', value=+200.),
	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
	lm.Parameter('p08_center', value=+440.),
	lm.Parameter('p09_center', expr ='p08_center + 0.5*(p06_center - p04_center)'),
	lm.Parameter('p01_fwhm',   expr ='p02_fwhm'),
	lm.Parameter('p02_fwhm',   value=+100., min=0.),
	lm.Parameter('p03_fwhm',   expr ='p02_fwhm'),
	lm.Parameter('p04_fwhm',   expr ='p05_fwhm'),
	lm.Parameter('p05_fwhm',   value=+30., min=0.),
	lm.Parameter('p06_fwhm',   expr ='p05_fwhm'),
	lm.Parameter('p07_fwhm',   expr ='p08_fwhm'),
	lm.Parameter('p08_fwhm',   value=+100., min=0.),
	lm.Parameter('p09_fwhm',   expr ='p08_fwhm'))
## Raman TOF = 17 ms
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
# 	lm.Parameter('p02_center', value=-460.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-210.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+220.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+490.),
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
## Raman TOF = 46 ms
# RamseyZFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.35, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.01, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.15, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.01, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.35, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p02_center', value=-1150.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-190.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+200.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+1200.),
# 	lm.Parameter('p09_center', expr ='p08_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p01_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p02_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p03_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p04_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p05_fwhm',   value=+40., min=0.),
# 	lm.Parameter('p06_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p07_fwhm',   expr ='p08_fwhm'),
# 	lm.Parameter('p08_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p09_fwhm',   expr ='p08_fwhm'))
## Raman TOF = 47 ms
# RamseyZFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.17, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.01, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.14, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.01, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.17, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p02_center', value=-1210.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-200.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+210.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+1250.),
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
## Raman TOF = 76 ms
# RamseyZFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.21, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.14, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.03, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.21, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.03, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', expr ='p02_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p02_center', value=-1915.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-195.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+200.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+1965.),
# 	lm.Parameter('p09_center', expr ='p08_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p01_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p02_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p03_fwhm',   expr ='p02_fwhm'),
# 	lm.Parameter('p04_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p05_fwhm',   value=+40., min=0.),
# 	lm.Parameter('p06_fwhm',   expr ='p05_fwhm'),
# 	lm.Parameter('p07_fwhm',   expr ='p08_fwhm'),
# 	lm.Parameter('p08_fwhm',   value=+100., min=0.),
# 	lm.Parameter('p09_fwhm',   expr ='p08_fwhm'))
## Raman TOF = 77 ms
# RamseyZFitPars.add_many(
# 	lm.Parameter('yOffset',    value=+0.01),
# 	lm.Parameter('p01_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': -1}),
# 	lm.Parameter('p02_height', value=+0.17, min=0., user_data={'Model': 'Moffat', 'Peak': 'kD',  'State':  0}),
# 	lm.Parameter('p03_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kD',  'State': +1}),
# 	lm.Parameter('p04_height', value=+0.01, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': -1}),
# 	lm.Parameter('p05_height', value=+0.14, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State':  0}),
# 	lm.Parameter('p06_height', value=+0.01, min=0., user_data={'Model': 'Sinc2',  'Peak': 'kCo', 'State': +1}),
# 	lm.Parameter('p07_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': -1}),
# 	lm.Parameter('p08_height', value=+0.17, min=0., user_data={'Model': 'Moffat', 'Peak': 'kU',  'State':  0}),
# 	lm.Parameter('p09_height', value=+0.01, min=0., user_data={'Model': 'Gauss',  'Peak': 'kU',  'State': +1}),
# 	lm.Parameter('p01_center', value=-1960.-210., vary=False),
# 	lm.Parameter('p02_center', value=-1960.),
# 	lm.Parameter('p03_center', expr ='p02_center + 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p04_center', value=-200.),
# 	lm.Parameter('p05_center', value=+0.),
# 	lm.Parameter('p06_center', value=+210.),
# 	lm.Parameter('p07_center', expr ='p08_center - 0.5*(p06_center - p04_center)'),
# 	lm.Parameter('p08_center', value=+2010.),
# 	lm.Parameter('p09_center', value=+2010.+210., vary=False),
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
	'RunPlotVariable':		'DetectTOF',	## Independent variable for run plots (an instance variable; e.g. 'Run', 'RamanT', 'RamanpiZ', 'RunTime')
	'SortVariableOrder':	'Ascending',	## Control for sorting RunPlotVariable (AnalysisLevel = 2, 'Ascending', 'Descending')
	'RunPlotColors':		['red', 'royalblue', 'green', 'darkorange', 'purple', 'darkcyan', 'deeppink', 'gold'],
	'TrackRunFitPars':		False,			## Flag for using fit results from one run as the initial parameters for the next
	'FitParameters':		[RamseyXFitPars, RamseyYFitPars, RamseyZFitPars], ## Fit parameters for each Raman axis
	'SetFitPlotXLimits':	False, 			## Flag for setting fit plot x-axis limits
	'FitPlotXLimits':		[-350., 350.], 	## Fit plot x-axis limits [xMin, xMax]
	'DetectCoeffs': 		[1., 0., 0.], 	## Detector mixing coefficients [Lower, Middle, Upper]
	'RemoveOutliers':		False,			## Flag for removing outliers from sinusoidal fits
	'OutlierThreshold':		3.5,			## Outlier threshold (in standard deviations of fit residuals)
	'PrintRunTiming':		False,			## Flag for printing run timing
	'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file
	'SaveAnalysisLevel2': 	True,			## Flag for saving analysis level 2 results to file (FitData = True)
	'SaveAnalysisLevel3':	True,			## Flag for saving analysis level 3 results to file
	'CopropagatingSpectrum':False, 			## Flag for setting which peaks to analyze (AnalysisLevel = 3)
	'ConvertToBField':		False,			## Flag for converting frequency to magnetic field (AnalysisLevel = 3,4)
	'PSD_Plot': 			False,			## Flag for plotting PSD of accelerations (AnalysisLevel = 4)
	'PSD_Method': 			'welch',		## Control for type of PSD algorithm ('periodogram', 'welch')
	'ADev_Plot': 			True,			## Flag for plotting Allan deviation of accelerations (AnalysisLevel = 4)
	'CorrelParameter':		'p06_center'	## Parameter name which to correlate with Monitor data (AnalysisLevel = 5)
}

RabiXFitPars = lm.Parameters()
RabiXFitPars.add_many(
	lm.Parameter('Omega',     value=+np.pi/9.),
	lm.Parameter('xOffset',   value=+0.0),
	lm.Parameter('yOffset',   value=+0.0),
	lm.Parameter('Amplitude', value=+0.50, min=0.),
	lm.Parameter('alpha',     value=+0.25, min=0., vary=True),
	lm.Parameter('beta',      value=+0.15, min=0., vary=False),
	lm.Parameter('gamma',     value=+0.10, min=0., vary=True))
RabiYFitPars = lm.Parameters()
RabiYFitPars.add_many(
	lm.Parameter('Omega',     value=+np.pi/9.),
	lm.Parameter('xOffset',   value=+0.0),
	lm.Parameter('yOffset',   value=+0.0),
	lm.Parameter('Amplitude', value=+0.50, min=0.),
	lm.Parameter('alpha',     value=+0.25, min=0., vary=True),
	lm.Parameter('beta',      value=+0.13, min=0., vary=False),
	lm.Parameter('gamma',     value=+0.10, min=0., vary=True))
RabiZFitPars = lm.Parameters()
# RabiZFitPars.add_many(
# 	lm.Parameter('Omega',     value=+np.pi/5.),
# 	lm.Parameter('xOffset',   value=+0.0),
# 	lm.Parameter('yOffset',   value=+0.0),
# 	lm.Parameter('Amplitude', value=+0.50, min=0.),
# 	lm.Parameter('alpha',     value=+0.25, min=0., vary=True),
# 	lm.Parameter('beta',      value=+0.10, min=0., vary=False),
# 	lm.Parameter('gamma',     value=+0.10, min=0., vary=True))
RabiZFitPars.add_many(
	lm.Parameter('xOffset',   value=+86.0),
	lm.Parameter('yOffset',   value=+0.0),
	lm.Parameter('Amplitude', value=+1.2,   min=0.),
	lm.Parameter('vInitial',  value=+0.0,   min=-0.05, max=+0.05, vary=False),
	lm.Parameter('rDetect',   value=+7.0,   min=0.,    max=10.,   vary=True),
	lm.Parameter('sigmaR',    value=+3.0,   min=0.,    max=10.,   vary=True),
	lm.Parameter('sigmaV',    value=+0.022, min=0.010, max=0.050, vary=False))

RabiOpts = {
	'FitData':				True,			## Flag for fitting data
	'ShowFitResults':		True,			## Flag for printing fit results
	'DetectVariable':		'NT',			## Dependent variable to analyze from detection data ('N2', 'NT', 'Ratio')
	'RunPlotVariable':		'Run',			## Independent variable for run plots (an instance variable; e.g. 'Run', 'RamanT', 'RamanpiZ', 'RunTime')
	'SortVariableOrder':	'Ascending',	## Control for sorting RunPlotVariable (AnalysisLevel = 2, 'Ascending', 'Descending')
	'RunPlotColors':		['red', 'royalblue', 'green', 'darkorange', 'purple', 'darkcyan', 'deeppink', 'gold'],
	'TrackRunFitPars':		True,			## Flag for using fit results from one run as the initial parameters for the next
	'FitParameters':		[RabiXFitPars, RabiYFitPars, RabiZFitPars], ## Fit parameters for each Raman axis
	'SetPlotLimits':		[False, False],	## Flags for setting plot limits [x-axis, y-axis]
	'PlotLimits':			[[70., 110.], [-0.05, 1.05]], ## Plot axis limits [[xMin, xMax], [yMin, yMax]]
	'SetFitPlotXLimits':	False, 			## Flag for setting fit plot x-axis limits
	'FitPlotXLimits':		[70., 105.], 	## Fit plot x-axis limits [xMin, xMax]
	'DetectCoeffs': 		[1., 0., 0.], 	## Detector mixing coefficients [Lower, Middle, Upper]
	'RemoveOutliers':		False,			## Flag for removing outliers from sinusoidal fits
	'OutlierThreshold':		3.5,			## Outlier threshold (in standard deviations of fit residuals)
	'PrintRunTiming':		False,			## Flag for printing run timing
	'SaveAnalysisLevel1':	True,			## Flag for saving analysis level 1 results to file
	'SaveAnalysisLevel2': 	True,			## Flag for saving analysis level 2 results to file (FitData = True)
	'SaveAnalysisLevel3':	True,			## Flag for saving analysis level 3 results to file
	'PSD_Plot': 			False,			## Flag for plotting PSD of accelerations (AnalysisLevel = 4)
	'PSD_Method': 			'welch',		## Control for type of PSD algorithm ('periodogram', 'welch')
	'ADev_Plot': 			True,			## Flag for plotting Allan deviation of accelerations (AnalysisLevel = 4)
	'ADev_ShowErrors':		False,			## Flag for showing error on Allan deviation (AnalysisLevel = 4)
	'CorrelParameter':		'Amplitude'		## Parameter name which to correlate with Monitor data (AnalysisLevel = 5)
}

TrackOpts = {
	'PSD_Plot':				False,			## Flag for plotting power spectral density
	'PSD_Method':			'welch', 		## Control for type of PSD algorithm ('periodogram', 'welch')
	'PSD_ShowErrorSignal':	True,			## Flag for computing PSD of error signals (AnalysisLevel = 1 only)
	'ADev_Plot':			True,			## Flag for plotting Allan deviation
	'ADev_Fit':				False,			## Flag for fitting k-dependent Allan deviation curves
	'ADev_Fit_FitExp':		True,			## Flag for varying exponential in ADev fit (False: fixed exponent = -0.5)
	'ADev_Fit_SetRange':	True,			## Flag for setting time range in ADev fit
	'ADev_Fit_Range':		[1.E2, 1.E3],	## Time range to use for fitting k-dependent Allan deviation curves 
	'ADev_ShowErrorSignal':	False,			## Flag for computing ADev of error signals (AnalysisLevel = 1 only)
	'ADev_taus':			'all',			## Control for steps of tau to use ('all', 'octave', 'decade')
	'ADev_ShowErrors':		True,			## Flag for showing ADev uncertainties
	'ADev_Errorstyle':		'Shaded',		## Control for ADev error style ('Bar' or 'Shaded')
	'ComputeMovingAvg':		False,			## Flag for computing moving average of tracking data (AnalysisLevel = 1,2,3)
	'MovingAvgWindow':		1000.,			## Time window (sec) for moving average (AnalysisLevel = 1,3)
	'SubtractTideModel':	False,			## Flag for overlaying tidal anomaly model (AnalysisLevel = 1,2)
	'RecomputeTideModel':	True,			## Flag for recomputing tidal anomaly model (should be done whenever TimeRange is changed)
	'SetPlotXLimits':		False, 			## Flag for setting x-axis plot limits for last figure axis
	'PlotXLimits':			[-20,250], 		## X-axis plot limits for last figure axis [xMin, xMax]
	'SetTimeRange':			True, 			## Flag for setting time range for tracking data
	'TimeRange':			[8000,210000],	## Time range for tracking data [tStart, tStop]
	'PlotQuantity':			'kDepAccels',	## Plot quantities for AnalysisLevels >= 2 ('AllPhases', 'AllAccels', 'kDepPhases', 'kDepAccels')
	'AccelUnit':			'm/s^2', 			## Unit for plotting accelerations ('ug', 'm/s^2')
	'AxisMisalignments':	np.array([0.,0.,0.])*1.E-4 ## Accelerometer misalignment correction factors (AnalysisLevel >= 2)
}

StreamOpts = {
	'PSD_Plot':				False,			## Flag for generating power spectral density plot
	'PSD_Method':			'welch', 		## Flag for type of PSD algorithm ('periodogram' or 'welch')
	'ADev_Plot':			True,			## Flag for generating total Allan deviation plot
	'ADev_ComputeErrors':	False,			## Flag for computing ADev errors
	'SetPlotXLimits':		False, 			## Flag for setting x-axis plot limits for last figure axis
	'PlotXLimits':			[-20,420], 		## X-axis plot limits for last figure axis [xMin, xMax]
	'SetStreamRange':		True, 			## Flag for setting range for accel stream data
	'StreamRange':			[0,100000,1], 	## Stream range [iStart, iStop, iStep]
	'PrintStreamStats': 	True			## Flag for printing statistics of stream data
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
	nRuns     = len(RunList)
	AnalysisCtrl['RunList'] = RunList

	if Folder == 'Streaming':
		if AnalysisCtrl['AnalysisLevels'][0] == 0:
		##================================================================
		## Load streamed accelerometer data and plot time series
		##================================================================	
			Strm = iXC_Stream.Stream(WorkDir, Folder, SubFolder, RunNum, StreamOpts)
			Strm.LoadAccelStream()
			Strm.PlotAccelStream()
	elif Folder == 'Monitor':
		if AnalysisCtrl['AnalysisLevels'][0] == 0:
		##================================================================
		## Load monitor data and plot time series
		##================================================================	
			Mon = iXC_Monitor.Monitor(WorkDir, Folder, RunList, MonitorOpts, PlotOpts)
			Mon.ProcessMonitorData()
			Mon.PlotMonitorData()
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

			if analysisLevel == 0:
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
					Mon.ProcessMonitorData(MonitorOpts['ComputeMovingAvg'], MonitorOpts['MovingAvgWindow'])
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
					iXC_Raman.RamanAnalysisLevel5(AnalysisCtrl, RamanOpts, PlotOpts, RunPars)

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

			elif RunPars.DataType == 'Tracking':

				for RunNum in RunList:
					if RunNum == RunList[0]:
						Trk = iXC_Track.Track(WorkDir, Folder, RunNum, TrackOpts, MonitorOpts, PlotOpts, False, RunPars.__dict__.items())
					else:
						Trk = iXC_Track.Track(WorkDir, Folder, RunNum, TrackOpts, MonitorOpts, PlotOpts)

					Trk.LoadAnalysisData(analysisLevel)

					if analysisLevel == 1:
					##================================================================
					## Load tracking ratio data and plot time series analysis
					##================================================================
						# Trk.PlotTrackData()
						Trk.PlotRatioData()

					elif analysisLevel == 2:
					##================================================================
					## Load tracking phase data and plot time series analysis
					##================================================================
						Trk.PlotPhaseData()

					elif analysisLevel == 3:
					##================================================================
					## Load mean accelerometer data and plot time series analysis
					##================================================================
						Trk.PlotAccelMean()

					elif analysisLevel == 4:
					##================================================================
					## Load tracking and monitor data, plot correlations
					##================================================================
						Trk.PlotCorrelations()

	logging.info('iX_Main::#######################################################################################')
	logging.info('iX_Main::Done!')

######################### End of main() #############################
#####################################################################

if __name__ == '__main__':
	main()