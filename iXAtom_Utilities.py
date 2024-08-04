#####################################################################
## Filename:    iXAtom_Utilities.py
## Author:      B. Barrett
## Description: Utilities for iXAtom analysis package, including:
##            - Dataframe custom export methods
##            - Mathplotlib custom plotting methods
##			  - General times series analysis
## Version:     3.2.5
## Last Mod:    25/08/2021
#####################################################################

import logging
import os
import pytz
import datetime 		 as dt
import numpy 			 as np
import pandas 			 as pd
import matplotlib.pyplot as plt
import allantools 		 as allan

from scipy.stats  import chi2
from scipy.signal import periodogram
from scipy.signal import welch

#####################################################################

def WriteDataFrameToFile(DataFrame, FolderPath, FilePath, ShowHeaders, ShowIndices, Format, Columns=None):
	"""Write contents of a pandas dataframe to file.
	ARGUMENTS:
	\t DataFrame   (df)   - Pandas dataframe
	\t FolderPath  (str)  - Folder path (one level above file path)
	\t FilePath    (str)  - Full file path
	\t ShowHeaders (bool) - Flag for writing headers to file
	\t ShowIndices (bool) - Flag for writing indices to file
	\t Format      (str)  - Float format
	\t Columns     (list) - Subset of columns to write to file.
							The order of the columns in this list is preserved in the file.
	"""

	if not os.path.exists(FolderPath):
		os.makedirs(FolderPath)

	DataFrame.to_csv(FilePath, sep='\t', header=ShowHeaders, index=ShowIndices, float_format=Format, columns=Columns)

################## End of WriteDataFrameToFile() ####################
#####################################################################

def SetDefaultPlotOptions():

	plt.rc('font', size=12, family='serif')
	plt.rc('axes', titlesize=12, labelsize=12, edgecolor='black', linewidth=1)
	plt.rc('lines', linewidth=1.5, markersize=8)
	plt.rc('legend', fontsize=10, frameon=False, handletextpad=0.4)

################# End of SetDefaultPlotOptions() ####################
#####################################################################

def CustomPlot(Ax, PlotOpts, xData, yData, yErr=[], LogScale=[False,False], MaxRelErr=2.):

	if len(yErr) > 0 and np.mean(yErr)/np.mean(abs(yData)) <= MaxRelErr:
		hideErrs = False
	else:
		hideErrs = True

	if not LogScale[0] and not LogScale[1]:
		# Linear scale on both axes
		if hideErrs:
			Ax.plot(xData, yData, label=PlotOpts['LegLabel'], color=PlotOpts['Color'],
				linestyle=PlotOpts['Linestyle'], marker=PlotOpts['Marker'])
		else:
			Ax.errorbar(xData, yData, yerr=yErr, label=PlotOpts['LegLabel'], color=PlotOpts['Color'], alpha=0.7,
				linestyle=PlotOpts['Linestyle'], ecolor=PlotOpts['Color'], fmt=PlotOpts['Marker'], capsize=2)
	elif LogScale[0] and not LogScale[1]:
		# Log scale on x-axis
		if hideErrs:
			Ax.semilogx(xData, yData, label=PlotOpts['LegLabel'], color=PlotOpts['Color'],
				linestyle=PlotOpts['Linestyle'], marker=PlotOpts['Marker'])
		else:
			Ax.set_xscale('log')
			Ax.errorbar(xData, yData, yerr=yErr, label=PlotOpts['LegLabel'], color=PlotOpts['Color'], 
				linestyle=PlotOpts['Linestyle'], ecolor=PlotOpts['Color'], fmt=PlotOpts['Marker'], capsize=2)
	elif not LogScale[0] and LogScale[1]:
		# Log scale on y-axis
		if hideErrs:
			Ax.semilogy(xData, yData, label=PlotOpts['LegLabel'], color=PlotOpts['Color'],
				linestyle=PlotOpts['Linestyle'], marker=PlotOpts['Marker'])
		else:
			Ax.set_yscale('log')
			yErr = np.maximum(1.0E-12, yErr) # To avoid negative numbers
			Ax.errorbar(xData, yData, yerr=yErr, label=PlotOpts['LegLabel'], color=PlotOpts['Color'],
				linestyle=PlotOpts['Linestyle'], ecolor=PlotOpts['Color'], fmt=PlotOpts['Marker'], capsize=2)
	else:
		# Log scale on both axes
		if hideErrs:
			Ax.loglog(xData, yData, label=PlotOpts['LegLabel'], color=PlotOpts['Color'],
				linestyle=PlotOpts['Linestyle'], marker=PlotOpts['Marker'])
		else:
			Ax.set_xscale('log')
			Ax.set_yscale('log')
			yErr = np.maximum(1.0E-12, yErr) # To avoid negative numbers
			Ax.errorbar(xData, yData, yerr=yErr, label=PlotOpts['LegLabel'], color=PlotOpts['Color'], 
				linestyle=PlotOpts['Linestyle'], ecolor=PlotOpts['Color'], fmt=PlotOpts['Marker'], capsize=2)

	if PlotOpts['Title'] != 'None':
		Ax.set_title(PlotOpts['Title'])
	# if PlotOpts['LegLabel'] != 'None':
	# 	Ax.set_label(PlotOpts['LegLabel'])
	if PlotOpts['xLabel'] != 'None':
		Ax.set_xlabel(PlotOpts['xLabel'])
	if PlotOpts['yLabel'] != 'None':    
		Ax.set_ylabel(PlotOpts['yLabel'])
	if PlotOpts['Legend']:
		try: 
			Ax.legend(loc=PlotOpts['LegLocation'])
		except:
			Ax.legend(loc='best')

	if LogScale[0] or LogScale[1]:
		Ax.grid(b=True, which='both', axis='both', color='0.75', linestyle='-')
	else:
		Ax.grid(b=True, which='major', axis='both', color='0.75', linestyle='-')

###################### End of CustomPlot() ##########################
#####################################################################

def TimestampToDatetime(Datestamp, Timestamp, Format='%d/%m/%Y %H:%M:%S.%f', Timezone=pytz.timezone('Europe/Paris')):
	"""Convert datestamp and timestamp strings to a datetime object.
	ARGUMENTS:
	Datestamp (str) - Datestamp string.
	Timestamp (str) - Timestamp string. 
	Format    (str) - Datetime format string. Default: 'DD/MM/YYYY HH:MM:SS.sss'
	Timezone  (tzinfo) - Timezone setting. Default: CET 'Europe/Paris'
	RETURN FORMAT:
	Datetime object
	"""

	dateTime = dt.datetime.strptime(Datestamp + ' ' + Timestamp, Format).replace(tzinfo=None)

	return dateTime.astimezone(tz=Timezone)

################## End of TimestampToDatetime() #####################
#####################################################################

def AllanDev(yList, taus='all', rate=1., ADevType='Total', tauMax=0.4, ComputeErr=True):
	"""Compute Allan deviation of yList.
	ARGUMENTS:
	yList    (np.array) - Input data. Assumed to be a unitless fractional-frequency-like data.
	taus     (np.array) - Values of tau (s) for which to compute ADev. 
						  One can also use values of 'all', 'octave', 'decade'. Defaults to 'all'.
	rate        (float) - Sampling rate of input data (Hz). Defaults to 1 Hz.
	ADevType      (str) - Control for type of Allan deviation to compute ('Overlapped', 'Total'). Defaults to 'Total'.
	tauMax      (float) - Maximum tau to plot (in units of the half data length when ADevType = 'Overlapped',
							and the full data length when ADevType = 'Total'). Defaults to 0.4.
	ComputeErr 	 (bool) - Flag for computing ADev uncertainties based on Chi2 distribution (expensive for large data sets).
	RETURN FORMAT: (Tau, ADev, ADevSigL, ADevSigU)
	tau      (np.array) - Values of taus in seconds at which total Allan deviation was computed.
	ADev     (np.array) - Values of total Allan deviation.
	ADevSigL (np.array) - Lower limit of statistical error on ADev.
	ADevSigU (np.array) - Upper limit of statistical error on ADev.
	"""

	if ADevType == 'Total':
		(tau, ADev, ADevErr, ADevN) = allan.totdev(yList, rate=rate, data_type='freq', taus=taus)
		tauMax = min(abs(tauMax), 1.0)
	else:
		(tau, ADev, ADevErr, ADevN) = allan.oadev(yList, rate=rate, data_type='freq', taus=taus)
		tauMax = min(abs(tauMax), 1.0)

	## Total number of data points
	N = len(yList)
	## Number of averaging times requested
	ntau = len(tau)

	## Keep only a subset of Allan deviation
	if taus == 'all':
		ntau = int(round(tauMax*ntau))
	elif taus == 'octave' or taus == 'decade':
		itau = 0
		while tau[itau] <= tauMax*tau[-1]:
			itau += 1
		ntau = itau+1

	tau     = tau[:ntau]
	ADev    = ADev[:ntau]
	ADevN   = ADevN[:ntau]
	ADevErr = ADevErr[:ntau]

	if ComputeErr:
		ADevErrL, ADevErrU = ComputeADevErrors(N, ntau, rate, tau, ADev, ADevErr, ADevType)[:2]
		return (tau, ADev, ADevErrL, ADevErrU)
	else:
		return (tau, ADev, ADevErr, ADevErr)

######################## End of AllanDev() ##########################
#####################################################################

def ComputeADevErrors(N, ntau, rate, tau, ADev, ADevErr, ADevType='Total', ModelType='chi2'):
	"""Compute 1-sigma confidence intervals and one- or two-sided uncertainties for Allan deviations.
	ARGUMENTS:
	N             (int) - Total number of data points
	ntau          (int) - Number of averaging times contained in 'tau'
	rate        (float) - Sampling rate of input data (Hz)
	tau      (np.array) - Averaging times
	ADev     (np.array) - Allan deviation
	ADevErr  (np.array) - One-sided uncertainty output by allantools
	ADevType      (str) - Type of Allan deviation
	ModelType     (str) - Type of model to use to compute uncertainty
	RETURN FORMAT: (ADevErrL, ADevErrU, ADevCIL, ADevCIU)
	ADevErrL (np.array) - Lower bound of ADev uncertainty
	ADevErrU (np.array) - Upper bound of ADev uncertainty
	ADevCIL  (np.array) - Lower bound of ADev confidence interval
	ADevCIU  (np.array) - Upper bound of ADev confidence interval
	"""

	if ModelType == 'chi2':
		## Compute Allan deviation confidence intervals based on chi2-distribution
		ADevErrL = np.zeros(ntau)
		ADevErrU = np.zeros(ntau)
		ADevCIL  = np.zeros(ntau)
		ADevCIU  = np.zeros(ntau)

		for i in range(ntau):
			## Averaging factor tau = m*tau0 = m/rate
			m = rate*tau[i]
			if ADevType == 'Total':
				## Equivalent degrees of freedom for total ADev (white frequency noise case)
				edf = 1.5*float(N)/m
			else:
				## Equivalent degrees of freedom for overlapped ADev (white frequency noise case)
				edf = (3*(float(N)-1)/(2*m) - 2*(float(N)-2)/float(N))*4*m**2/(4*m**2 + 5)
			## Chi-Squared values corresponding to +/- sigma
			(chi2L, chi2U) = ChiSquaredModel(edf)
			## Estimate two-sided confidence interval
			ADevCIL[i]  = np.sqrt(edf/chi2U)*ADev[i]
			ADevCIU[i]  = np.sqrt(edf/chi2L)*ADev[i]
			ADevErrL[i] = np.abs(ADevCIL[i] - ADev[i])
			ADevErrU[i] = np.abs(ADevCIU[i] - ADev[i])
	else:
		## Simple one-sided confidence interval output by allantools
		##   ADevErr = ADev/sqrt(n), where n is the number of data pairs used to compute each ADev
		ADevErrL = ADevErr
		ADevErrU = ADevErr
		ADevCIL  = ADev - ADevErr
		ADevCIU  = ADev + ADevErr

	return (ADevErrL, ADevErrU, ADevCIL, ADevCIU)

################### End of ComputeADevErrors() ######################
#####################################################################

def ChiSquaredModel(dof, p=0.683):
	"""Compute chi-squared distribution for a given number of degrees of freedom (dof)
	at confidence level p .
	ARGUMENTS:
	dof   (float) - Number of degrees of freedom
	p     (float) - Value between 0 and 1 corresponding to confidence level.
					Defaults to 0.683 corresponding to 1 sigma
	RETURN FORMAT: (chi2U, chi2L)
	chi2U (float) - Upper limit of chi2 distribution
	chi2L (float) - Lower limit of chi2 distribution
	"""

	chi2L = chi2.ppf(0.5*(1-p), dof) ## Percent point function (inverse of cdf)
	chi2U = chi2.ppf(0.5*(1+p), dof) 

	return (chi2L, chi2U)

#################### End of ChiSquaredModel() #######################
#####################################################################

def AnalyzeTimeSeries(tRange, yData, yErr, Options):
	"""Analyze and plot summary of time series data.
	ARGUMENTS:
	tRange  (list)    - List of values characterizing time range and sampling rate [tStart, tStop, tStep]
	yData   (2D list) - 2D list of time series datasets with shape [[y1,y2,...,yN], [z1,z2,...,zM], ...]
						Each 'yi' is a 1D np.array containing data to plot on a single graph.
						The 'zi's are plotted on a graph below the 'yi's and so on.
						PSDs and ADevs of each dataset are also computed (up to a number MaxSubSet
						defined in 'Options') and displayed on separate graphs.
	yErr    (2D list) - Errors associated with each data set within yData.
	Options (dict)    - Key:value pairs controlling analysis options:
		'SavePlot'          (bool)  - Flag for saving (True) or showing (False) plot
		'PlotFolderPath'    (str)   - Folder in which to save plot
		'PlotFileName'      (str)   - Filename for plot (including extension)
		'ColumnDim'         (tuple) - Dimensions of each plot column (width, height)
		'Colors'            (list)  - List of M strings containing plot colors for each variable
		'Linestyle'			(str)   - Control for time series plot line style
		'Linewidth'			(float) - Control for time series plot line width
		'Marker'			(str)   - Control for time series plot marker style
		'Markersize'		(int)   - Control for time series plot marker size
		'ShowErrors'        (bool)  - Flag for showing error bars on time series plots
		'SampleRate'	    (float) - Sample rate of data sets contained in yData
		'xLabels'           (list)  - List of M strings containing plot labels for each x-axis
		'yLabels'           (list)  - List of M+2 plot labels for each y-axis (PSD + ADev included)
		'yScales'           (list)  - List of M scale factors for each y-axis
		'ShowFigureLabels'  (bool)  - Flag for showing figure labels
		'FigureLabels'		(list)  - List of M+2 strings containing figure labels (PSD + ADev included)
		'ShowLegend'        (list)  - List of flags for showing legend on (PSD,ADev) plots
		'LegendLabels'      (list)  - List of M strings containing legend labels for each variable
		'LegendLocations'   (list)  - List of legend locations ('outside' puts legends outside the plot box)
		'LegendFontSize'    (int)   - Font size for legend text
		'SetPlotLimits'     (list)  - List of flags to set X and Y limits of time-series plots
		'PlotXLimits'       (list)  - List of X limits for all time-series plots
		'PlotYLimits'       (list)  - List of Y limits for each time-series plots
		'PSD_Plot'          (bool)  - Flag for plotting power spectral density
		'PSD_PlotSubSets'   (list)  - Flags for plotting PSD of selected elements of yData
		'PSD_Method'        (str)   - Control for selecting PSD algorithm ('welch' or 'periodogram')
		'ADev_Plot'         (bool)  - Flag for plotting Allan deviation
		'ADev_PlotSubSets'  (list)  - Flags for plotting Allan deviation of selected elements of yData
		'ADev_Type'         (str)   - Controls for which ADev types to compute
		'ADev_taus'         (str)   - Control for set of averaging times to use ('all', 'decade', 'octave')
		'ADev_ShowErrors'   (bool)  - Flag for computing and displaying uncertainties on ADev graph
		'ADev_Errorstyle'   (str)   - Control for error bar style in ADev plot ('Bar', 'Shaded')
		'ADev_Linestyle'    (str)   - Control for linestyle in ADev plot
		'ADev_Marker'       (str)   - Control for marker type in ADev plot
		'ADev_SetLimits'    (list)  - Flags for setting [xlimits, ylimits] of ADev plot
		'ADev_XLimits'      (list)  - List containing [xmin, xmax] for ADev plot
		'ADev_YLimits'      (list)  - List containing [ymin, ymax] for ADev plot
		'ADev_Fit'          (list)  - List of flag for fitting Allan deviation
		'ADev_Fit_XLimits'  (list)  - List containing [xmin, xmax] for ADev fit
		'ADev_Fit_SetRange'	(list)  - List of flags for setting independent fit x-ranges,
		'ADev_Fit_Range'	(list)  - List of fit x-ranges,
		'ADev_Fit_FixExp'	(list)  - List of flags for fixing the exponent 'alpha' in ADev fits
	RETURN FORMAT: 
	    [t, f, PSD, tau, ADev, ADevErrL, ADevErrU]
	"""

	logging.info('iXUtils::Analyzing time series data...')

	plt.rc('legend', fontsize=Options['LegendFontSize'], handletextpad=0.3)
	plt.rc('lines', linewidth=Options['Linewidth'], markersize=Options['Markersize'])

	nRows    = len(yData)
	nSamps   = len(yData[0][0])

	f        = np.empty(nRows, dtype=object)
	PSD      = np.empty(nRows, dtype=object)
	tau      = np.empty(nRows, dtype=object)
	ADev     = np.empty(nRows, dtype=object)
	ADevErrL = np.empty(nRows, dtype=object)
	ADevErrU = np.empty(nRows, dtype=object)

	if Options['ADev_Plot'] and Options['PSD_Plot']:
		## Plot time series + ADev + PSD
		nCol = 3
		[rPSD, rADev] = [nRows, nRows+1]
	elif Options['ADev_Plot'] or Options['PSD_Plot']:
		## Plot time series + ADev or PSD
		nCol = 2
		if Options['ADev_Plot']:
			rADev = nRows
		else:
			rPSD  = nRows
	else:
		## Plot time series only
		nCol = 1			

	(colW, colH) = Options['ColumnDim']
	fig = plt.figure(figsize=(nCol*colW, colH), constrained_layout=True)
	gs  = fig.add_gridspec(nRows, nCol)
	axs = []
	for r in range(nRows):
		axs.append(fig.add_subplot(gs[r, 0])) # Left column, independent rows

	if nRows > 1:
		axs[0].get_shared_x_axes().join(*axs) # Share temporal axis of all rows

	if Options['ADev_Plot'] and Options['PSD_Plot']:
		axs.append(fig.add_subplot(gs[:, 1])) # Middle column spans all rows
		axs.append(fig.add_subplot(gs[:, 2])) # Right column spans all rows
	elif Options['ADev_Plot'] or Options['PSD_Plot']:
		axs.append(fig.add_subplot(gs[:, 1])) # Right column spans all rows

	t = np.linspace(tRange[0], tRange[1], num=nSamps, endpoint=True)

	## Default plot options
	PlotOpts = {'Color': 'red', 'Linestyle': Options['Linestyle'], 'Marker': Options['Marker'],
		'Title': 'None', 'xLabel': 'None', 'yLabel': 'None',
		'LegLabel': 'None', 'Legend': False, 'LegLocation': 'best'}

	for r in range(nRows):
		for j in range(len(yData[r])):
			PlotOpts['Color']    = Options['Colors'][r][j]
			PlotOpts['xLabel']   = Options['xLabels'][r]
			PlotOpts['yLabel']   = Options['yLabels'][r]
			PlotOpts['LegLabel'] = Options['LegendLabels'][r][j]

			tSub = t.copy()
 			## Ensures length of x & y are the same
			nSamps_rj = len(yData[r][j])
			if nSamps_rj != nSamps:
				logging.warning('iXUtils::AnalyzeTimeSeries::Length of yData[{}][{}] ({}) does not match length of t ({})...'.format(r, j, nSamps_rj, nSamps))
				if nSamps_rj < nSamps:
					tSub = t[:nSamps+1]
				elif nSamps_rj > nSamps:
					yData[r][j] = yData[r][j][:nSamps+1]
					yErr[r][j]  = yErr[r][j][:nSamps+1]
				
			CustomPlot(axs[r], PlotOpts, tSub, Options['yScales'][r][j]*yData[r][j])
			if Options['ShowErrors'] and np.linalg.norm(yErr[r][j]) > 0:
				axs[r].fill_between(tSub, Options['yScales'][r][j]*(yData[r][j] - yErr[r][j]), Options['yScales'][r][j]*(yData[r][j] + yErr[r][j]), alpha=0.4, color=Options['Colors'][r][j])

		## Remove axis labels, but keep ticks, for all but last row
		plt.setp(axs[r].get_xticklabels(), visible=False if r < nRows-1 else True)

	if Options['PSD_Plot']:
		plt.rc('lines', linewidth=1.5)
		PlotOpts['xLabel'] = r'$f$  (Hz)'
		PlotOpts['yLabel'] = Options['yLabels'][nRows]
		logScale = [True, True]

		for r in range(nRows):
			for j in range(len(yData[r])):
				if Options['PSD_PlotSubSets'][r][j]:
					PlotOpts['Color']    = Options['Colors'][r][j]
					PlotOpts['LegLabel'] = Options['LegendLabels'][r][j]

					if Options['PSD_Method'] == 'welch':
						f[r], PSD[r] = welch(yData[r][j], fs=Options['SampleRate'], return_onesided=True, scaling='density')
						f[r]   = f[r][:-1]
						PSD[r] = PSD[r][:-1]
					else:
						f[r], PSD[r] = periodogram(yData[r][j], fs=Options['SampleRate'], return_onesided=True, scaling='density')
						f[r]   = f[r][1:-1]
						PSD[r] = PSD[r][1:-1]

					PSD[r] = np.sqrt(PSD[r])
					CustomPlot(axs[rPSD], PlotOpts, f[r], PSD[r], [], logScale)

	if Options['ADev_Plot']:
		plt.rc('lines', linewidth=1.5)

		PlotOpts['xLabel'] = r'$\tau$  (s)'
		PlotOpts['yLabel'] = Options['yLabels'][nRows+1]

		logScale = [True, True]

		for r in range(nRows):
			for j in range(len(yData[r])):
				if Options['ADev_PlotSubSets'][r][j]:
					PlotOpts['Linestyle'] = Options['ADev_Linestyle']
					PlotOpts['Marker']    = Options['ADev_Marker']
					PlotOpts['Color']     = Options['Colors'][r][j]
					PlotOpts['eColor']    = Options['Colors'][r][j]
					PlotOpts['LegLabel']  = Options['LegendLabels'][r][j]

					(tau[r], ADev[r], ADevErrL[r], ADevErrU[r]) = AllanDev(yData[r][j], taus=Options['ADev_taus'], rate=Options['SampleRate'], 
							ADevType=Options['ADev_Type'], ComputeErr=Options['ADev_ShowErrors'])

					if Options['ADev_ShowErrors']:
						if Options['ADev_Errorstyle'] == 'Bar':
							CustomPlot(axs[rADev], PlotOpts, tau[r], ADev[r], yErr=np.array([ADevErrL[r], ADevErrU[r]]), LogScale=logScale)
						else:
							CustomPlot(axs[rADev], PlotOpts, tau[r], ADev[r], yErr=[], LogScale=logScale)
							axs[rADev].fill_between(tau[r], ADev[r] - ADevErrL[r], ADev[r] + ADevErrU[r], color=PlotOpts['Color'], alpha=0.3)
					else:
						CustomPlot(axs[rADev], PlotOpts, tau[r], ADev[r], yErr=[], LogScale=logScale)

					if Options['ADev_Fit'][r][j]:
						if Options['ADev_Fit_SetRange'][r][j]:
							xFit  = np.array([])
							yFit  = np.array([])
							dyFit = np.array([])
							for ix in range(len(tau[r])):
								if tau[r][ix] > Options['ADev_Fit_Range'][r][j][0] and tau[r][ix] < Options['ADev_Fit_Range'][r][j][1]:
									xFit  = np.append(xFit, tau[r][ix])
									yFit  = np.append(yFit, ADev[r][ix])
									dyFit = np.append(dyFit, 0.5*(ADevErrL[r][ix] + ADevErrU[r][ix]))
						else:
							xFit  = tau[r].copy()
							yFit  = ADev[r].copy()
							dyFit = 0.5*(ADevErrL[r] + ADevErrU[r])

						logx, logdy = np.log10(xFit), dyFit/(np.log(10)*yFit)
						if not Options['ADev_Fit_FixExp'][r][j]:
							logy = np.log10(yFit)
							pOpt, pCov = np.polyfit(logx, logy, 1, w=1/logdy, full=False, cov=True)
							pErr = np.sqrt(np.diag(pCov))
							alpha, logy0   = pOpt
							dalpha, dlogy0 = pErr
						else:
							logy = np.log10(yFit) + 0.5*logx
							pOpt, pCov = np.polyfit(logx, logy, 0, w=1/logdy, full=False, cov=True)
							pErr = np.sqrt(np.diag(pCov))
							alpha, dalpha = -0.5, 0.
							logy0, dlogy0 = pOpt[0], pErr[0]

						sigma, dsigma = 10**logy0, np.log(10)*10**logy0*dlogy0
						print('alpha = {:.2e} +/- {:.2e}'.format(alpha, dalpha))
						print('sigma = {:.2e} +/- {:.2e}'.format(sigma, dsigma))

						x = np.linspace(Options['ADev_Fit_XLimits'][0], Options['ADev_Fit_XLimits'][1], num=10, endpoint=True)

						PlotOpts['Color']     = Options['Colors'][r][j]
						PlotOpts['Linestyle'] = '-'
						PlotOpts['Marker']    = 'None'
						PlotOpts['LegLabel']  = None
						CustomPlot(axs[rADev], PlotOpts, x, sigma*x**alpha, LogScale=logScale)

	for r in range(nRows):
		## Turn off offset on y-axis tick labels of time series plots
		axs[r].get_yaxis().get_major_formatter().set_useOffset(False)

	if Options['SetPlotLimits'][0]:
		## Set X-limits for time-series plots
		for r in range(nRows):
			axs[r].set_xlim(*Options['PlotXLimits'])

	if Options['SetPlotLimits'][1]:
		## Set Y-limits for time-series plots
		for r in range(nRows):
			axs[r].set_ylim(*Options['PlotYLimits'][r])

	if Options['ADev_Plot'] and Options['ADev_SetLimits'][0]:
		## Set xlimits for ADev plot
		axs[rADev].set_xlim(*Options['ADev_XLimits'])

	if Options['ADev_Plot'] and Options['ADev_SetLimits'][1]:
		## Set ylimits for ADev plot
		axs[rADev].set_ylim(*Options['ADev_YLimits'])

	if Options['ShowLegend'][0]:
		if Options['LegendLocations'][0] == 'outside':
			## Set legend location outside of plot
			for r in range(nRows):
				axs[r].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		else:
			## Set legend location to upper right
			for r in range(nRows):
				axs[r].legend(loc=Options['LegendLocations'][0])

	if Options['PSD_Plot'] and Options['ShowLegend'][1]:
		if Options['LegendLocations'][1] == 'outside':
			## Set legend location outside of plot
			axs[rPSD].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		else:
			## Set legend location to upper right
			axs[rPSD].legend(loc=Options['LegendLocations'][1])

	if Options['ADev_Plot'] and Options['ShowLegend'][2]:
		if Options['LegendLocations'][2] == 'outside':
			## Set legend location outside of plot for last plot
			axs[rADev].legend(loc='upper left', bbox_to_anchor=(1.01,1.0))
		else:
			## Set legend location to upper right
			axs[rADev].legend(loc=Options['LegendLocations'][2])

	if Options['ShowFigureLabels']:
		## Set figure label locations
		for r in range(nRows):
			axs[r].text(-0.18, 0.91, Options['FigureLabels'][r], transform=axs[r].transAxes, color='black', fontsize=16)
		if Options['PSD_Plot']:
			axs[rPSD].text(-0.21, 0.97, Options['FigureLabels'][rPSD], transform=axs[rPSD].transAxes, color='black', fontsize=16)
		if Options['ADev_Plot']:
			axs[rADev].text(-0.21, 0.97, Options['FigureLabels'][rADev], transform=axs[rADev].transAxes, color='black', fontsize=16)

	if Options['SavePlot']:
		if not os.path.exists(Options['PlotFolderPath']):
			os.makedirs(Options['PlotFolderPath'])

		plotPath = os.path.join(Options['PlotFolderPath'], Options['PlotFileName'])

		plt.savefig(plotPath, dpi=150)
		logging.info('iXUtils::Time series data plot saved to:')
		logging.info('iXUtils::  {}'.format(plotPath))
	else:
		plt.show()

	return [t, f, PSD, tau, ADev, ADevErrL, ADevErrU]

#################### End of AnalyzeTimeSeries() #####################
#####################################################################