#####################################################################
## Filename:	iXAtom_Systematics.py
## Author:		B. Barrett
## Version:		3.2.5
## Description:	Main implementation for evaluation of iXAtom systematic effects.
## Last Mod:	29/11/2020
#####################################################################

import logging
import numpy as np
import os
import pandas as pd

import iXAtom_Utilities 		  as iXUtils
import iXAtom_Class_RunParameters as iXC_RunPars
import iXAtom_Class_Systematics	  as iXC_Sys

##===================================================================
##  Set control modes
##===================================================================

mode	= 'EvalSys'						## Control for evaluating selected systematics or parameters ('EvalSys', 'EvalPar')
# mode	= 'EvalPar'						## Control for evaluating selected systematics or parameters ('EvalSys', 'EvalPar')

# orient	= 'Vertical'					## Control for orientation-dependent parameters ('Vertical', 'Tilted')
orient	= 'Tilted'						## Control for orientation-dependent parameters ('Vertical', 'Tilted')

iaxList	= [2]							## Control for Raman axes at which to evaluate systematics (0,1,2 = X,Y,Z)
ksList  = [1]							## Control for momentum transfer at which to evaluate parameters (-1,0,+1 = -keff,kco,+keff)

## Root directory containing data
rootDir = 'C:\\Bryns Goodies\\Work-iXAtom\\Data 2020\\'

dataSets = [None]
# dataSets = [
# 	{'Month': 'August', 'Day': 21, 'Runs': [16,27,36]}, ## 'TiltX': 340, 'TiltZ': -30,  'RamanTs': [ 4., 5., 5.]
# 	{'Month': 'August', 'Day': 21, 'Runs': [45,54,63]}, ## 'TiltX': 320, 'TiltZ': -30,  'RamanTs': [ 5., 5., 5.]

# 	{'Month': 'August', 'Day': 28, 'Runs': [ 8, 8, 8]}, ## 'TiltX':  30, 'TiltZ':  30,  'RamanTs': [10.,10.,10.]
# 	{'Month': 'August', 'Day': 29, 'Runs': [ 8, 8, 8]}, ## 'TiltX': 330, 'TiltZ':  30,  'RamanTs': [10.,10.,10.]
# 	{'Month': 'August', 'Day': 29, 'Runs': [16,16,16]}, ## 'TiltX': 300, 'TiltZ':  30,  'RamanTs': [10.,10.,10.]
# 	{'Month': 'August', 'Day': 29, 'Runs': [24,24,24]}, ## 'TiltX':  40, 'TiltZ':  30, 'RamanTs': [10.,10.,10.]
# 	{'Month': 'August', 'Day': 31, 'Runs': [ 8, 8, 8]}, ## 'TiltX':  20, 'TiltZ':  30,  'RamanTs': [10.,10.,10.]
# 	{'Month': 'August', 'Day': 31, 'Runs': [18,18,18]}, ## 'TiltX': 340, 'TiltZ':  30,  'RamanTs': [10.,10.,10.]

# 	{'Month': 'September', 'Day': 1, 'Runs': [ 8, 8, 8]}, ## 'TiltX':  30, 'TiltZ': -15,  'RamanTs': [10.,10.,10.]
# 	{'Month': 'September', 'Day': 1, 'Runs': [16,16,16]}, ## 'TiltX': 320, 'TiltZ': -15, 'RamanTs': [10.,10.,10.]
# 	{'Month': 'September', 'Day': 1, 'Runs': [24,24,24]}, ## 'TiltX': 300, 'TiltZ': -15, 'RamanTs': [10.,10.,10.]
# 	{'Month': 'September', 'Day': 1, 'Runs': [32,32,32]}, ## 'TiltX':  50, 'TiltZ': -15,  'RamanTs': [10.,10.,10.]

# 	{'Month': 'September', 'Day': 1, 'Runs': [40,40,40]}, ## 'TiltX':  20, 'TiltZ': -45, 'RamanTs': [10.,10.,10.]
# 	{'Month': 'September', 'Day': 1, 'Runs': [48,48,48]}, ## 'TiltX':  40, 'TiltZ': -45, 'RamanTs': [10.,10.,10.]
# 	{'Month': 'September', 'Day': 1, 'Runs': [50,50,50]}, ## 'TiltX':  56, 'TiltZ': -45, 'RamanTs': [10.,10.,10.]
# 	{'Month': 'September', 'Day': 3, 'Runs': [ 1, 1, 1]}] ## 'TiltX': 315, 'TiltZ': -45, 'RamanTs': [10.,10.,10.]

##===================================================================
##  Set evaluation options
##===================================================================

sysOpts = {
	# 'SysList':		['All'],		## List of systematics to evaulate ('OPLS', 'TPLS', 'QZ', 'BF', 'GG', 'Cor', 'PL', 'WD', 'MZA', 'SF', 'NL', 'All')
	# 'SysList':		['SF','WC','TPLS','PL','Cor','OPLS','QZ','NL'],
	'SysList':		['SF', 'WC', 'TPLS', 'Cor'],
	'Print':        True,				## Flag for printing table of systematics

	'UseBCal':		True,				## Use B-field calibration vs tilt angles for the B-field model
	'TPLS_Form':	'Exact',			## Two-photon light shift: expression form ('Exact' or 'Approx')
	'PL_EOMType':	'IQ',				## Parasitic lines: electro-optic modulator type ('PM' or 'IQ')
	'RabiModel':	'Gaussian',			## Rabi frequency model ('None', 'Gaussian', 'PL')
	'DetuningType':	'OPLS', 			## Type of Raman detuning model to use ('Full', 'OPLS', 'Basic')
	'RTFreq':		'On',				## RT frequency feedback switch in Raman detuning ('On', 'Off')
	'Units':        'ug',				## Evaluation units ('rad', 'mrad', 'm/s^2', 'ug', 'Hz/s')

	'Plot':        	'Off',				## Flag for plotting systematics ('1D', '2D', 'Off')
	'ShowPlot':		True,				## Flag for diplaying plot
	'Plot2D_Var':	'kDep',				## 2D systematic plot variable ('kU', 'kD', 'kInd', 'kDep') 
	'PlotVariables':['T'],				## Independent variable to plot systematics against ('T', 'TOF', 'v0', 'zM', 'Rabi1_kU', 'TiltX', 'TiltZ')
	'PlotRanges':   [[1.E-3, 20.E-3]],	## Range  of independent variables
	'PlotPoints': 	[101],				## Number of independent variables
	# 'PlotVariables':	['TOF'],
	# 'PlotRanges':    [[10.E-3, 50.E-3]],
	# 'PlotPoints':   [101],
	# 'PlotVariables':	['zM'],
	# 'PlotRanges':    [[-200.E-3, 0.E-3]],
	# 'PlotPoints':   [201],
	# 'PlotVariables': ['TiltX', 'TiltZ'],
	# 'PlotRanges':   [[-90., 90.], [-90.,90.]],
	# 'PlotPoints':   [201,201], 

	'Export':		False,				## Flag for exporting systematics in rad and SI units (2D systematics always exported in rad)
	## Path for exporting 2D plot
	'Folder':		'C:\\Bryns Goodies\\Dropbox\\Python Code\\iXAtom Analysis\\Systematics\\', ## Folder to store systematics
	'FilePrefix':	'Final-2D'			## Systematic file prefix (e.g. FilePrefix+'-Z-kDep-vs-zM.txt')

	## Path for exporting tables
	# 'Folder':		'C:\\Bryns Goodies\\Dropbox\\Python Code\\iXAtom Analysis\\Accel Calibration\\Systematics\\',
	# 'FilePrefix':	'TiltX+340_TiltZ-30'
}

parOpts = {
	# 'ParList':		['All'],			## List of parameters to evaluate ('All' = ['rCOM', 'vCOM', 'B', 'BGrad', 'OmegaR', 'Contrast'])
	# 'ParList':		['B', 'BGrad'],		## List of parameters to evaluate ('rCOM', 'vCOM', 'B', 'BGrad', 'OmegaR', 'delta1', 'delta2', 'delta3', 'Contrast')
	# 'ParList':		['rCOM', 'vCOM', 'OmegaR', 'Contrast'],		## List of parameters to evaluate
	'ParList':		['Contrast'],		## List of parameters to evaluate
	'Print':        True,				## Flag for printing table of parameters

	'UseBCal':		True,				## Use B-field calibration vs tilt angles for the B-field model
	'TPLS_Form':	'Exact',			## Two-photon light shift: expression form ('Exact' or 'Approx')
	'PL_EOMType':	'PM',				## Parasitic lines: electro-optic modulator type ('PM' or 'IQ')
	'RabiModel':	'Gaussian',			## Type of model to use for Rabi frequency ('None', 'PL', 'Gaussian')
	'DetuningType':	'OPLS', 			## Type of Raman detuning model to use ('Full', 'OPLS', 'Basic')
	'RTFreq':		'On',				## RT frequency feedback switch in Raman detuning ('On', 'Off')
	'RamanRegime':	'DD',				## Raman single- or double-diffraction regime ('SD', 'DD')
	'kCoTrans':		'On',				## Include co-propagating transitions in Contrast model ('On', 'Off')
	'VelocityAvg':	'Off',				## Velocity averaging for Contrast model ('On', 'Off')
	
	'Plot':         True,				## Flag for plotting parameters
	# 'PlotVariable':	'TOF',				## Independent variable to plot parameters against ('t', 'TOF', 'zM', 'TiltX', 'TiltZ')
	# 'PlotRange':    [10.E-3, 90.E-3],	## Range  of independent variables
	# 'PlotPoints':   101,					## Number of independent variables
	# 'PlotVariable':	'zM',				## Independent variable to plot systematics against ('t', 'TOF', 'zM', 'TiltX', 'TiltZ')
	# 'PlotRange':    [-200.E-3, 0.E-3],	## Range  of independent variables
	# 'PlotPoints':   201,				## Number of independent variables
	'PlotVariable':	'TiltX',			## Independent variable to plot parameters against ('t', 'TOF', 'zM', 'TiltX', 'TiltZ')
	'PlotRange':    [-60., 60.],		## Range  of independent variables
	'PlotPoints':   101,				## Number of independent variables

	'Export':		False,				## Flag for exporting parameters in SI units (Plot = True)
	'Folder':		'C:\\Bryns Goodies\\Dropbox\\Python Code\\iXAtom Analysis\\Contrast Model\\', ## Folder to store parameters
	# 'FilePrefix':	'Contrast-T=2.5ms-RTOn-kCo'	## Parameter file prefix (e.g. FilePrefix+'-Z-kU-vs-zM.txt')
	# 'FilePrefix':	'Contrast-T=5.0ms-RTOn-kCo' ## Parameter file prefix
	'FilePrefix':	'Contrast-T=10.0ms-RTOn-kCo'	## Parameter file prefix
	# 'FilePrefix':	'Contrast-T=10.0ms-RTOff-kCo'	## Parameter file prefix
	# 'FilePrefix':	'Contrast-T=10.0ms-RTOff-vAvg'	## Parameter file prefix
}

##===================================================================
##  Configure logger
##===================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s::%(message)s')

##===================================================================
## Set default plot options
##===================================================================

iXUtils.SetDefaultPlotOptions()

#####################################################################
def main():

	##----------------------- Initialization ------------------------
	# logging.disable(level=logging.INFO) # Disable logger for info & debug levels

	if dataSets[0] is None or mode == 'EvalPar':
		class RunPars:
			def __init__(self):
				self.SoftwareVersion	= 3.4
				self.RamanDetuning		= -1.144E9
				if orient == 'Tilted':
					## Parameters from Oct 15 2020:
					self.TiltX   		= 54.7
					self.TiltZ   		= -45.0
					self.RamanTOF		= 19.9E-3
					self.RamanT 		= 10.0E-3
					self.RamanpiX		= 12.0E-6
					self.RamanpiY		= 11.0E-6
					self.RamanpiZ		= 10.0E-6
					self.RamankUFreq  	= [6.835000793E+9, 6.834402851E+9, 6.834988786E+9]
					self.RamankDFreq   	= [6.834390844E+9, 6.834983983E+9, 6.834398048E+9]
					self.RamankUChirp  	= [+1.472810000E+7, -1.436040000E+7, +1.444660000E+7]
					self.RamankDChirp 	= [-1.472810000E+7, +1.436040000E+7, -1.444660000E+7]
					self.SelectionFreqs = [6.834457611E+9, 6.834907611E+9]
				else: ## orient == 'Vertical'
					## Parameters from Oct 22 2020:
					self.TiltX   		= 0.
					self.TiltZ   		= 0.
					self.RamanTOF		= 19.9E-3
					self.RamanT 		= 10.0E-3
					self.RamanpiX		= 6.08E-6
					self.RamanpiY       = 11.0E-6
					self.RamanpiZ       =  9.0E-6
					self.RamankUFreq  	= [6.835381931E+9, 6.835381931E+9, 6.835381931E+9]
					self.RamankDFreq   	= [6.834018665E+9, 6.834018665E+9, 6.834018665E+9]
					self.RamankUChirp  	= [+2.5134928E+7, +2.5134928E+7, +2.5134928E+7]
					self.RamankDChirp 	= [-2.5134928E+7, -2.5134928E+7, -2.5134928E+7]
					self.SelectionFreqs	= [6.834457611E+9, 6.834907611E+9]

		## Initialize run parameters
		runPars = RunPars()

	if orient == 'Tilted':
		## Parameters from Oct 15-19 2020:
		sysInput = {
			'kCorrCoeff':	np.array([0.90, 0.90, 0.90]),		## Correlation coefficient for kInd phase shifts
			'r0':			np.array([
								[ 0.E+0, 2.E-4],
								[-8.E-4, 2.E-4],
								[ 0.E+0, 2.E-4]]),				## Initial position relative to mirrors (m)
			'v0':			np.array([
								[0., 3.E-3],
								[0., 3.E-3],
								[0., 3.E-3]]),					## Initial velocity (after molasses release) (m/s)
			'Temp':			np.array([
								[3.5E-6, 1.0E-6],
								[3.5E-6, 1.0E-6],
								[3.5E-6, 1.0E-6]]),				## Sample temperature (K)
			'Rabi1_kU': 	np.array([
								[2.991E-01, 7.5E-03], #[3.005E-01, 1.9E-03], [2.854E-01, 1.79E-02]
								[2.734E-01, 5.5E-03], #[2.732E-01, 4.5E-03], [2.743E-01, 1.61E-02]
								[3.159E-01, 7.5E-03]])*1.E6, #[3.191E-01, 2.6E-03], [3.041E-01, 9.56E-03] ## Counter-propagating Rabi frequency at t = t1 (rad/s)
			'Rabi3_kU': 	np.array([
								[1.384E-01, 2.0E-03], #[1.385E-01, 1.7E-02], [1.381E-01, 8.5E-02]
								[1.653E-01, 5.1E-03], #[1.658E-01, 4.4E-03], [1.555E-01, 9.6E-02]
								[1.995E-01, 7.8E-03]])*1.E6, #[1.992E-01, 9.4E-04], [2.147E-01, 4.8E-02]	## Counter-propagating Rabi frequency at t = t3 (rad/s)
			'Rabi1_kCo': 	np.array([
								[1.921E-01, 1.6E-02],
								[3.617E-01, 2.0E-02],
								[2.258E-01, 7.5E-03]])*1.E6,	## Co-propagating Rabi frequency for Delta mF = 0 at t = t1 (rad/s)
			'Rabi3_kCo': 	np.array([
								[9.622E-02, 3.8E-02],
								[1.844E-01, 1.1E-02],
								[1.623E-01, 7.3E-03]])*1.E6,	## Co-propagating Rabi frequency for Delta mF = 0 at t = t3 (rad/s)
			'beta0':		np.array([
								[0.13941, 0.00038],
								[0.13770, 0.00027],
								[0.14771, 0.00073]]),			## Magnetic bias field (G)
			'beta1':		np.array([
								[-0.045, 0.019],
								[+0.211, 0.011],
								[-0.204, 0.011]]),				## Magnetic field gradient (G/m)
			'beta2':		np.array([
								[0.000, 0.001],
								[0.000, 0.001],
								[0.000, 0.001]]),				## Magnetic field curvature (G/m^2)
			'wBeam':		np.array([0.011, 0.000]),			## 1/e^2 beam waist (m)
			'DM':			np.array([0.050, 0.001]),			## Mirror diameter (m)
			'FlatM':		np.array([1.E-6, 5.0E-7]),			## Mirror flatness (m)
			'IRatio': 		np.array([0.568, 0.005]),			## Intensity ratio of primary lines (I_1->2' / I_2->2')
			'zR':			np.array([
								[30.0, 5.],
								[30.0, 5.],
								[30.0, 5.]]),					## Rayleigh range of the Raman beams (m)
			'zM':			np.array([
								[-0.10125, 0.00040],
								[-0.11925, 0.00040],
								[-0.12196, 0.00018]]),			## Relative mirror position (m)
			'zC':			np.array([
								[+0.077, 0.010],
								[+0.094, 0.010],
								[+0.091, 0.010]]),				## Relative collimator position (m)
		}
	else: ## orient == 'Vertical'
		sysInput = {
			'kCorrCoeff':	np.array([0.90, 0.90, 0.90]),		## Correlation coefficient for kInd phase shifts
			'r0':			np.array([
								[0.000, 0.],
								[-8.E-4, 0.],
								[0.000, 0.]]),						## Initial position relative to mirrors (m)
			'v0':			np.array([
								[0., 1.E-3],
								[0., 1.E-3],
								[0., 1.E-3]]),					## Initial velocity (after molasses release) (m/s)
								# [-0.0155, 3.1E-3]]),			## Initial velocity (after molasses release) (m/s)
			'Temp':			np.array([
								[3.5E-6, 0.5E-6],
								[3.5E-6, 0.5E-6],
								[3.5E-6, 0.5E-6]]),				## Sample temperature (K)
			'Rabi1_kU': 	np.array([
								[0.431, 0.008],
								[0.253, 0.008],
								[0.314, 0.008]])*1.E6,			## Counter-propagating Rabi frequency at t = t1 (rad/s)
			'Rabi3_kU': 	np.array([
								[0.431*0.9, 0.008],
								[0.253*0.9, 0.008],
								[0.314*0.9, 0.008]])*1.E6,			## Counter-propagating Rabi frequency at t = t3 (rad/s)
			'Rabi1_kCo': 	np.array([
								[1.921E-01, 1.6E-02],
								[3.617E-01, 2.0E-02],
								[2.258E-01, 7.5E-03]])*1.E6,	## Co-propagating Rabi frequency for Delta mF = 0 at t = t1 (rad/s)
			'Rabi3_kCo': 	np.array([
								[9.620E-02, 3.8E-02],
								[1.844E-01, 1.1E-02],
								[1.623E-01, 7.3E-03]])*1.E6,	## Co-propagating Rabi frequency for Delta mF = 0 at t = t3 (rad/s)
			'beta0':		np.array([
								[0.15087, 0.00012],
								[0.15087, 0.00012],
								[0.15087, 0.00012]]),			## Magnetic bias field (G)
			'beta1':		np.array([
								[0.0507, 0.0069],
								[0.0507, 0.0069],
								[0.0507, 0.0069]]),				## Magnetic field gradient (G/m)
			'beta2':		np.array([
								[0.000, 0.001],
								[0.000, 0.001],
								[0.000, 0.001]]),				## Magnetic field curvature (G/m^2)
			'wBeam':		np.array([0.009, 0.000]),			## 1/e^2 beam waist (m)
			'DM':			np.array([0.050, 0.001]),			## Mirror diameter (m)
			'FlatM':		np.array([1.E-6, 0.5E-6]),			## Mirror flatness (m)
			'IRatio': 		np.array([0.568, 0.005]),			## Intensity ratio of primary lines (I_1->2' / I_2->2')
			'zR':			np.array([
								[30.0, 10.],
								[30.0, 10.],
								[30.0, 10.]]),					## Rayleigh range of the Raman beams (m)
			'zM':			np.array([
								[-0.10125, 0.0004],
								[-0.11925, 0.0004],
								[-0.12196, 0.00018]]),			## Relative mirror position (m)
			'zC':			np.array([
								[+0.077, 0.01],
								[+0.094, 0.01],
								[+0.091, 0.01]]),				## Relative collimator position (m)
		}

	if mode == 'EvalSys':
		sysTables = [pd.DataFrame([]) for iax in range(3)]

		for dataSet in dataSets:
			for iax in iaxList:
				logging.info('iX_Main::#######################################################################################')
				logging.info('iX_Main::                          Evaluating Systematics (iax = {})'.format(iax))
				logging.info('iX_Main::#######################################################################################')

				if dataSet is not None:
					workDir = os.path.join(rootDir, dataSet['Month'], '{:02d}'.format(dataSet['Day']))
					runPars = iXC_RunPars.RunParameters(workDir, 'Raman', dataSet['Runs'][iax])
					runPars.LoadRunParameters()

				sys = iXC_Sys.Systematics(runPars, sysInput, sysOpts)

				print(sys.Seff)
				print(sys.keff*(sys.T + sys.taupi)*(sys.T + 2.*sys.taupi/np.pi))
				print(sys.Seff/(sys.keff*(sys.T + sys.taupi)*(sys.T + 2.*sys.taupi/np.pi)))
				print(np.mean(sys.Seff/(sys.keff*(sys.T + sys.taupi)*(sys.T + 2.*sys.taupi/np.pi))))

				if dataSet is not None:# or sysOpts['Plot'] == '2D':
					sysOpts['FilePrefix'] = 'TiltX{:+}_TiltZ{:+}'.format(int(np.round(sys.TiltX)), int(np.round(sys.TiltZ)))
					sysInput['Rabi1_kU'][:,0]  = np.pi/sys.tau2
					sysInput['Rabi1_kU'][:,1]  = 0.05*sysInput['Rabi1_kU'][:,0]
					sysInput['Rabi1_kCo'][:,0] = 0.68*sysInput['Rabi1_kU'][:,0]
					sysInput['Rabi1_kCo'][:,1] = 0.10*sysInput['Rabi1_kCo'][:,0]

				if sysOpts['Print']:
					sysTables[iax] = sys.PrintSystematics(iax)

				if sysOpts['Plot'] == '1D':
					sys.PlotSystematics1D(iax)
				elif sysOpts['Plot'] == '2D':
					sys.PlotSystematics2D(iax)

			if sysOpts['Print'] and iaxList == [0,1,2]:
				iXC_Sys.PrintTriadSystematics(sysTables, sysOpts, runPars)

	elif mode == 'EvalPar':
		sys = iXC_Sys.Systematics(runPars, sysInput, parOpts)

		for ks in ksList:
			logging.info('iX_Main::#######################################################################################')
			logging.info('iX_Main::                            Evaluating Parameters (ks = {})'.format(ks))
			logging.info('iX_Main::#######################################################################################')

			if parOpts['Print']:
				sys.PrintParameters(iaxList, ks)

			if parOpts['Plot']:
				sys.PlotParameters(iaxList, ks)

	else:

		logging.error('iX_Main::Mode = {} not recognized...'.format(mode))

	logging.info('iX_Main::#######################################################################################')
	logging.info('iX_Main::Done!')

######################### End of main() #############################
#####################################################################

if __name__ == '__main__':
	main()