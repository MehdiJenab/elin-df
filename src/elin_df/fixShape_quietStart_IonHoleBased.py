



#******************************************************************************
#BEGIN ReadMe
	# Mehdi.Jenab@Chalmers.se
	# starts with a predetermined nonlinear shape of phi(x) and v_sol
	# 1) Find Elin DF of electron
	# 	\phi(x) |	---> E(x) ---> \ro(x)	|
	#	v_sol 	|=== Schamel DF ---> n_i(x)	| ===> n_e(x) --- Elin DF ---> *DF_e*

	# 2) fine Elin DF of ions
	# n_e(x) = n_i(x) guiding eq. quiet start--|
	# v_hole = 0.0 ----------------------------|=== Elin DF ----> *DF_i*

	# 3) Running Gkyll
	# DF_e |
	# DF_i | --- input Gkyll
#END
#******************************************************************************

#******************************************************************************
#BEGIN imports and dependencies
#==============================================================================
import 		numpy 				as np
import 		matplotlib.pyplot 	as plt
import 		sys
import 		os
import 		json
from 		scipy 				import integrate

from 		matplotlib 			import rc

#==============================================================================


#==============================================================================
from .Elin_Distribution_Function import ElinDistributionFunction as distributionClass

from .MathFunctions 						import mathClass 				as math
from .find_beta 						import FindBeta
from .global_parameter import global_parameters

from elin_df.file_utils import (
    remove_trailing_comma, 
    write_json_section_end,
    get_script_directory
)

p = global_parameters()
m  = math()


rc('font', size=15)
rc('legend', fontsize=15)
#==============================================================================
#END
#******************************************************************************

#******************************************************************************
#BEGIN global functions and expressions
tab = "	"
beginLine = "\n" + tab + tab

# Get script directory for relative file paths
import os
SCRIPT_DIR = get_script_directory(__file__)
def pause():
    programPause = input("Press to continue...")
#END
#******************************************************************************


#******************************************************************************
class QuietStart(object):
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN initialization and reading data
	#==========================================================================
	def __init__(self):
		print("Quiet Start starts ...")

		self.nVx_Gq 			= [1000,200] # since nvx should be large so searching beta can converge, nVx_Gq used for output
		self.iSpeciesElc 		= 0
		self.iSpeciesIon 		= 1





		iS 						= self.iSpeciesElc
		self.vGqElc 			= m.get_vGqArray(p.Vx_min[iS],p.Vx_max[iS],p.dVx[iS],self.nVx_Gq[iS])
		iS						= self.iSpeciesIon
		self.vGqIon 			= m.get_vGqArray(p.Vx_min[iS],p.Vx_max[iS],p.dVx[iS],self.nVx_Gq[iS])


		self.xGqArray 			= m.get_vGqArray(p.X_min,p.X_max,p.d_X,p.nX)


		#self.dataUnperturbed, self.dataSoliton, self.limitXsoliton, self.pd = self.read_input_file()
		self.dataUnperturbed,self.pd  	= self.set_dataUnperturbed()
		self.write_quietStartFile()

		#self.v_soliton 			= self.dataSoliton[0][self.pd["U0"]]
		self.x_df_output		= []
		self.flag_print = True
		if (self.flag_print==True):
			self.fig_target,   self.ax_target 	= plt.subplots(1) #plt.subplots() sharex=True,
			self.fig_fixed,    self.ax_fixed    = plt.subplots(1)
			self.fig_dfRef,   self.ax_dfRef 	= plt.subplots(1)
	#==========================================================================


	#==========================================================================
	def set_dataUnperturbed(self):
		dataUnperturbed 		= []
		with open(os.path.join(SCRIPT_DIR, 'inputPattern.json')) as data_file:
			data0 				= json.load(data_file)
		inputPattern 			= data0['inputPattern']
		pd = {} #pattern dictionary
		for idx in range(len(inputPattern)):
			pd [str(inputPattern[idx])] = idx
			dataUnperturbed.append([])

		dfTrapped		= distributionClass(i_species=self.iSpeciesElc, v_soliton = 0.0)
		df2dArr, vx2dArr, ex2dArr, en2dArr  = dfTrapped.return_DfVxExEn_4arrays()
		df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)
		dfGqArrayElc = m.get_interpolation(vx1dArr, self.vGqElc,df1dArr)

		moments 		= dfTrapped.getMomentsTemp()
		NameSc 			= "elc"
		dataUnperturbed[pd["Density"		+"_"+NameSc]] = moments[0]
		dataUnperturbed[pd["FirstMoment"	+"_"+NameSc]] = moments[1]
		dataUnperturbed[pd["SecondMoment"	+"_"+NameSc]] = moments[2]
		dataUnperturbed[pd["Temperature"	+"_"+NameSc]] = moments[3]

		dfReflected		= distributionClass(i_species=self.iSpeciesIon, v_soliton = 0.0)
		df2dArr, vx2dArr, ex2dArr, en2dArr  = dfReflected.return_DfVxExEn_4arrays()
		df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)
		dfGqArrayIon = m.get_interpolation(vx1dArr, self.vGqIon,df1dArr)

		moments 		= dfReflected.getMomentsTemp()
		NameSc 			= p.Name[self.iSpeciesIon]
		dataUnperturbed[pd["Density"		+"_"+NameSc]] = moments[0]
		dataUnperturbed[pd["FirstMoment"	+"_"+NameSc]] = moments[1]
		dataUnperturbed[pd["SecondMoment"	+"_"+NameSc]] = moments[2]
		dataUnperturbed[pd["Temperature"	+"_"+NameSc]] = moments[3]
		dataUnperturbed[pd["U0"]]						  = 0.0#"ToDo v_soliton

		dataUnperturbed[pd["X"]]						  = 0.0
		dataUnperturbed[pd["PHI"]]						  = 0.0
		dataUnperturbed[pd["Ex"]]						  = 0.0

		dataUnperturbed[pd["DF_ion"]]					  = dfGqArrayIon
		dataUnperturbed[pd["DF_elc"]]					  = dfGqArrayElc
		#print inputPattern
		#print dataUnperturbed
		return dataUnperturbed, pd
	#==========================================================================

	#==========================================================================
	def setFixedSoliton(self,**kwargs):
		a		 	= kwargs.get('a', "")
		l		 	= kwargs.get('l', "")
		x0		 	= kwargs.get('x0', "")
		vHoleElc	= kwargs.get('vElc', "")
		vHoleIon	= kwargs.get('vIon', "")

		phiFixed = []
		elfFixed = []
		rhoFixed = []
		xvlFixed = []

		epsilon = 1e-5
		n 		= 1 # power of sech
		def get_phi_elf_rho(x,a,n):
			phi 				= a * (1.0/np.cosh(x))**n
			elf 				= - (n* phi/l) * np.tanh(x)
			elf 				= - elf # minus from Gauss law
			#rho 				=   4 * a * (1.0/np.cosh(x))**2 * (np.tanh(x))**2 - 2*a* ((1.0/np.cosh(x)))**4
			rho 				= (n * phi/l**2) * ( n * np.tanh(x)**2 - (1.0/np.cosh(x))**2)
			rho 				= - rho # minus from Gauss law

			return phi,elf, rho
		for ix in range(len(self.xGqArray)):
			xR 					= self.xGqArray[ix]
			x 					= (xR-x0)/l
			phi0,elf0, rho0 	= get_phi_elf_rho(x,a,n)
			#x1		 			= x +1.5 # to add wiggle for supersolitons
			#phi1,elf1, rho1 	= get_phi_elf_rho(x1,0.1*a,15)

			phi 				= phi0 #+ phi1
			elf 				= elf0 #+ elf1
			rho 				= rho0 #+ rho1


			if phi<epsilon and rho<epsilon and elf<epsilon:
				pass
			else:
				phiFixed.append(phi)
				elfFixed.append(elf)
				rhoFixed.append(rho)
				xvlFixed.append(xR )


		self.ax_fixed.plot(xvlFixed,phiFixed,"-*")
		self.ax_fixed.plot(xvlFixed,elfFixed,"-*")
		self.ax_fixed.plot(xvlFixed,rhoFixed,"-*")
		#plt.pause(0.1)
		#m.pause()


		dataFixed = [[],[],[],[],[],[],[]]
		dic = {}

		dic ["X"] 	= 1
		dic ["PHI"] = 2
		dic ["Ex"] 	= 3
		dic ["RHO"] = 4
		dic ["vHoleElc"] = 5
		dic ["vHoleIon"] = 6
		dataFixed[0] = dic
		dataFixed[1] = xvlFixed
		dataFixed[2] = phiFixed
		dataFixed[3] = elfFixed
		dataFixed[4] = rhoFixed
		dataFixed[5] = vHoleElc
		dataFixed[6] = vHoleIon


		return dataFixed
	#==========================================================================




	#==========================================================================
	def write_header(self,msg):
		stg  = tab 		+ "{" +				"\n"
		stg += tab+tab	+ "\""+msg+"\":"+	"\n"
		stg += tab+tab	+ "[" +				"\n"
		stg += tab+tab
		return stg

	def write_headerEnding(self, file_DF):
		"""
		Close a JSON section by:
		1) Removing the trailing ', ' from the last element
		2) Appending the standard section ending:
			] 
			},
		"""
		# make sure everything written so far is on disk
		file_DF.flush()

		# get the underlying file path
		filepath = file_DF.name

		# remove the trailing ", " at the very end of the file, if present
		remove_trailing_comma(filepath, trailing=", ")

		# append:
		#   \n
		#   \t\t]\n
		#   \t},\n\n
		write_json_section_end(filepath, indent_level=2)

	def write_array(self,array):
		stg = ""
		for value in array:
			stg += str(value)+", "
		return stg[:-2]

	def write_pair(self,title,array,fileName):
		beginLine ='\n'+tab+tab+tab+tab
		stg  = beginLine + "{"
		stg += beginLine + '"'+title+'":'
		stg += beginLine + "[\n"
		stg += beginLine
		stg  = stg + self.write_array(array)
		fileName.write(stg)

		stg  = "\n"
		stg += beginLine + "]"
		stg += beginLine + "}, "
		fileName.write(stg)

	#==========================================================================


	#==========================================================================
	def write_quietStartFile(self):
		# writing the output file ---------------------------------------------
		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'w')

		stg  = "{\n\"kinetic\":\n[\n\n"
		stg += self.write_header("input_pattern")
		output =  sorted(self.pd, key=self.pd.__getitem__) #sort the dictionary based on its keys
		for parameter in output:
			stg += '"'+parameter+'"'+", "
		stg += '"df_' + p.Name[self.iSpeciesElc]+'", '
		file_DF.write(stg)
		self.write_headerEnding(file_DF)

		#------------------------------
		stg = self.write_header("Unpertured_data")
		for value in self.dataUnperturbed:
			if isinstance(value, list):
				stg += "\n[\n"
				for df in value:
					stg += str(df)+", "
				stg = stg[:-2]
				stg += "\n]\n, "
			else:
				stg += str(value)+", "

		file_DF.write(stg)
		self.write_headerEnding(file_DF)
		#------------------------------


		stg = self.write_header("velocity")
		file_DF.write(stg)

		self.write_pair("velocity_elc",self.vGqElc,file_DF)
		self.write_pair("velocity_ion",self.vGqIon,file_DF)
		self.write_headerEnding(file_DF)

		file_DF.write( self.write_header("solitons")  )
		file_DF.close()
		#------------------------------
	#==========================================================================

	#==========================================================================
	def write_Moments_DF_FixedSoliton(self, momentsElc,momentsIon,dfGqElc,dfGqIon,nameElc,nameIon,ix,dataFixed):
		dataXpoint = []
		for item in range (len(self.pd)-2):
			dataXpoint.append([])
		dataXpoint[self.pd["X"]] 	= dataFixed[dataFixed[0]["X"]][ix]
		dataXpoint[self.pd["PHI"]] 	= dataFixed[dataFixed[0]["PHI"]][ix]
		dataXpoint[self.pd["Ex"]] 	= dataFixed[dataFixed[0]["Ex"]][ix]
		dataXpoint[self.pd["U0"]] 	= dataFixed[dataFixed[0]["vHoleElc"]] # v_soliton
		dataXpoint[self.pd["Density"		+"_"+nameElc]] = momentsElc[0]
		dataXpoint[self.pd["FirstMoment"	+"_"+nameElc]] = momentsElc[1]
		dataXpoint[self.pd["SecondMoment"	+"_"+nameElc]] = momentsElc[2]
		dataXpoint[self.pd["Temperature"	+"_"+nameElc]] = momentsElc[3]

		dataXpoint[self.pd["Density"		+"_"+nameIon]] = momentsIon[0]
		dataXpoint[self.pd["FirstMoment"	+"_"+nameIon]] = momentsIon[1]
		dataXpoint[self.pd["SecondMoment"	+"_"+nameIon]] = momentsIon[2]
		dataXpoint[self.pd["Temperature"	+"_"+nameIon]] = momentsIon[3]

		beginLine = "\n" + tab + tab
		#------------------------------
		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		stg  = "\n"
		stg += beginLine +"["
		stg += beginLine
		for value in dataXpoint:
			stg += str(value)+", "

		file_DF.write(stg)

		self.write_pair('df_'+nameElc,dfGqElc,file_DF)
		self.write_pair('df_'+nameIon,dfGqIon,file_DF)
		# remove the trailing comma/space left by write_pair before closing this section
		remove_trailing_comma(file_DF.name, trailing=", ")


		stg = beginLine + "],  "
		file_DF.write(stg)
		file_DF.close()
	#==========================================================================


	#==========================================================================
	def write_Moments_DF(self,dataXpoint,momentsTrapped,dfGqArray,NameSc):
		dataXpoint[self.pd["Density"		+"_"+NameSc]] = momentsTrapped[0]
		dataXpoint[self.pd["FirstMoment"	+"_"+NameSc]] = momentsTrapped[1]
		dataXpoint[self.pd["SecondMoment"	+"_"+NameSc]] = momentsTrapped[2]
		dataXpoint[self.pd["Temperature"	+"_"+NameSc]] = momentsTrapped[3]
		beginLine = "\n" + tab + tab
		#------------------------------
		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		stg  = "\n"
		stg += beginLine +"["
		stg += beginLine
		for value in dataXpoint:
			stg += str(value)+", "

		file_DF.write(stg)

		stg  = beginLine + "{"
		stg += beginLine + '"df":'
		stg += beginLine + "["
		stg += beginLine
		stg += stg + self.write_array(dfGqArray)
		file_DF.write(stg)

		stg  = "\n"
		stg += beginLine + "]"
		stg += beginLine + "}"
		stg += beginLine + "], "
		file_DF.write(stg)

		file_DF.close()
		#------------------------------
	#==========================================================================


	#==========================================================================
	def finalize_write_quietStart(self):
		file_DF = open("QuietStart_Input.json", "a", encoding="utf-8")

		# ... whatever you already write before the final cleanup ...

		# make sure all writes are flushed
		file_DF.flush()

		# use the file path, not the file object
		remove_trailing_comma(file_DF.name, trailing=", ")

		# now close the top-level JSON object/array as you want
		stg = "\n]\n}\n"
		file_DF.write(stg)

		file_DF.close()
	#==========================================================================



	#==========================================================================
	def read_input_file(self):
		dataSoliton 			= []
		with open(os.path.join(SCRIPT_DIR, 'Fluid_Input_SelfConsistent.json')) as data_file:
			data0 				= json.load(data_file)

		inputPattern 			= data0['fluid'][0]['input_pattern']
		dataInput				= data0['fluid'][1]['input']

		pd = {} #pattern dictionary
		for idx in range(len(inputPattern)):
			pd [str(inputPattern[idx])] = idx
		iPh 					= pd["PHI"]
		iXx 					= pd["X"]

		dataUnperturbed 		= dataInput[0]
		dataUnperturbed[iXx]= 0.0 # x = 0.0

		# findind the both ends of soliton
		cnt 			= 0
		while True:
			cnt +=1
			if dataInput[cnt][iPh] != 0.0:
					ixStartSoliton 	= cnt
					break

		cnt = ixStartSoliton
		while True:
			cnt +=1
			if dataInput[cnt][iPh] == 0.0:
					ixEndSoliton 	= cnt
					break


		# writing the soliton information
		for cnt in range(ixStartSoliton, ixEndSoliton):
				dataSoliton.append(dataInput[cnt])
		# starting X and ending X of soliton is written at the end of the data
		limitXsoliton = [dataInput[ixStartSoliton][iXx], dataInput[ixEndSoliton][iXx]]


		dataUnperturbed[pd["U0"]]= dataSoliton[0][pd["U0"]] # find v_soliton for unpertubed data

		return dataUnperturbed, dataSoliton, limitXsoliton,pd
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN fitting Elin DF on top of ni=ne

	def insert_fixedSoliton(self):
		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		a 		= 40.0#20.0
		l 		= 22.5#15.0
		x0		= 300
		machNumber  = 45
		v_soliton 	= machNumber * p.ion_sound_velocity
		v_holeIon 	= 0.0

		file_DF.write( self.write_header("soliton_Data"))
		file_DF.close()
		print(("vSoliton=",v_soliton))
		# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		dataFixed = self.setFixedSoliton(a=a,l=l,x0=x0,vElc=v_soliton,vIon=v_holeIon)
		self.get_ElinDF_elc_ion(dataFixed)# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		self.write_headerEnding(file_DF)
		file_DF.close()
		#pause()


		#----------------------------------------------------------------------

		mid = dataFixed[dataFixed[0]["X"]][-1]
		bgn = dataFixed[dataFixed[0]["X"]][ 0]
		x0 = mid + (mid-bgn) + 100
		a 		= 20.0
		l 		= 15.0
		machNumber  = 30
		v_soliton 	= machNumber * p.ion_sound_velocity
		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		file_DF.write( self.write_header("soliton_Data"))
		file_DF.close()
		dataFixed = self.setFixedSoliton(a=a,l=l,x0=x0,vElc=v_soliton,vIon=v_holeIon)
		self.get_ElinDF_elc_ion(dataFixed) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		self.write_headerEnding(file_DF)
		file_DF.close()
		#----------------------------------------------------------------------

		#----------------------------------------------------------------------
		'''
		mid = dataFixed[dataFixed[0]["X"]][-1]
		bgn = dataFixed[dataFixed[0]["X"]][ 0]
		x0 = mid + (mid-bgn) + 100
		a 		= 20.0
		l 		= 15.0
		machNumber  = 30
		v_soliton 	= machNumber * p.ion_sound_velocity
		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		file_DF.write( self.write_header("soliton_Data"))
		file_DF.close()
		dataFixed = self.setFixedSoliton(a=a,l=l,x0=x0,vElc=v_soliton,vIon=v_holeIon)
		self.get_ElinDF_elc_ion(dataFixed) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

		file_DF = open(os.path.join(SCRIPT_DIR, 'QuietStart_Input.json'), 'a')
		self.write_headerEnding(file_DF)
		file_DF.close()
		#----------------------------------------------------------------------
		'''

		self.finalize_write_quietStart()
	#==========================================================================
	def get_ElinDF_elc_ion(self,dataFixed):
		xvlFixed = dataFixed[dataFixed[0]["X"]]
		endSoliton 	= xvlFixed[-1]
		bgnSoliton 	= xvlFixed[ 0]
		midSoliton  = bgnSoliton + (endSoliton-bgnSoliton)/2.0
		phi0 		= 0.0
		phiArr 		= []

		v_soliton 	= dataFixed[dataFixed[0]["vHoleElc"]]
		vHoleIon 	= dataFixed[dataFixed[0]["vHoleIon"]]

		dfRefIon	= distributionClass(i_species=self.iSpeciesIon, v_soliton = v_soliton)
		dfTrpElc	= distributionClass(i_species=self.iSpeciesElc, v_soliton = v_soliton)
		dfTrpIon	= distributionClass(i_species=self.iSpeciesIon, v_soliton = vHoleIon)


		scalingFactorIons = -0.01 # -1 no scaling, minus sign is to produce ion hole

		nameElc = p.Name[self.iSpeciesElc]
		nameIon = p.Name[self.iSpeciesIon]
		findBetaArray = [FindBeta(dfTrpElc,self.vGqElc), FindBeta(dfTrpIon,self.vGqIon)]
		#for findBeta in findBetaArray:
			#findBeta.ax_beta[1].set_xlim([0,10])
			#findBeta.ax_beta[1].set_ylim([0.9,1.2])
			#findBeta.ax_beta[0].set_xlim([0,10])
			#findBeta.ax_beta[0].set_ylim([-1000,100])
		#self.ax_target.set_ylim([0,0.01])
		#self.ax_target.set_xlim([0, 10 ])#dataFixed[dataFixed[0]["PHI"]][ len(xvlFixed)/2   ]

		diff = []
		idx = 0
		phiMax = np.amax(dataFixed[dataFixed[0]["PHI"]])
		while True:


			phi = dataFixed[dataFixed[0]["PHI"]][idx]
			rho = dataFixed[dataFixed[0]["RHO"]][idx]
			xvl = dataFixed[dataFixed[0]["X"]]	[idx]

			increment = 1
			if idx>49:
				increment = 5
			if phi>phiMax * 0.9:
				increment = 1

			idx += increment
			if idx>(len(xvlFixed)/2)-1:
				break

			#if phi<dataFixed[dataFixed[0]["PHI"]][idx-1]:
				#break

			deltaPhi 		= phi - phi0

			dfRefIon.reset_VxDf_array()
			dfRefIon.reflected_VxDfMoments(phi)
			momentsRefIon 		= dfRefIon.getMomentsTemp()

			if idx%100==0.0 or idx>(len(xvlFixed)/2)-2:
				df2dArr, vx2dArr, ex2dArr, en2dArr  = dfRefIon.return_DfVxExEn_4arrays()
				df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)
				#dfGqArray = m.get_interpolation(vx1dArr, self.vGqArray,df1dArr)
				m.live_plot(self.ax_dfRef, 		vx1dArr,df1dArr, x="V_x",y="distribution function",title="df",s="-")
				#m.live_plot(self.ax_dfRef 	 , self.vGqArray, dfGqArray			, x=r"V_x"	,y="DF"	  ,title="df"	 ,s="-o")


			ni 					= momentsRefIon[0]
			sys.stdout.flush()
			print("\n (", "ix=",idx,"x=",xvl, "dphi=", deltaPhi,"|phi=", phi, end=' ')
			ne 			= - (rho - ni)
			valueTarget = ne
			if deltaPhi>0.0:
				print(" --- electron ", end=' ')
				findBeta = findBetaArray[0]
				#momentsElc,dfGqElc = findBeta.find_fitting_beta(valueTarget,self.iSpeciesElc,idx,deltaPhi,phi,xvl)
				momentsElc,dfGqElc=findBeta.densitySearch(valueTarget,self.iSpeciesElc,idx,deltaPhi,phi,xvl)
				flagNgDF =  self.check_negative_df(dfGqElc,"electron")


				valueTargetIon = momentsElc[0]
				print(" --- ion ", end=' ')
				findBeta = findBetaArray[1]
				#momentsIon,dfGqIon = findBeta.find_fitting_beta(valueTarget,self.iSpeciesIon,idx,scalingFactorIons * deltaPhi,scalingFactorIons * phi,xvl)
				momentsIon,dfGqIon = findBeta.densitySearch(valueTargetIon,self.iSpeciesIon,idx,scalingFactorIons * deltaPhi,scalingFactorIons * phi,xvl)
				#m.pause()
				flagNgDF =  self.check_negative_df(dfGqIon,"ion")

			else:
				print("what the heck has just happened?")
				break

			if idx%100==0.0 or idx>(len(xvlFixed)/2)-2:
			#if findBetaArray[0].phiArr[-1]>9.5:
				m.live_plot(self.ax_target, phiArr       ,diff,c="blue")#,vmin=,vmax=)#ylog='symlog') #xlog='log'
				self.ax_target.set_xlim([np.amin(phiArr),np.amax(phiArr)])
				self.ax_target.set_ylim([np.amin(diff),np.amax(diff)])
				#pause()


			if isinstance(momentsElc, list) and isinstance(momentsIon, list):
				diff.append( momentsIon[0]-momentsElc[0])
				phiArr.append(phi0)
				self.write_Moments_DF_FixedSoliton(momentsElc,momentsIon,dfGqElc,dfGqIon,nameElc,nameIon,idx,dataFixed)
				phi0 = phi 	#if dphi=phi[i]-phi[i+1] does not have any beta associated, then dphi=phi[i]-phi[i+2],
							#basically jumping over phi[i+1]
			else:
				diff.append(0.0)
				phiArr.append(phi)

		m.live_plot(self.ax_target, phiArr       ,diff,c="blue")#,vmin=,vmax=)#ylog='symlog') #xlog='log'
		self.ax_target.set_xlim([np.amin(phiArr),np.amax(phiArr)])
		self.ax_target.set_ylim([np.amin(diff),np.amax(diff)])

	#==========================================================================

	#==========================================================================
	def check_negative_df(self,dfarray,msg):
		flagNgDF = False
		for f in dfarray:
			if f<0.0:
				flagNgDF = True

		if flagNgDF == True:
			print("negative DF for",msg)
		return flagNgDF

	#==========================================================================


	'''
	#==========================================================================
	def get_ElinDF(self):
		print "vSoliton=", self.v_soliton, "grid points of soliton=",len(self.dataSoliton)
		dfTrapped			= distributionClass(i_species=self.iSpeciesElc, v_soliton = self.v_soliton)
		phi0 = 0.0
		betaArr = []
		phiArr  = []
		niArr   = []
		flagGuidedSearch = True # dummy, to print message once
		i_species = self.iSpeciesElc
		#for idx in range(0,len(self.dataSoliton)/2, 10):
		idx = 0
		while True:
			if idx>=len(self.dataSoliton)/2:
				break
			if idx<50:
				idx += 1
			else:
				idx += 50
			phi 			= self.dataSoliton[idx][self.pd["PHI"]]
			Efd 			= self.dataSoliton[idx][self.pd["Ex"]]
			deltaPhi 		= phi - phi0
			ni 				= self.dataSoliton[idx][self.pd["Density_ion"]]
			#FirstMoment 	= self.dataSoliton[idx][self.pd["FirstMoment_ion"]]
			xValue 			= self.dataSoliton[idx][self.pd["X"]]
			valueTarget 	= ni #FirstMoment #

			#if ni-1.0>0.000000001:
			#------------------------------------------------------------------
			niArr.append(valueTarget)

			phiArr.append(phi)
			print "(", "x=",xValue,
			sys.stdout.flush()
			dfTrapped.update_VxMoments_initialize_VxEkTrapp(deltaPhi)

			betaAppro, valueTargetAppro, flagNotFound =  self.find_fitting_beta(dfTrapped,valueTarget,i_species)
			betaArr.append(np.sign(betaAppro)*np.log10(np.abs(betaAppro)))
			if idx>100:
				if flagGuidedSearch == True:
					print ("\n \n *** guided search on beta *** \n \n ")
					flagGuidedSearch = False
				self.betaUpIni = betaAppro+100
				self.betaDnIni = betaAppro-100

			print "beta=",betaAppro#, "appro=",valueTargetAppro, " ~", valueTarget, ", phi=", phi, ",E=",Efd
			transMoments = dfTrapped.produce_TranTrappedDisFun_itsMoments(betaAppro)
			dfTrapped.add_DfVxMoments_Tran_Permanent()

			momentsTrapped = dfTrapped.getMomentsTemp()
			df2dArr, vx2dArr, ex2dArr, en2dArr  = dfTrapped.return_DfVxExEn_4arrays()

			df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)

			dfGqArray = m.get_interpolation(vx1dArr, self.vGqArray,df1dArr)
			self.write_Moments_DF(self.dataSoliton[idx],momentsTrapped,dfGqArray,NameSc)
			#self.print_DF(i_species, df1dArr, vx1dArr, self.ax_df,array="1d")
			#self.print_DF(i_species, dfGqArray, self.vGqArray, self.ax_df,array="1d")
			if idx%100==0:
				m.live_plot(self.ax_df, vx1dArr,df1dArr, x="V_x",y="distribution function",title="df",s="-*")
				m.live_plot(self.ax_df,   	self.vGqArray, dfGqArray, x="V_x",y="DF"	,title="df"	 ,s="-o")
				m.live_plot(self.ax_beta[0], phiArr       , betaArr  , x="phi",y="Beta"	,title="beta",s="-o",ylog='log',xlog='log')
				m.live_plot(self.ax_beta[1], phiArr       , niArr  	, x="phi"  ,y="n_i"				 ,s="-o")


			#self.print_DF(i_species, df2dArr, vx2dArr, self.ax_df)
			#if idx%100==0:
				#pause()
			self.x_df_output.append([xValue,dfGqArray])
			#------------------------------------------------------------------

			phi0 = phi
		self.finalize_write_quietStart()
		m.live_plot(self.ax_df, vx1dArr,df1dArr, x="V_x",y="distribution function",title="df",s="-*")
		m.live_plot(self.ax_df,   	self.vGqArray, dfGqArray, x="V_x",y="DF"	,title="df"	 ,s="-o")
		m.live_plot(self.ax_beta[0], phiArr       , betaArr  , x="phi",y="Beta"	,title="beta",s="-o",ylog='log',xlog='log')
		m.live_plot(self.ax_beta[1], phiArr       , niArr  	, x="phi"  ,y="n_i"				 ,s="-o")
	#==========================================================================
	'''



	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN visualization

	#==========================================================================
	def print_DF(self,i_species,yIn, xIn,axIn,**kwargs):
		flagFastPrint 			= True
		flagColorSperatedPrint 	= False

		arrayType		 	= kwargs.get('array', "2d")
		if arrayType == "1d":
			m.live_plot(axIn[i_species], xIn,yIn, x="V_x",y="distribution function",title="df")

		if arrayType == "2d":
			if flagFastPrint == True:
				y1d, x1d = m.convert_2D_1D(yIn, xIn)
				m.live_plot(axIn[i_species], x1d,y1d, x="V_x",y="distribution function",title="df")#,s="-*"

			if flagColorSperatedPrint == True:
				for iSlice in range(len(yIn)):
					m.live_plot(axIn[i_species], xIn[iSlice],yIn[iSlice], x="V_x",y="distribution function",title="df")
	#==========================================================================


	#==========================================================================
	def print_moments(self,i_species):
			df = self.dfArray[i_species]
			if i_species ==0:
				color = "red"
			if i_species ==1:
				color = "blue"
			df2dArr, vx2dArr, ex2dArr, en2dArr  = df.return_DfVxExEn_4arrays()
			self.print_DF(i_species,df2dArr, vx2dArr,self.ax_df)
			#self.print_DF(i_species,ex2dArr, en2dArr,self.ax_ex)
			#for iSlice in range(len(df2dArr)):
				#DF_array = df2dArr[iSlice]
                #Vx_array = vx2dArr[iSlice]
				#m.live_plot(self.ax_df[i_species], Vx_array,np.log10(DF_array), x="V_x",y="distribution function",title="df")
				#m.live_plot(self.ax_ex[i_species], en2dArr[iSlice],ex2dArr[iSlice], x="V_x",y="distribution function",title="df")

			m.live_plot(self.ax_mom[0], self.phi_array,self.ZeroMoment_out[i_species], c=color,  s='*-', diff=True) #x="V_x",y="ZeroMoment",title="ion"
			m.live_plot(self.ax_mom[1], self.phi_array,self.FirstMoment_out[i_species], c=color,  s='*-', diff=True)
			m.live_plot(self.ax_mom[2], self.phi_array,self.SecondMoment_out[i_species], c=color,  s='*-', diff=True)
			m.live_plot(self.ax_mom[3], self.phi_array,self.Temperature_out[i_species], c=color,  s='*-', diff=True)
			#m.live_plot(self.ax_trg,self.phi_array[-1],Temperature_electron_Target, c='orange',s="H")#,x="phi", y="Temperature",title="guiding value", c='orange',s="H")
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#******************************************************************************


#******************************************************************************
#BEGIN test and usage
QSobject   = QuietStart()
##QSobject.get_ElinDF()
QSobject.insert_fixedSoliton()
pause()
#END
#******************************************************************************
