



#******************************************************************************
#BEGIN ReadMe
#------------------------------------------------------------------------------
# ELIN distribution function 
# by Mehdi Jenab (mehdi.jenab@chalmers.se)
# project started on 3 Feb 2020, 
# goal: produce mutli_Beta distribution function on top of Schamel approach
#------------------------------------------------------------------------------
#END
#******************************************************************************


#******************************************************************************
#BEGIN imports and dependencies
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import random 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
from .global_parameter import global_parameters 
p = global_parameters()

import warnings
warnings.filterwarnings("error")
#------------------------------------------------------------------------------
#END
#******************************************************************************

#******************************************************************************
class ElinDistributionFunction():
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN initialization, set and get arrays
	#==========================================================================
	def __init__(self, *arg, **kwargs): 
		self.iSpecies 	= kwargs.get('i_species', "")
		self.vSoliton 	= kwargs.get('v_soliton', "")

		self.M 			= p.Mass[self.iSpecies]
		self.gasGamma 	= kwargs.get('gasGamma', 3)
		ZeroUnperturb 	= [1.0,1.0]
		FirstUnperturb	= [0.0, 0.0]
		SecondUnperturb = [(2.0/p.Mass[0])*(p.Temp[0]/(p.gasGamma[0]-1.0)), (2.0/p.Mass[1])*(p.Temp[1]/(p.gasGamma[1]-1.0))]
		self.df2dArr	= []
		self.vx2dArr	= []
		self.Ex2dArr 	= [] # Exponent
		self.En2dArr 	= [] # energy
		self.moments	= [ZeroUnperturb[self.iSpecies],FirstUnperturb[self.iSpecies],SecondUnperturb[self.iSpecies],p.Temp[self.iSpecies]] # [ZeroMoment, 1stMoment, 2ndMoment, Temperature]

		self.dfTranArr	= []
		self.vxTranArr	= []
		self.momentsTran= [0.0,0.0,0.0,0.0] # [ZeroMoment, 1stMoment, 2ndMoment, Temperature]
		self.EkTranArr  = []

		self.dVxScalinng= 10.0
		self.dVx 		= p.dVx		[self.iSpecies]
		self.dVxTrapped = self.dVx/self.dVxScalinng
		self.beta_before = 0.0

		self.produce_initial_DF()
	#==========================================================================

	#==========================================================================
	def set_Ex2dArr(self):
		self.Ex2dArr = []
		for dfSlice in self.df2dArr:
			ExArr= []
			for df in dfSlice:
				ExArr.append(np.log(df))
			self.Ex2dArr.append(ExArr)
			del ExArr
	#==========================================================================

	#==========================================================================
	def set_En2dArr(self):
		self.En2dArr = []
		for vxSlice in self.vx2dArr:
			EnArr = self.core_vx_to_energy(vxSlice)
			self.En2dArr.append(EnArr)
	#==========================================================================

	#==========================================================================
	def return_DfVxExEn_4arrays(self):
		self.set_Ex2dArr()
		self.set_En2dArr()
		#print len(self.df2dArr), len(self.vx2dArr), len(self.Ex2dArr), len(self.En2dArr)
		return self.df2dArr, self.vx2dArr, self.Ex2dArr, self.En2dArr
	#==========================================================================

	#==========================================================================
	def getMomentsTemp(self):
		self.core_getMoments(self.vx2dArr,self.df2dArr,self.moments)
		self.moments[3] = self.getTemperature(self.moments)
		return self.moments
	#==========================================================================
	
	#==========================================================================
	def getMomentsTrap(self):
		momentsTrap = [0,0,0,-1]
		self.core_getMoments(self.vxTranArr,self.dfTranArr,momentsTrap)
		return momentsTrap
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN core functions
	#==========================================================================
	def core_shift_velocity(self,vx0,phi,vxArr):
		M = p.Mass 		[self.iSpecies]
		Q = p.Charge	[self.iSpecies]

		def Energy_kinetic(vx0):
			return 0.5* M * vx0**2


		engPotential 					= Q * phi

		vx_SolFrame 					= vx0 - self.vSoliton
		engKinetic_SolFrame 			= Energy_kinetic(vx_SolFrame)

		engKinetic_shiftedSF 			= engKinetic_SolFrame  - engPotential #  UnShifted ---> Shifted (in SolitonFrame)

		vPowered2 						= 2.0 * engKinetic_shiftedSF / M

		if vPowered2<0: # this should not happen
			print(("<<<  ??? Error ??? core_shift_velocity >>>", "v^2 < 0",vx0))
			#vPowered2 = -vPowered2
			programPause = input("Press to continue...")
		
		if vx_SolFrame == 0: # exception case: vx0 = vSoliton ==> vx_SolFrame=0 ==> np.sign(vx_SolFrame)=0.0 
			vx_SolFrame_before 				=  vxArr[int(len(vxArr)/2)] - self.vSoliton
			vx_shiftedSF 					=  np.sign(vx_SolFrame_before) * np.sqrt(vPowered2)
		else:
			vx_shiftedSF 					= np.sign(vx_SolFrame) * np.sqrt(vPowered2)
		vx_shifted 							= vx_shiftedSF  + self.vSoliton
		return vx_shifted
	#==========================================================================


	#==========================================================================
	def core_vx_to_energy(self,vxArr):
		enArr 	= []
		for vx in vxArr:
			vx = vx -self.vSoliton
			en = 0.5 * self.M * vx**2
			if vx<0:
				enArr.append(-en)
			else:
				enArr.append(en)
		return enArr
	#==========================================================================


	#==========================================================================
	def core_VxArrayProduction_VsolInRange(self):
		vMin 	= p.Vx_min	[self.iSpecies]
		vMax 	= p.Vx_max	[self.iSpecies]
		v1 = self.vSoliton - (self.dVxScalinng * self.dVx)
		v2 = self.vSoliton + (self.dVxScalinng * self.dVx)
		
		vxArr, enArr = self.core_FillVxArray(vMin,v1,self.dVx)
		self.vx2dArr.append(vxArr)
		self.En2dArr.append(enArr)


		vxArr, enArr = self.core_FillVxArray(v2,vMax,self.dVx)
		self.vx2dArr.append(vxArr)
		self.En2dArr.append(enArr)

		vxArr, enArr = self.core_FillVxArray(v1,self.vSoliton,self.dVxTrapped)
		self.vx2dArr.append(vxArr)
		self.En2dArr.append(enArr)

		vxArr, enArr = self.core_FillVxArray(self.vSoliton,v2,self.dVxTrapped)
		self.vx2dArr.append(vxArr)
		self.En2dArr.append(enArr)
	#==========================================================================




	#==========================================================================
	def core_FillVxArray(self,vMin,vMax,dVx):
		vx0 	= vMin
		vxArr 	= []
		
		
		while True:
			if vx0>vMax:
				break
			vxArr.append(vx0)
			vx0 += dVx
		vxArr.append(vMax)

		enArr 	= self.core_vx_to_energy(vxArr)

		return vxArr, enArr
	#==========================================================================

	#==========================================================================
	def core_vLabFrame_to_oppositeValueLabFrame(self,vLabFrame,phi,vxArr,**kwargs):
		# takes in v in lab frame (either shifted or unshifted)
		# through soliton frame, 
		# produce value v in lab frame for opposite ( unshifted or shifted)
		
		# 3 cases: 
		#		1) vLfShf --> vLfUsh 			for reflected population (like ions , phi>0), 								out=v,  coeff = +1
		#		2) vLfShf --> engKinetic_SfUsh 	for inside the hole population of trapped species ( like electrons, phi>0), out=en, coeff = +1
		#		3) vLfUns --> vLfShf 			for free population of trapped species ( like electrons, phi>0), 			out=v,  coeff = -1
		# * for the 2 case, engKinetic_SfUsh is negative 
		
		
		
		out 		 	= kwargs.get('out', "v") # default, returns vLfUsh
		coeff 		 	= kwargs.get('coeff', 1) # default, goes from Shifted ---> UnShifted (in SolitonFrame)
		# coeff = -1 goes from UnShifted ---> Shifted (in SolitonFrame)
		M = p.Mass 		[self.iSpecies]
		Q = p.Charge	[self.iSpecies]


		def print_error_inDetail():
			print(("<<<  ??? Error ??? engKin_SolFrame < 0  ||| ",vSolFrame,out,"|||"), end=' ')
			if Q<0: 
				print(("electrons, "), end=' ')
			else:
				print(("ions, "), end=' ')

			if Q*phi<0: 
				print(("trapped,"), end=' ')
			else:
				print(("reflected,"), end=' ')

			if coeff<0:
				print(("free population of trapped"), end=' ')
			else:
				if Q*phi<0:
					print(("inside hole of trapped"), end=' ')
				else:
					print(("reflected population"), end=' ')
			programPause = input("...")

		def Energy_kinetic(v):
			return 0.5* M * v**2


		engPotential 					= Q * phi

		vSolFrame         				= vLabFrame - self.vSoliton
		engKin_SolFrame 				= Energy_kinetic(vSolFrame)


		# ---- coeff = -1, Uns ---> Shf , coeff = +1, Shf ---> Uns
		engKin_SolFrame 				= engKin_SolFrame  + (coeff *  engPotential) #  
		# -------------------------------------------------------------------


					
		if out =="en":
			outPut = engKin_SolFrame
		else:
			v2_SolFrame 						= 2.0 * engKin_SolFrame / M

			if v2_SolFrame<0: # this should not happen
				print_error_inDetail()
				#print ("<<<  ??? Error ??? shifting velocity>>> ", "v^2 < 0",vLabFrame)
				#programPause = raw_input("Press to continue...")
				#v2_SolFrame = -v2_SolFrame
			
			if vSolFrame == 0: # exception case: vLfShf = vSoliton ==> vSfShf=0 ==> np.sign(vSfShf)=0.0 
				vx_SolFrame_before 			=  vxArr[int(len(vxArr)/2)] - self.vSoliton
				vSolFrameOut 				=  np.sign(vx_SolFrame_before) * np.sqrt(v2_SolFrame)
			else:
				vSolFrameOut 				= np.sign(vSolFrame) * np.sqrt(v2_SolFrame)
			vLabFrameOut 					= vSolFrameOut  + self.vSoliton
			#if out =="v":
			outPut = vLabFrameOut

		return outPut
	#==========================================================================



	#==========================================================================
	def core_getMoments(self,vxInput2d, dfInput2d, moments):
		moments[0] = self.getMoment(0, vxInput2d, dfInput2d)
		moments[1] = self.getMoment(1, vxInput2d, dfInput2d)
		moments[2] = self.getMoment(2, vxInput2d, dfInput2d)
	#==========================================================================

	#==========================================================================
	def getMoment(self,order, vxInput2d, dfInput2d):
		integral = 0.0
		for iSlice in range(len(vxInput2d)):
			integrand = []
			for iVx in range(len(vxInput2d[iSlice])):
				integrand.append(dfInput2d[iSlice][iVx] * vxInput2d[iSlice][iVx]**order)
			integral += np.trapz(integrand,x=vxInput2d[iSlice])
			#integral = integrate.simps(integrand,x=vxInput2d[iSlice])
			#integral = integrate.romb(integrand)
			del integrand
		return integral
	#==========================================================================

	#==========================================================================
	def getTemperature(self,moments):
		density 	= moments[0]
		J 			= moments[1]
		moment2nd 	= moments[2]
		Mass      	= p.Mass[self.iSpecies]
		#density   	= self.getMoment(0)#n
		#rho_gkyl  	= Mass*density
		#xmom_gkyl 	= Mass*self.getFirstMomentum()
		#J 			= self.getMoment(1)#nu
		#U_gkyl    	= self.getAveragedVelocity()
		
		E_gkyl    	= 0.5*Mass* moment2nd# nmu2 

		Temp		=  ((self.gasGamma-1)/density) * (  E_gkyl - (0.5 * Mass * J**2 / (density) ) )
		#Temp		=  ((self.gasGamma-1)/density) * (  E_gkyl - (xmom_gkyl**2 / (2.0*rho_gkyl) ) )
		#Temp		=  ((self.gasGamma-1)/density) * (  E_gkyl - (0.5 * rho_gkyl  *U_gkyl**2) ) 
		return Temp
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN functions
	#==========================================================================
	def produce_initial_DF(self):
		A 		= (p.Den_Ratio_b[self.iSpecies]) * np.sqrt(1.0/(2.0*np.pi)) * np.sqrt(p.norm_factor[self.iSpecies])

		Q 		= p.Charge	[self.iSpecies]
		M 		= p.Mass	[self.iSpecies]
		T 		= p.Temp	[self.iSpecies]
		vMin 	= p.Vx_min	[self.iSpecies]
		vMax 	= p.Vx_max	[self.iSpecies]
		
		if self.vSoliton<vMax and self.vSoliton>vMin:
			self.core_VxArrayProduction_VsolInRange()
		else:
			vxArr, enArr = self.core_FillVxArray(vMin,vMax,self.dVx)
			self.vx2dArr.append(vxArr)
			self.En2dArr.append(enArr)
		


		for iSlice in range(len(self.vx2dArr)):
			dfArr 	= []
			for iVx in range(len(self.vx2dArr[iSlice])):
				vx0 		= self.vx2dArr[iSlice][iVx]
				E 	= 0.5 * M *(vx0**2) # engergy kinetic
				df0 = A * (np.exp(-E/T)) # Maxwellian distribution function
				dfArr.append(df0)

			self.df2dArr.append(dfArr)
			del dfArr
	#==========================================================================


	#==========================================================================
	def update_VxMoments_initialize_VxEkTrapp(self,phi):
		self.update_Vx2dArray(phi)
		self.getMomentsTemp()

		#initialize_trapped_Vx_Ek
		self.EkTranArr = []
		self.vxTranArr = []
		self.produce_Trapped_VxArray(phi)
		self.produce_TranEkArr(phi)
	#==========================================================================


	#==========================================================================
	def update_Vx2dArray(self,phi):
		M = p.Mass 		[self.iSpecies]
		for iSlice in range(len(self.vx2dArr)):
			for iVx in range(len(self.vx2dArr[iSlice])):
				vLfUns 		= self.vx2dArr[iSlice][iVx] # velocity in Lab Frame and unshifted
				vLfShf  	= self.core_vLabFrame_to_oppositeValueLabFrame(vLfUns,phi,self.vx2dArr[iSlice],coeff=-1)   # velocity in Lab Frame and shifted
				self.vx2dArr[iSlice][iVx] = vLfShf
			enArr = self.core_vx_to_energy(self.vx2dArr[iSlice])
			for iVx in range(len(self.vx2dArr[iSlice])):
					self.En2dArr[iSlice][iVx] = enArr[iVx]
	#==========================================================================

	#==========================================================================
	def produce_TranEkArr(self,phi): # produce new population of trapped particles (inside the hole)
		for iSlice in range(len(self.vxTranArr)):
			EkArr = []
			for iVx in range(len(self.vxTranArr[iSlice])):
				vLfShf 	    = self.vxTranArr[iSlice][iVx] # velocity in Lab Frame and Shifted
				EkSfUsh 	= self.core_vLabFrame_to_oppositeValueLabFrame(vLfShf,phi,self.vxTranArr[iSlice],out="en")
				EkArr.append(EkSfUsh) # energy kinetic in Soliton Frame and UnShifted
			self.EkTranArr.append(EkArr)
			del EkArr
	#==========================================================================

	#==========================================================================
	def produce_Trapped_VxArray(self,phi):
		vMin 	= self.vx2dArr[-2][-1] # this needs vx array to be added on top of each other accordingly
		vMax 	= self.vx2dArr[-1][0]

		dVxAux = np.abs(vMax-vMin)/10.0#self.dVxScalinng
		if dVxAux>self.dVxTrapped:
			dVxAux = self.dVxTrapped

		vxArr, enArr 	= self.core_FillVxArray(vMin,self.vSoliton,dVxAux)
		self.vxTranArr.append(vxArr)

		vxArr, enArr 	= self.core_FillVxArray(self.vSoliton,vMax, dVxAux)
		self.vxTranArr.append(vxArr)
	#==========================================================================

	#==========================================================================
	def produce_TranTrappedZero(self):
		self.dfTranArr = []
		for iSlice in range(len(self.vxTranArr)):
			dfArr = []
			for iVx in range(len(self.vxTranArr[iSlice])):
				dfArr.append			( 0.0) # to set up DF inside hole equal to zero
			self.dfTranArr.append(dfArr)
			del dfArr
	#==========================================================================


		
	#==========================================================================
	def produce_TranTrappedZero_itsMoments(self): 
		#Note that this is NOT equal to 
		#	1. update_VxMoments_initialize_VxEkTrapp and 
		#	2. getMomentsTemp
		# if you need a DF=0 in the hole area use this method and not combination of 1 and 2.
		self.produce_TranTrappedZero()
		self.momentsTran = [0,0,0,0]
		self.moments[3]	=	-1.0 # never use this value since it is wrong temperature
		self.core_getMoments(self.vxTranArr,self.dfTranArr,self.momentsTran)
		for iMoment in range(len(self.momentsTran)-1):
			self.momentsTran[iMoment] += self.moments[iMoment]
		self.momentsTran[3] = self.getTemperature(self.momentsTran) # Temp is not equal to TempFree + TempTrapped, hence transient Temp needs to be calculated here.
		return self.momentsTran
	#==========================================================================

	#==========================================================================
	def produce_TranTrappedDisFun(self,beta):
		T = p.Temp[self.iSpecies]
		
		dfBase = self.df2dArr[-2][-1]
		self.dfTranArr = []
		for iSlice in range(len(self.vxTranArr)):
			dfArr = []
			for iVx in range(len(self.vxTranArr[iSlice])):
				engKinetic_shiftedSF 	= self.EkTranArr[iSlice][iVx]
				Energy 					= beta * engKinetic_shiftedSF
				try:
					dfShape 				= np.exp(-Energy/T) 
				except RuntimeWarning:
					print(" Runtime Error, Energy=",Energy,"beta=",beta)
					dfShape 				= np.exp(-Energy/T) 
				dfArr.append			( dfBase * dfShape)
				#dfBase = dfArr[-1] # each point in velocity direction uses the previous DF as df_base
			self.dfTranArr.append(dfArr)
			del dfArr
	#==========================================================================

	#==========================================================================
	def produce_TranTrappedDisFun_itsMoments(self,beta): # only place where beta has affects
		self.produce_TranTrappedDisFun(beta)

		'''

		dfBase = self.df2dArr[-2][-1] #self.df2dArr[-2][-1]
		self.dfTranArr = []
		iSlice = 0
		dfArr = []

		if self.beta_before==0:
			deltaBeta = 0.0
			self.beta_before =  beta
		else:
			deltaBeta = (beta-self.beta_before)/len(self.vxTranArr[iSlice])
		beta0 = beta #self.beta_before
		for iVx in range(len(self.vxTranArr[iSlice])):
			vx0 					= self.vxTranArr[iSlice][iVx]
			engKinetic_shiftedSF 	= self.EkTranArr[iSlice][iVx]
			Energy 					= beta0 * engKinetic_shiftedSF
			dfShape 				= np.exp(-Energy/T) 
			dfArr.append			( dfBase* dfShape)
			#beta0 					+= deltaBeta
			#dfBase = dfArr[-1] # each point in velocity direction uses the previous DF as df_base
		self.dfTranArr.append(dfArr)
		dfArrRverse = []
		for iVx in range(len(dfArr)-1,-1,-1):
			dfArrRverse.append(dfArr[iVx])
		self.dfTranArr.append(dfArrRverse)
		del dfArr
		del dfArrRverse

		#iSlice = 1
		#dfBase = self.df2dArr[-2][-1]
		#dfArr = []
		#for iVx in range(len(self.vxTranArr[iSlice])-1,-1,-1):
			#vx0 = self.vxTranArr[iSlice][iVx]
			#engKinetic_shiftedSF 	= self.EkTranArr[iSlice][iVx]
			#Energy 					= beta * engKinetic_shiftedSF
			#dfShape 				= np.exp(-Energy/T) 
			#dfArr.append			( dfBase * dfShape)
			##dfBase = dfArr[-1] # each point in velocity direction uses the previous DF as df_base
		#reversed_dfArr = dfArr[::-1]
		#self.dfTranArr.append(reversed_dfArr)
		#del dfArr
		'''

		#----------- to produce moments ---------------------------------------
		self.momentsTran = [0,0,0,0]
		self.moments[3]	=	-1.0 # never use this value since it is wrong temperature
		self.core_getMoments(self.vxTranArr,self.dfTranArr,self.momentsTran)
		for iMoment in range(len(self.momentsTran)-1):
			self.momentsTran[iMoment] += self.moments[iMoment]
		self.momentsTran[3] = self.getTemperature(self.momentsTran) # Temp is not equal to TempFree + TempTrapped, hence transient Temp needs to be calculated here.
		#print beta,"--",self.momentsTran, "--",self.moments
		self.beta_before = beta
		
		return self.momentsTran
	#==========================================================================


	#==========================================================================
	def add_DfVxMoments_Tran_Permanent(self):
		M = p.Mass 		[self.iSpecies]
		for iSlice in range(len(self.vxTranArr)):
			self.vx2dArr.append(self.vxTranArr[iSlice])
			self.df2dArr.append(self.dfTranArr[iSlice])

			exArr = []
			for iElement in range(len(self.vxTranArr[iSlice])):	
				df = self.dfTranArr[iSlice][iElement]
				exArr.append(np.log(df))
			enArr = self.core_vx_to_energy(self.vxTranArr[iSlice])
			self.Ex2dArr.append(exArr)
			self.En2dArr.append(enArr)
			del exArr

		#for iMoment in range(len(self.moments)):
			#self.moments[iMoment] = self.momentsTran[iMoment]
	#==========================================================================


	#~~~~~~~~~~~~~~~~~~ reflected population ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#==========================================================================
	def reset_VxDf_array(self):
		del self.vx2dArr
		self.vx2dArr = []
		del self.df2dArr
		self.df2dArr = []
		del self.Ex2dArr 
		self.Ex2dArr = []
		del self.En2dArr 
		self.En2dArr = []
		#for iSlice in range(len(self.vx2dArr)):
			#self.vx2dArr[iSlice] = []
			#self.df2dArr[iSlice] = []
	#==========================================================================

	#==========================================================================
	def reflected_VxDfMoments(self,phi):
		# vx_SolFrame --> vx_LabFrame --> df_SolFrame
		A 		= (p.Den_Ratio_b[self.iSpecies]) * np.sqrt(1.0/(2.0*np.pi)) * np.sqrt(p.norm_factor[self.iSpecies])

		Q 		= p.Charge	[self.iSpecies]
		M 		= p.Mass	[self.iSpecies]
		T 		= p.Temp	[self.iSpecies]
		vMin 	= p.Vx_min	[self.iSpecies]
		vMax 	= p.Vx_max	[self.iSpecies]


		if self.vSoliton<vMax and self.vSoliton>vMin:
			self.core_VxArrayProduction_VsolInRange()
		else:
			vxArr, enArr = self.core_FillVxArray(vMin,vMax,self.dVx)
			self.vx2dArr.append(vxArr)
			self.En2dArr.append(enArr)


		for iSlice in range(len(self.vx2dArr)):
			dfArr 	= []
			exArr 	= []
			for iVx in range(len(self.vx2dArr[iSlice])):
				#vLfUsh 		= self.vx2dArr[iSlice][iVx]
				#vLfShf 		= self.core_vLabFrame_to_oppositeValueLabFrame(vLfUsh,phi,self.vx2dArr[iSlice],coeff = -1)
				#self.vx2dArr[iSlice][iVx] = vLfShf
                
				vLfShf		= self.vx2dArr[iSlice][iVx] # velocity in Lab Frame and shifted
				vLfUsh 		= self.core_vLabFrame_to_oppositeValueLabFrame(vLfShf,phi,self.vx2dArr[iSlice],coeff = +1) ## velocity in Lab Frame and unshifted
				E 			= 0.5 * M *(vLfUsh**2) # engergy kinetic
				exponent 	= -E/T
				df0 		= A * (np.exp(exponent)) # Maxwellian distribution function
				#self.vx2dArr[iSlice][iVx] = vLfUsh
				dfArr.append(df0)
				exArr.append(exponent)

			self.df2dArr.append(dfArr)
			self.Ex2dArr.append(exArr)
			del dfArr
			del exArr
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN visualisation
	#==========================================================================
	def print_DF(self,*args,**kwargs):
		ax_in			= args[0]
		X_array			= args[1]
		Y_array			= args[2]
		X_title		 	= kwargs.get('x', "")
		Y_title		 	= kwargs.get('y', "")
		plot_title	 	= kwargs.get('title', "")
		color		 	= kwargs.get('c', "")
		style		 	= kwargs.get('s', '-')
		xlimit		 	= kwargs.get('xlimit', "")
		ylimit		 	= kwargs.get('ylimit', "")
		log			 	= kwargs.get('log', False)
		colors = iter(p.color_line)
		for iSlice in range(len(X_array)):
			if not color:
				r0 = random.randint(0, len(p.color_line)-1)
				c = p.color_line[r0]
				#icolor = iSlice%len(p.color_line)
				#c = p.color_line[icolor]
			else:
				c = color
			if log == True:
				Y = np.log10(Y_array[iSlice] )
			else:
				Y = Y_array[iSlice]
			ax_in.plot(X_array[iSlice],Y,style,c=c )	#, linewidth =linewidth.next()
		ax_in.grid(True)
		plt.pause(0.01)
	#==========================================================================

	#==========================================================================
	def print_DFTranTrapp(self,ax):
		for iSlice in range(len(self.vxTranArr)):
			ax.plot(self.vxTranArr[iSlice],(self.dfTranArr[iSlice]),marker = "o", linestyle='-' )#, linewidth =linewidth.next()
		ax.grid(True)
		plt.pause(0.01)
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#******************************************************************************


#******************************************************************************
#BEGIN Test case
#==============================================================================
def test_case_Schamel():
    # here piecewise df is compared to Schamel df (mulitple pieces should fit one piece when beta is constant)
	v_soliton 						= 522.0#1000.0#21.42
	i_species_input 				= 0

	fig_DfVx, ax_DfVx 				= plt.subplots(1) 
	fig_ExEn, ax_ExEn 				= plt.subplots(1) # shows exponent versus energy (in SolitonFrame)
	flag_reflected 					= False

	# in case of ions
	if i_species_input == 1:
		flag_reflected 				= True

	# 1) create initial df (Maxwellian), phi = 0.0
	df = ElinDistributionFunction(i_species=i_species_input, v_soliton = v_soliton)
	#df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr)
	df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr,s=".",c="blue")#,log=True)
	df.set_Ex2dArr()
	df.print_DF(ax_ExEn,df.En2dArr,df.Ex2dArr,s=".",c="blue",drawProperEnergy=True)
	
	# 2) create Schamel DF (one piece)
	beta0 							= -1.0
	phi0							= 10.0
	if flag_reflected==False:
		df.update_VxMoments_initialize_VxEkTrapp(phi0)
		df.produce_TranTrappedDisFun_itsMoments(beta0)
		df.add_DfVxMoments_Tran_Permanent() 
		#df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr,s="-",c="black")
		#df.print_DF(ax_ExEn,df.En2dArr,df.Ex2dArr,s="-",c="black")
		#plt.pause(5.0)


		del df

		# 3) produce multiple-piece DF using same beta
		df = ElinDistributionFunction(i_species=i_species_input, v_soliton = v_soliton)
		df.set_Ex2dArr()

	fig_beta, ax_beta 				= plt.subplots(1)
	nPhi 							= 40
	deltaPhi 						= phi0 / nPhi
	phi0							= deltaPhi
	phiArr 							= []
	betaArray 						= []#[-10,10,-5,5,0]
	color 							= ["black","brown","red","orange","green"]
	plt.pause(1.0)
	for iphi in range(nPhi):
		if flag_reflected==False:
			phiArr.append(phi0)
			beta0 					=-10.0/phi0#betaArray[iphi]#beta0 # 0.95*beta0 # 
			betaArray.append(beta0)
			print((phi0,beta0, ))
			df.update_VxMoments_initialize_VxEkTrapp(deltaPhi)
			df.produce_TranTrappedDisFun_itsMoments(beta0)
			#for iB in range(1):
				#df.produce_TranTrappedDisFun_itsMoments(beta0*iB) 
				#df.print_DFTranTrapp(ax_DfVx)
			df.add_DfVxMoments_Tran_Permanent() 
			#df.getMomentsTemp()
			#print (df.moments[3])
			#df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr,c=color[iphi])
			#df.print_DF(ax_ExEn,df.En2dArr,df.Ex2dArr)
			#programPause = raw_input("Press to continue...")
			phi0 					+= deltaPhi


		else:
			phi0 += deltaPhi
			print(("phi = ",phi0))
			df.reset_VxDf_array()
			df.reflected_VxDfMoments(phi0)
			df.getMomentsTemp()
			df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr,s="o")#,log=True)
			#print len(df.En2dArr), len(df.Ex2dArr)
			df.print_DF(ax_ExEn,df.En2dArr,df.Ex2dArr)

	ax_beta.plot(phiArr,betaArray)	
	df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr)
	df.print_DF(ax_DfVx,df.vx2dArr,df.df2dArr,c="green")
	plt.show()
#==============================================================================

#test_case_Schamel()
#END
#******************************************************************************
