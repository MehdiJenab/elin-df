

#******************************************************************************
#BEGIN ReadMe 
	# Mehdi.Jenab@Chalmers.se 
	# class handels the fitting Beta for different guiding equation,
	# 13 May 2020: density search (guiding equation) is tested and proved.
#END
#******************************************************************************

#******************************************************************************
#BEGIN imports and dependencies
import 	sys
import 	numpy 				as np
import 	matplotlib.pyplot 	as plt
from 	.Elin_Distribution_Function 	import ElinDistributionFunction as distributionClass
from 	.MathFunctions 				import mathClass 				as math
from 	.global_parameter 			import global_parameters 
from 	scipy.interpolate 			import interp1d

p = global_parameters()
m  = math()
#END
#******************************************************************************


#******************************************************************************
#BEGIN CLASS find Beta
class FindBeta(object):
	#==========================================================================
	def __init__(self,df,vGqArray):
		self.iMomentTarget		= 0 # density, ne=ni
		self.betaDnIniTotal 	= -1e10
		self.betaUpIniTotal 	=  1e10
		self.betaDnIni			= -1e3
		self.betaUpIni 			=  1e5
		self.df 				= df
		self.target 			= 0.0
		self.vGqArray 			= vGqArray

		self.valuePercentageEpsilon 	= 0.1/100#1e-8
		self.betaEpsilon 				= 0.1#1e-6
		self.lastBeta 					= 0.0
		
		self.flagGuidedSearch 			= True # dummy, to print message once

		self.flag_print = True
		if (self.flag_print==True):
			self.fig_df,   self.ax_df 	= plt.subplots(1, figsize=[8,4]) #plt.subplots() sharex=True,
			self.fig_beta, self.ax_beta	= plt.subplots(2,figsize=[8,4]) #plt.subplots() ,,sharex=True
			self.ax_beta[0].ticklabel_format(useOffset=False)
			self.ax_beta[1].ticklabel_format(useOffset=False)
		
		self.betaArr 		= []
		self.phiArr  		= []
		self.valueAppArr   	= []
		self.xvlArr 		= []
		self.targetArr 		= []
	#==========================================================================

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN density search tools
	#==========================================================================
	def densitySearch(self,denTarget,iS,idx,dPhi,phi,xvl):
		flagAdd  = True
		flagShow = False
		iDen = 0
		deltaDenTarget, flagNegative = self.initialization_search	      (dPhi,iDen,denTarget)
		
		#print "deltaDenTarget=",deltaDenTarget
		if flagNegative == True:
			#flagShow = True
			beta_before = self.betaArr[-1]
			print("using beta_before=",beta_before, end=' ')
			betaAppr = beta_before
			valueAppr, flagNaN = self.getDesity(betaAppr) #<<<<<<<<<<<
		else:
			dDenArr, betaArr  	= self.initial_swep_beta_domain_simple(deltaDenTarget,iDen) #<<<<<<<<<<<
			#print "deltaDenTarget=",deltaDenTarget, "dDenArr=",dDenArr, "betaArr=",betaArr,
			if dDenArr[0]==0.0 and dDenArr[-1]==0.0:
				flagAdd = False
				print("small phi warning")
				momentsTrapped 	= 0
				dfGqArray 		= 0
				#flagShow = True
			else:
				betaAppr, valueAppr 	= self.closing_on_dDen_value(betaArr,dDenArr,deltaDenTarget) #<<<<<<<<<<<



		if flagAdd == True:
			diff = valueAppr-deltaDenTarget
			print("B=",betaAppr, end=' ')# " Target=",deltaDenTarget, "diff=", diff

			transMoments = self.df.produce_TranTrappedDisFun_itsMoments(betaAppr)
			self.df.add_DfVxMoments_Tran_Permanent()
			momentsTrapped = self.df.getMomentsTemp()

			dfGqArray,df1dArr, vx1dArr = self.get_dfGqArray()

			if idx%100==0 or idx>490:
				flagShow = True


			if flagShow == True:
				self.print_search_engine_density(idx,phi,dfGqArray,df1dArr, vx1dArr)
				#m.pause()
			
			self.betaArr.append(betaAppr)#np.sign(betaAppr)*np.log10(np.abs(betaAppr))
			self.valueAppArr.append(valueAppr)

			self.phiArr.append(phi)
			self.xvlArr.append(xvl)
			self.targetArr.append(deltaDenTarget)#deltaDenTarget

		sys.stdout.flush()
		return momentsTrapped,dfGqArray
	#==========================================================================

	#==========================================================================
	def initialization_search(self,dPhi,iDen,denTarget):
		flagNegative = False
		self.df.update_VxMoments_initialize_VxEkTrapp(dPhi)
		moments = self.df.getMomentsTemp()
		
		deltaDenTarget = denTarget - moments[iDen]
		#self.df.produce_TranTrappedDisFun(0.0)
		#deltaMoment = self.df.getMomentsTrap()
		#self.df.print_DFTranTrapp(self.ax_df)
		if deltaDenTarget<0.0:
			print("\n \n target is negative, out of reach denTarget=", denTarget, ", Den=",moments[iDen], " dPhi=",dPhi, end=' ')
			flagNegative = True
		return deltaDenTarget, flagNegative
	#==========================================================================

	#==========================================================================
	def initial_swep_beta_domain_simple (self,deltaDenTarget,iDen):
		dDenArr = []
		betaArr = []
		power 	= -13
		beta 	= -10**(-power) # note python screws up for integer value of -10e19, hence Beta has to be always larger than this number.

		while True:
			Value_output, flagNaN = self.getDesity(beta)
			if flagNaN == False:
				betaArr.append(beta)
				dDenArr.append(Value_output)
			else: 
				print("Nan in swept for", beta)
			if len(dDenArr)>2 and dDenArr[-1]>deltaDenTarget:
				break
			if beta>10e9:
				break
			power += 1
			if power<0:
				beta   = -10**(-power)
			else:
				beta   =  10**(power)
			#print beta,",",deltaMoment[iDen], "|",
		valueSmall = dDenArr[0]
		betaSmall  = betaArr[0]
		for ival in range(len(dDenArr)):
			if dDenArr[ival]<deltaDenTarget:
				if dDenArr[ival]>valueSmall:
					valueSmall = dDenArr[ival]
					betaSmall  = betaArr[ival]
		dDenArrOut = [valueSmall,deltaDenTarget,dDenArr[-1]]
		betaArrOut = [betaSmall,betaArr[-1]]
		return dDenArrOut, betaArrOut
	#==========================================================================

	#==========================================================================
	def closing_on_dDen_value(self,betaDomain,valueDomain,deltaDenTarget):
		i_iteration = 0
		while True: # 2nd step: closing on the target value
			i_iteration += 1

			valueAppr, betaAppr ,flagNaN = self.getTargetDen(betaDomain)

			if flagNaN == False:
				if valueAppr >valueDomain[0] and valueAppr< valueDomain[1]:
					betaDomain[0] = betaAppr
				if valueAppr >valueDomain[1] and valueAppr< valueDomain[2]:
					betaDomain[1] = betaAppr
				#print "--- betas = ", betaAppr, "in",betaDomain,  "valueAppr =" , valueAppr, ", deltaDenTarget=",deltaDenTarget, " approximation=", np.abs((deltaDenTarget-valueAppr)/deltaDenTarget), ", beta approximation=", np.abs(betaDomain[1] - betaDomain[0])
				#print_all_DF()
				if  np.abs((deltaDenTarget-valueAppr)/deltaDenTarget)<self.valuePercentageEpsilon and np.abs(betaDomain[1] - betaDomain[0])<self.betaEpsilon:
					break
			else:
				print("Nan in closing for", beta)
			if i_iteration>100000: # in most cases it converges in 10 steps
				if np.abs(betaDomain[1] - betaDomain[0])<1.0:
					if np.abs(betaDomain[1] - betaDomain[0]) != 0.0:
						print(("<Weak warning>, log dB=",int(np.log10(np.abs(betaDomain[1] - betaDomain[0])) ) ), end=' ')
					else:
						print(("delta beta= 0",betaDomain[1] - betaDomain[0]), end=' ')
				else:
					print(("  <<< ??? ERROR ??? not converged, beta = (",betaAppr,")", "iterations=", i_iteration, "value approximation=",np.abs((deltaDenTarget-valueAppr)/deltaDenTarget)))
				break
		#----------------------------------------------------------------------
		#print "[itr=", i_iteration,"]",
		return betaAppr, valueAppr
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN visualization tools
	#==========================================================================
	def get_dfGqArray(self):
		df2dArr, vx2dArr, ex2dArr, en2dArr  = self.df.return_DfVxExEn_4arrays()
		df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)
		dfGqArray = m.get_interpolation(vx1dArr, self.vGqArray,df1dArr)
		return dfGqArray,df1dArr, vx1dArr
	#==========================================================================

	#==========================================================================
	def print_search_engine_density(self,idx,phi,dfGqArray,df1dArr, vx1dArr):
		#if idx%100==0:
		#if self.phiArr[-1]>9.5:
			if phi<0 : # this is for printing purposes
				xprintArr =  np.negative(self.phiArr) #self.xvlArr
				xprintArr = xprintArr.tolist()
			else:
				xprintArr =  self.phiArr #self.xvlArr
			
			m.live_plot(self.ax_df, 		vx1dArr,df1dArr, x="V_x",y="distribution function",title="df",s="-*")
			m.live_plot(self.ax_df 	 , self.vGqArray, dfGqArray			, x=r"V_x"	,y="DF"	  ,title="df"	 ,s="-o")
			m.live_plot(self.ax_beta[0], xprintArr  , self.betaArr  	, x=r"$\phi$",y=r"$\beta$" ,title=r'$\beta$',s="-o",ylog='symlog', ylim=True,xlim=True)#,),xlog='symlog'
			m.live_plot(self.ax_beta[1], xprintArr  , self.valueAppArr  , x=r"$\phi$",y="$n_i$" ,s="-o", c="blue",ylim=True,xlim=True)
			#m.live_plot(self.ax_beta[1], xprintArr   ,self.targetArr   	, x=r"$\phi$",y="$n_i$" ,s="-*", c="black",ylim=True,xlim=True)

			
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#BEGIN core functions
	#==========================================================================
	def getDesity(self,beta):
		self.df.produce_TranTrappedDisFun(beta)
		deltaMoment = self.df.getMomentsTrap()
		Value_output = deltaMoment[self.iMomentTarget]
		flagNaN = self.check_Nan_in_moments(deltaMoment)
		return Value_output, flagNaN
	#==========================================================================

	#==========================================================================
	def getTargetDen(self,betaDomain):
		np.random.seed()
		if betaDomain[1]>betaDomain[0]:
			beta_min = betaDomain[0]
			beta_max = betaDomain[1]
		else:
			beta_min = betaDomain[1]
			beta_max = betaDomain[0]
		beta_random = np.random.uniform(beta_min,beta_max)
		Value_output, flagNaN = self.getDesity(beta_random)
		return Value_output, beta_random, flagNaN
	#==========================================================================

	#==========================================================================
	def check_Nan_in_moments(self,moments):
		flagNaN = False
		for M in moments:
			if flagNaN == False:
				if np.isnan(M) or np.isinf(M):
					flagNaN = True
					print('!', end=' ')#"<<NaN in moments>>", moments, beta_random, betaDomain
					#pause()
		return flagNaN
	#==========================================================================
	#END
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
	#==========================================================================
	def closing_on_target_value(self,betaDomain,valueDomain): 
		i_iteration = 0
		while True: # 2nd step: closing on the target value
			i_iteration += 1

			valueAppr, betaAppr ,flagNaN = self.getTargetValue(betaDomain)

			if flagNaN == False:
				if valueAppr >valueDomain[0] and valueAppr< valueDomain[1]:
					betaDomain[0] = betaAppr
				if valueAppr >valueDomain[1] and valueAppr< valueDomain[2]:
					betaDomain[1] = betaAppr
				#print "--- betas = ", betaAppr, "in",betaDomain,  "valueAppr =" , valueAppr, ", self.target=",self.target, " approximation=", np.abs((self.target-valueAppr)/self.target), ", beta approximation=", np.abs(betaDomain[1] - betaDomain[0])
				#print_all_DF()
				if  np.abs((self.target-valueAppr)/self.target)<self.valuePercentageEpsilon and np.abs(betaDomain[1] - betaDomain[0])<self.betaEpsilon:
					break

			if i_iteration>100: # in most cases it converges in 10 steps
				if np.abs(betaDomain[1] - betaDomain[0])<1.0:
					if np.abs(betaDomain[1] - betaDomain[0]) != 0.0:
						print ("<Weak warning>, log dB=",int(np.log10(np.abs(betaDomain[1] - betaDomain[0])) ) ),
					else:
						print ("delta beta= 0",betaDomain[1] - betaDomain[0]),
				else:
					print ("  <<< ??? ERROR ??? not converged, beta = (",betaAppr,")", "iterations=", i_iteration, "value approximation=",np.abs((self.target-valueAppr)/self.target))
				break
		#----------------------------------------------------------------------
		print "[itr=", i_iteration,"]",
		return betaAppr, valueAppr
	#==========================================================================


	#==========================================================================
	def print_all_DF():
			sys.stdout.flush()
			df2dArr, vx2dArr, ex2dArr, en2dArr  = self.df.return_DfVxExEn_4arrays()
			self.print_DF(i_species,df2dArr, vx2dArr,self.ax_df)
			self.df.print_DFTranTrapp(self.ax_df[0])
			p.pause_function()
	#==========================================================================


	#==========================================================================
	def crude_guess(self):
		print "\n		 << crude Guess >> \n 			"
		flagNotFound 			= False

		betaDown 			= self.betaDnIniTotal
		betaUp 				= self.betaUpIniTotal

		valueUp 			= 0.0
		valueDown 			= 0.0
		flagUp 				= False
		flagDown			= False

		#print "\n \n \n   negative end   "
		beta 			= self.betaDnIniTotal
		while True:
			betaDomain 			= [beta,beta+1]
			valueAppr, betaAppr ,flagNaN = self.getTargetValue(betaDomain)
			if flagNaN == False:
				print betaAppr,valueAppr, "|v",
				if valueAppr>self.target:
					flagUp 		= True
					valueUp 	= valueAppr
					betaUp 		= betaAppr
				else:
					flagDown 	= True
					valueDown 	= valueAppr
					betaDown  	= betaAppr
				break
			else:
				beta = beta *0.1

		#print "\n \n \n  ****** positive end **************** \n \n"
		beta				= self.betaUpIniTotal
		while True:
			betaDomain 			= [beta-1,beta]
			valueAppr, betaAppr ,flagNaN = self.getTargetValue(betaDomain)
			if flagNaN == False:
				print betaAppr,valueAppr, "|^",
				if valueAppr>self.target:
					flagUp 		= True
					valueUp 	= valueAppr
					betaUp 		= betaAppr
				else:
					flagDown 	= True
					valueDown 	= valueAppr
					betaDown  	= betaAppr
				break
			else:
				beta = beta * 0.1
		if flagDown == True and flagUp == True:
			print ("Pass, solution in domain"),
		else: 
			print (" 	???		ERROR, solution is not in the domain		"),
			if flagDown == True:
				print ("value < valueTarget in whole domain")
			if flagUp == True:
				print ("value > valueTarget in whole domain")
			flagNotFound = True

			
		valueDomain 		= [valueDown,self.target, valueUp]
		betaDomain 			= [betaDown,betaUp]
		return betaDomain, valueDomain, flagNotFound
	#==========================================================================


	#==========================================================================
	def find_values_on_two_sides(self):
		betaDown 			= self.betaDnIni
		betaUp				= self.betaUpIni
		betaDomain 			= [betaDown,betaUp]
		flag_up 			= False
		flag_down 			= False
		valueUp 			= 0 # dummy value
		valueDown 			= 0 # dummy value
		i_iteration 		= 0
		flagNotFound 		= False
		#----------------------------------------------------------------------
		while True: # 1st step: to find two values, one on each side of target
			i_iteration 	+=1

			valueAppr, betaAppr ,flagNaN = self.getTargetValue(betaDomain)

			if flagNaN == False:
				if valueAppr>self.target:
					flag_up 	= True
					valueUp 	= valueAppr
					betaUp 		= betaAppr
				else:
					flag_down 	= True
					valueDown 	= valueAppr
					betaDown  	= betaAppr
				betaDomain 		= [betaDown,betaUp] # guided search for beta on opposite sides
				#print "--- betas = ", betaAppr, "in",betaDomain, " values = ",valueUp, ">",self.target, ">", valueDown
				#print_all_DF()

			if flag_down == True and flag_up == True:
				break

			# what to do when the solution is not converging
			if i_iteration%100==0: #  sloppier implementation  (i_iteration>100: break)
				print (" failed to find find values on two sides, betas = ("  + str(betaDomain[0]) + " " +str(betaDomain[1]),")" + "iterations=" + str(i_iteration)) #", flag_up=",flag_up, " , flag_down=", flag_down,
				#print (", valueAppr=" + str(valueAppr) + ", self.target = " + str(self.target)+",B=",betaAppr),
				if betaDomain[0]== betaDomain[1]:
					print ("falling on crude guess, devating from guided search")
					betaDomain,valueDomain, flagNotFound 	= self.crude_guess()
					if flagNotFound == True:
						return betaDomain, valueDomain, flagNotFound
					self.betaDnIni = self.betaDnIniTotal
					self.betaUpIni = self.betaUpIniTotal
					#if betaDomain[0]== betaDomain[1]:
					# try to have random choice among beta domain, deactivate guided search
					if i_iteration>100:
						print (" 		???		 Error 		, can not find values on two sides")
						flagNotFound = True
						return betaDomain, valueDomain,flagNotFound
					
		#----------------------------------------------------------------------
		valueDomain 	= [valueDown,self.target, valueUp]
		betaDomain 		= [betaDown,betaUp]
		print "[itr=", i_iteration,"]",
		return betaDomain, valueDomain, flagNotFound
	#==========================================================================


	#==========================================================================
	def initial_swep_beta_domain(self, deltaDenTarget, iDen):
		dDenArr  = []
		betaArr = []
		power = -13
		beta = -10**(-power) # note python screws up for integer value of -10e19, hence Beta has to be always larger than this number.
		sgn  = +1
		while True:
			self.df.produce_TranTrappedDisFun(beta)
			deltaMoment = self.df.getMomentsTrap()
			if deltaMoment[iDen] > deltaDenTarget+2.0:
				sgn = -1
			else: 
				sgn = +1
			
			#if np.abs((deltaMoment[iDen] - deltaDenTarget)/deltaDenTarget)*100<80 :
			# inf and nan moments has to be taken out
			betaArr.append(beta)
			dDenArr.append(deltaMoment[iDen])
			
			if  len(dDenArr)>2:
				if dDenArr[-1]-dDenArr[-2]>0.0001:
					power = power + sgn * 0.1
				else:
					power = power + sgn * 1
			else:
				power += 1
			if power<0:
				beta   = -10**(-power)
			else:
				beta   =  10**(power)
			if  len(dDenArr)>5:
				itrValue = 0
				for val in dDenArr:
					if val>deltaDenTarget+2:
						itrValue += 1
				if itrValue>5:
					break
			if beta>10e6:
				break
			#print beta, deltaMoment[iDen], deltaDenTarget
		return dDenArr, betaArr
	#==========================================================================


	#==========================================================================
	def plot_target_search(self,dDenArr,betaArr):
		m.live_plot(self.ax_beta[0], dDenArr, betaArr   	, x=r"$n_e$",y=r'$\beta$' ,s="-*", c="black",ylog='symlog',xlog='log')
		m.live_plot(self.ax_beta[1], betaArr, dDenArr      	, x=r'$\beta$',y=r"$n_e$" ,s="-*", c="black",ylog='log',xlog='symlog')

		
		#m.live_plot(self.ax_beta[0], dmTrimmed, betaTrimmed   , x=r"$n_e$",y=r'$\beta$' ,s="-*", c="blue",ylog='symlog',xlog='log')
		#m.live_plot(self.ax_beta[1], betaTrimmed, dmTrimmed   , x=r'$\beta$',y=r"$n_e$" ,s="-*", c="blue",ylog='log',xlog='symlog')
	#==========================================================================


	#==========================================================================
	def densitySearch_basedon_interpolation(self,denTarget,iS,idx,dPhi,phi,xvl):
		#failed since some beta(n) are almost detla-like function around certain value and 
		#hence the interplation can not produce reasonable value for beta
		def interplating(betaApp,betaArrIn,dDenArrIn,itr):
			dDenArr  = []
			betaArr = []
			enp     = 3-itr
			if enp<-1:
				enp = -1
			domain  = 10**enp
			betaMin = betaApp - domain
			betaMax = betaApp + domain
			beta 	= betaMin
			while True:
				if beta>betaMax:
					break
				betaArr.append(beta)
				beta += domain/2.0
			for beta in betaArr:
				self.df.produce_TranTrappedDisFun(beta)
				deltaMoment = self.df.getMomentsTrap()
				dDenArr.append(deltaMoment[iDen])
			#print "      domain=",betaArr

			dDenArr = dDenArr + dDenArrIn
			betaArr = betaArr + betaArrIn
			#print betaArr
			m.live_plot(self.ax_beta[0], dDenArr, betaArr   	, x=r"$n_e$",y=r'$\beta$' ,s="-*",ylog='symlog',xlog='log')
			m.live_plot(self.ax_beta[1], betaArr, dDenArr      	, x=r'$\beta$',y=r"$n_e$" ,s="-*",ylog='log',xlog='symlog')
			
			f_cubic_itp   	= interp1d(dDenArr,betaArr, kind='cubic',assume_sorted=False)# beta(n_e)
			betaApp 		= f_cubic_itp(deltaDenTarget)

			self.df.produce_TranTrappedDisFun(betaApp)
			deltaMoment = self.df.getMomentsTrap()
			diff = deltaMoment[0]-deltaDenTarget
			print "B=",betaApp, "Den=",deltaMoment[0]," Target=",deltaDenTarget, "diff=", diff
			return betaApp, betaArr, dDenArr, diff
		
		
		

		#avl_dDen = deltaDenTarget-deltaMoment[iDen]
		#if avl_dDen>0: # then it needs to search in the positive beta
			#print ("searching POSITVE beta"),
		#else: # search negative beta
			#print ("searching negative beta"),
		
		#dDenArr  = []
		#betaArr = []

		iDen = 0
		deltaDenTarget, flagNegative = self.initialization_search(dPhi,iDen,denTarget)
		dDenArr, betaArr = self.initial_swep_beta_domain(deltaDenTarget, iDen)
		print betaArr
		print deltaDenTarget, dDenArr
		dmTrimmed 	= []
		betaTrimmed = []
		nElements  = len(dDenArr)
		for idx in range(nElements-1):
			if np.abs(dDenArr[idx]-dDenArr[idx+1])>10e-6:
				dmTrimmed.append(dDenArr[idx])
				betaTrimmed.append(betaArr[idx])
		idx = nElements-1
		if np.abs(dDenArr[idx]-dDenArr[idx-1])>10e-6:
				dmTrimmed.append(dDenArr[idx])
				betaTrimmed.append(betaArr[idx])
				
		self.plot_target_search(dDenArr,betaArr)
		m.pause()
		#f_cubic_itp   	= interp1d(dmTrimmed,betaTrimmed, kind='cubic')# beta(n_e)
		f_cubic_itp   	= interp1d(dDenArr,betaArr, kind='cubic')
		betaApp 		= f_cubic_itp(deltaDenTarget)
		print "B=",betaApp,

		self.df.produce_TranTrappedDisFun(betaApp)
		deltaMoment = self.df.getMomentsTrap()
		diff = deltaMoment[0]-deltaDenTarget
		print "Den=",deltaMoment[0]," Target=",deltaDenTarget,"diff=", diff, "initial guess"
		m.pause()
		betaArr = betaTrimmed
		dDenArr = dmTrimmed
		itr 	= 0
		while True:
			itr += 1
			if diff == 0.0:
				break
			betaApp_before = betaApp
			betaApp,betaArr,dDenArr, diff  = interplating(betaApp,betaArr,dDenArr,itr)
			#print betaApp, betaArr
			if np.abs( (betaApp - betaApp_before)/betaApp_before )*100>0.001:
				break

		momentsTrapped = self.df.getMomentsTemp()
		self.print_search_engine_density(idx,phi,dfGqArray)

		print "====================="
		return momentsTrapped,dfGqArray
	#==========================================================================


	#==========================================================================
	def find_fitting_beta(self,value_target, i_species,idx,deltaPhi,phi,xvl):
		self.df.update_VxMoments_initialize_VxEkTrapp(deltaPhi)
		self.target = value_target
		np.random.seed()
		valueAppr 				= 0.0
		betaAppr				= 0.0




		if idx>100:
			if self.flagGuidedSearch == True:
				print ("\n \n *** guided search on beta *** \n \n ")
				self.flagGuidedSearch = False
			self.betaUpIni = self.lastBeta+100
			self.betaDnIni = self.lastBeta-100


		betaDomain,valueDomain, flagNotFound 	= self.find_values_on_two_sides()
		#betaDomain,valueDomain 	= crude_guess()
		if flagNotFound == False:
			betaAppr, valueAppr 	= self.closing_on_target_value(betaDomain,valueDomain)
			self.lastBeta = betaAppr
		else: # in most cases when dphi is too small beta can not be found and the last beta found is good enough
			if deltaPhi/phi<1.0/100.0:
					betaAppr = self.lastBeta
					print ("warning ... using the last beta")



		self.betaArr.append(betaAppr)#np.sign(betaAppr)*np.log10(np.abs(betaAppr))
		self.valueAppArr.append(valueAppr)

		self.phiArr.append(phi)
		self.xvlArr.append(xvl)
		self.targetArr.append(self.target)
		print "vTarget=",self.target,
		sys.stdout.flush()
		
		
		print "beta=",betaAppr#, "appro=",valueTargetAppro, " ~", self.target, ", phi=", phi, ",E=",Efd
		transMoments = self.df.produce_TranTrappedDisFun_itsMoments(betaAppr)
		self.df.add_DfVxMoments_Tran_Permanent()

		momentsTrapped = self.df.getMomentsTemp()
		df2dArr, vx2dArr, ex2dArr, en2dArr  = self.df.return_DfVxExEn_4arrays()

		df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)

		dfGqArray = m.get_interpolation(vx1dArr, self.vGqArray,df1dArr)





		if idx%100==0:
		#if self.phiArr[-1]>9.5:
			if phi<0 : # this is for printing purposes
				xprintArr =  self.phiArr #self.xvlArr
			else:
				xprintArr =  self.phiArr #self.xvlArr
			#m.live_plot(self.ax_df, 		vx1dArr,df1dArr, x="V_x",y="distribution function",title="df",s="-*")
			m.live_plot(self.ax_df 	 , self.vGqArray, dfGqArray			, x=r"V_x"	,y="DF"	  ,title="df"	 ,s="-o")
			m.live_plot(self.ax_beta[0], xprintArr  , self.betaArr  	, x=r"$\phi$",y=r"$\beta$" ,title=r'$\beta$',s="-o")#,ylog='symlog',xlog='symlog')
			m.live_plot(self.ax_beta[1], xprintArr  , self.valueAppArr  , x=r"$\phi$",y="$n_i$" ,s="-o", c="blue")
			m.live_plot(self.ax_beta[1], xprintArr   ,self.targetArr   	, x=r"$\phi$",y="$n_i$" ,s="-*", c="black")

			
		return momentsTrapped,dfGqArray
	#==========================================================================

	#==========================================================================
	def getTargetValue(self,betaDomain):
		flagNaN = False
		np.random.seed()
		if betaDomain[1]>betaDomain[0]:
			beta_min = betaDomain[0]
			beta_max = betaDomain[1]
		else:
			beta_min = betaDomain[1]
			beta_max = betaDomain[0]
		beta_random = np.random.uniform(beta_min,beta_max)
		transMoments = self.df.produce_TranTrappedDisFun_itsMoments(beta_random)
		Value_output = transMoments[self.iMomentTarget]
		for M in transMoments:
			if flagNaN == False:
				if np.isnan(M) or np.isinf(M):
					flagNaN = True
					print "\n \n <<NaN in moments>>", transMoments, beta_random, betaDomain
					#pause()
		return Value_output, beta_random, flagNaN
	#==========================================================================
'''
#END
#******************************************************************************

#******************************************************************************
#BEGIN Test case
#==============================================================================
def test_case():
	iS  = 0
	nVx_Gq = 120
	vGqElc 		= m.get_vGqArray(p.Vx_min[iS],p.Vx_max[iS],p.dVx[iS],nVx_Gq)
		
	dfTrpElc 	= distributionClass(i_species=0, v_soliton = 500)
	vx 			= vGqElc

	phiArr 		= [ 1.0 ,  2.0,  3.0,  4.0, 5.0]
	betaArr 	= [-10.0, -5  , -2.5, -1.0, -0.5]
	phi0 		= 0
	for iPhi in range (len(phiArr)):
		phi 		= phiArr[iPhi]
		deltaPhi 	= phi - phi0
		dfTrpElc.update_VxMoments_initialize_VxEkTrapp(deltaPhi)
		transMoments = dfTrpElc.produce_TranTrappedDisFun_itsMoments(betaArr[iPhi])
		dfTrpElc.add_DfVxMoments_Tran_Permanent()
		phi0 = phi

	fig_df, ax_df = plt.subplots(1, figsize=[8,4])
	betaTests = [-100,-10,-1]
	deltaPhi = 10.01#10e-6
	dfTrpElc.update_VxMoments_initialize_VxEkTrapp(deltaPhi)
	nTest = 16
	nHalf = int(nTest/2)-7
	for iBeta in range(nTest):
		sgn = np.sign(nHalf - iBeta)
		epn = np.abs(nHalf - iBeta)
		beta = sgn * 10**(epn)
		
		if epn<13 :
			transMoments = dfTrpElc.produce_TranTrappedDisFun_itsMoments(beta)
		else:
			#transMoments = dfTrpElc.produce_TranTrappedZeor_itsMoments()
			transMoments  = dfTrpElc.getMomentsTemp()
			beta 		 = 0.0
		
		print(transMoments,beta,epn)
		
		df2dArr, vx2dArr, ex2dArr, en2dArr  = dfTrpElc.return_DfVxExEn_4arrays()
		df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)
		dfGqArray = m.get_interpolation(vx1dArr, vx,df1dArr)
		dfTrpElc.print_DFTranTrapp(ax_df)
	m.live_plot(ax_df, vx1dArr, df1dArr, 		s="-*")
	m.live_plot(ax_df, vx		, dfGqArray, 	s="-o")

	# note python screws up for integer value of -10e19, hence Beta has to be always larger than this number. 
#==============================================================================

#==============================================================================
def test_densitySearch():
	iS  = 0
	nVx_Gq = 120
	vGqElc 		= m.get_vGqArray(p.Vx_min[iS],p.Vx_max[iS],p.dVx[iS],nVx_Gq)
		
	dfTrpElc 	= distributionClass(i_species=0, v_soliton = 10)
	vx 			= vGqElc

	phiArr 		= [ 1.0 ,  2.0,  3.0,  4.0, 5.0]
	betaArr 	= [-10.0, -5  , -2.5, -1.0, -0.5]
	phi0 		= 0
	for iPhi in range (len(phiArr)):
		phi 		= phiArr[iPhi]
		deltaPhi 	= phi - phi0
		dfTrpElc.update_VxMoments_initialize_VxEkTrapp(deltaPhi)
		transMoments = dfTrpElc.produce_TranTrappedDisFun_itsMoments(betaArr[iPhi])
		dfTrpElc.add_DfVxMoments_Tran_Permanent()
		phi0 = phi

	betaTests 		= [-100,-10,-1]
	
	deltaPhi 		= 0.01
	denTarget 		= 1.002
	phi 			= phiArr[-1]+deltaPhi
	idx 			= 0
	xvl 			= 0
	
	findBeta 	= FindBeta(dfTrpElc,vx)
	beta_before = betaArr[-1]
	findBeta.densitySearch(denTarget,iS,idx,deltaPhi,phi,xvl) #<<<<<<<<<<<<<<
	
	df2dArr, vx2dArr, ex2dArr, en2dArr  = dfTrpElc.return_DfVxExEn_4arrays()
	df1dArr, vx1dArr = m.convert_2D_1D(df2dArr, vx2dArr)
	dfGqArray = m.get_interpolation(vx1dArr, vx,df1dArr)
		
		
	m.live_plot(findBeta.ax_df, vx1dArr, df1dArr, 		s="-*")
	m.live_plot(findBeta.ax_df, vx		, dfGqArray, 	s="-o")
#==============================================================================
#END
#******************************************************************************

#******************************************************************************
#BEGIN
#test_case()
#test_densitySearch()
#m.pause()
#END
#******************************************************************************
