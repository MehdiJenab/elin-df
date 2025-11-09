

import 		numpy 				as np
import 		matplotlib.pyplot 	as plt
from 		scipy.interpolate 	import interp1d
from 		random 				import randint

from .global_parameter 			import global_parameters 
p = global_parameters()
#******************************************************************************
class mathClass():
	def pause(self):
		programPause = input("...")
	#==========================================================================
	def convert_2D_1D(self,yIn, xIn):
		y_left 		= []
		y_right 	= []
		x_left 		= []
		x_right 	= []

		for iSlice in range(len(yIn)):
			if iSlice%2==0:
				for ivx in range(len(yIn[iSlice])-1):
					y_left.append(yIn[iSlice][ivx])
					x_left.append(xIn[iSlice][ivx])

		for iSlice in range(len(yIn)-1,0,-1):
			if iSlice%2==1:
				for ivx in range(len(yIn[iSlice])-1):
					y_right.append(yIn[iSlice][ivx])
					x_right.append(xIn[iSlice][ivx])

		if len(y_right)>0:
			for ivx in range(len(y_right)):
				y_left.append(y_right[ivx])
				x_left.append(x_right[ivx])


		# remove same points from vx arrays, neighboring slices in ElIN df have overlapping points on the borders
		# which makes interp1d fail since x array is not sorted! (if it has repetative points)
		xOut = []
		yOut = []
		for ivx in range(1,len(x_left)):
			if x_left[ivx-1]!=x_left[ivx]:
				xOut.append(x_left[ivx-1])
				yOut.append(y_left[ivx-1])
		xOut.append(x_left[-1])
		yOut.append(y_left[-1])
		return yOut, xOut
	#==========================================================================



	#==========================================================================
	def get_interpolation(self,X_InArray, X_OutArray, Y_InArray):
		f_cubic_itp   = interp1d(X_InArray, Y_InArray, kind='cubic',fill_value=0.0, assume_sorted=False, bounds_error=False)
		Y_OutArray = []
		for X in X_OutArray:
			Y_OutArray.append(f_cubic_itp(X))
		return Y_OutArray
	#==========================================================================


	#==========================================================================
	def get_vGqArray(self,xMin, xMax,xDltIn,nx):
		xDlt					= (xMax-xMin) / nx
		vGqArray				= []
		x 						= xMin
		flag_loop 				= True

		#----------------------------------------------------------------------
		def get_GaussianQuadrature(xStart,xEnd):
			GqOrder 		= 5
			GQ0 			= 0
			GQ1 			= (1.0/3.0) * np.sqrt(5.0 - (2.0 * np.sqrt(10.0/7.0)) ) #0.25
			GQ2 			= (1.0/3.0) * np.sqrt(5.0 + (2.0 * np.sqrt(10.0/7.0)) ) #0.5 

			GQ_minus1to1 	= [-GQ2, -GQ1, GQ0, GQ1, GQ2]

			GQ_0To1 		= []

			b 				= xEnd
			a 				= xStart
			for x in GQ_minus1to1:
				t = ((b-a)*x + b + a)/2.0
				GQ_0To1.append(t)
			#self.xMiddleAdoptive = 224.624910759 # to be deleted in the roll-out version
			#GQ_0To1 = [0.0,0.25,0.5,0.75] # to have regular sampling independent of GQ nodes
			#self.GQ_0To1 = GQ_0To1
			return GQ_0To1
		#----------------------------------------------------------------------

		GQ_0To1 = get_GaussianQuadrature(0,xDlt)

		while True:
			for iq in range(len(GQ_0To1)):
				xGQ = x + GQ_0To1[iq]
				if xGQ>=xMax:
					flag_loop = False
				else:
					vGqArray.append(xGQ)
			if flag_loop == False:
				break
			x += xDlt
		return vGqArray
	#==========================================================================


	#==========================================================================
	def live_plot(self,*args,**kwargs):
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
		ylog		 	= kwargs.get('ylog', "")
		xlog		 	= kwargs.get('xlog', "")
		diff		 	= kwargs.get('diff', False)
		xlim 			= kwargs.get('xlim', False)
		ylim 			= kwargs.get('ylim', False)

		if xlim == True:
			ax_in.set_xlim([np.amin(X_array),np.amax(X_array)])
			
		if ylim == True:
			ax_in.set_ylim([np.amin(Y_array),np.amax(Y_array)])
			
		flag_scatter 	= True
		if xlimit:
			ax_in.set_xlim(xlimit[0],xlimit[1])
		if ylimit:
			ax_in.set_ylim(ylimit[0],ylimit[1])
		if isinstance(X_array, list):
			flag_scatter = False


		if not color:
			color = p.color_line[randint(0, 9)]
		Y_array1 = []
		
		if diff == True:
			for iy in range(len(Y_array)):
				Y_array1.append( Y_array[iy] - Y_array[0])
		else: 
				Y_array1 =  Y_array
		#if (X_title == "V_x"):
			#Y_array = np.log10(Y_array)

		#ax_in.axis([np.nanmin(X_array) , np.nanmax(X_array) , np.nanmin(Y_array), np.nanmax(Y_array)+0.1*np.nanmax(Y_array)])

		if flag_scatter == True:
			ax_in.scatter(X_array,Y_array1,c=color)
		else:
			ax_in.plot(X_array,Y_array1, style,c=color, label='the data')

		if X_title:
			ax_in.set_xlabel(X_title)
		if Y_title:
			ax_in.set_ylabel(Y_title)
		if plot_title:
			ax_in.set_title(plot_title)
		if ylog :
			ax_in.set_yscale(ylog)
		if xlog :
			ax_in.set_xscale(xlog)
		ax_in.grid(True)
		#plt.pause(0.00001)
	#==========================================================================
#******************************************************************************
#math = mathClass()
