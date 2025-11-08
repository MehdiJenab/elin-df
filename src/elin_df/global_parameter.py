


import os
import json
import numpy as np





#==============================================================================
class global_parameters(object):
	path_src = os.path.dirname(os.path.abspath(__file__))
	gasGamma_electron = 3.0
	gasGamma_ion      = 3.0
	gasGamma           = [gasGamma_electron,gasGamma_ion]
	ion_sound_velocity = 1.0
	#==================================================================================================

	config_path = os.path.join(path_src, "0_config.json")
	with open(config_path) as data_file:
		data = json.load(data_file)

	N_species = data['config'][0]['N_species']
	N_Time    = data['config'][3]['Time'][0]['N_Time']
	Num_Spatial_Output =    data['config'][3]['Time'][2]['Num_Spatial_Output'] + 1 # +1 because of writing the initial step into Spatial.h5
	if (Num_Spatial_Output>N_Time+1):
		print ("    *** Error Message: Num_Spatial_Output>N_Time ***")
		Num_Spatial_Output = 1

	Num_Contour =    data['config'][3]['Time'][3]['Num_Contour'] + 1
	if (Num_Contour>N_Time+1):
		print ("    *** Error Message:Num_Contour>N_Time ***")
		Num_Contour = 1

	d_Time =              data['config'][3]['Time'][1]['d_Time']
	Time_Length =    d_Time * N_Time



	nX =       data['config'][2]['Grid'][0]['nX']
	X_min =                data['config'][2]['Grid'][1]['X_min']
	X_max =                data['config'][2]['Grid'][2]['X_max']
	x_Length =         X_max - X_min
	d_X = x_Length / nX
	Name =  [None]*(N_species+1)
	Mass =  [None]*(N_species+1)
	Density=        [None]*(N_species+1)
	Charge= [None]*(N_species+1)
	Temp =  [None]*(N_species+1)
	Den_Ratio_b =  [None]*(N_species+1)
	norm_factor =   [None]*(N_species+1)

	NoW =   [None]*(N_species+1)
	nVx =   [None]*(N_species+1)
	Vx_min= [None]*(N_species+1)
	Vx_max= [None]*(N_species+1)
	dVx   = [None]*(N_species+1)
	Alpha=  [None]*(N_species+1)
	Kappa_DF =  [None]*(N_species+1)
	DF_Vx_MK =  [None]*(N_species+1)
	DF_NotShifted_Shifted = [None]*(N_species+1)
	for i in range(0,N_species+1):
					Name  [i]=      data['config'][1]['PhasePoint'][i]['Species'][0]['Name']
					Mass  [i]=      data['config'][1]['PhasePoint'][i]['Species'][1]['Mass']
					Charge[i]=      data['config'][1]['PhasePoint'][i]['Species'][2]['Charge']        
					Temp  [i]=       data['config'][1]['PhasePoint'][i]['Species'][3]['Temperature']
					norm_factor[i] = Mass[i]/Temp[i]
					Den_Ratio_b  [i]=       data['config'][1]['PhasePoint'][i]['Species'][4]['Den_Ratio_b']
					Density  [i]=         data['config'][1]['PhasePoint'][i]['Species'][5]['Density']
					Alpha [i]=      data['config'][1]['PhasePoint'][i]['Species'][6]['Alpha']
					NoW   [i]=      data['config'][1]['PhasePoint'][i]['Species'][7]['NoW']
					nVx   [i]=      data['config'][1]['PhasePoint'][i]['Species'][10]['nVx']
					Vx_min[i] =    data['config'][1]['PhasePoint'][i]['Species'][11]['Vx_min']
					Vx_min[i] = Vx_min[i]/np.sqrt(norm_factor[i])
					Vx_max[i] =    data['config'][1]['PhasePoint'][i]['Species'][12]['Vx_max']
					Vx_max[i] = Vx_max[i]/np.sqrt(norm_factor[i])
					dVx[i] = ( abs( Vx_min[i] - Vx_max[i] ) )  / ( float( nVx[i] ) )
					DF_Vx_MK[i] = data["config"][1]["PhasePoint"][i]["Species"][14]["DF_Vx_MK"]
					Kappa_DF[i] =    data['config'][1]['PhasePoint'][i]['Species'][15]['Kappa_DF']
					DF_NotShifted_Shifted[i] =    data['config'][1]['PhasePoint'][i]['Species'][17]['DF_NotShifted_Shifted']
        
	#ion_sound_velocity = np.sqrt(1.0+Temp[0]/Temp[1])
	ion_sound_velocity = np.sqrt((gasGamma_electron*Temp[0]+gasGamma_ion*Temp[1])/Mass[1])#np.sqrt(1.0+Temp[0]/Temp[1])
	jump_number_time_spatial = (N_Time)/(Num_Spatial_Output-1)  
	jump_number_time_contour = (N_Time)/(Num_Contour-1)
	color_line=['black','red','blue','brown','green','cyan','magenta','navy','darkorchid','indigo','purple','gold','tan','salmon','red','blue','black','brown','yellow','green','cyan','magenta','navy','darkorchid','indigo','purple','gold','tan','salmon']
    
    
	#---------------------------------------------------------
	config_path_solitons = os.path.join(path_src, "0_config_solitons.json")
	with open(config_path_solitons) as data_file:
			data_config_solitons = json.load(data_file)
	#---------------------------------------------------------

	#---------------------------------------------------------
	Number_Solitons =  data_config_solitons["config_solitons"][0]["Number_Solitons"] 
	
	Mach_Number_array_input =  []
	Beta_Schamel_array_input =  []
	Phi_max_array_input =  []
	Alpha_Schamel_array_input =  []
	left_side_of_soliton_array_input =  []
	v_soliton_array_input =  []
	for i_counter in range(1,Number_Solitons+1):
					Mach_Number_array_input.append(data_config_solitons["config_solitons"][i_counter]["Soliton"][0]["Mach_Number"])
					Beta_Schamel_array_input.append(data_config_solitons["config_solitons"][i_counter]["Soliton"][1]["Beta_Schamel"])
					Phi_max_array_input.append(data_config_solitons["config_solitons"][i_counter]["Soliton"][2]["Phi_max"])
					Alpha_Schamel_array_input.append(data_config_solitons["config_solitons"][i_counter]["Soliton"][3]["Alpha_Schamel"])
					left_side_of_soliton_array_input.append(data_config_solitons["config_solitons"][i_counter]["Soliton"][4]["left_side_of_soliton"])
					v_soliton_array_input.append(ion_sound_velocity * Mach_Number_array_input[-1] )
	#---------------------------------------------------------
	
	
	
	#---------------------------------------------------------
	config_path_output = os.path.join(path_src, "0_config_output.json")
	with open(config_path_output) as data_file:    
			data_config_output = json.load(data_file)
	#---------------------------------------------------------

	#---------------------------------------------------------
	jason_key = data_config_output["config_output"][0]["general"][0]["X_limit"] 
	
	if jason_key[0] ==0:
		X_limit = [X_min ,X_max]
	else:
		X_limit = [jason_key[1] ,jason_key[2]]
	
	
	jason_key = data_config_output["config_output"][0]["general"][1]["moving_frame_velocity"] 
	if jason_key[0] ==0:
		moving_frame_velocity = v_soliton_array_input[0]
	else:
		moving_frame_velocity = jason_key[1]
	jason_key = data_config_output["config_output"][0]["general"][2]["time_interval_print"]	
	if jason_key[0] ==0:
		time_interval_print = 1
	else:
		time_interval_print = jason_key[1]
	time_interval_print_contour = time_interval_print * jump_number_time_contour
	time_interval_print_spatial = time_interval_print * jump_number_time_spatial
	jason_key = data_config_output["config_output"][0]["general"][3]["start_end_time"]	
	if jason_key[0] ==0:
		start_time = 0
		end_time = N_Time
	else:
		start_time = jason_key[1]
		end_time =  jason_key[2]
	
	jason_key = data_config_output["config_output"][0]["general"][4]["conservation_analysis"]
	if jason_key[0] ==0:
		conservation_analysis = False
	else:
		conservation_analysis = True
	
	jason_key = data_config_output["config_output"][0]["general"][5]["parameter_name_array"]
	parameter_name_array = [None]*(jason_key[0])
	for i_counter in range (0, jason_key[0]):
		parameter_name_array[i_counter]=jason_key[i_counter+1]
		
	number_random_sampling = data_config_output["config_output"][0]["general"][6]["number_random_sampling"]
	
	Velocity_limit = [None]*(N_species+1)
	for i_species in range (0, N_species+1):
		jason_key = data_config_output["config_output"][1]["species_dependent"][i_species]["Velocity_limit"]
		if jason_key[0] ==0:
			Velocity_limit[i_species] = [Vx_min[i_species] ,Vx_min[i_species]]
		else:
			Velocity_limit[i_species] = [jason_key[1]/np.sqrt(norm_factor[i_species]) ,jason_key[2]/np.sqrt(norm_factor[i_species])]


	AdiosTime = 0

	
	#==================================================================================================
	def pause_function(self):
		input_string = input(" ... paused ... ")
		#print "Received input is : ", input_string
#==============================================================================
#==================================================================================================

