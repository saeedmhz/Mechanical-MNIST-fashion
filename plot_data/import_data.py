import numpy as np
import matplotlib.pyplot as plt
##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       DATA IMPORT            ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################

##########################################################################################
# bitmap input -- might have to change file destination
##########################################################################################
bitmap_test = np.loadtxt('example_outputs/input_test_fashion_MNIST_first100.txt')
bitmap_train = np.loadtxt('example_outputs/input_train_fashion_MNIST_first100.txt')

##########################################################################################
# strain energy -- might have to change file destination
##########################################################################################
psi_test = np.loadtxt('example_outputs/UE_psi_test_first100.txt')
psi_train = np.loadtxt('example_outputs/UE_psi_train_first100.txt')

##########################################################################################
# reaction force -- might have to change file destination
##########################################################################################
F_test = np.loadtxt('example_outputs/UE_rxnF_test_first100.txt')
Fx_test = F_test[:,0]; Fy_test = F_test[:,1]
F_train = np.loadtxt('example_outputs/UE_rxnF_train_first100.txt')
Fx_train = F_train[:,0]; Fy_train = F_train[:,1]

##########################################################################################
# displacement -- might have to change file destination
##########################################################################################
disp_x_test = np.loadtxt('example_outputs/UE_disp_x_test_first100.txt')
disp_y_test = np.loadtxt('example_outputs/UE_disp_y_test_first100.txt')
disp_x_train = np.loadtxt('example_outputs/UE_disp_x_train_first100.txt')
disp_y_train = np.loadtxt('example_outputs/UE_disp_y_train_first100.txt')

# note -- each row can be changed into the format of a bitmap w/ .reshape(28,28)

##########################################################################################
# ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~       MAKE PLOTS            ~ - ~ - ~ - ~ - ~ - ~ - ~ - ~
##########################################################################################

##########################################################################################
def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')

# plot reaction force wrt strain energy -- just to see
plt.figure()

ax = plt.gca()
data = [Fx_test, Fy_test, Fx_train, Fy_train]
ax.violinplot(data) 

set_axis_style(ax,['Fx test','Fy test', 'Fx train', 'Fy train'])

plt.ylabel('reaction force')
plt.title('visualize data reaction forces')
plt.savefig('visualize_F')

plt.figure()

ax = plt.gca()
data = [psi_test, psi_train]
ax.violinplot(data) 

set_axis_style(ax,['psi test', 'psi train'])

plt.ylabel('reaction force')
plt.title('visualize data delta psi')
plt.savefig('visualize_psi')

##########################################################################################



##########################################################################################
# set up displacement information 
# plot displacement -- right on 
def flip_data(input): 
	data_to_plot_flipped = input
	data_to_plot = np.zeros(data_to_plot_flipped.shape)
	for jj in range(0,data_to_plot.shape[0]):
		for kk in range(0,data_to_plot.shape[1]):
			data_to_plot[kk,jj] = data_to_plot_flipped[int(27.0-kk),jj]
	return data_to_plot

def define_colorfield(data):
	max = np.max(data)
	min = np.min(data)
	color_data = (data - min)/(max-min)
	return color_data


for example_to_show in range(0,5):
	bitmap_actual = bitmap_test[example_to_show,:].reshape(28,28)
	disp_actual_x = disp_x_test[example_to_show,:].reshape(28,28)
	disp_actual_y = disp_y_test[example_to_show,:].reshape(28,28)

	init_x = np.zeros((28,28))
	init_y = np.zeros((28,28))

	for kk in range(0,28):
		for jj in range(0,28):
			init_x[kk,jj] = jj + 0.5 # x is columns, 0 is lower corner 
			init_y[kk,jj] = kk + 0.5 # y is rows 

	##########################################################################################

	fig = plt.figure(figsize=(4,4))

	x_positions = init_x + disp_actual_x
	y_positions = init_y + disp_actual_y
	idx_bitmap_actual_flip = flip_data(bitmap_actual)
	color_field = define_colorfield(idx_bitmap_actual_flip)

	plt.plot([-5,33],[-5,45],'w.')

	for kk in range(0,28):
		for jj in range(0,28):
			plt.plot(x_positions[kk,jj] , y_positions[kk,jj] , 's',markersize=2.5,color = (color_field[kk,jj],0,0))

	plt.title('displacement', y=-0.01)
	plt.axis('equal')
	plt.axis('off')

	plt.savefig('disp_example_%i'%(example_to_show))