results & values:

##########

by the way: datasets are already splited to training, validation and testing sets.


model 1: ( keras_cifar10_trained_model )
model 2: ( keras_cifar10_trained_sub_model )
model 3: ( keras_cifar10_trained_model_128_cell )
model 4: ( keras_cifar10_trained_sub_model_128_cell_64_batch )
model 5: ( keras_cifar10_trained_adam_opt )
model 9: ( final model )

*IMPORTANT NOTE*  I will always run 10 epochs and check the accuracy and the loss, if it looks promessing and giving a promissing values I will run 100 epochs then.

the best results are:

			* results *
	model   trial	 num_epochs		Parameter_name		previous_value		new_value		train_accuracy		test_accuracy		train_loss		test_loss		action
	1		0			100				--					--					--				0.79772			 0.7779 			0.6367213		0.70282867 		original_model
	2		0			10 				--					--					--				0.65908 		 0.6542 			0.9567476 		0.97330696 		original_model

	4		6 			100 			--		 			--					-- 				0.87464 		 0.8366 			0.37794 		0.50077 		old_model
	2		3 			10 				--		 			-- 					-- 				0.69456 		 0.6881 			0.8667949 		0.88892 		old_model
	
	5		8 			100				opt		 			-- 					-- 				0.92114 		 0.8506 			0.238647 		0.4417384 		cur_model
	5		7 			10 				--		 			-- 					-- 				0.6884 			 0.6796 			0.90928 		0.9351026 		cur_model

	6		10			100			 	drop	 			-- 					-- 			 	0.91444 		0.8524 				0.25846557 		0.4306899 		under_training
	6		9			10 			 	--		 			-- 					-- 				0.69276 		 0.684 				0.88099228 		0.90294918  	under_training

our trials: 

			* done *
	2		1			10 				batch_size			32					64 				0.57726			 0.575 				1.2242891 		1.2297011 		neglect
	2		2			10 		 num_initial_filters		32 					128 			0.7216 			 0.7158 			0.79991521 		0.82979323 		apply
	2		3 			10 				batch_size 			32 					64 				0.69456 		 0.6881 			0.8667949 		0.88891986 		promissing to check
	3		4 			30 	 	 num_initial_filters		32 					128 			0.8011 			 0.7614 			0.591712 		0.704423		apply
	3		5 			100 	 num_initial_filters		32 					128 			0.76056 		 0.741 				0.813926 		0.910277 		apply early stopping
	4		6 			100 			batch_size 			32					64 				0.87464 		 0.8366 			0.37794 		0.50077 		apply


	5		7			10 				optimizer 			rmsprop				adam			0.6884 			 0.6796 			0.90928 		0.9351026 		promissing to check
	5		8 			100 			optimizer 			rmsprop 			adam 			0.92114 		 0.8506 			0.238647 		0.4417384 		apply
	6		9			10 			dropout_layer			adding dropout in_layer 0.15 	 	0.69276 		 0.684 				0.88099228 		0.90294918 		promissing to check
	6		10			100 		dropout_layer			adding dropout in_layer 0.15 	 	0.91444 		 0.8524				0.25846557 		0.4306899 	promissing to inc. dropout 
	6		11			100 		dropout_layer			adding dropout in_layer 0.25 		0.90466 		 0.8462 			0.28886295 		0.456715 		neglect
	6		12			10 		dropout 0.25 reg_term		-- 					L2 		 		0.682 			 0.6715 			0.94805615 		0.9720314 		not clear, inc. epochs
	6		13			30 		dropout 0.25 reg_term		-- 					L2 		 		0.79546 		 0.7727 			0.6204847 		0.68845 	reg&droupout 0.25 not good
	6		14			10		dropout 0.15 reg_term		-- 					L2 		 		0.68486 		 0.6786 			0.92278288 		0.9466334 		reg&dropout 0.15 better
	6		15			30		dropout 0.15 reg_term		-- 					L2 		 		0.81056 		 0.7863 			0.5809 			0.6574507 		promissing to check
	6		16			100		dropout 0.15 reg_term		-- 					L2 		 		0.91084 		 0.8484 			0.31013 		0.47525568 		neglect
	6		17			30		without dropout reg_term	-- 					L2 		 		0.79682 		 0.7736 			0.613128 		0.68806 		neglect

	7		18			10 		data_augmentation with drop	True 				False 			0.73772			 0.7033 			0.80808064 		0.8855037 		promissing to apply
	8		19			10		data_augmentation no drop	True 				False 			0.75288 		 0.7174 			0.75784913 		0.8365525 	best accuracy with 10 epochs



	8		20 			10 		no DA, no dropout doubled filters number with reg				0.86334 		 0.7847 			0.47977725 		0.6887 	trick in n_filters, will overfit
	8		21 			10 		no DA, input dropout .15 doubled filters number with reg		0.83168			 0.761 				0.556263 		0.76045 		may overfit
	
	8		23 			10 		with DA, input dropout .15 doubled filters with reg	rmsprop 	0.7346			 0.74612 			0.78452533 		0.83092948		adam better than rmspro	8		29 			10		with DA, input dropout .25 doubled filters number with reg		0.76396 		 0.7409 			0.729281 		0.798403  reduce diff. but dec. accurac	8		24 			10 		no DA, input drop .15 doubled filters number with reg			0.85088			 0.7684 			0.5175825 		0.72770359 		neglect, will overfit
	8		27 			10 		no DA, input drop .15 doubled filters , no reg 					0.8525 			 0.7664 			0.4637492 		0.6905263 		neglect, will overfit
	8		25 			10 		no DA, no dropout doubled filters , no reg 						0.8674 			 0.7844 			0.41034758 		0.6296873 		may overfit
	8		26 			50 		no DA, no dropout doubled filters , no reg 						0.99988 		 0.8289				0.00258 		0.7562815 	overfitted, early stop at 52


	9		22 			10		with DA, input dropout .15 doubled filters with reg adam		0.77528			 0.7516 			0.70745424 		0.7716177 	DA term prevent overfitting
	9 		28 			100 	with DA, input dropout .15 doubled filters with reg adam		0.98764 		 0.8904 			0.130619		0.428236

extra steps if we have time, we may try num of filters 64 instead of 128


 