def create_matrix_train_set(max_number_of_images): # el parametro es para pruebas con un set de datos menor, en la version de produccion quitar para que tome todas
	
	if max_number_of_images <= 0:
		max_number_of_images = 5635;
		
	#image = tifffile.imread('/media/luisf/media/devongt/Kagle/UltrasoundNerveSegmentation/train/9_44.tif', key=0); #devuelve un array de pixeles de  420*580
	#image = image.flatten();
	#print np.shape(image);

	image_list = os.listdir(train_data_path);
	training_size = max_number_of_images; #len(image_list)/2; quitar el parametro y dejar el actual comentario 
	names_array = np.ndarray(shape=(training_size),dtype=np.object_);
	images_array = np.ndarray(shape=(training_size,image_rows,image_cols,1),dtype=np.uint8);#crear un array vacio para un row por cada imagen y una columna por cada pixel de la imagen
	masks_array = np.ndarray(shape=(training_size,image_rows,image_cols,1),dtype=np.uint8); #similar crear un array vacio para las mascaras
	contains_nerve_outputs = np.ndarray(shape=(training_size,1),dtype=np.uint8); #crea un vector de output que indica si la imagen tiene o no nervio

	image_counter = 0;
	for image in image_list[0:(max_number_of_images*2)-2]:#para hacer pruebas con un subset de imagenes, quitar el subsetting en full mosh
		if 'mask' in image: #ejecutar el codigo del ciclo solo si es imagen y no mascara
			continue;
		image_name = image.split(".")[0];
		mask_name = image_name+"_mask.tif"; 
		print "Reading image "+str( image_counter)+":"+image_name;
		names_array[image_counter] = image_name;
		image_array = tifffile.imread(train_data_path+"/"+image);
		#image_array = image_array.flatten();

		print "Reading mask:"+mask_name;
		mask_array = tifffile.imread(train_data_path+"/"+mask_name); #algunas mascaras por ejemplo la 10_48_mask.tif da warnings 
		#mask_array = mask_array.flatten();

		images_array[image_counter] = image_array;
		masks_array[image_counter] = mask_array;
		print "Calculating contains nerve flag for:"+image_name;	
		contains_nerve_outputs[image_counter] = create_contains_nerve_output(mask_array.flatten());	
		
		image_counter+=1;
		print " ";

	print "Saving numpy arrays";
	np.save(train_data_path+"/training_set_names_matrix.npy",names_array);
	np.save(train_data_path+"/training_set_images_matrix.npy",images_array);
	np.save(train_data_path+"/traing_set_masks_matrix.npy",masks_array);
	np.save(train_data_path+"/contains_nerve_outputs_matrix.npy",contains_nerve_outputs);

def read_matrix_train_set(size):  #agregar que lea imagenes aleatoriamente con   np.random.choice(len(data), size, False)
     names_array =   np.load(train_data_path+"/training_set_names.npy");
     images_array = np.load(train_data_path+"/training_set_images.npy");
     masks_array = np.load(train_data_path+"/traing_set_masks.npy");
     contains_nerve_outputs = np.load(train_data_path+"/contains_nerve_outputs.npy");
     trainsize = size if size>0 else len(images_array);
     data_indices = np.random.choice(len(images_array), trainsize,False);
     names_array =   names_array[data_indices];
     images_array = images_array[data_indices];
     masks_array = masks_array[data_indices];
     contains_nerve_outputs = contains_nerve_outputs[data_indices];

	images_array = (images_array.reshape(-1,image_rows,image_cols,1)-128.0)/255.0; #aplicacion de normalization
     print("Train tensor shape:",images_array.shape);
     print("Train mean:",np.mean(images_array));
     print("Train standard deviation",np.std(images_array));
	masks_array = masks_array.reshape(-1,image_rows,image_cols,1);
	
	return names_array,images_array,masks_array,contains_nerve_outputs;