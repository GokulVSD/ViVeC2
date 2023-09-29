README File for CSE 515 - Multimedia & Web Databases Course Project - Phase 1 Execution

1. About Phase 1 of the Project



2. Directory Structure
	- <root_dir>
		- Code
			- Datasets (will automatically be created when main.py is executed for the first time)
			- caltech101_dataset_feature_descriptors.pkl (Pickle File containing Feature Descriptor Information for Caltech101 Dataset)
			- color_moments.py
			- data_utils.py
			- distance_measures.py
			- feature_descriptors.py
			- gen_utils.py
			- hog.py
			- image_comparator.py
			- main.py
			- resnet_fd.py
			- task_manager.py
			- user_input_manager.py
		- Outputs (Stores the Resultant Outputs for Task 3 of the Phase 1 Specifications) [example format provided below]
			- <image_id> directory
				- <feature_descriptor> directory (expands for each feature descriptor)
					- <resulting images in png format>
				- <source input image in png format>
		- Report
			- Final Phase 1 Report in PDF Format

3. Running the Codebase
	Step 1: Install Python 3.9.0 and respective Pip (Python Package Installer & Manager).
	Step 2: Navigate to the Code directory (as mentioned above) & open a designated terminal. 
	Step 3: Run the following command to install the necessary dependencies for the Project:
		pip install -r requirements.txt
	Step 4: Start the Main Driver Program using the following command
		Windows Users: python main.py
		Linux/Mac Users: python3 main.py
	Step 5: You will be prompted to choose a particular task as designated by the Task Specification Document. Choose a valid Task Number.
		- Task 1: Visualizes the Input Image and Displays the Feature Descriptors for a valid Input Image ID.
		- Task 2: Creates a Pickle File containing the Feature Descriptors for the entire Dataset (Caltech101 Dataset). {Please note that this will take a significant amount of 			  time for completion}
		- Task 3: Displays the Top "k" Images & Distance Scores based on each Feature Descriptor (based on a hard-coded Distance Measure selected for each Descriptor).
			  The Resultant Images are stored in the Outputs folder (in the format depicted above) rank-wise. Each Image is titled with its respective Image ID and Distance 			  Score with respect to the Input Image.
	
	Important Note: Images with only a Single Color Channel DO NOT Produce any Feature Descriptors (Task 1) or resultant Image Comparisons (Task 3), and are omitted from the feature			descriptors pickle file (Task 2).