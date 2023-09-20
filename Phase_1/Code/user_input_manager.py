# Retrieves a Valid Task Number for Phase 1 of the Project
def retrieve_task_number(valid_tasks):
    input_received = False
    while not input_received:
        print()
        print("Please enter a valid Task Number from the following options:")
        for task in valid_tasks:
            print(task, "->", valid_tasks[task])
        try:
            task_number = int(input("Enter your Choice: "))
            if task_number in valid_tasks:
                input_received = True
            else:
                print("Invalid Task Number Entered!")
        except Exception as e:  # Throws an Exception if
            print("Invalid Input Entered! Please try again. Exception:", e)
    return task_number


# Retrieve a Mandatory Image-ID based on the Dataset Size
def retrieve_input_image_id(last_dataset_idx):
    input_received = False
    while not input_received:
        try:
            image_id = \
                int(input("Enter a Valid Image ID for CalTech101 Dataset [0 - {}]: ".format(last_dataset_idx)))
            if 0 <= image_id <= last_dataset_idx:
                input_received = True
            else:
                print("Image ID out of Range! Please try again.")
        except Exception as e:
            print("Invalid Input Entered! Please try again. Exception:", e)
    return image_id


# Retrieves a Valid 'k' Value
def retrieve_k_value():
    input_received = False
    while not input_received:
        try:
            k = int(input("Enter a 'k' value: "))
            input_received = True
        except Exception as e:
            print("Invalid Input Entered! Please try again. Exception:", e)
    return k


# Warning Message - with a Yes/No resolution from the user
def display_warning(message):
    input_received = False
    while not input_received:
        warning = str(input(message + " Do you still wish to continue? (y/n) "))
        if warning.lower() in ["y", "n"]:
            response = True if warning == "y" else False
            input_received = True
        else:
            input_received = False
    return response

