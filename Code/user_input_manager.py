def retrieve_selection(valid_options, messages):
    try:
        selection_message = messages["SELECTION"]
        invalid_message = messages["INVALID"]
        return_message = messages["RETURN"]
    except Exception as e:
        print("Drop-Down Menu Messages incorrectly configured!", e)
        print("This functionality is yet to be completed!")
        return "E"

    input_received = False
    while not input_received:
        print()
        print(selection_message)  # Initial Selection Message
        for option in valid_options:
            print(option, "->", valid_options[option])
        print(return_message)  # Message to End Execution / Go Back to Previous Menu
        try:
            option_number = input("Enter your Choice: ")
            if option_number in valid_options or option_number.upper() == "E":
                input_received = True
            else:
                print(invalid_message)
        except Exception as e:  # Throws an Exception if the Input Format is Incorrect
            print("Invalid Input Entered! Please try again. Exception:", e)

    return option_number


# Retrieves a Valid Task Number for a particular Phase of the Project
def retrieve_phase_number(valid_phases):
    messages = {
        "SELECTION": "Please select the Phase of the Project:",
        "INVALID": "Invalid Phase Entered!",
        "RETURN": "Press E to Exit Execution"
    }
    return retrieve_selection(valid_phases, messages)


# Retrieves a Valid Task Number for a particular Phase of the Project
def retrieve_task_number(valid_tasks):
    messages = {
        "SELECTION": "Please enter a valid Task Number from the following options:",
        "INVALID": "Invalid Task Number Entered!",
        "RETURN": "Press E to Go Back to Phase Selection Menu"
    }
    return retrieve_selection(valid_tasks, messages)


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

