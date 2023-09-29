from phase_one import phase_one_driver
from phase_two import phase_two_driver
from phase_three import phase_three_driver
from user_input_manager import retrieve_phase_number

PHASES = {
    "1": "Phase 1",
    "2": "Phase 2",
    "3": "Phase 3"
}

PHASE_DRIVERS = {
    "Phase 1": phase_one_driver,
    "Phase 2": phase_two_driver,
    "Phase 3": phase_three_driver
}


def phase_manager(directories, data):
    print("Starting Project Phase Manager...")
    exec_complete = False
    while not exec_complete:
        phase_number = retrieve_phase_number(PHASES)
        if phase_number.lower() == "e":
            print("Exiting Phase Selection Menu...")
            exec_complete = True
        else:
            exec_complete = PHASE_DRIVERS[PHASES[phase_number]](directories, data)
    print("Project Phase Manager Execution Complete!")
