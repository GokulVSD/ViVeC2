import util as util
import PIL


def task0a(caltechDB, resNetModel):
    pass


def task0b(caltechDB, resNetModel, featureDB):
    # given an (even or odd numbered) imageID or an image file, a user selected feature space, positive integer k,
    inputOption = input("Do you want to enter imageID [A] or Image File Full Path [B]: ").lower()
    if inputOption == "A":
        imageID = input("Enter Image ID")
        image = caltechDB[int(imageID)][0]
    elif inputOption == "B":
        imagePath = input("Enter Image Path")
        try:
            # Reference: https://www.tutorialspoint.com/python_pillow/python_pillow_using_image_module.htm
            image = PIL.Image.open(imagePath)
        except:
            print("Invalid Path")
            exit()
    else:
        print("Error")
        exit()
    featureSpace = input("Enter Feature Space: ")
    k = abs(int(input("Enter K for Similar Images: ")))

    # identifies and visualizes the most similar k images, along with their scores, under the selected feature space


def main():
    task = input("Please Enter Task (0a): ").lower()
    print("User Entered: " + task)

    # Create Caltech database and ResNet model to be used in all tasks
    caltechDB = torchvision.datasets.Caltech101("./", download=True)
    resNetModel = torchvision.models.resnet50(progress=True, weights=torchvision.models.ResNet50_Weights.DEFAULT)  # pretrained is not supported
    resNetModel.zero_grad(set_to_none=True)
    resNetModel.eval()

    if task == "0a":
        task0a(caltechDB, resNetModel)
    elif task == "0b":
        task0b(caltechDB, resNetModel, featureDB)
    elif task == "1":
        task1()
    elif task == "2a":
        task2a()
    elif task == "2b":
        task2b()
    elif task == "3":
        task3()
    elif task == "4":
        task4()
    elif task == "5":
        task5()
    elif task == "6":
        task6()
    elif task == "7":
        task7()
    elif task == "8":
        task8()
    elif task == "9":
        task9()
    elif task == "10":
        task10()
    elif task == "11":
        task11()
    else:
        print("Error")
        exit()


# To test a task, just call function here:
# task0a():
# Otherwise just call main()
# main():
