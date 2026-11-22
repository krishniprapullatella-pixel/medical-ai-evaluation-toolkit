correct_predictions = int(input("Enter the number of correct predictions: "))
total_images = int(input("Enter the total number of images: "))
if(total_images <= 0):
    print("ERROR! TOTAL IMAGES MUST BE GREATER THAN ZERO")
elif(correct_predictions > total_images or correct_predictions < 0):
    print("ERROR! INVALID NUMBER OF CORRECT PREDICTIONS")
else:
    incorrect_predictions = total_images - correct_predictions
    accuracy = (correct_predictions / total_images) * 100


    print("\nMEDICAL AI MODEL REPORT")
    print("-----------------------")
    print("Total images:", total_images)
    print("Correct predictions:", correct_predictions)
    print("Incorrect predictions:", incorrect_predictions)
    print(f"Accuracy: {accuracy:.2f}%")