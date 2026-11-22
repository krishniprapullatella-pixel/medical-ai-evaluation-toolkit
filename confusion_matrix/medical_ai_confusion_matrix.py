tp = int(input("Enter True Positives: ")) # correctly identified diseased
tn = int(input("Enter True Negatives: ")) # correctly healthy people
fp = int(input("Enter False Positives: ")) # incorrectly diseased people 
fn = int(input("Enter False Negatives: ")) # missed diseased cases

if(tn < 0 or tp < 0 or fp < 0 or fn < 0):
    print("ERROR! Values are incorrect")
else:
    total = tp + tn + fp + fn
    
    if(total == 0):
        print("ERROR! Values are incorrect")
    else:
        accuracy = ((tp + tn) / total) * 100
        if(tp + fp == 0):
            print("ERROR! Precision undefined")
        else:
             precision = (tp / (tp + fp)) * 100   # how many actually have the disease

        print("\nMEDICAL AI MODEL REPORT")
        print("-----------------------")
        print("Total images:", total)
        print("True Positives:", tp)
        print("True Negatives:", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}%")
        