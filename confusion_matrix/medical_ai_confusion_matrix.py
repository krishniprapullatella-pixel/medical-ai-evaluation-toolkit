
tp = int(input("Enter True Positives: "))
tn = int(input("Enter True Negatives: "))
fp = int(input("Enter False Positives: "))
fn = int(input("Enter False Negatives: "))

if tp < 0 or tn < 0 or fp < 0 or fn < 0:
    print("ERROR! Values cannot be negative.")

else:
    total = tp + tn + fp + fn

    if total == 0:
        print("ERROR! Total images cannot be zero.")

    else:
        accuracy = ((tp + tn) / total) * 100

        # PRECISION
        if (tp + fp) == 0:
            precision = None
        else:
            precision = (tp / (tp + fp)) * 100

        # RECALL
        if (tp + fn) == 0:
            recall = None
        else:
            recall = (tp / (tp + fn)) * 100

        # SPECIFICITY
        if (tn + fp) == 0:
            specificity = None
        else:
            specificity = (tn / (tn + fp)) * 100

        # F1-SCORE
        if precision is None or recall is None:
            f1_score = None
        elif (precision + recall) == 0:
            f1_score = None
        else:
            f1_score = (2 * precision * recall) / (precision + recall)

        # MEDICAL AI REPORT
        print("\nMEDICAL AI MODEL REPORT")
        print("-----------------------")
        print("Total images:", total)
        print("True Positives:", tp)
        print("True Negatives:", tn)
        print("False Positives:", fp)
        print("False Negatives:", fn)

        print(f"Accuracy: {accuracy:.2f}%")

        if precision is None:
            print("Precision: Undefined")
        else:
            print(f"Precision: {precision:.2f}%")

        if recall is None:
            print("Recall: Undefined")
        else:
            print(f"Recall: {recall:.2f}%")

        if specificity is None:
            print("Specificity: Undefined")
        else:
            print(f"Specificity: {specificity:.2f}%")

        if f1_score is None:
            print("F1-score: Undefined")
        else:
            print(f"F1-score: {f1_score:.2f}%")
