try:
    tp = int(input("Enter the tp value: "))
    if tp < 0:
        print("ERROR: Value cannot be negative.")

    else:
        print("Valid input:", tp)
except ValueError:
    print("Error! Wrong value")