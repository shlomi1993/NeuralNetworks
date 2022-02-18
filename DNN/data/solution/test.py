def final_test():
       
    file1 = open("test_y_my_best", "r")
    my_answers = file1.read()
    file1.close()
    my_answers = my_answers.split("\n")[:-1]
    
    file2 = open("test_y_answers", "r")
    right_answers = file2.read()
    file2.close()
    right_answers = right_answers.split("\n")[:-1]
        
    total = len(right_answers)
    if len(my_answers) != total:
        print("Wrong number of answers")
        return
    
    correct = 0
    for i in range(total):
        if my_answers[i] == right_answers[i]:
            correct += 1

    print(f"Result: {correct}/{total} ({round((100 * correct) / total, 4)}%)")
    return correct / total


final_test()
