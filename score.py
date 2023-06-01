def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        numbers = [list(map(int, line.strip().split())) for line in lines]
    return numbers

def calculate_percentage(file_a, file_b):
    numbers_a = read_file(file_a)
    numbers_b = read_file(file_b)
    total_numbers = 0
    correct_numbers = 0

    for i in range(min(len(numbers_a), len(numbers_b))):
        row_a = numbers_a[i]
        row_b = numbers_b[i]
        for j in range(min(len(row_a), len(row_b))):
            total_numbers += 1
            if row_a[j] == row_b[j]:
                correct_numbers += 1

    percentage = (correct_numbers / total_numbers) * 100
    return percentage

file_a = 'accuracy.txt'
file_b = 'myScore.txt'
percentage = calculate_percentage(file_a, file_b)
print(f"Percentage of correct numbers printed: {percentage:.2f}%")