import timeit

# Define a function to square a number
def square(x):
    return x ** 2

# Create a large list of numbers
my_list = list(range(1000000))  # A list containing numbers from 0 to 999,999

# Using List Comprehension
list_comp_time = timeit.timeit('[square(x) for x in my_list]', globals=globals(), number=100)

# Using map() function
map_time = timeit.timeit('list(map(square, my_list))', globals=globals(), number=100)

print(f"List Comprehension Time: {list_comp_time} seconds")
print(f"map() Function Time: {map_time} seconds")
