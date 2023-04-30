# print("hello world 😺")
# print("hype " * 10)
# hello = "hello world 😺"

# variable1 = False

# print(variable1)


# news = " heyya whats \\\\ up"
# print(news)


# formatted_string = f"{variable1} {news}"
# print(formatted_string)

# print(hello.title())


# x = input("x: ")
# print(type(x))
# y = float(x) + 1
# print(type(y))

# name = "Bartholomew"
# print(name[-2:4])


# count = 0
# for number in range(1, 10):
#     if (number % 2) == 0:
#         print(number)
#         count += 1

# print(f"We have {count} even numbers")


def test(a):
    a = a+1
    b = a*100
    c = a + b
    return a, b, (c-1000)


a = 1
print(a)
a, b, c = test(a)
print(a)
print(b)
print(c)
