def fizz_buzz(input):
    if (((input % 3) == 0) and ((input % 5) == 0)):
        return "FizzBuzz"
    elif (input % 3) == 0:
        return "Fizz"
    elif (input % 5) == 0:
        return "Buzz"
    else:
        return f"{input}"


print(fizz_buzz(7))
