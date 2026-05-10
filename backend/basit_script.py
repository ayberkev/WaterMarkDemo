# basic_script.py

def greet(name):
    return f"Hello, {name}! Welcome to Python."

if __name__ == "__main__":
    user_name = input("Enter your name: ")
    message = greet(user_name)
    print(message)
