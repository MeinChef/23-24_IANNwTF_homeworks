from cat import Cat

def meows(n):
    calls = 0
    while calls < n:
        yield "meow " * calls * 2
        calls += 1

if __name__ == "__main__":

    a = Cat("Kitty")
    c = Cat("Katthew")
    print(a.meow())
    print(a.meow_at(c))

    # task 2
    print("\n\r#######################################################")

    liste =  [i*i for i in range(101)]
    print(liste)

    #task 3

    print("\n\r#######################################################")

    for i in meows(10):
        print(i)
