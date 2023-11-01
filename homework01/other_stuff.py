import numpy as np

def meows(n):
    calls = 0
    prints = 1
    while calls < n:
        yield "meow " * prints
        calls += 1
        prints *= 2

if __name__ == "__main__":
    # task 2
    print("\n\r#######################################################")

    liste =  [i*i for i in range(101)]
    print(liste)
    liste_even = [i * i for i in range(101) if i % 2 == 0]
    print(liste_even)

    #task 3

    print("\n\r#######################################################") # CATS

    for i in meows(6):
        print(i)

    # task 4
    arr = np.random.normal(0,1,25)

    print(arr)

    for i in range(len(arr)):
        if arr[i] < 0.09:
            arr[i] = 42
        else:
            arr[i] = pow(arr[i], 2)
            
    arr = arr.reshape((5,5))
    print(arr)
    print(arr[:,3])

    # task 5