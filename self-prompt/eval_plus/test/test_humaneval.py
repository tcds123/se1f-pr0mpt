METADATA = {}

def prime_fib1(n: int) -> int:
    a, b = 0, 1
    count = 0
    while count < n:
        a, b = b, a + b
        if is_prime1(b):
            count += 1
    return b


def is_prime1(num: int) -> bool:
    if num <= 1:
        return False
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True


def prime_fib2(n: int) -> int:
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def fib(n: int) -> int:
        if n == 1:
            return 0
        elif n == 2:
            return 1
        else:
            return fib(n-1) + fib(n-2)

    temp = 0
    count = 0
    num = 1
    while count < n:
        temp = fib(num)
        if is_prime(temp):
            count += 1
        num += 1
    return temp

def prime_fib3(n: int) -> int:
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def fib(n: int) -> int:
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fib(n - 1) + fib(n - 2)

    temp = 0
    count = 0
    num = 1
    while count < n:
        temp = fib(num)
        if is_prime(temp):
            count += 1
        num += 1

    return temp

def prime_fib4(n: int) -> int:
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    a, b = 0, 1
    count = 0
    while count < n:
        a, b = b, a + b
        if is_prime(b):
            count += 1
    return b

def prime_fib5(n: int) -> int:
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
    """
    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def fib(n: int) -> int:
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return fib(n - 1) + fib(n - 2)

    temp = 0
    count = 0
    num = 1

    while count < n:
        temp = fib(num)
        if is_prime(temp):
            count += 1
        num += 1

    return temp


def prime_fib6(n: int) -> int:

    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    a, b = 0, 1
    count = 0
    while count < n:
        a, b = b, a + b
        if is_prime(b):
            count += 1

    return b


def prime_fib7(n: int) -> int:

    def is_prime(num: int) -> bool:
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True

    def fib(n: int) -> int:
        if n == 1:
            return 0
        elif n == 2:
            return 1
        else:
            return fib(n - 1) + fib(n - 2)

    temp = 0
    count = 0
    for i in range(1, n + 1):
        if is_prime(fib(i)):
            temp += fib(i)
            count += 1

    return temp


def check(candidate):
    assert candidate(1) == 2
    assert candidate(2) == 3
    assert candidate(3) == 5
    assert candidate(4) == 13
    assert candidate(5) == 89
    assert candidate(6) == 233
    assert candidate(7) == 1597
    assert candidate(8) == 28657
    assert candidate(9) == 514229
    assert candidate(10) == 433494437

if __name__ == "__main__":
    fun1 = prime_fib7
    check(fun1)
