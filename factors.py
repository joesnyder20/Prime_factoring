#    The purpose of this progam is to demonstrate different methods to factor, as well as graphing the times it takes,
#    to demoonstrate the difficulty of factoring large numbers
#
#    Author: Joe Snyder


# 1. Brute Force
# 2. Pollards p-1
# 3. Sympy Factoring
# 4. Ellpitic curve

import math
import random
import time
import threading
import matplotlib.pyplot as plt
from sympy import factorint
import gmpy2
from gmpy2 import mpz
from queue import Queue
import numpy as np


# =========================================
#     Helper Functions
# =========================================

thread_lock = threading.Lock()  # Lock for sync between threads 
timeout = False  # Flag to indicate if the function times out
event = threading.Event()
exec_time = [0, 0, 0,0]
result_queue = Queue()

def calculate_k(probability):
    k = math.ceil(math.log2(1/probability))
    return k


def miller_rabin(n, k):
    if n == 2 or n == 3:
        return True

    if n <= 1 or n % 2 == 0:
        return False

    # Find r and d such that n = 2^r * d + 1
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Witness loop
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_large_prime(start, end, k):
    while True:
        p = random.randint(start, end)
        if miller_rabin(p, k):
            return p

def generate_composite(length, k):
    # Convert length from decimal digits to equivalent binary bits 
    length_bits = math.ceil(length * math.log2(10))

    # approximately length_bits-bit prime numbers
    start = 1 << (length_bits - 1)
    end = (1 << length_bits) - 1

    p1 = generate_large_prime(start, end, k)
    p2 = generate_large_prime(start, end, k)

    return p1 * p2

# Thread functions
def run_brute_force():
    global timeout 
    global exec_time # access the execution time variable
    start_time = time.time()
    result = brute_force_factorization(number)
    if time.time() - start_time > 60:
        timeout = True
    with thread_lock:
        if result[0] is not None:
            exec_time[0] = result[1] # record the execution time
            result_queue.put(('Brute Force',result))
    event.set()

def run_pollards_p_minus_1():
    global exec_time # access the execution time variable
    result = pollards_p_minus_1(number)
    with thread_lock:
        
        exec_time[1] = result[1] # record the execution time
        result_queue.put(('Pollards',result))
    event.set()

def run_sympy():
    global exec_time # access the execution time variable
    result = factorize_sympy(number)
    with thread_lock:
        exec_time[2] = result[1] # record the execution time
        result_queue.put(('Sympy',result))
    event.set()

def run_ecc():
    global exec_time # access the execution time variable
    result = full_factorize_ecm(number)
    with thread_lock:
        exec_time[3] = result[1] # record the execution time
        result_queue.put(('ECC',result))

    event.set()



def factorize_concurrently(number):
    
    thread_brute_force = threading.Thread(target=run_brute_force)
    thread_pollards_p_minus_1 = threading.Thread(target=run_pollards_p_minus_1)
    thread_factorize_sympy = threading.Thread(target=run_sympy)
    thread_factorize_ecc = threading.Thread(target=run_ecc)
    



    thread_brute_force.setDaemon = True # I dont really understand this but will later
    thread_pollards_p_minus_1.setDaemon = True
    thread_factorize_sympy.setDaemon = True
    thread_factorize_ecc.setDaemon = True
    

    thread_brute_force.start()
    thread_pollards_p_minus_1.start()
    thread_factorize_sympy.start()
    thread_factorize_ecc.start()
    

   
    thread_brute_force.join(timeout=45) 
    thread_pollards_p_minus_1.join(timeout=45)
    thread_factorize_sympy.join(timeout=45) 
    thread_factorize_ecc.join(timeout=45)

    while not result_queue.empty():
        method_name, result = result_queue.get()
        print(f"---------{method_name}-----------\n")
        print("Prime Factors:", result[0])
        print("Time Taken:", result[1], "seconds\n")
        
    

    # Code to check if threads are still alive
    if thread_brute_force.is_alive() or thread_pollards_p_minus_1.is_alive() or thread_factorize_sympy.is_alive() or thread_factorize_ecc.is_alive():
        timeout = True
    else:
        timeout = False

    if timeout:
        print('One of the algorithms was unsuccessful or slow.')
    else:
        print('All methods were successful.')

    algorithm_names = ["Brute Force", "Pollard's p-1", "SymPy Method", "ECC"]
    create_bar_graph(algorithm_names, exec_time)

def create_bar_graph(algorithm_names, exec_time):
    bars = plt.bar(algorithm_names, exec_time)

    diff = max(exec_time) - min(exec_time)
    plt.yticks(np.arange(min(exec_time), max(exec_time) + diff / 10, diff / 50))  

    plt.xlabel('Algorithm')
    plt.ylabel('Time of Execution (seconds)')
    num_length = len(str(number))
    plt.title(f'Factored Number: {number:,} (Length: {num_length})')



    for idx, rect in enumerate(bars):  # iterate over the bars
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(exec_time[idx]), ha='center', va='bottom')

    plt.show()

# def generate_number(length: str):
    
#     length = int(length)
    
#     lower_limit = 10 ** (length - 1)
#     upper_limit = 10 ** length - 1
#     return random.randint(lower_limit, upper_limit)


# ==================================================
# 1. Brute Force
# ==================================================

def is_prime(num):
    if num < 2:
        return False
    
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    
    return True
    
def brute_force_factorization(n,max_length=19): #tends to time out after this 
    if len(str(n)) > max_length:  # check the length of the number
        print("Number is too large for brute force method.\n")
        return None, 0 , 0
    factors = []
    prime_factors = []
    counter = 0

    start_time = time.time()

    for i in range(2, int(math.sqrt(n)) + 1):
        counter += 1
        while n % i == 0:
            factors.append(i)
            if is_prime(i):
                prime_factors.append(i)
            n //= i
            

    if n > 1:
        factors.append(n)
        if is_prime(n):
            prime_factors.append(n)



    elapsed_time = time.time() - start_time


    return prime_factors, elapsed_time
   


# 17 digit number starts to get execution times about 5 seconds
# 18 digit number is about 16
# 19 takes 49


#===============================================================
# 2. Pollard p-1 Algorithum
#===============================================================

def pollards_p_minus_1(n):
    smoothness_bound = int(math.log2(n) ** 2)  #determines the range of potential factors to consider
    def gcd(a, b):
        while b != 0:
            a, b = b, a % b
        return a

    def pollards_p_minus_1_factorization():
        a = 2
        counter = 0

        for p in range(2, smoothness_bound):
            a = pow(a, p, n)  # a**p mod n
            counter += 1
            factor = gcd(a - 1, n)   #most of the time, this will return 1 or n, we live for the time it doesnt
            if 1 < factor < n:
                return factor, counter
        
        return None, counter

    start_time = time.time()
    prime_factors = []
    attempts = 0

    while True:
        factor, counter = pollards_p_minus_1_factorization()
        attempts += counter
        if factor is None:  
            break
        if is_prime(factor):
            prime_factors.append(factor)
        else:
            # Get prime factors of the composite factor
            composite_prime_factors = brute_force_factorization(factor)[0]
            prime_factors.extend(composite_prime_factors)
        n //= factor

    time_taken = time.time() - start_time
    

    return prime_factors, time_taken


# ==================================
# 3. Sympy factoring
# =================================

def factorize_sympy(n):
    start_time = time.time()

    factors = factorint(n)

    end_time = time.time()
    elapsed_time = end_time - start_time



    factors_only = list(factors.keys())
    
    return factors_only, elapsed_time


# ===================================
# 4. Ellitic Curve
# ===================================

def elliptic_add(p1, p2, A, n):
    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        lam = ((3 * x1 * x1 + A) * gmpy2.invert(2 * y1, n)) % n
    else:
        lam = ((y2 - y1) * gmpy2.invert(x2 - x1, n)) % n

    x3 = (lam * lam - x1 - x2) % n
    y3 = ((x1 - x3) * lam - y1) % n
    return (x3, y3)

def lenstra_ecm(n, iterations=1000):
    for _ in range(iterations):
        a = mpz(random.randint(0, n - 1))
        x = mpz(random.randint(0, n - 1))
        y = mpz(random.randint(0, n - 1))
        curve_a = (y*y - x*x*x - a*x) % n
        point = (x, y)

        for _ in range(iterations):
            try:
                point = elliptic_add(point, point, curve_a, n)
            except ZeroDivisionError:
                factor = math.gcd(point[0] - point[1], n)
                if factor != 1 and factor != n:
                    return factor, n // factor 

    return None

def full_factorize_ecm(n):
    factors = []

    # creating a helper function for recursion

    start_time = time.time()
    def factor_helper(num):
        if gmpy2.is_prime(num):    # base case, append the prime number and return
            factors.append(num)
            return
        else:
            result = lenstra_ecm(num)  # if not prime, get a factor
            if result:
                factor, cofactor = result
                factor_helper(factor)  # recursively factorize the factor
                factor_helper(cofactor)  # recursively factorize the cofactor

    # calling the helper function to start factorization
    factor_helper(n)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return factors, elapsed_time


# ===============================
#  Main Execution 
# ===============================


prob = input('Enter Probability (recommended 1% for time, but can enter decimals): ')
prob_k = calculate_k(float(prob))

length_p = input('Enter length of p: ')
print('q will be roughly the same length...\n')
length_p_int = int(length_p)

number = generate_composite(length_p_int, prob_k)

print(number)

factorize_concurrently(number)


# 19 is limit for Brute Force
# 35 executes fine for all
#60 runs
#Synpy timed out at 100 one time, 4 seconds anohter
#130 runs
#150 runs
#170 45 seconds for pollards, 7 for ecc but low factors, sympy timed out
#200 pollards takes 70 seconds, ECC runs but trivial factor so not really


