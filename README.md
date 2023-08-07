Factors.py will generate two prime numbers, p & q, and multiply them in to create a large composite number N. It will then run four different algorithms and compare the different times it takes to factor the number in a bar graph. 

It will first ask you the desired probability that a generated prime number is not prime, by way of the Miller-Rabid Primality test. 

It will then ask for a desired length of p, and it will generate a prime number q that is about the same length. 




factors_comparison.py is basically factors.py, except it asks for a lower bound of p and an upper bound. It will run facotrs.py through that range, and show how as the length of p and q increase, the length of time to factor the number increase. 



==================================================
1. Brute Force


Brute Force is the most inefficient method for factoring a number, 
but it is the easiest to understand


General Idea: We will take are large number n, and find the square root, then round up 1.
This provides us our range of numbers that we will brute force to get our factors


====================================================

2. Pollard p-1 Algorithm


Unique Prime Factorization - This states that all numbers have a unique combination of prime factors
Suppose we have a large composite number N, we find two large primes, p, and q that are both factors of N 

This is sometimes referred to as the Monte Carlo Method

for lack of better syntax, %= will refer to 'congruent to 

Pollards p-1 Algorithm, not to be confused with Pollards Rho Algorithm
abuses Fermants little theory, which states:

Given gcd(a,p) = 1 then a**(p-1) %= 1 mod p     where p is prime

supp0se p-1 is a factor of some number M 

M = (p-1) * k 

then a**m = (a**p-1)**k %= 1 mod p 

Therefore, is p divided by a**m -1, then p also divides n

The trick is to find an M to give a nontrivial factor, i.e not 1 or N itself

So how do we do that? Great question 

First, let's discuss what a 'smooth number" is

All a smooth number is a number that can be expressed as small primes. 

ex: 20 = 2**2 * 5     30 = 2**1 * 3**1 * 5**1

utilizing smooth numbers will help us make our algorithm more efficient 
and eliminate "unnecessary" steps. Some may disagree with the usage of 
unnecessary, rather we can find numbers that are far more likely to yield a factor

it is somewhat typical to set a = 2 to satisfy gcd(a,p) = 1

in my opinion, the most confusing part about this is the smoothness_bound function
Choosing a smoothness bound involves finding a balance between efficiency and likelihood
of finding factors within a reasonable range

in general, a larger smoothness bound allows for the consideration of larger prime factors,
but requires more computational resources and may make the algo slower

a common method is to set the bound off of the number being factored, and it
works well for the script kitties cause they don't need to understand it!

in this case, I used log^2(n) as the smoothness bound, material to why this is
and other potential bounds you could set based on what you are doing will be attached

============================

3. Sympy factoring


This uses several methods it would seem. It is written entirely in Python according to a quick Google search.


===========================

4. Elliptic Curve

Also known as Lenstra's Algorithm
1. Choose a random integer d and an elliptic curve E. For each integer d, there is a corresponding elliptic curve E given by the equation y² = x³ + ax + b, where a and b are chosen such that 4a³ + 27b² ≠ 0 (mod n).
2. Choose a random point P on the curve E.
3. Compute the multiple [m]P = P + P + ... + P (m times), for some suitably large m.
 	▪	If at any point in the computation of [m]P, you find a non-trivial root of unity r in the ring Z/nZ (which can occur if you find that the denominator of a rational number is not invertible modulo n), then gcd(r,n) is a non-trivial factor of n.
 	▪	If no root of unity is found (which can occur if the group of points on E over F_p is cyclic of order q that is smooth), then return to step 1 and choose a new curve and point.
The success of ECM depends on the fact that randomly chosen elliptic curves will have a high probability of having smooth order over the ring of integers Z/nZ.
This method can be faster than Pollard's p-1 method and Pollard's rho method for integers that have a factor p such that the largest prime factor of p-1 is small, but is a bit more complicated to implement.
