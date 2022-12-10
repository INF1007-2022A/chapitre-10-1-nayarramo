#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import scipy as sp
import scipy.interpolate as interp
from scipy.integrate import quad
from math import sqrt
import matplotlib.pyplot as plt
import random



# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(start=-1.3, stop=2.5, num=64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([(np.sqrt(x[0]**2 + x[1]**2), np.arctan2(x[1], x[0])) for x in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    difference = np.empty(len(values))
    for i in range(len(values)):
        difference[i] = abs(values[i] - number)

    minimum = difference.argmin()
    return minimum

def f(x):
    return x * np.sin(1/(x**2)) + x

def graph_sin():
    xp = np.linspace(-1, 1, 250)
    yp = f(xp)
    plot = interp.interp1d(xp, yp)
    xplot = np.arange(-1, 1, 10)
    yplot = plot(xplot)
    plt.plot(xplot, yplot, xp, yp)
    plt.show()
    

def monte_carlo():
  
    circle_points = 0
    square_points = 0
 
    for _ in range(1000000):
        
        # Random uniformaly generated x and y from -1 to 1
        rand_x = random.uniform(-1, 1)
        rand_y = random.uniform(-1, 1)
 
        # Distance from origin
        origin_dist = rand_x**2 + rand_y**2
 
        # Checking if point in circle
        if origin_dist <= 1:
            circle_points += 1
        square_points += 1

    pi = 4* circle_points / square_points
    return pi

def g(x):
    return np.e**(-(x**2))

def integrale():
    integral = quad(g, np.NINF, np.inf)
    return integral

def graph_integrale():
    # integral = quad(g, -4 , 4)
    xplot = np.linspace(-5, 5, 1000)
    plt.plot(xplot, g(xplot), color= 'black')
    plt.axhline(color= 'black')
    plt.fill_between(xplot, g(xplot), where= [{x>-4} and {x<4} for x in xplot], color= 'green')
    plt.show()

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print("The value of pi estimated using monte carlo method is", monte_carlo())
    graph_sin()
    graph_integrale()
    
