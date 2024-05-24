import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.linalg import sqrtm
import sys
from mpi4py import MPI

def Z_out(y, omega, V, mult):

    Va = V[:2, :2]
    Vb = V[:2, 1:]
    Vc = V[1:, 1:]

    I = 0

    def func_PPP(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument))


    def func_PMM(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument))


    def func_PPM(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument))


    def func_PMP(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument))


    def func_MPM(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument))


    def func_MMP(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument))


    def func_MPP(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument))


    def func_MMM(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument))


    if y > 1:
        I += quad(func_PPP, 0, mult)[0]
        I += quad(func_PMM, -mult, 0)[0]
        I += quad(func_MPM, -mult, 0)[0]
        I += quad(func_MMP, 0, mult)[0]

    I += quad(func_PPM, -mult,0)[0]
    I += quad(func_PMP, 0, mult)[0]
    I += quad(func_MPP, 0, mult)[0]
    I += quad(func_MMM, -mult, 0)[0]

    return I


def z1_term(y, omega, V, mult):
    
    Va = V[:2, :2]
    Vb = V[:2, 1:]
    Vc = V[1:, 1:]

    I = 0

    def func_PPP(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument))


    def func_PMM(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument))


    def func_PPM(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument))


    def func_PMP(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument))


    def func_MPM(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return -coeff * np.exp(exponent) * (1+erf(argument))


    def func_MMP(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return -coeff * np.exp(exponent) * (1-erf(argument))


    def func_MPP(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return -coeff * np.exp(exponent) * (1-erf(argument))


    def func_MMM(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return -coeff * np.exp(exponent) * (1+erf(argument))


    if y > 1:
        I += quad(func_PPP, 0, mult)[0]
        I += quad(func_PMM, -mult, 0)[0]
        I += quad(func_MPM, -mult, 0)[0]
        I += quad(func_MMP, 0, mult)[0]

    I += quad(func_PPM, -mult,0)[0]
    I += quad(func_PMP, 0, mult)[0]
    I += quad(func_MPP, 0, mult)[0]
    I += quad(func_MMM, -mult, 0)[0]

    return I


def z2_term(y, omega, V, mult):

    Va = V[:2, :2]
    Vb = V[:2, 1:]
    Vc = V[1:, 1:]

    I = 0

    def func_PPP(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([X, -omega[1], Y])
        b = V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]
        return -coeff*np.exp(exponent)*(1-erf(argument)) * b/V[1,1] + np.exp(-0.5 * vec@V@vec ) / (4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y-1))


    def func_PMM(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([X, -omega[1], Y])
        b = V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]
        return -coeff*np.exp(exponent)*(1+erf(argument)) * b/V[1,1] - np.exp(-0.5 * vec@V@vec )/(4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y-1))


    def func_PPM(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([X, -omega[1], Y])
        b = V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]
        return -coeff*np.exp(exponent)*(1-erf(argument)) * b/V[1,1] + np.exp(-0.5 * vec@V@vec )/(4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y+1))


    def func_PMP(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([X, -omega[1], Y])
        b = V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]
        return -coeff*np.exp(exponent)*(1+erf(argument)) * b/V[1,1] - np.exp(-0.5 * vec@V@vec )/(4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y+1))


    def func_MPM(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([-X, -omega[1], Y])
        b = -V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]
        return coeff*np.exp(exponent)*(1+erf(argument)) * b/V[1,1] + np.exp(-0.5 * vec@V@vec ) / (4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y-1))


    def func_MMP(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([-X, -omega[1], Y])
        b = -V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]
        return coeff*np.exp(exponent)*(1-erf(argument)) * b/V[1,1] - np.exp(-0.5 * vec@V@vec )/(4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y-1))


    def func_MPP(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([-X, -omega[1], Y])
        b = V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]
        return -coeff*np.exp(exponent)*(1-erf(argument)) * b/V[1,1] + np.exp(-0.5 * vec@V@vec )/(4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y+1))


    def func_MMM(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])

        vec = np.array([-X, -omega[1], Y])
        b = V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]
        return -coeff*np.exp(exponent)*(1+erf(argument)) * b/V[1,1] - np.exp(-0.5 * vec@V@vec )/(4*V[1,1]*np.sqrt(2)*np.pi**1.5*np.sqrt(y+1))


    if y > 1:
        I += quad(func_PPP, 0, mult)[0]
        I += quad(func_PMM, -mult, 0)[0]
        I += quad(func_MPM, -mult, 0)[0]
        I += quad(func_MMP, 0, mult)[0]

    I += quad(func_PPM, -mult,0)[0]
    I += quad(func_PMP, 0, mult)[0]
    I += quad(func_MPP, 0, mult)[0]
    I += quad(func_MMM, -mult, 0)[0]
    
    return I


def z3_term(y, omega, V, mult):

    Va = V[:2, :2]
    Vb = V[:2, 1:]
    Vc = V[1:, 1:]

    I = 0

    def func_PPP(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument)) * z


    def func_PMM(z):
        X = np.sqrt(y-1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument)) * z


    def func_PPM(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument)) * z


    def func_PMP(z):
        X = np.sqrt(y+1) - omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) - 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y + V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument)) * z


    def func_MPM(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument)) * z


    def func_MMP(z):
        X = np.sqrt(y-1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y-1)))
        argument = (-V[1,2]*Y + V[0,1]*X + V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument)) * z


    def func_MPP(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1-erf(argument)) * z


    def func_MMM(z):
        X = np.sqrt(y+1) + omega[0]
        Y = z - omega[2]

        exponent = - (X**2*np.linalg.det(Va) + 2*X*Y*np.linalg.det(Vb) + Y**2*np.linalg.det(Vc)) / 2/V[1,1]
        coeff = 1 / (8*np.pi*np.sqrt(V[1,1]*(y+1)))
        argument = (V[1,2]*Y - V[0,1]*X - V[1,1]*omega[1]) / np.sqrt(2 * V[1,1])
        return coeff * np.exp(exponent) * (1+erf(argument)) * z


    if y > 1:
        I += quad(func_PPP, 0, mult)[0]
        I += quad(func_PMM, -mult, 0)[0]
        I += quad(func_MPM, -mult, 0)[0]
        I += quad(func_MMP, 0, mult)[0]

    I += quad(func_PPM, -mult,0)[0]
    I += quad(func_PMP, 0, mult)[0]
    I += quad(func_MPP, 0, mult)[0]
    I += quad(func_MMM, -mult, 0)[0]

    return I


def g_out(y,omega,V, mult = 10, eps = 1e-6):
    V_inv = np.linalg.inv(V + eps*np.eye(3))
    z = np.array([z1_term(y, omega, V_inv, mult), z2_term(y, omega, V_inv, mult), z3_term(y, omega, V_inv, mult)]) / Z_out(y, omega, V_inv, mult)
    return V_inv @ (z - omega)


def Q_func(Q_hat):
    return Q_hat @ np.linalg.inv(np.eye(3) + Q_hat)


def Q_hat_func_MCMC(alpha, Q, samples):
    Q_hat = np.zeros((3,3))
    for _ in range(samples):

        Z = np.random.normal(0,1, 3)
        U = np.random.normal(0,1, 3)

        sqrt_Q = sqrtm(Q)
        sqrt_one_minus_Q = sqrtm(np.eye(3) - Q)
        omega = sqrt_Q @ Z
        preact = sqrt_Q@Z + sqrt_one_minus_Q@U
        y = preact[0]**2 + np.sign( np.prod(preact) ) 
        V = np.eye(3) - Q

        g_out_vec = g_out(y, omega, V)
        Q_hat += alpha * np.outer(g_out_vec, g_out_vec)

    return Q_hat / samples


def main(alpha, Q, samples, iter, damping=.7):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        Q_list = []

    for i in range(iter):
        Q_hat = Q_hat_func_MCMC(alpha, Q, samples)
        
        if rank != 0:
            comm.send(Q_hat, dest=0)
            Q = comm.recv(source=0)
        
        if rank == 0:
            Q_hat_all = np.zeros((size, 3, 3), dtype=np.float64)

            Q_hat_all[0] = Q_hat
            for j in range(1, size):
                Q_hat_all[j] = comm.recv(source=j)

            Q_hat = np.mean(Q_hat_all, axis=0)
            Q = damping*Q_func(Q_hat) + (1-damping)*Q
            Q_list.append(Q)
            np.save(f"data_multi_index_phase_sign/Q_list_alpha_{alpha}_samples_{int(size*samples)}.npy", Q_list)

            Q = (Q + Q.T) / 2
            for j in range(1, size):
                comm.send(Q, dest=j)



        

if __name__=="__main__":
    alpha = float(sys.argv[1])
    iter = int(sys.argv[2])
    samples = int(sys.argv[3])
    Q = np.array([[.5,.0,.0],[.0,.5,.0],[.0,.0,.5]])


    main(alpha, Q, samples, iter)