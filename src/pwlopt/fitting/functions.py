from numpy import arctan, cos, exp, float64, pi, sin
from numpy.typing import NDArray
from pyintval import abs as ivabs
from pyintval import atan as ivatan
from pyintval import cos as ivcos
from pyintval import exp as ivexp
from pyintval import sign as ivsign
from pyintval import sin as ivsin
from pyintval import sqrt as ivsqrt


def fun1(p: NDArray[float64]):
    x = p[..., 0]
    y = p[..., 1]
    dx = x - 0.5
    dy = y - 0.5

    r2 = dx**2 + dy**2
    theta = arctan(dy / dx)

    d = 1.0 + 0.3 * sin(5.0 * theta)

    return exp(-5.0 * r2 / d**2)


def ivfun1_gradnorm(x, y):
    dx = x - 0.5
    dy = y - 0.5

    r2 = dx**2 + dy**2
    theta = ivatan(dy / dx)

    sin5 = ivsin(5.0 * theta)
    cos5 = ivcos(5.0 * theta)

    d = 1.0 + 0.3 * sin5

    f = ivexp(-5.0 * r2 / d**2)

    # theta_x = -(y - 0.5) / r^2
    # theta_y =  (x - 0.5) / r^2
    theta_x = -dy / r2
    theta_y = dx / r2

    # d_x and d_y
    d_x = 1.5 * cos5 * theta_x
    d_y = 1.5 * cos5 * theta_y

    # q = r^2 / d^2
    q_x = 2.0 * dx / d**2 - 2.0 * r2 * d_x / d**3
    q_y = 2.0 * dy / d**2 - 2.0 * r2 * d_y / d**3

    return ivsqrt((-5.0 * f * q_x)**2 + (-5.0 * f * q_y)**2)



def fun2(p: NDArray[float64]):
    x = p[..., 0]
    y = p[..., 1]
    a = 6.0 * pi * x + 0.5 * y
    b = 5.0 * pi * y + x

    e1 = exp(
        -10.0 * ((x - 0.4)**2 + (y - 0.3)**2)
    )

    e2 = exp(
        -12.0 * ((x - 0.7)**2 + (y - 0.8)**2)
    )

    return (
        sin(a) * e1
        + cos(b) * e2
        + 0.1 * sin(3.0 * pi * x * y)
    )

def ivfun2_gradnorm(x, y):
    a = 6.0 * pi * x + 0.5 * y
    b = 5.0 * pi * y + x

    e1 = ivexp(
        -10.0 * ((x - 0.4)**2 + (y - 0.3)**2)
    )

    e2 = ivexp(
        -12.0 * ((x - 0.7)**2 + (y - 0.8)**2)
    )

    # First term
    df1_dx = (
        6.0 * pi * ivcos(a)
        - 20.0 * (x - 0.4) * ivsin(a)
    ) * e1

    df1_dy = (
        0.5 * ivcos(a)
        - 20.0 * (y - 0.3) * ivsin(a)
    ) * e1

    # Second term
    df2_dx = (
        -ivsin(b)
        - 24.0 * (x - 0.7) * ivcos(b)
    ) * e2

    df2_dy = (
        -5.0 * pi * ivsin(b)
        - 24.0 * (y - 0.8) * ivcos(b)
    ) * e2

    # Third term
    df3_dx = (
        0.3 * pi * y * ivcos(3.0 * pi * x * y)
    )

    df3_dy = (
        0.3 * pi * x * ivcos(3.0 * pi * x * y)
    )

    return ivsqrt((df1_dx + df2_dx + df3_dx)**2 + (df1_dy + df2_dy + df3_dy)**2)


def fun3(p: NDArray[float64]):
    x = p[..., 0]
    y = p[..., 1]
    c = cos(2.0 * pi * abs(y - 0.5))
    return sin(3.0 * pi * x) * c * (x + y)

def ivfun3_gradnorm(x,y):
    z = y - 0.5
    a = 3.0 * pi * x
    c = ivcos(2.0 * pi * ivabs(z))

    s = ivsin(a)

    df_dx = (
        3.0 * pi * ivcos(a) * c * (x + y)
        + s * c
    )

    # derivative of cos(2*pi*abs(y-.5))
    dc_dy = (
        -2.0 * pi
        * ivsin(2.0 * pi * ivabs(z))
        * ivsign(z)
    )

    df_dy = s * (
        dc_dy * (x + y) + c
    )

    return ivsqrt(df_dx**2 + df_dy**2) 


def fun4(p: NDArray[float64]) -> NDArray[float64]:
    x = p[..., 0]
    y = p[..., 1]

    r2 = x**2 + (y - 0.5)**2

    return sin(50 *( x**2 + (y - 0.5)**2) ** .5) * exp(-10 * r2)

def ivfun4_gradnorm(p):
    x, y = p
    r2 = x**2 + (y - 0.5)**2
    r = ivsqrt(r2)

    return ivexp(-10 * r2) * ivabs(
        50 * ivcos(50 * r) - 20 * r * ivsin(50 * r)
    )


def fun5(p: NDArray[float64]):
    x = p[..., 0]
    y = p[..., 1]

    return (
        sin(5.0 * pi * x) * cos(5.0 * pi * y)
        + 0.5 * sin(10.0 * pi * x * y)
        + 0.2 * cos(15.0 * (x**2 + y**2))
    )

def ivfun5_gradnorm(x, y):
    a = 5.0 * pi * x
    b = 5.0 * pi * y
    c = 10.0 * pi * x * y
    d = 15.0 * (x**2 + y**2)

    df_dx = (
        5.0 * pi * ivcos(a) * ivcos(b)
        + 5.0 * pi * y * ivcos(c)
        - 6.0 * x * ivsin(d)
    )

    df_dy = (
        -5.0 * pi * ivsin(a) * ivsin(b)
        + 5.0 * pi * x * ivcos(c)
        - 6.0 * y * ivsin(d)
    )

    return ivsqrt(df_dx**2 + df_dy**2)
