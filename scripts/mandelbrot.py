import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z * z + c
        n += 1
    return n

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1, r2, np.array([[mandelbrot(complex(r, i), max_iter) for r in r1] for i in r2]))

def display_fractal(xmin, xmax, ymin, ymax, width=800, height=800, max_iter=256):
    dpi = 80
    img_width = dpi * width // 100
    img_height = dpi * height // 100
    plt.figure(figsize=(width / 100, height / 100), dpi=dpi)
    ticks = np.arange(0, img_width, 3 * dpi)
    x_ticks = xmin + (xmax - xmin) * ticks / img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax - ymin) * ticks / img_width
    plt.yticks(ticks, y_ticks)
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.title("Mandelbrot Set")
    r1, r2, m_set = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter)
    plt.imshow(m_set.T, extent=[xmin, xmax, ymin, ymax], cmap='twilight_shifted', origin='lower', aspect='auto')
    plt.show()

# Display the Mandelbrot fractal
display_fractal(-2.0, 1.0, -1.5, 1.5)

