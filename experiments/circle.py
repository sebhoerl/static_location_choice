import numpy as np
import matplotlib.pyplot as plt

def get_coords(center1, center2, distance1, distance2):
    d = np.sqrt(np.sum((center1 - center2)**2))

    if not (d < abs(distance1 - distance2) or d > distance1 + distance2):
        a = 0.5 * (distance1**2 - distance2**2 + d**2) / d
        h = np.sqrt(distance1**2 - a**2)

        p2 = center1 + a * (center2 - center1) / d

        return [(
                p2[0] + h * (center2[1] - center1[1]) / d,
                p2[1] - h * (center2[0] - center1[0]) / d
            ), (
                p2[0] - h * (center2[1] - center1[1]) / d,
                p2[1] + h * (center2[0] - center1[0]) / d
            )]
    else:
        f = distance1 / (distance1 + distance2)
        return [center1 + (center2 - center1) * f]

center1 = np.array((1.0, 1.0))
center2 = np.array((2.8, 3.6))

center2 = center1

distance1 = 4.4
distance2 = 1.0

coords = get_coords(center1, center2, distance1, distance2)

angle = np.linspace(0, 2 * np.pi, 1000)
x1 = distance1 * np.cos(angle) + center1[0]
y1 = distance1 * np.sin(angle) + center1[1]
x2 = distance2 * np.cos(angle) + center2[0]
y2 = distance2 * np.sin(angle) + center2[1]

plt.figure()

plt.plot(center1[0], center1[1], "or")
plt.plot(center2[0], center2[1], "ob")

plt.plot(x1, y1, "r")
plt.plot(x2, y2, "b")

for coord in coords:
    plt.plot(coord[0], coord[1], "ok")

plt.show()

#
