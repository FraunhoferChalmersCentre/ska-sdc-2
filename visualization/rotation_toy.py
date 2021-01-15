import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x_var = np.random.uniform(0, 3)
y_var = np.random.uniform(0, 3)
correlation = np.random.uniform(-1, 1)
true_cov_mat = np.array([[x_var, correlation * np.sqrt(x_var * y_var)], [correlation * np.sqrt(x_var * y_var), y_var]])
rv = multivariate_normal([0, 0], cov=true_cov_mat)

im_size = 21
im_limit = 5
x, y = np.meshgrid(np.linspace(-im_limit, im_limit, im_size), np.linspace(-im_limit, im_limit, im_size))
pos = np.dstack((x, y))

densities = rv.pdf(pos)

image = np.array([densities* i / 100 for i in range(100)])

center = int((im_size + 1) / 2 - 1)
vertical_indices = np.array([(center - i, 0) for i in range(im_size)]).astype(int)
vertical_line = vertical_indices + center
angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
rotated = np.array([rotation_matrix @ v.T for v in vertical_indices]).astype(int)
rotated = rotated + center

fig, axes = plt.subplots(3, 1)
axes[0].imshow(densities, cmap=plt.cm.gray)
axes[0].plot([vertical_line[0, 0], vertical_line[-1, 0]], [vertical_line[0, 1], vertical_line[-1, 1]])
axes[0].plot([rotated[0, 0], rotated[-1, 0]], [rotated[0, 1], rotated[-1, 1]])

axes[1].imshow(image[:, vertical_line[:, 0], vertical_line[:, 1]].T)

axes[2].imshow(image[:, rotated[:, 0], rotated[:, 1]].T)
